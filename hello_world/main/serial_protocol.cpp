/*
 * serial_protocol.cpp
 * ===================
 * UART communication between ESP32-S3 and PC.
 *
 * Protocol (PC → ESP32):
 *   1. PC sends "START\n"
 *   2. PC sends image count as 4-byte big-endian uint32
 *   3. For each image:
 *        "IMG" + 4-byte big-endian length + raw RGB bytes
 *
 * Protocol (ESP32 → PC):
 *   - Log lines at any time
 *   - "OK\n"   after each image received
 *   - "RESULT,id,class,confidence,latency_ms,temp_c,ram_free_kb,energy_mah\n"
 *   - CSV block + "DONE\n" at end
 */

#include "serial_protocol.h"

#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#include "driver/uart.h"
#include "driver/gpio.h"   // ← fixes GPIO_NUM_43 / GPIO_NUM_44 not declared
#include "esp_log.h"

static const char* TAG = "serial";

#define UART_NUM        UART_NUM_0
#define UART_TX_PIN     43   // ESP32-S3 default TX (use int, not GPIO_NUM_x)
#define UART_RX_PIN     44   // ESP32-S3 default RX
#define UART_BUF_SIZE   (4 * 1024)

// ─────────────────────────────────────────────────────────────────────────────

void serial_init(int baud_rate) {
    uart_config_t cfg = {
        .baud_rate           = baud_rate,
        .data_bits           = UART_DATA_8_BITS,
        .parity              = UART_PARITY_DISABLE,
        .stop_bits           = UART_STOP_BITS_1,
        .flow_ctrl           = UART_HW_FLOWCTRL_DISABLE,
        .rx_flow_ctrl_thresh = 0,
        .source_clk          = UART_SCLK_DEFAULT,
        .flags               = { 0 },
    };
    uart_param_config(UART_NUM, &cfg);
    uart_set_pin(UART_NUM, UART_TX_PIN, UART_RX_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    uart_driver_install(UART_NUM, UART_BUF_SIZE, UART_BUF_SIZE, 0, nullptr, 0);
    ESP_LOGI(TAG, "UART%d ready at %d baud", UART_NUM, baud_rate);
}


// ── Helpers ──────────────────────────────────────────────────────────────────

static void uart_write_str(const char* s) {
    uart_write_bytes(UART_NUM, s, strlen(s));
}

static int uart_read_exact(uint8_t* buf, size_t n, int timeout_ms = 10000) {
    size_t received = 0;
    while (received < n) {
        int r = uart_read_bytes(UART_NUM, buf + received, n - received, pdMS_TO_TICKS(timeout_ms));
        if (r <= 0) {
            ESP_LOGE(TAG, "uart_read_exact timeout (%zu/%zu bytes)", received, n);
            return -1;
        }
        received += r;
    }
    return (int) received;
}

static int uart_read_line(char* buf, size_t max_len, int timeout_ms = 10000) {
    size_t i = 0;
    while (i < max_len - 1) {
        uint8_t c;
        int r = uart_read_bytes(UART_NUM, &c, 1, pdMS_TO_TICKS(timeout_ms));
        if (r <= 0) return -1;
        if (c == '\n') break;
        if (c != '\r') buf[i++] = c;
    }
    buf[i] = '\0';
    return (int) i;
}


// ── Public API ───────────────────────────────────────────────────────────────

void serial_log(const char* fmt, ...) {
    char buf[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    uart_write_str(buf);
    uart_write_str("\n");
}


void serial_wait_for_start() {
    serial_log("READY");
    char line[32];
    while (true) {
        int n = uart_read_line(line, sizeof(line));
        if (n > 0 && strcmp(line, "START") == 0) {
            serial_log("[ESP32] Received START");
            return;
        }
    }
}


size_t serial_receive_image(uint8_t* buf, size_t buf_size) {
    // Wait for "IMG" header
    uint8_t hdr[3] = {};
    while (true) {
        if (uart_read_exact(hdr, 3) < 0) return 0;
        if (memcmp(hdr, "IMG", 3) == 0) break;
        // Re-sync: shift one byte and try again
        hdr[0] = hdr[1];
        hdr[1] = hdr[2];
        if (uart_read_exact(&hdr[2], 1) < 0) return 0;
    }

    // Read 4-byte big-endian length
    uint8_t len_bytes[4];
    if (uart_read_exact(len_bytes, 4) < 0) return 0;
    uint32_t length = ((uint32_t)len_bytes[0] << 24) |
                      ((uint32_t)len_bytes[1] << 16) |
                      ((uint32_t)len_bytes[2] << 8)  |
                       (uint32_t)len_bytes[3];

    if (length > buf_size) {
        ESP_LOGE(TAG, "Image too large: %u bytes (buffer: %zu)", length, buf_size);
        return 0;
    }

    // Read image data
    if (uart_read_exact(buf, length) < 0) return 0;

    uart_write_str("OK\n");
    return length;
}


void serial_send_result(int index, int total, const InferenceResult& r) {
    char buf[128];
    snprintf(buf, sizeof(buf),
        "RESULT,%d,%d,%.4f,%.3f,%.1f,%d,%.8f",
        index,
        r.predicted_class,
        r.confidence,
        r.latency_ms,
        r.temp_c,
        r.ram_free_kb,
        r.energy_mah
    );
    uart_write_str(buf);
    uart_write_str("\n");

    // Also print human-readable progress
    char progress[128];
    snprintf(progress, sizeof(progress),
        "[%d/%d] class=%d conf=%.3f latency=%.2fms temp=%.1fC ram_free=%dKB",
        index, total,
        r.predicted_class, r.confidence,
        r.latency_ms, r.temp_c, r.ram_free_kb
    );
    uart_write_str(progress);
    uart_write_str("\n");
}


void serial_send_csv(const InferenceResult* results, int count) {
    uart_write_str("── CSV ──────────────────────────────────────────────────\n");
    uart_write_str("id,predicted_class,confidence,latency_ms,temp_c,ram_free_kb,energy_mah\n");

    for (int i = 0; i < count; i++) {
        const InferenceResult& r = results[i];
        char row[128];
        snprintf(row, sizeof(row),
            "%d,%d,%.4f,%.3f,%.1f,%d,%.8f\n",
            i + 1,
            r.predicted_class,
            r.confidence,
            r.latency_ms,
            r.temp_c,
            r.ram_free_kb,
            r.energy_mah
        );
        uart_write_str(row);
    }
}


void serial_send_done() {
    uart_write_str("DONE\n");
}