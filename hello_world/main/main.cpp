/*
 * main.cpp
 * ========
 * ESP32-S3 Inference Benchmark — ESP-IDF entry point.
 *
 * Flow:
 *   1. Init UART + TFLite
 *   2. Signal PC "READY", wait for "START"
 *   3. Receive image count
 *   4. For each image: receive → infer → send result
 *   5. Send CSV summary → "DONE"
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "driver/uart.h"

#include "inference.h"
#include "serial_protocol.h"
#include "model_data.h"

static const char* TAG = "main";

// ── Configuration ─────────────────────────────────────────────────────────────
#define BAUD_RATE       115200
#define IMAGE_W         224
#define IMAGE_H         224
#define IMAGE_CHANNELS  3
#define IMAGE_BYTES     (IMAGE_W * IMAGE_H * IMAGE_CHANNELS)  // 12288

// Maximum number of images to process in one session
#define MAX_IMAGES      500


// ─────────────────────────────────────────────────────────────────────────────
extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "==============================================");
    ESP_LOGI(TAG, "  ESP32-S3 Inference Benchmark — ESP-IDF");
    ESP_LOGI(TAG, "==============================================");

    // ── 1. Init serial ───────────────────────────────────────────────────────
    serial_init(BAUD_RATE);
    serial_log("==============================================");
    serial_log("  ESP32-S3 Inference Benchmark — ESP-IDF");
    serial_log("  Image size : %dx%d RGB (%d bytes)", IMAGE_W, IMAGE_H, IMAGE_BYTES);
    serial_log("  Free heap  : %u KB", esp_get_free_heap_size() / 1024);
    serial_log("==============================================");

    // ── 2. Load and init TFLite model ────────────────────────────────────────
    serial_log("[Init] Loading model (%u bytes) ...", g_model_data_len);
    if (!inference_init(g_model_data, g_model_data_len)) {
        serial_log("[ERROR] Failed to init inference engine. Halting.");
        while (1) vTaskDelay(pdMS_TO_TICKS(1000));
    }
    serial_log("[Init] ✓ Model loaded");

    // ── 3. Allocate image buffer (prefer SPIRAM) ──────────────────────────────
    uint8_t* img_buf = (uint8_t*) heap_caps_malloc(
        IMAGE_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );
    if (!img_buf) {
        img_buf = (uint8_t*) malloc(IMAGE_BYTES);
    }
    if (!img_buf) {
        serial_log("[ERROR] Cannot allocate image buffer (%d bytes)", IMAGE_BYTES);
        while (1) vTaskDelay(pdMS_TO_TICKS(1000));
    }

    // ── 4. Allocate results array ─────────────────────────────────────────────
    InferenceResult* results = (InferenceResult*) calloc(MAX_IMAGES, sizeof(InferenceResult));
    if (!results) {
        serial_log("[ERROR] Cannot allocate results array");
        free(img_buf);
        while (1) vTaskDelay(pdMS_TO_TICKS(1000));
    }

    // ── 5. Handshake with PC ──────────────────────────────────────────────────
    serial_wait_for_start();

    // Receive total image count (4-byte big-endian)
    uint8_t count_bytes[4];
    uart_read_bytes(UART_NUM_0, count_bytes, 4, pdMS_TO_TICKS(5000));
    int num_images = ((int)count_bytes[0] << 24) |
                     ((int)count_bytes[1] << 16) |
                     ((int)count_bytes[2] << 8)  |
                      (int)count_bytes[3];

    if (num_images <= 0 || num_images > MAX_IMAGES) {
        serial_log("[ERROR] Invalid image count: %d (max %d)", num_images, MAX_IMAGES);
        serial_send_done();
        goto cleanup;
    }
    serial_log("[Info] Expecting %d images", num_images);

    // ── 6. Process images one by one ─────────────────────────────────────────
    for (int i = 0; i < num_images; i++) {
        size_t received = serial_receive_image(img_buf, IMAGE_BYTES);
        if (received == 0) {
            serial_log("[ERROR] Failed to receive image %d", i + 1);
            break;
        }

        InferenceResult r = inference_run(img_buf, received);
        results[i] = r;

        serial_send_result(i + 1, num_images, r);
    }

    // ── 7. Send summary CSV + DONE ────────────────────────────────────────────
    serial_send_csv(results, num_images);
    serial_send_done();

cleanup:
    free(img_buf);
    free(results);
    inference_deinit();

    serial_log("[Done] Session complete. Reset board to run again.");
    ESP_LOGI(TAG, "Session complete.");
}
