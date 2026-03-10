/*
 * inference.cpp
 * =============
 * TensorFlow Lite Micro inference wrapper for ESP32-S3.
 */

#include "inference.h"

#include <string.h>
#include <math.h>
#include <algorithm>        // ← fixes std::min

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_system.h"     // ← fixes esp_get_free_heap_size()
#include "driver/temperature_sensor.h"

// TFLite-Micro headers (provided by the tensorflow-lite-micro component)
// Note: micro_error_reporter.h was removed in TFLite Micro >= 1.0
// Use micro_log.h instead — logging is now handled via MicroPrintf
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char* TAG = "inference";

// ── TFLite globals ────────────────────────────────────────────────────────────
// Adjust tensor arena size if you get OOM errors (try 200*1024 for larger models)
constexpr size_t kTensorArenaSize = 150 * 1024;
static uint8_t* tensor_arena = nullptr;

static tflite::MicroInterpreter*    interpreter  = nullptr;
static const tflite::Model*         model        = nullptr;

// ── Temperature sensor ────────────────────────────────────────────────────────
static temperature_sensor_handle_t temp_handle = nullptr;

static void init_temp_sensor() {
    temperature_sensor_config_t cfg = TEMPERATURE_SENSOR_CONFIG_DEFAULT(10, 80);
    esp_err_t err = temperature_sensor_install(&cfg, &temp_handle);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "Temperature sensor init failed: %s", esp_err_to_name(err));
        temp_handle = nullptr;
        return;
    }
    temperature_sensor_enable(temp_handle);
}

static float read_temp() {
    if (!temp_handle) return -1.0f;
    float t = -1.0f;
    temperature_sensor_get_celsius(temp_handle, &t);
    return t;
}

// ── Energy estimation ─────────────────────────────────────────────────────────
// Typical active current for ESP32-S3 at 240 MHz during inference
static const float CURRENT_MA   = 180.0f;
static int64_t     energy_start = 0;   // µs timestamp

static float elapsed_energy_mah(int64_t duration_us) {
    float hours = duration_us / 3600000000.0f;
    return CURRENT_MA * hours;
}

// ─────────────────────────────────────────────────────────────────────────────
// PUBLIC API
// ─────────────────────────────────────────────────────────────────────────────

bool inference_init(const uint8_t* model_data, size_t model_len) {
    ESP_LOGI(TAG, "Initialising TFLite-Micro ...");

    // Allocate tensor arena in SPIRAM if available, else internal RAM
    tensor_arena = (uint8_t*) heap_caps_malloc(
        kTensorArenaSize,
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
    );
    if (!tensor_arena) {
        ESP_LOGW(TAG, "SPIRAM alloc failed, trying internal RAM");
        tensor_arena = (uint8_t*) malloc(kTensorArenaSize);
    }
    if (!tensor_arena) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena (%zu bytes)", kTensorArenaSize);
        return false;
    }

    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema mismatch: got %u, expected %d",
                 model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }

    // ── Register only the ops your model needs ────────────────────────────
    // Add or remove ops to match your model. Common ones listed below.
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddReshape();
    resolver.AddSoftmax();
    resolver.AddMaxPool2D();
    resolver.AddAveragePool2D();
    resolver.AddAdd();
    resolver.AddMul();
    resolver.AddQuantize();
    // resolver.AddDequantize();  // uncomment if your model needs it

    interpreter = new tflite::MicroInterpreter(
        model, resolver, tensor_arena, kTensorArenaSize
    );

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed");
        return false;
    }

    TfLiteTensor* input  = interpreter->input(0);
    TfLiteTensor* output = interpreter->output(0);

    ESP_LOGI(TAG, "Input  shape: [%d, %d, %d, %d]  type=%d",
             input->dims->data[0], input->dims->data[1],
             input->dims->data[2], input->dims->data[3],
             input->type);
    ESP_LOGI(TAG, "Output shape: [%d, %d]  type=%d",
             output->dims->data[0], output->dims->data[1],
             output->type);
    ESP_LOGI(TAG, "Tensor arena used: %zu bytes", interpreter->arena_used_bytes());

    init_temp_sensor();
    energy_start = esp_timer_get_time();

    ESP_LOGI(TAG, "✓ Inference engine ready");
    return true;
}


InferenceResult inference_run(const uint8_t* image, size_t image_size) {
    InferenceResult result = {};

    TfLiteTensor* input = interpreter->input(0);

    // Copy image bytes into model input tensor
    if (image_size != (size_t) input->bytes) {
        ESP_LOGW(TAG, "Image size mismatch: got %zu, expected %zu",
                 image_size, (size_t) input->bytes);
    }
    memcpy(input->data.uint8, image, std::min(image_size, (size_t) input->bytes));

    // ── Run inference ─────────────────────────────────────────────────────
    int64_t t_start = esp_timer_get_time();

    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Invoke() failed");
        return result;
    }

    int64_t t_end      = esp_timer_get_time();
    int64_t duration_us = t_end - t_start;
    result.latency_ms  = duration_us / 1000.0f;

    // ── Read output ───────────────────────────────────────────────────────
    TfLiteTensor* output    = interpreter->output(0);
    int           n_classes = output->dims->data[output->dims->size - 1];

    int   best_class = 0;
    float best_score = -1e9f;

    for (int i = 0; i < n_classes; i++) {
        float score;
        if (output->type == kTfLiteUInt8) {
            // Dequantize: real_value = scale * (q - zero_point)
            score = output->params.scale *
                    ((float) output->data.uint8[i] - output->params.zero_point);
        } else if (output->type == kTfLiteInt8) {
            score = output->params.scale *
                    ((float) output->data.int8[i] - output->params.zero_point);
        } else {
            score = output->data.f[i];
        }
        if (score > best_score) {
            best_score = score;
            best_class = i;
        }
    }

    result.predicted_class = best_class;
    result.confidence      = best_score;

    // ── System metrics ────────────────────────────────────────────────────
    result.temp_c      = read_temp();
    result.ram_free_kb = (int)(esp_get_free_heap_size()            / 1024);
    result.ram_used_kb = (int)((heap_caps_get_total_size(MALLOC_CAP_DEFAULT)
                                - esp_get_free_heap_size())        / 1024);
    result.energy_mah  = elapsed_energy_mah(t_end - energy_start);

    return result;
}


void inference_deinit() {
    if (interpreter) { delete interpreter; interpreter = nullptr; }
    if (tensor_arena) { free(tensor_arena); tensor_arena = nullptr; }
    if (temp_handle)  {
        temperature_sensor_disable(temp_handle);
        temperature_sensor_uninstall(temp_handle);
        temp_handle = nullptr;
    }
}