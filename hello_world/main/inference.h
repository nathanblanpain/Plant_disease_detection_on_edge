#pragma once
#include <stdint.h>
#include <stddef.h>

// ── Result returned after each inference ─────────────────────────────────────
struct InferenceResult {
    int     predicted_class;    // index of highest-confidence output
    float   confidence;         // confidence score 0.0–1.0
    float   latency_ms;         // inference time in milliseconds
    float   temp_c;             // chip temperature at inference time
    int     ram_free_kb;        // free heap after inference
    int     ram_used_kb;        // used heap after inference
    float   energy_mah;         // estimated energy consumed (mAh)
};

// Call once at startup
bool inference_init(const uint8_t* model_data, size_t model_len);

// Run one inference on a flat RGB image (H x W x 3, uint8)
// image_size = total bytes = H * W * 3
InferenceResult inference_run(const uint8_t* image, size_t image_size);

// Free TFLite resources
void inference_deinit();
