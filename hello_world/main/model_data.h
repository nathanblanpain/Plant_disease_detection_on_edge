#pragma once
#include <stddef.h>

// The model is embedded at compile time from model.tflite
// Generate model_data.cpp with:
//   xxd -i model.tflite > model_data.cpp
// Then rename the array and length to match below.

extern const unsigned char g_model_data[];
extern const unsigned int  g_model_data_len;
