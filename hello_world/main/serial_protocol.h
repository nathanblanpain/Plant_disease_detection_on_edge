#pragma once
#include <stdint.h>
#include <stddef.h>
#include "inference.h"

// Initialise UART
void serial_init(int baud_rate);

// Send a log line to PC (adds \n automatically)
void serial_log(const char* fmt, ...);

// Block until PC sends "START\n"
void serial_wait_for_start();

// Read one image from PC over serial
// Protocol: "IMG" + 4-byte big-endian length + raw RGB bytes
// Returns number of bytes read, 0 on error
size_t serial_receive_image(uint8_t* buf, size_t buf_size);

// Send inference result back to PC
void serial_send_result(int index, int total, const InferenceResult& r);

// Send final CSV summary
void serial_send_csv(const InferenceResult* results, int count);

// Send "DONE\n" to signal end of session
void serial_send_done();
