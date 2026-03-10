"""
pc_sender.py
============
Run this on your PC. Sends images one by one to the ESP32 over serial,
then receives and prints the benchmark results.

Install dependencies:
    pip install pyserial Pillow
"""

import serial
import serial.tools.list_ports
import struct
import time
import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────────────────────────────────────
IMAGES_FOLDER = r"C:/Users/Utilisateur/Documents/travail/Suede/IoT/datasets/PlantVillage/224x224/test/Tomato_Yellow"
IMAGE_SIZE    = (224, 224)   # must match IMAGE_W/IMAGE_H in main.cpp
BAUD_RATE     = 115200       # must match CONFIG_CONSOLE_UART_BAUDRATE in sdkconfig
PORT          = "COM14"        # None could auto detect in theory


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-DETECT PORT
# ─────────────────────────────────────────────────────────────────────────────
def find_esp32_port():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        desc = (p.description or "").lower()
        if any(k in desc for k in ["esp32", "cp210", "ch340", "uart", "usb serial"]):
            print(f"  [Serial] Auto-detected: {p.device} ({p.description})")
            return p.device
    # fallback: just return first available port
    if ports:
        print(f"  [Serial] Using first available port: {ports[0].device}")
        return ports[0].device
    raise Exception("No serial port found. Connect your ESP32 and try again.")


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def load_and_prepare(image_path):
    """
    Load any image (JPG/PNG/BMP), resize to IMAGE_SIZE, return raw RGB bytes.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Run: pip install Pillow")

    img = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    return bytes(img.tobytes())  # flat RGB bytes, length = W*H*3


def get_image_files(folder):
    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted([
        f for f in Path(folder).iterdir()
        if f.suffix.lower() in supported
    ])
    if not files:
        raise Exception(f"No images found in '{folder}'")
    return files


# ─────────────────────────────────────────────────────────────────────────────
# PROTOCOL
#
# PC  → ESP32 :  "IMG"  + 4-byte length (big-endian) + raw RGB bytes
# ESP32 → PC  :  "OK/n" when ready for next image
# ESP32 → PC  :  result lines ending with "DONE/n" after last image
# ─────────────────────────────────────────────────────────────────────────────
def send_image(ser, img_bytes, index, total):
    length = len(img_bytes)
    header = b"IMG" + struct.pack(">I", length)

    ser.write(header)
    ser.write(img_bytes)
    ser.flush()
    print(f"  [{index+1}/{total}] Sent {length} bytes", end="", flush=True)

    # Wait for ESP32 acknowledgement
    ack = ser.readline().decode(errors="replace").strip()
    if ack == "OK":
        print(f"  → ACK received")
    else:
        print(f"  → Unexpected response: '{ack}'")


def receive_results(ser):
    """Read all result lines until ESP32 sends 'DONE'."""
    print("\n── Results from ESP32 ──────────────────────────────────────────")
    lines = []
    while True:
        line = ser.readline().decode(errors="replace").strip()
        if not line:
            continue
        if line == "DONE":
            break
        print(f"  {line}")
        lines.append(line)
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    port = PORT or find_esp32_port()
    images = get_image_files(IMAGES_FOLDER)

    print("=" * 55)
    print("  PC Image Sender → ESP32-S3")
    print(f"  Port     : {port} @ {BAUD_RATE} baud")
    print(f"  Images   : {len(images)} files in '{IMAGES_FOLDER}'")
    print(f"  Img size : {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]} RGB")
    print("=" * 55)

    with serial.Serial(port, BAUD_RATE, timeout=30) as ser:
        time.sleep(2)  # wait for ESP32 to boot
        ser.reset_input_buffer()
        time.sleep(0.5)  # give it a moment after reset
        # Wait for ESP32 to signal it's ready
                # Wait for ESP32 to signal it's ready
        print("\n  Waiting for ESP32 to be ready …")
        ready_found = False
        start_time = time.time()
        while time.time() - start_time < 10:  # 10 second timeout
            line = ser.readline().decode(errors="replace").strip()
            if line == "READY":
                print("  ✓ ESP32 is ready\n")
                ready_found = True
                break
            if line:
                print(f"  ESP32: {line}")
        if not ready_found:
            print("[ERROR] ESP32 did not send 'READY' within 10 seconds. Check connection and firmware.")
            return

        # Send total image count first
        ser.write(struct.pack(">I", len(images)))
        ser.flush()

        # Send images one by one
        for i, img_path in enumerate(images):
            print(f"\n  Image: {img_path.name}")
            try:
                img_bytes = load_and_prepare(img_path)
                send_image(ser, img_bytes, i, len(images))
            except Exception as e:
                print(f"  [ERROR] {e}")
                break

        # Receive all results
        results = receive_results(ser)

    print("\n" + "=" * 55)
    print("  Transfer complete.")
    print("=" * 55)


if __name__ == "__main__":
    main()