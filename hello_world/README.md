# ESP32-S3 Inference Benchmark — ESP-IDF

## Project structure

```
esp32_inference/
├── CMakeLists.txt
├── sdkconfig.defaults
├── main/
│   ├── CMakeLists.txt
│   ├── main.cpp              ← entry point
│   ├── inference.cpp/h       ← TFLite engine
│   ├── serial_protocol.cpp/h ← UART comms with PC
│   ├── model_data.cpp/h      ← embedded model bytes
└── components/
    └── tensorflow-lite-micro/  ← see step 3 below
```

---

## Step 1 — Convert your model to a C array

Run this once on your PC from the folder containing `model.tflite`:

```bash
# Linux / Mac
xxd -i model.tflite > main/model_data.cpp

# Then open model_data.cpp and rename:
#   unsigned char model_tflite[]     → const unsigned char g_model_data[]
#   unsigned int model_tflite_len    → const unsigned int  g_model_data_len
# Also add at top: #include "model_data.h"

# Windows — use the Python script below
python convert_model.py
```

### convert_model.py (Windows helper)
```python
data = open('model.tflite', 'rb').read()
with open('main/model_data.cpp', 'w') as f:
    f.write('#include "model_data.h"\n\n')
    f.write('const unsigned char g_model_data[] = {\n  ')
    f.write(', '.join(f'0x{b:02x}' for b in data))
    f.write(f'\n}};\nconst unsigned int g_model_data_len = {len(data)};\n')
print(f"Done: {len(data)} bytes")
```

---

## Step 2 — Set up ESP-IDF

```bash
# Install ESP-IDF v5.x if not already done
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
./install.sh esp32s3
source ./export.sh     # run this every new terminal session
```

---

## Step 3 — Add TensorFlow Lite Micro component

```bash
cd esp32_inference/components
git clone https://github.com/espressif/esp-tflite-micro.git tensorflow-lite-micro
```

---

## Step 4 — Set target and build

```bash
cd esp32_inference
idf.py set-target esp32s3
idf.py build
```

---

## Step 5 — Flash

```bash
# Hold BOOT, press RESET, release RESET, release BOOT
idf.py -p /dev/ttyACM0 flash monitor
```

---

## Step 6 — Run from PC

```bash
pip install pyserial Pillow
python pc_sender.py
```

Edit `pc_sender.py` to point to your images folder:
```python
IMAGES_FOLDER = "./images"
IMAGE_SIZE    = (64, 64)   # must match IMAGE_W/IMAGE_H in main.cpp
PORT          = None       # auto-detects, or set e.g. "COM3"
```

---

## Adjusting for your model

In `main.cpp`:
```c
#define IMAGE_W        64   // ← change to your model input width
#define IMAGE_H        64   // ← change to your model input height
#define IMAGE_CHANNELS 3    // ← 3 for RGB, 1 for grayscale
```

In `inference.cpp`, add or remove ops to match your model:
```cpp
resolver.AddConv2D();
resolver.AddDepthwiseConv2D();
resolver.AddFullyConnected();
// etc.
```

In `sdkconfig.defaults`, adjust tensor arena if you get OOM:
```
kTensorArenaSize = 150 * 1024   // try 200*1024 for larger models
```
