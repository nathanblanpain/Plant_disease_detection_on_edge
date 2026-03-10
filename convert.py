data = open('yolov8n_int8.tflite', 'rb').read()
with open('hello_world/main/model_data.cpp', 'w') as f:
    f.write('#include "model_data.h"\n\n')
    f.write('const unsigned char g_model_data[] = {\n  ')
    f.write(', '.join(f'0x{b:02x}' for b in data))
    f.write(f'\n}};\nconst unsigned int g_model_data_len = {len(data)};\n')
print(f"Done: {len(data)} bytes")