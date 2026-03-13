from ultralytics import YOLO

# Load a standard YOLO26 model
model = YOLO("runs/detect/my_experiment2/weights/best.pt")

# Export to TFLite format with INT8 quantization
# The 'int8' argument triggers Post-Training Quantization
# 'data' provides the calibration dataset needed for mapping values
model.export(format="tflite", int8=True, data="PlantDoc/dataset.yaml")
