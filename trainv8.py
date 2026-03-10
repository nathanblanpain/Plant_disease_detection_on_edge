from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # load pretrained model (downloads automatically)

results = model.train(
    data='PlantDoc\dataset.yaml',
    epochs=100,
    imgsz=224,
    batch=10,
    name='my_experiment'
)

print(results)