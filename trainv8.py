from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # load pretrained model (downloads automatically)

results = model.train(
    data='PlantDoc/dataset.yaml',
    epochs=80,
    imgsz=416,
    batch=10,
    name='my_experiment'
)

