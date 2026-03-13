from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # load pretrained model (downloads automatically)

results = model.train(
    data='PlantDoc/dataset.yaml',
    epochs=80,
    imgsz=224,
    batch=10,
    name='my_experiment'
)

print(results)