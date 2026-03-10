from ultralytics import YOLO

# Charger TON modèle entraîné
model = YOLO("runs/detect/my_yolo11_run/weights/best.pt")

# Analyser une image
results = model.predict("datasets/PlantVillage/224x224/test/Bell_Pepper_Leaf/002f87b7-e1a5-49e5-a422-bb423630ded5___JR_HL 8068.JPG")

# Afficher les résultats
results[0].show()  # Ouvre une fenêtre avec les détections