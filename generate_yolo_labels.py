"""
Script YOLO - Génération de fichiers labels
Crée un fichier .txt par image avec : class_id 0 0 width height
"""

import os
from PIL import Image

# Mapping dossiers -> numéro de classe
FOLDERS = [
    "Bell_Pepper_Leaf",       # 0
    "Bell_Pepper_Spot",        # 1
    "Potato_Early",           # 2
    "Potato_Late",            # 3
    "Tomato_Mosaic",          # 4
    "Tomato_Septoria_Spot",    # 5
]

# Extensions d'images supportées
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ⚙️ Chemin racine de votre dataset (modifiez ce chemin)
ROOT_DIR = "./PlantDoc/416x416/contexts"

def generate_labels(root_dir):
    total_created = 0
    total_skipped = 0

    for class_id, folder_name in enumerate(FOLDERS):
        folder_path = os.path.join(root_dir, folder_name)

        if not os.path.isdir(folder_path):
            print(f"⚠️  Dossier introuvable : {folder_path}")
            continue

        images = [
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ]

        if not images:
            print(f"⚠️  Aucune image dans : {folder_name}")
            continue

        for img_file in images:
            img_path = os.path.join(folder_path, img_file)
            base_name = os.path.splitext(img_file)[0]
            txt_path = os.path.join(folder_path, base_name + ".txt")

            try:
                with Image.open(img_path) as img:
                    width, height = img.size

                # Format : class_id 0.5 0.5 1 1
                label_content = f"{class_id} 0.5 0.5 1 1\n"

                with open(txt_path, "w") as f:
                    f.write(label_content)

                print(f"✅ [{class_id:2d}] {folder_name}/{base_name}.txt  ({width}x{height})")
                total_created += 1

            except Exception as e:
                print(f"❌ Erreur sur {img_file} : {e}")
                total_skipped += 1

    print(f"\n{'='*50}")
    print(f"✅ Fichiers créés  : {total_created}")
    print(f"❌ Fichiers ignorés : {total_skipped}")
    print(f"{'='*50}")


if __name__ == "__main__":
    generate_labels(ROOT_DIR)
