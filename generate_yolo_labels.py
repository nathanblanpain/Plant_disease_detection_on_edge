"""
Script YOLO - Génération de fichiers labels
Crée un fichier .txt par image avec : class_id 0 0 width height
"""

import os
from PIL import Image

# Mapping dossiers -> numéro de classe
FOLDERS = [
    "Bell_Pepper_leaf",       # 0
    "Bell_Pepper_spot",        # 1
    "Potato_early",           # 2
    "Potato_late",            # 3
    "Tomato_Bacteria_Spot",   # 4
    "Tomato_Early",           # 5
    "Tomato_Late",            # 6
    "Tomato_Leaf",            # 7
    "Tomato_mold",            # 8
    "Tomato_Mosaic",          # 9
    "Tomato_Septoria_Spot",    # 10
    "Tomato_Yellow",          # 11
]

# Extensions d'images supportées
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ⚙️ Chemin racine de votre dataset (modifiez ce chemin)
ROOT_DIR = ".\Plantdoc\\320x320\Train"

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

                # Format : class_id 0 0 width height
                label_content = f"{class_id} 0 0 {width} {height}\n"

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
