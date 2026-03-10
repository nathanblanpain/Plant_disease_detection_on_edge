"""
Corrige les fichiers labels YOLO :
- Format correct : class_id 0.5 0.5 1.0 1.0 (boîte pleine image, normalisée)
- Parcourt labels/train/ et labels/val/
"""

import os

# ⚙️ MODIFIEZ CE CHEMIN
LABELS_BASE_DIR = r"C:\Users\Utilisateur\Documents\travail\Suede\IoT\datasets\PlantDoc\416x416\label"

SPLITS = ["train", "test"]  # ou ["train", "test"] selon votre structure

# -------------------------------------------------------

def fix_labels(labels_base_dir):
    total_fixed = 0
    total_skipped = 0

    for split in SPLITS:
        split_dir = os.path.join(labels_base_dir, split)

        if not os.path.isdir(split_dir):
            print(f"⚠️  Dossier introuvable : {split_dir}")
            continue

        txt_files = [f for f in os.listdir(split_dir) if f.endswith(".txt")]

        for txt_file in txt_files:
            txt_path = os.path.join(split_dir, txt_file)

            try:
                with open(txt_path, "r") as f:
                    lines = f.readlines()

                new_lines = []
                changed = False

                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        new_lines.append(line)
                        continue

                    class_id = parts[0]
                    cx, cy, w, h = parts[1], parts[2], parts[3], parts[4]

                    # Si les valeurs ne sont pas normalisées (> 1.0), corriger
                    if float(cx) != 0.5 or float(cy) != 0.5 or float(w) != 1.0 or float(h) != 1.0:
                        new_lines.append(f"{class_id} 0.5 0.5 1.0 1.0\n")
                        changed = True
                    else:
                        new_lines.append(line)

                if changed:
                    with open(txt_path, "w") as f:
                        f.writelines(new_lines)
                    total_fixed += 1
                else:
                    total_skipped += 1

            except Exception as e:
                print(f"❌ Erreur sur {txt_file} : {e}")

        print(f"✅ [{split}] {len(txt_files)} fichiers traités")

    print(f"\n{'='*50}")
    print(f"✅ Corrigés  : {total_fixed}")
    print(f"⏭️  Déjà OK  : {total_skipped}")
    print(f"{'='*50}")
    print(f"\n⚠️  N'oubliez pas de supprimer les .cache avant de relancer :")
    print(f"   del \"{os.path.join(labels_base_dir, 'train.cache')}\"")
    print(f"   del \"{os.path.join(labels_base_dir, 'val.cache')}\"")

if __name__ == "__main__":
    fix_labels(LABELS_BASE_DIR)
