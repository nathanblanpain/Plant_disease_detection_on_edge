import os
import shutil
import random

resolution_folder = "PlantDoc/416x416"
val_ratio = 0.20

train_images = os.path.join(resolution_folder, "train", "images")
train_labels = os.path.join(resolution_folder, "train", "labels")
val_images = os.path.join(resolution_folder, "val", "images")
val_labels = os.path.join(resolution_folder, "val", "labels")
os.makedirs(val_images, exist_ok=True)
os.makedirs(val_labels, exist_ok=True)

# Chemins complets des images
all_images = [
    os.path.join(root, f)
    for root, dirs, files in os.walk(train_images)
    for f in files
    if f.lower().endswith((".png", ".jpg"))
]
print(len(all_images), "images trouvées dans le dossier train/images.")

num_val = int(len(all_images) * val_ratio)
val_images_list = random.sample(all_images, num_val)

for img_path in val_images_list:
    # Chemin relatif depuis train_images (ex: "Potato_Early/image.jpg")
    rel_path = os.path.relpath(img_path, train_images)

    # Image
    src_img = img_path
    dst_img = os.path.join(val_images, rel_path)
    os.makedirs(os.path.dirname(dst_img), exist_ok=True)
    shutil.move(src_img, dst_img)

    # Label correspondant
    rel_label = os.path.splitext(rel_path)[0] + ".txt"
    src_label = os.path.join(train_labels, rel_label)
    dst_label = os.path.join(val_labels, rel_label)
    if os.path.exists(src_label):
        os.makedirs(os.path.dirname(dst_label), exist_ok=True)
        shutil.move(src_label, dst_label)

print(f"{len(val_images_list)} images déplacées dans val ({val_ratio*100:.0f}%).")