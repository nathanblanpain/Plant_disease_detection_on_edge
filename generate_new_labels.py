import os
import shutil

new_images_dir = "224/contexts/"
old_images_dir = "224/test"
old_labels_dir = "224/test"
new_labels_dir = "224/contexts/"  # where to save the copied labels

os.makedirs(new_labels_dir, exist_ok=True)

new_images = [f for f in os.listdir(new_images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
old_labels = [f for f in os.listdir(old_labels_dir) if f.endswith('.txt')]

matched = 0
unmatched = []

for new_img in new_images:
    new_stem = os.path.splitext(new_img)[0]  # e.g. "cam1_frame_0042"
    
    # Find an old label whose stem is contained in the new image name
    found = False
    for old_lbl in old_labels:
        old_stem = os.path.splitext(old_lbl)[0]  # e.g. "frame_0042"
        
        if old_stem in new_stem:  # partial match
            src = os.path.join(old_labels_dir, old_lbl)
            dst = os.path.join(new_labels_dir, new_stem + ".txt")
            shutil.copy2(src, dst)
            print(f"✓ {old_lbl} → {new_stem}.txt")
            matched += 1
            found = True
            break
    
    if not found:
        unmatched.append(new_img)

print(f"\nMatched: {matched}, Unmatched: {len(unmatched)}")
if unmatched:
    print("No label found for:", *unmatched, sep="\n  ")