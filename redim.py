from PIL import Image
import os
from pathlib import Path

def resize_dataset(input_dir, output_dir, target_size=(320, 320), 
                   maintain_aspect_ratio=False):
    """
    Redimensionne toutes les images d'un dataset
    
    Args:
        input_dir: Dossier source avec images
        output_dir: Dossier destination
        target_size: Tuple (width, height) en pixels
        maintain_aspect_ratio: Si True, garde ratio + padding noir
    """
    
    # Créer dossier de sortie
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extensions supportées
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Parcourir toutes les images
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if Path(file).suffix.lower() in valid_extensions:
                image_files.append(os.path.join(root, file))
    
    print(f"Trouvé {len(image_files)} images à redimensionner...")
    
    for i, img_path in enumerate(image_files):
        try:
            # Ouvrir l'image
            img = Image.open(img_path)
            
            # Construire chemin de sortie (garde structure dossiers)
            rel_path = os.path.relpath(img_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Redimensionner
            if maintain_aspect_ratio:
                # Garde le ratio avec padding
                img.thumbnail(target_size, Image.Resampling.LANCZOS)
                # Créer image noire de la taille cible
                new_img = Image.new('RGB', target_size, (0, 0, 0))
                # Centrer l'image redimensionnée
                offset = ((target_size[0] - img.size[0]) // 2,
                         (target_size[1] - img.size[1]) // 2)
                new_img.paste(img, offset)
                img = new_img
            else:
                # Resize sans garder ratio (stretch)
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            
            # Sauvegarder
            img.save(output_path, quality=95)
            
            if (i + 1) % 100 == 0:
                print(f"Traité {i + 1}/{len(image_files)} images...")
                
        except Exception as e:
            print(f"Erreur avec {img_path}: {e}")
    
    print(f"Terminé ! {len(image_files)} images redimensionnées dans {output_dir}")

# UTILISATION
if __name__ == "__main__":

    ################################
    #PLANT VILLAGE
    ################################

    # 224x224 TRAIN
    resize_dataset(
        input_dir="PlantVillage/original/train",
        output_dir="PlantVillage/224x224/train",
        target_size=(224, 224),
        maintain_aspect_ratio=False
    )

    # 224x224 TEST
    resize_dataset(
        input_dir="PlantVillage/original/test",
        output_dir="PlantVillage/224x224/test",
        target_size=(224, 224),
        maintain_aspect_ratio=False
    )
    
    # 320x320 TRAIN
    resize_dataset(
        input_dir="PlantVillage/original/train",
        output_dir="PlantVillage/320x320/train",
        target_size=(320, 320),
        maintain_aspect_ratio=False
    )

    # 320x320 TEST
    resize_dataset(
        input_dir="PlantVillage/original/test",
        output_dir="PlantVillage/320x320/test",
        target_size=(320, 320),
        maintain_aspect_ratio=False
    )
    
    # 416x416 TRAIN
    resize_dataset(
        input_dir="PlantVillage/original/train",
        output_dir="PlantVillage/416x416/train",
        target_size=(416, 416),
        maintain_aspect_ratio=False
    )

    # 416x416 TEST
    resize_dataset(
        input_dir="PlantVillage/original/test",
        output_dir="PlantVillage/416x416/test",
        target_size=(416, 416),
        maintain_aspect_ratio=False
    )


    ################################
    #PLANT DOC
    ################################

    # 224x224 TRAIN
    resize_dataset(
        input_dir="PlantDoc/original/train",
        output_dir="PlantDoc/224x224/train",
        target_size=(224, 224),
        maintain_aspect_ratio=False
    )

    # 224x224 TEST
    resize_dataset(
        input_dir="PlantDoc/original/test",
        output_dir="PlantDoc/224x224/test",
        target_size=(224, 224),
        maintain_aspect_ratio=False
    )
    
    # 320x320 TRAIN
    resize_dataset(
        input_dir="PlantDoc/original/train",
        output_dir="PlantDoc/320x320/train",
        target_size=(320, 320),
        maintain_aspect_ratio=False
    )

    # 320x320 TEST
    resize_dataset(
        input_dir="PlantDoc/original/test",
        output_dir="PlantDoc/320x320/test",
        target_size=(320, 320),
        maintain_aspect_ratio=False
    )
    
    # 416x416 TRAIN
    resize_dataset(
        input_dir="PlantDoc/original/train",
        output_dir="PlantDoc/416x416/train",
        target_size=(416, 416),
        maintain_aspect_ratio=False
    )

    # 416x416 TEST
    resize_dataset(
        input_dir="PlantDoc/original/test",
        output_dir="PlantDoc/416x416/test",
        target_size=(416, 416),
        maintain_aspect_ratio=False
    )