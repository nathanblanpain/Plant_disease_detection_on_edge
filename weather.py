import albumentations as A
import cv2
import os
from pathlib import Path

def create_augmented_datasets(input_dir, output_base_dir):
    """
    Crée 6 versions du dataset : normal + 5 contextes
    Source : Albumentations library [20]
    """
    
    # Définir transformations
    contexts = {

        'normal': A.Compose([]),
        
        'rain': A.Compose([
            A.RandomRain(slant_lower=-10, slant_upper=10, 
                        drop_length=20, drop_width=1,
                        drop_color=(200, 200, 200), blur_value=3,
                        brightness_coefficient=0.8, rain_type='drizzle', p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(-0.3, -0.1), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=0.5)
        ]),
        
        #'night': A.Compose([
        #    A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.4),
        #                              contrast_limit=(-0.3, -0.1), p=1.0),
        #    A.GaussNoise(var_limit=(30, 70), p=1.0),
        #    A.RGBShift(r_shift_limit=(-20, -10), g_shift_limit=(-10, 0),
        #              b_shift_limit=(10, 20), p=0.7),
        #    A.Blur(blur_limit=3, p=0.3)
        #]),
        
        'fog': A.Compose([
            A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.6, 
                       alpha_coef=0.1, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                      contrast_limit=(-0.3, -0.2), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=0.8)
        ]),
        
        #'shadow': A.Compose([
        #    A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1,
        #                  num_shadows_upper=3, shadow_dimension=6, p=1.0),
        #    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.1), p=0.5)
        #]),
        #
        #'occluded': A.Compose([
        #    A.CoarseDropout(max_holes=5, max_height=50, max_width=50,
        #                   min_holes=2, min_height=20, min_width=20,
        #                   fill_value=(34, 139, 34), p=1.0)
        #])
    }
    
    # Trouver toutes les images
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    
    print(f"Trouvé {len(image_files)} images")
    print(f"Création de {len(contexts)} contextes...")
    
    # Pour chaque contexte
    for context_name, transform in contexts.items():
        output_dir = os.path.join(output_base_dir, context_name)
        print(f"\n>>> Génération contexte '{context_name}'...")
        
        for i, img_path in enumerate(image_files, 1):
            try:
                # Lire image
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Appliquer transformation
                transformed = transform(image=img_rgb)['image']
                transformed_bgr = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
                
                # Sauvegarder
                # Sauvegarder avec renommage
                rel_path = os.path.relpath(img_path, input_dir)
                rel_dir = os.path.dirname(rel_path)
                
                filename = os.path.basename(rel_path)
                name, ext = os.path.splitext(filename)
                
                # normal -> original
                prefix = "original" if context_name == "normal" else context_name
                
                new_filename = f"{prefix}_{name}{ext}"
                
                out_dir = os.path.join(output_base_dir, rel_dir)
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                
                out_path = os.path.join(out_dir, new_filename)
                
                cv2.imwrite(out_path, transformed_bgr)
                
                if i % 500 == 0:
                    print(f"    {i}/{len(image_files)}...")
            
            except Exception as e:
                print(f"    Erreur {os.path.basename(img_path)}: {e}")
        
        print(f"    ✓ Terminé")
    
    print(f"\n✓ Tous contextes créés dans {output_base_dir}")

# UTILISATION
if __name__ == "__main__":
    create_augmented_datasets(
        input_dir="PlantDoc/224x224/test",
        output_base_dir="PlantDoc/224x224/contexts" 
    )

    create_augmented_datasets(
        input_dir="PlantDoc/320x320/test",
        output_base_dir="PlantDoc/320x320/contexts" 
    )
    
    create_augmented_datasets(
        input_dir="PlantDoc/416x416/test",
        output_base_dir="PlantDoc/416x416/contexts" 
    )
    