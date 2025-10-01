import os
import shutil
import random

original_dataset_dir = 'dataset/PlantVillage'
 # Your dataset folder
base_dir = 'dataset'                   # Base folder to create train/validation folders
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

# Create the train and validation directories if they don't exist
for directory in [train_dir, val_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Get all class folders
for disease in os.listdir(original_dataset_dir):
    disease_path = os.path.join(original_dataset_dir, disease)
    if not os.path.isdir(disease_path):
        continue
    
    os.makedirs(os.path.join(train_dir, disease), exist_ok=True)
    os.makedirs(os.path.join(val_dir, disease), exist_ok=True)
    
    all_images = os.listdir(disease_path)
    random.shuffle(all_images)
    
    split = int(len(all_images) * 0.8)
    train_images = all_images[:split]
    val_images = all_images[split:]
    
    for img in train_images:
        src = os.path.join(disease_path, img)
        dst = os.path.join(train_dir, disease, img)
        shutil.copy(src, dst)
    for img in val_images:
        src = os.path.join(disease_path, img)
        dst = os.path.join(val_dir, disease, img)
        shutil.copy(src, dst)

print("Dataset split is done!")
