import os
from PIL import Image

def clean_folder(folder):
    removed = 0
    for root, _, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Try to open the file as an image
                with Image.open(file_path) as img:
                    img.verify()
            except Exception:
                print("Removing corrupted or non-image file:", file_path)
                removed += 1
                os.remove(file_path)
    print(f"Removed {removed} files from {folder}")

clean_folder('dataset/train')
clean_folder('dataset/validation')
