# Import libraries
from PIL import Image
import os

# Define target size
TARGET_SIZE = (224, 224) 

# Function to check if an image is valid
def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False


# Function to resize and save an image
def resize_image(path, target_size=TARGET_SIZE):
    try:
        img = Image.open(path).convert("RGB")  # Ensure consistent colour mode
        img = img.resize(target_size, Image.LANCZOS)
        img.save(path)
        return True
    except Exception as e:
        print(f"Failed to resize {path}: {e}")
        return False

# Function to load valid images from a folder
def load_valid_images(input_folder, delete_corrupt=False, extensions=(".jpg", ".jpeg", ".png")):

    image_items = []

    for file in os.listdir(input_folder):
        path = os.path.join(input_folder, file)

        if file.lower().endswith(extensions):

            if is_valid_image(path):
                image_items.append({
                    "file": file,
                    "path": path
                })
            else:
                print(f"Corrupt image: {file}")

                if delete_corrupt:
                    os.remove(path)

    return image_items