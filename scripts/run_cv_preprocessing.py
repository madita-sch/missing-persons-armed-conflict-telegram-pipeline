# Import libraries
from src.preprocessing.image_preprocessing import load_valid_images, resize_image

# Configuration: replace with folder containing images
INPUT_FOLDER = "data/images"

# Load valid images
images = load_valid_images(INPUT_FOLDER, delete_corrupt=False)

print(f"Found {len(images)} valid images.")

# Resize all valid images
for item in images:
    success = resize_image(item["path"])

    if success:
        print(f"Resized: {item['file']}")

print("Preprocessing complete.")