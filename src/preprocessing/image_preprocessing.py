from PIL import Image
import os


def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False


def load_valid_images(input_folder, delete_corrupt=False, extensions=(".jpg", ".jpeg", ".png")):
    """
    Returns:
        List of dicts:
        {
            "file": filename,
            "path": full_path
        }
    """

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
                print(f"❌ Corrupt image: {file}")

                if delete_corrupt:
                    os.remove(path)

    return image_items