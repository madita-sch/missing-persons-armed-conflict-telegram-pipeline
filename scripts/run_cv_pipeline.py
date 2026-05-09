# Import libraries
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.image_preprocessing import load_valid_images
from src.cv.face_detection import detect_face
from src.cv.shirt_color import (
    extract_shirt_region,
    dominant_color,
    rgb_to_hsv,
    hsv_to_color_name,
    rgb_to_hex
)
from src.cv.image_similarity import get_embedding

# Load data
image_items = load_valid_images("data/images")

# Initialize results list and embeddings list
results = []
embeddings = []

# Process images iteratively
for item in tqdm(image_items):

    file = item["file"]
    path = item["path"]

    img = cv2.imread(path)
    if img is None:
        continue

    # Face detection
    face_bbox, detector_used = detect_face(img)

    # Shirt region
    shirt = None
    if face_bbox is not None:
        shirt = extract_shirt_region(img, face_bbox)

    # Color extraction
    if shirt is None:
        rgb = None
        hsv = None
        name = "no_region"
    else:
        rgb = dominant_color(shirt)
        hsv = rgb_to_hsv(rgb)
        name = hsv_to_color_name(hsv)

    # Embedding for similarity
    embeddings.append(get_embedding(img))

    results.append({
        "file": file,
        "face_detected": face_bbox is not None,
        "detector_used": detector_used,
        "shirt_color_rgb": rgb,
        "shirt_color_hex": rgb_to_hex(rgb),
        "shirt_color_hsv": hsv.tolist() if hsv is not None else None,
        "shirt_color_name": name
    })

# Save CV results
df = pd.DataFrame(results)
df.to_csv("outputs/cv_results.csv", index=False)


# Find image similarities
X = np.array(embeddings)
sim_matrix = cosine_similarity(X)

similar_pairs = []

for i in range(len(image_items)):
    for j in range(i + 1, len(image_items)):
        if sim_matrix[i, j] > 0.9:
            similar_pairs.append({
                "img1": image_items[i]["file"],
                "img2": image_items[j]["file"],
                "similarity": sim_matrix[i, j]
            })

# Save image similarity results
pd.DataFrame(similar_pairs).to_csv("outputs/image_similarity.csv", index=False)