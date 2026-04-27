# Import libraries 
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity

# Define embedding and similarity functions
def get_embedding(image):
    image = cv2.resize(image, (128, 128))
    return image.flatten() / 255.0


def compute_similarity_matrix(images):
    X = np.array([get_embedding(img) for img in images])
    return cosine_similarity(X)