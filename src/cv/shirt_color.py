import cv2
import numpy as np


def extract_shirt_region(image, face_bbox):
    x, y, w, h = face_bbox
    height, width, _ = image.shape

    for scale in [1.5, 2.5, 3.5, 5.0]:

        y_start = max(0, y + int(0.6 * h))
        y_end = min(height, y + int(scale * h))

        x_start = max(0, x - int(1.0 * w))
        x_end = min(width, x + int(2.0 * w))

        region = image[y_start:y_end, x_start:x_end]

        if region is not None and region.size > 0:
            return region

    return None


def dominant_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = image.reshape(-1, 3).astype(np.float32)

    _, labels, centers = cv2.kmeans(
        pixels,
        3,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    counts = np.bincount(labels.flatten())
    return tuple(map(int, centers[np.argmax(counts)]))


def rgb_to_hsv(rgb):
    rgb_array = np.uint8([[rgb]])
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)[0][0]


def hsv_to_color_name(hsv):
    h, s, v = hsv

    if v < 50:
        return "black"
    if s < 40 and v > 200:
        return "white"
    if s < 40:
        return "gray"
    if h < 10 or h > 170:
        return "red"
    elif h < 25:
        return "orange"
    elif h < 35:
        return "yellow"
    elif h < 85:
        return "green"
    elif h < 130:
        return "blue"
    elif h < 160:
        return "purple"
    else:
        return "pink"


def rgb_to_hex(rgb):
    if rgb is None:
        return None
    return '#{:02x}{:02x}{:02x}'.format(*rgb)