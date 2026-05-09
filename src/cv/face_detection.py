# Import libraries
import cv2
from ultralytics import YOLO

# Define both models
yolo_model = YOLO("yolov8n.pt")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Define yolo detection function
def detect_face_yolo(image):
    results = yolo_model(image)[0]

    best_box = None
    best_conf = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls == 0 and conf > best_conf:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            best_box = (x1, y1, x2 - x1, y2 - y1)
            best_conf = conf

    return best_box

# Define haar cascade detection function
def detect_face_haar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    return faces[0]

# Define main detection function that tries YOLO first, then Haar
def detect_face(image):
    bbox = detect_face_yolo(image)
    if bbox is not None:
        return bbox, "yolo"

    bbox = detect_face_haar(image)
    if bbox is not None:
        return bbox, "haar"

    return None, "none"