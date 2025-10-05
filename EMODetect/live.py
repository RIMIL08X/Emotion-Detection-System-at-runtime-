

import os
import cv2
import numpy as np
import time

#gpu disabling 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable TensorFlow GPU usage

from tensorflow.keras.models import load_model
from ultralytics import YOLO
import torch

#basic configs
FER_MODEL_PATH = "model/fer2013_cnn_model.h5"   # FER2013 CNN model
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
WEBCAM_INDEX = 0
SHOW_FPS = True

#Loading the Model
print("[INFO] Loading FER model (CPU only)...")
if not os.path.exists(FER_MODEL_PATH):
    raise FileNotFoundError(f"FER model not found at: {FER_MODEL_PATH}")

emotion_model = load_model(FER_MODEL_PATH)

print("[INFO] Loading YOLOv8 face detector (PyTorch)...")
yolo_face_model = YOLO("yolov8n-face.pt")

# Log device info
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] YOLO running on: {device.upper()} | TensorFlow on: CPU")

#prep function
def preprocess_face(face_img):
    """
    Convert a face image to 48x48 grayscale, normalized for CNN input
    """
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, 48, 48, 1))
    return reshaped

# web-cam start
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera index.")

prev_time = time.time()
print("[INFO] Starting live detection. Press 'q' to quit.")

# live loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Failed to grab frame.")
        break

    # Detect faces with YOLOv8
    results = yolo_face_model(frame, verbose=False)

    # Loop through all detected boxes
    for result in results:
        for box in result.boxes.xyxy:  # bounding boxes (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box)
            face_crop = frame[y1:y2, x1:x2]

            # Skip empty crops
            if face_crop.size == 0:
                continue

            # Preprocess for CNN
            processed_face = preprocess_face(face_crop)

            # Predict emotion (TensorFlow, CPU)
            prediction = emotion_model.predict(processed_face, verbose=0)
            emotion_idx = np.argmax(prediction)
            emotion_text = EMOTION_LABELS[emotion_idx]

            # Draw bounding box + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, emotion_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Calculate and display FPS
    if SHOW_FPS:
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("YOLO + FER Live Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#clearance
cap.release()
cv2.destroyAllWindows()
print("[INFO] Live detection stopped.")
