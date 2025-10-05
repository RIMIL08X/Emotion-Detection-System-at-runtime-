# Real-Time Facial Emotion Recognition

A real-time **Facial Emotion Recognition (FER)** system built using **YOLOv8** for face detection, **OpenCV** for video capture, and a **custom CNN (.h5) model** trained on FER datasets for emotion classification.

---

## 🚀 Features

* Real-time emotion detection from webcam or video feed.
* Uses **YOLOv8** for high-accuracy face detection.
* Loads a **pretrained CNN (.h5)** model for emotion classification.
* Bounding boxes with emotion labels drawn using OpenCV.
* Works on CPU or GPU.

---

## 🧠 Model Overview

1. **YOLOv8 (Face Detection)**

   * Detects human faces in real time.
   * Model file: `yolov8n-face.pt`
   * Source: [YOLOv8n-Face Weights (Hugging Face)](https://huggingface.co/arnabdhar/YOLOv8n-face/resolve/main/yolov8n-face.pt)

2. **FER Model (Emotion Classification)**

   * Trained CNN stored as `model.h5`.
   * Accepts cropped face images and classifies emotions such as:
     `['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']`

---

## 🛠️ Requirements

Create a virtual environment (recommended):

```bash
python3 -m venv emodetect-venv
source emodetect-venv/bin/activate
```

Install dependencies:

```bash
pip install torch torchvision ultralytics tensorflow opencv-python numpy
```

If TensorFlow GPU support is needed:

```bash
pip install tensorflow[and-cuda]
```

---

## 📁 Project Structure

```
EMODetect/
├── model.h5                # Pretrained emotion classification model
├── yolov8n-face.pt         # YOLO face detection model
├── live.py                 # Main real-time inference script
├── preprocess.py           # (Optional) Dataset preprocessing script
├── train.py                # (Optional) Model training script
├── requirements.txt        # Project dependencies
└── README.md               # Documentation
```

---

## 🎥 Running the Live Detection

```bash
python live.py
```

**What happens:**

* YOLOv8 detects faces.
* Each detected face is cropped and resized to the CNN input size.
* The CNN predicts the emotion label.
* Bounding boxes and labels appear live on your webcam feed.

Press **'q'** to exit the live window.

---

## 🧩 Example Output

When running `live.py`, you’ll see bounding boxes with emotion labels:

```
[Happy]  😊
[Angry]  😠
[Neutral] 😐
```

Each face in the frame will be annotated with a box and emotion tag.

---

## ⚙️ Troubleshooting

| Issue                                     | Solution                                                                 |
| ----------------------------------------- | ------------------------------------------------------------------------ |
| `FileNotFoundError: YOLO model not found` | Download `yolov8n-face.pt` and place in project root.                    |
| TensorRT warnings                         | Ignore if not using GPU acceleration.                                    |
| Webcam not opening                        | Change `cv2.VideoCapture(0)` to another index like `cv2.VideoCapture(1)` |

---

## 📚 References

* [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com)
* [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
* [OpenCV Documentation](https://docs.opencv.org)

---

## 🧑‍💻 Author

**Storm (Rimil)**
CSE Student @ SRM Kattankulathur
Focus: Computer Vision | Deep Learning | Reinforcement Learning

---

## 🧾 License

This project is licensed under the **MIT License** – free to use and modify for research and educational purposes.
