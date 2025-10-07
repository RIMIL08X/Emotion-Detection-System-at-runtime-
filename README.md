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

   * Trained CNN stored as `fer2013_cnn_model.h5`.
   * Accepts cropped face images and classifies emotions such as:
     `['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']`

---

## 🧩 Model Training Results

Below are terminal screenshots of the model training process (accuracy and loss progression):

<p align="center">
  <img src="https://raw.githubusercontent.com/RIMIL08X/Real-Time-Emotion-Detection-System/main/assets/training_terminal_1.png" alt="Training Log 1" width="600">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/RIMIL08X/Real-Time-Emotion-Detection-System/main/assets/training_terminal_2.png" alt="Training Log 2" width="600">
</p>

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/RIMIL08X/EMODetect.git
cd EMODetect
```

### 2. Install Python Version using pyenv

```bash
pyenv install 3.11.9
pyenv local 3.11.9
```

### 3. Create and Activate Virtual Environment

```bash
python3 -m venv emodetect-venv
source emodetect-venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r EMODetect/requirements.txt
```

### 5. Run Real-Time Detection

```bash
python EMODetect/live.py
```

**What happens:**

* YOLOv8 detects faces.
* Each detected face is cropped and resized to the CNN input size.
* The CNN predicts the emotion label.
* Bounding boxes and labels appear live on your webcam feed.

Press **'q'** to exit the live window.

---

## 📁 Project Structure

```
.
├── assets/                    # Assets folder for documentation and visuals
│   ├── training_terminal_1.png
│   └── training_terminal_2.png
│
├── EMODetect/
│   ├── __pycache__/           # Cached Python files
│   ├── model/                 # Directory containing trained model files
│   │   └── fer2013_cnn_model.h5
│   ├── live.py                # Real-time detection script (YOLO + CNN + OpenCV)
│   ├── preprocessing.py        # Dataset preprocessing utilities
│   ├── train.py                # Script to train the CNN on FER2013
│   ├── yolov8n-face.pt         # YOLOv8 lightweight face detection model
│   └── requirements.txt        # List of dependencies
│
└── README.md                   # This file
```

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
