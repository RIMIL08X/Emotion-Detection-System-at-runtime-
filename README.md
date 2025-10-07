# Real-Time Facial Emotion Recognition 😄

A real-time **Facial Emotion Recognition (FER)** system built using **YOLOv8** for face detection, **OpenCV** for video capture, and a **custom CNN (`.h5`) model** trained on FER datasets for emotion classification.

---

## 🚀 Features

* Real-time emotion detection from webcam or video feed.
* Uses **YOLOv8** for accurate and fast face detection.
* Loads a **custom-trained CNN (.h5)** model for emotion classification.
* Bounding boxes with emotion labels drawn using OpenCV.
* Works on **CPU** or **GPU**.

---

## 🧠 Model Overview

### 1. **YOLOv8 (Face Detection)**

* Detects faces in real-time.
* Model file: `yolov8n-face.pt`
* Source: [YOLOv8n-Face Weights (Hugging Face)](https://huggingface.co/arnabdhar/YOLOv8n-face/resolve/main/yolov8n-face.pt)

### 2. **FER Model (Emotion Classification)**

* Trained **CNN** stored as `fer2013_cnn_model.h5`.
* Accepts cropped face images and classifies into:
  `['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']`

---

## ⚙️ Setup Instructions

Follow these steps to set up and run the project locally.

### 1️⃣ Install `pyenv` and Python 3.11+

If you don’t have Python 3.11 installed:

```bash
# Install pyenv (Ubuntu/Debian example)
curl https://pyenv.run | bash

# Add pyenv to PATH (add these lines to ~/.bashrc or ~/.zshrc)
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
```

Now install Python 3.11:

```bash
pyenv install 3.11.6
pyenv local 3.11.6
```

---

### 2️⃣ Create and Activate Virtual Environment

```bash
python -m venv emodetect-venv
source emodetect-venv/bin/activate
```

Or (if you prefer pyenv’s virtualenv):

```bash
pyenv virtualenv 3.11.6 emodetect-venv
pyenv activate emodetect-venv
```

---

### 3️⃣ Install Dependencies

All dependencies are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

If you need GPU acceleration:

```bash
pip install tensorflow[and-cuda]
```

---

### 4️⃣ Run the Real-Time Detector

Once setup is complete, start the live detection:

```bash
python live.py
```

**What happens:**

* YOLOv8 detects faces in the webcam feed.
* Each detected face is cropped and fed into the CNN.
* The CNN predicts the emotion label.
* Bounding boxes with emotion names are drawn live on the frame.

Press **`q`** to quit.

---

## 📁 Project Structure

```
EMODetect/
│
├── __pycache__/                # Cached Python files
├── model/                      # Directory containing trained model files
│   └── fer2013_cnn_model.h5    # CNN model trained on FER2013 dataset
│
├── live.py                     # Real-time detection (YOLO + CNN + OpenCV)
├── preprocessing.py             # Dataset preprocessing utilities
├── train.py                    # CNN training script
├── yolov8n-face.pt             # YOLOv8 lightweight face detection model
├── requirements.txt             # List of dependencies
└── README.md                   # This file
```

---

## 📊 Model Training Results

The CNN model was trained on the **FER2013 dataset**. Below are **two terminal snapshots** showing the training process, accuracy, and loss progression across epochs:

<p align="center">
  <img src="https://raw.githubusercontent.com/stormrimil/EMODetect/main/assets/training_terminal_1.png" alt="Training Log 1" width="600">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/stormrimil/EMODetect/main/assets/training_terminal_2.png" alt="Training Log 2" width="600">
</p>

> *(Ensure both screenshots are uploaded to your GitHub repo at `assets/` path for direct rendering.)*

Example performance: **~85% validation accuracy** after fine-tuning.

---

## 🎥 Example Live Output

When you run `live.py`, you’ll see real-time bounding boxes and labels like:

```
[Happy]  😊
[Angry]  😠
[Neutral] 😐
```

---

## 🧩 Troubleshooting

| Issue                                          | Solution                                                          |
| ---------------------------------------------- | ----------------------------------------------------------------- |
| `FileNotFoundError: yolov8n-face.pt not found` | Download the YOLOv8n-Face model and place it in the project root. |
| TensorRT / CUDA warnings                       | Ignore if not using GPU acceleration.                             |
| Webcam not opening                             | Change `cv2.VideoCapture(0)` to another index (like `1` or `2`).  |

---

## 📚 References

* [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com)
* [FER2013 Dataset (Kaggle)](https://www.kaggle.com/datasets/msambare/fer2013)
* [OpenCV Official Documentation](https://docs.opencv.org)

---

## 👨‍💻 Author

**Storm (Rimil)**
CSE Student @ SRM Kattankulathur
Focus: Computer Vision | Deep Learning | Reinforcement Learning

---

## 🧾 License

This project is licensed under the **MIT License** – free to use and modify for research and educational purposes.

---

### ✅ Quick Summary

| Step                   | Command                                                               |
| ---------------------- | --------------------------------------------------------------------- |
| Setup Python via pyenv | `pyenv install 3.11.6`                                                |
| Create & activate venv | `python -m venv emodetect-venv && source emodetect-venv/bin/activate` |
| Install dependencies   | `pip install -r requirements.txt`                                     |
| Run the project        | `python live.py`                                                      |
