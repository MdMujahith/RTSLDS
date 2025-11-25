# 🧠 Real-Time Sign Language Detection System (RTSLDS)

A **Flask-based real-time AI application** that detects and interprets sign language gestures using a webcam. The system supports both **ASL (American Sign Language)** and **ISL (Indian Sign Language)** using trained YOLO models. Detected signs are instantly converted into **text and speech**, enabling smooth and inclusive communication.

## 🎥 Demo Video

See the project in action!

[**Click here to watch the Demo Video**](Demo_Video.mp4)

*(Note: Ensure the file `Demo_Video.mp4` is present in the main project folder)*

## 🚀 Features

*   **Real-Time Detection:** Live webcam feed processing with zero lag.
*   **Multi-Model Support:** Switch instantly between ASL (YOLOv8) and ISL (YOLOv5) models.
*   **Text-to-Speech (TTS):** Automatically speaks the detected sentences using Google Text-to-Speech (gTTS).
*   **Web Interface:** Clean, responsive UI built with Bootstrap and Socket.IO.
*   **Smart Debouncing:** Prevents flickering text by stabilizing detection outputs.

## 🔧 Setup Instructions

### 1. Prerequisites

Ensure you have the following installed:

*   Python **3.10** or **3.11** (Recommended for AI libraries)
*   **Git** (for cloning)
*   A webcam

### 2. Clone the Repository

```bash
git clone https://github.com/MdMujahith/RTSLDS.git
cd RTSLDS
```
### 3. Create a Virtual Environment

It is highly recommended to use a virtual environment to keep dependencies clean.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```
### 4. Install Dependencies

Install all required Python packages.
pip install -r requirements.txt
Critical Step for YOLOv5: You must also clone the YOLOv5 repository inside your project folder manually because it is a required dependency for the ISL model logic.
```bash
git clone https://github.com/ultralytics/yolov5
pip install -r yolov5/requirements.txt
```

### 5. Add Trained Models

You must have your trained .pt model files ready. Create a models folder and place them there.

Required Folder Structure:
```bash
/RTSLDS
├── models/
│   ├── yolov8_asl_best.pt  <-- Your ASL Model
│   └── yolov5_isl_best.pt  <-- Your ISL Model
├── yolov5/                 <-- Cloned from GitHub
├── app.py
└── requirements.txt
```
### 6. Run the Application

Start the Flask server:
```bash
python app.py
```
Once running, open your browser and go to: 👉 http://127.0.0.1:5000/

🎥 Usage Guide
- Select Mode: Click the "ASL" or "ISL" button on the dashboard to load the correct model.
- Start Camera: Click "Start Detection". The system will initialize your webcam.
- Perform Gestures: Show hand signs to the camera.

**View Results:**
1. Live Label: Shows the currently detected letter/word.
2. Confidence Bar: Shows how sure the AI is.
3. Translation: Words are assembled into sentences in the text box.
4. Audio: The system will speak the sentence aloud automatically.

### ⚙️ Troubleshooting
- ❌ Error: "ModuleNotFoundError: No module named 'yolov5'"
Fix: You forgot to clone the YOLOv5 repo. Run git clone https://github.com/ultralytics/yolov5 inside your project folder.
- ❌ Error: "Model not found"
Fix: Ensure your model files are named exactly yolov8_asl_best.pt and yolov5_isl_best.pt and are inside the models/ folder.
- ❌ Webcam Not Opening
Fix: Ensure no other app (like Zoom/Teams) is using the camera. If you have multiple cameras, try changing cv2.VideoCapture(0) to 1 in app.py.
- ❌ Audio Not Playing
Fix: Browsers often block auto-playing audio. Click anywhere on the webpage to interact with it, which usually enables audio permissions.

### 🧩 Model Training (Optional)

If you want to train your own models from scratch:
Dataset: Collect images using Roboflow or Kaggle.
Training:
ASL (YOLOv8): Use the ultralytics library.
ISL (YOLOv5): Use the yolov5 repository scripts.

👨‍💻 **Developed by** [Mohamed Mujahith](https://github.com/MdMujahith) and [Mohamed Ashbek ](https://github.com/ashbekT)

