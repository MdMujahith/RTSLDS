# 🧠 Real-Time Sign Language Detection System  

A **Flask-based real-time application** that detects and interprets **sign language gestures** using your webcam.  
The system supports both **ASL (American Sign Language)** and **ISL (Indian Sign Language)** and uses **trained YOLO models** for gesture recognition.  
Detected signs are instantly converted into **text and speech**, enabling smooth and inclusive communication.  

---

## 🔧 Setup Instructions  

### ✅ Prerequisites  
Make sure you have the following installed:  
- Python **3.8+**  
- **Webcam** (built-in or external)  
- **pip** (Python package manager)  
- **git** (for cloning the repository)  

---

### ✅ Step 1: Clone the Repository  
```
git clone https://github.com/MdMujahith/RTSLDS.git
cd RTSLDS
```
### ✅ Step 2: Create a Virtual Environment (Recommended)
python -m venv venv
Activate the virtual environment:

Windows:
```
venv\Scripts\activate

```
macOS/Linux:
```
source venv/bin/activate

```
### ✅ Step 3: Install Dependencies
Install all required Python packages:

```
pip install -r requirements.txt

```
If the file is missing, install manually:

```
pip install flask opencv-python pyttsx3 ultralytics torch torchvision

```
### ✅ Step 4: Add Trained Models
Download or train your YOLOv5/YOLOv8 models for ASL and ISL detection.

Place them in the models/ folder:
```
models/
 ├── asl_model.pt
 └── isl_model.pt
 ```
💡 You can train models using Roboflow + Google Colab and export them in YOLO format.

### ✅ Step 5: Run the Application
Start the Flask server:
```
python app.py
```
Once running, open your browser and go to:
👉 http://127.0.0.1:5000/

### ✅ Step 6: Test the Detection
Select ASL or ISL mode.

Allow camera access in your browser.

Show a trained gesture to the webcam.

The detected sign will appear as text and be spoken aloud.

## 🎥 Demo Video  
Experience the project in action:  

🎬 [**Watch Demo Video (MP4)**](Demo_Video.mp4)

> The demo video is included in this repository under `Demo_Video.mp4`.

 ## ⚙️ Common Issues & Fixes
❌ Webcam Not Opening
Try changing cv2.VideoCapture(0) to cv2.VideoCapture(1)

Ensure no other app is using your webcam

❌ No Detection
Improve lighting conditions

Confirm the gesture exists in your trained dataset

❌ Text-to-Speech Not Working
pyttsx3 works offline, but ensure it initializes properly

Install missing dependencies:
```
pip install espeak pyaudio
```
Alternatively, use ```gTTS``` for online text-to-speech

#### 🧩 Train Your Own Model
Go to Roboflow

Upload your sign gesture images

Annotate your dataset and export it in YOLOv5 format

Train using YOLOv5 or YOLOv8 on Google Colab

Download the trained best.pt file and place it inside the models/ folder

#### 🚀 Future Enhancements
Add sentence-level recognition

Deploy via Streamlit Cloud or Render

Add mobile support

Improve UI/UX with feedback and analytics
