# Real-Time Sign Language Detection System

This is a Flask-based project that detects sign language in real-time using a webcam. It supports ASL (American Sign Language) and ISL (Indian Sign Language), detects gestures using trained YOLO models, and converts the detected signs into both text and speech. 
==========================
🔧 SETUP INSTRUCTIONS
==========================

✅ Prerequisites:
------------------
- Python 3.8 or later
- A webcam (built-in or USB)
- pip (Python package manager)
- git (to clone the repo)

✅ Step 1: Clone the Repository
-------------------------------
Open your terminal or command prompt and run:

    git clone https://github.com/MdMujahith/RTSLDS.git
    cd RTSLDS

✅ Step 2: Create a Virtual Environment (Recommended)
-----------------------------------------------------
    python -m venv venv

Activate the virtual environment:

- On Windows:
    venv\Scripts\activate

- On macOS/Linux:
    source venv/bin/activate

✅ Step 3: Install Python Dependencies
--------------------------------------
Install all required packages:

    pip install -r requirements.txt

If the file is missing, manually install:

    pip install flask opencv-python pyttsx3 ultralytics torch torchvision

✅ Step 4: Add Trained Models
-----------------------------
Download or train your YOLOv5 or YOLOv8 models for ASL and ISL.

- Save the ASL model as: `models/asl_model.pt`
- Save the ISL model as: `models/isl_model.pt`

(You can train models using Roboflow + Google Colab and export them in YOLO format.)

✅ Step 5: Run the Project
--------------------------
Start the Flask server:

    python app.py

Now open your browser and visit:

    http://127.0.0.1:5000/

✅ Step 6: Test the Detection
-----------------------------
- Choose ASL or ISL
- Allow your camera to start
- Show any trained sign to the webcam
- The detected word will be shown and spoken

==========================
==========================
📌 COMMON ISSUES
==========================

❌ Webcam not opening:
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`
- Ensure no other app is using the camera

❌ No detection happening:
- Make sure lighting is good
- Confirm the sign used is from the trained model

❌ Text-to-speech not working:
- pyttsx3 works offline, but check if engine is initializing
- Try installing `espeak`, `pyaudio`, or use gTTS for online TTS

==========================
📚 TRAINING YOUR OWN MODEL
==========================

1. Go to https://roboflow.com
2. Upload your sign gesture images
3. Annotate and export in YOLOv5 format
4. Use Google Colab and train using YOLOv5 or YOLOv8
5. Download the `best.pt` model and place it in `models/`

==========================
🚀 FUTURE IDEAS
==========================

- Add more complex sentence recognition
- Deploy on Streamlit Cloud or Render
- Add mobile support
- Improve UI/UX with feedback system
