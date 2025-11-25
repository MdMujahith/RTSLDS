import sys
import os
import cv2
import torch
import numpy as np
import pathlib
import threading
import base64
import time
import uuid
import tempfile
import logging
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from gtts import gTTS

# --- PATCH FOR PYTHON 3.13 & NUMPY 2.0 ---
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool

# --- FIX WINDOWS PATHS ---
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# --- SETUP YOLOv5 PATHS (CRITICAL FIX) ---
# This tells Python: "The code you need is inside the 'yolov5' folder"
yolov5_path = os.path.join(os.getcwd(), 'yolov5')
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

# --- IMPORT AI LIBRARIES ---
print(f"Checking for YOLOv5 at: {yolov5_path}")
try:
    from ultralytics import YOLO
    # Now we import directly from the 'models' folder inside yolov5
    from models.experimental import attempt_load 
    from utils.general import non_max_suppression
    from utils.torch_utils import select_device
    print("✅ AI Libraries imported successfully!")
except ImportError as e:
    print(f"\n❌ CRITICAL IMPORT ERROR: {e}")
    print("----------------------------------------------------")
    print(f"1. Is the folder named 'yolov5' inside '{os.getcwd()}'?")
    print("2. Does 'yolov5/models/experimental.py' exist?")
    print("3. Did you run 'pip install -r yolov5/requirements.txt'?")
    print("----------------------------------------------------")
    sys.exit(1)

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SignLingo")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- Main Logic Class ---
class SignLanguageSystem:
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.mode = "asl"  # 'asl' or 'isl'
        
        # State tracking
        self.detected_chars = []
        self.current_sentence = ""
        self.last_audio_time = 0
        self.audio_cooldown = 4.0 # Seconds to wait before speaking again
        
        # Background worker for audio (Prevents video freeze)
        self.audio_executor = ThreadPoolExecutor(max_workers=1)
        
        # Load Models
        self.device = select_device('0' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.load_models()

    def load_models(self):
        """Loads models safely with error handling"""
        try:
            # UPDATE PATHS: Pointing to the 'models' folder
            logger.info("Loading YOLOv8 (ASL)...")
            asl_path = os.path.join("models", "yolov8_asl_best.pt")
            if os.path.exists(asl_path):
                self.models['asl'] = YOLO(asl_path)
            else:
                logger.error(f"❌ ASL Model missing: {asl_path}")
            
            logger.info("Loading YOLOv5 (ISL)...")
            isl_path = os.path.join("models", "yolov5_isl_best.pt")
            
            if os.path.exists(isl_path):
                self.models['isl'] = attempt_load(isl_path, device=self.device)
                
                # Get class names for ISL
                try:
                    self.isl_names = self.models['isl'].module.names if hasattr(self.models['isl'], 'module') else self.models['isl'].names
                except:
                    self.isl_names = [f"Class_{i}" for i in range(100)]
            else:
                logger.error(f"❌ ISL Model missing: {isl_path}")
                
            logger.info(f"✅ Models loaded. Device: {self.device}")
        except Exception as e:
            logger.error(f"❌ Critical Error loading models: {e}")
            import traceback
            traceback.print_exc()

    def start_camera(self):
        if self.camera is None or not self.camera.isOpened():
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                logger.error("Could not open webcam.")
                return False
        self.is_running = True
        return True

    def stop_camera(self):
        self.is_running = False
        if self.camera:
            self.camera.release()
            self.camera = None

    def process_text_logic(self, raw_text):
        """Converts raw labels like 'HELLO' into sentences"""
        mappings = {
            "HELLO": "Hello, nice to meet you",
            "THANK": "Thank you very much",
            "HELP": "I need help please",
            "YES": "Yes, that is correct",
            "NO": "No, I don't think so"
        }
        return mappings.get(raw_text, raw_text)

    def generate_and_send_audio(self, text):
        """Runs in background thread. Does NOT block video."""
        try:
            tts = gTTS(text=text, lang='en')
            filename = f"audio_{uuid.uuid4()}.mp3"
            filepath = os.path.join(tempfile.gettempdir(), filename)
            
            tts.save(filepath)
            
            with open(filepath, "rb") as audio_file:
                audio_b64 = base64.b64encode(audio_file.read()).decode('utf-8')
                socketio.emit('audio_update', {'audio_data': audio_b64})
            
            os.remove(filepath) # Cleanup
            logger.info(f"Audio sent for: {text}")
        except Exception as e:
            logger.error(f"TTS Error: {e}")

    def detect_loop(self):
        """Main Loop: Reads Camera -> Predicts -> Emits Results"""
        logger.info("Starting detection loop")
        
        while self.is_running and self.camera:
            success, frame = self.camera.read()
            if not success:
                break

            annotated_frame = frame.copy()
            detection_text = ""
            conf_score = 0.0

            # --- ASL LOGIC (YOLOv8) ---
            if self.mode == 'asl' and 'asl' in self.models:
                try:
                    results = self.models['asl'](frame, verbose=False, conf=0.5)
                    for r in results:
                        annotated_frame = r.plot() 
                        if len(r.boxes) > 0:
                            box = r.boxes[0]
                            detection_text = self.models['asl'].names[int(box.cls)]
                            conf_score = float(box.conf)
                except Exception as e:
                    logger.error(f"ASL Prediction Error: {e}")

            # --- ISL LOGIC (YOLOv5) ---
            elif self.mode == 'isl' and 'isl' in self.models:
                try:
                    # Preprocess
                    img = cv2.resize(frame, (640, 640))
                    img = img.transpose((2, 0, 1))[::-1]  
                    img = np.ascontiguousarray(img)
                    img = torch.from_numpy(img).to(self.device).float() / 255.0
                    if img.ndimension() == 3: img = img.unsqueeze(0)

                    # Inference
                    pred = self.models['isl'](img)[0]
                    pred = non_max_suppression(pred, 0.25, 0.45)

                    for det in pred:
                        if len(det):
                            det[:, :4] = self.scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                            for *xyxy, conf, cls in det:
                                label = f'{self.isl_names[int(cls)]}'
                                # Draw Box
                                cv2.rectangle(annotated_frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                                cv2.putText(annotated_frame, label, (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                
                                if conf > 0.5:
                                    detection_text = self.isl_names[int(cls)]
                                    conf_score = float(conf)
                except Exception as e:
                    logger.error(f"ISL Prediction Error: {e}")

            # --- LOGIC ---
            processed_text = ""
            if detection_text:
                # Debounce: Only accept if different from last or 1 sec passed
                if not self.detected_chars or self.detected_chars[-1] != detection_text:
                    self.detected_chars.append(detection_text)
                    if len(self.detected_chars) > 5: self.detected_chars.pop(0)
                    
                    processed_text = self.process_text_logic(detection_text)
                    
                    # Audio Trigger (Threaded)
                    if time.time() - self.last_audio_time > self.audio_cooldown:
                        self.last_audio_time = time.time()
                        self.audio_executor.submit(self.generate_and_send_audio, processed_text)

            # --- STREAMING ---
            try:
                _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                frame_b64 = base64.b64encode(buffer).decode('utf-8')

                socketio.emit('video_frame', {'frame': frame_b64})
                
                if detection_text:
                    socketio.emit('detection_update', {
                        'raw_detection': detection_text,
                        'processed_text': processed_text if processed_text else "...",
                        'confidence': conf_score
                    })
            except Exception as e:
                logger.error(f"Streaming Error: {e}")

            socketio.sleep(0.01) # Yield to other threads

    def scale_coords(self, img1_shape, coords, img0_shape):
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
        coords[:, [0, 2]] -= pad[0]
        coords[:, [1, 3]] -= pad[1]
        coords[:, :4] /= gain
        coords[:, :4] = coords[:, :4].clip(min=0)
        return coords

# --- Initialize ---
system = SignLanguageSystem()

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def connect():
    emit('status', {'message': 'Server Connected'})

@socketio.on('start_detection')
def start():
    if system.start_camera():
        emit('status', {'message': 'Camera Started'})
        socketio.start_background_task(system.detect_loop)
    else:
        emit('status', {'message': 'Camera Failed'})

@socketio.on('stop_detection')
def stop():
    system.stop_camera()
    emit('status', {'message': 'Camera Stopped'})

@socketio.on('set_mode')
def set_mode(data):
    system.mode = data.get('mode', 'asl')
    system.detected_chars = [] 
    emit('status', {'message': f'Switched to {system.mode.upper()}'})

@socketio.on('reset_detection')
def reset():
    system.detected_chars = []
    emit('detection_update', {'raw_detection': '', 'processed_text': '', 'confidence': 0})

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    socketio.run(app, debug=True, port=5000)