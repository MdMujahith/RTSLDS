from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO, emit
import cv2
import torch
import numpy as np
from pathlib import Path
import pathlib
import sys
import os
import base64
import json
import threading
import time
from gtts import gTTS
import tempfile
import uuid
from ultralytics import YOLO
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
from yolov5.models.experimental import attempt_load

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Fix for Windows paths
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Global variables
detection_mode = "asl"  # Default to ASL
cap = None
is_running = False
current_detection = ""
detected_chars = []
last_audio_time = 0
audio_cooldown = 3  # seconds between audio playbacks

def scale_coords(img1_shape, coords, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    coords[:, [0, 2]] -= pad[0] 
    coords[:, [1, 3]] -= pad[1] 
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clip(min=0) 
    return coords

# Load models
try:
    print("Loading YOLOv5 model for ISL...")
    isl_model_path = Path(r"C:\Users\mujah\FYP\Detection\isl.pt")
    device = select_device('cpu') 
    isl_model = attempt_load(str(isl_model_path), device)
    isl_model.eval()
    
    try:
        isl_class_names = isl_model.names
    except:
        isl_class_names = [f"ISL_{i}" for i in range(1000)]
    
    print(f"Loaded YOLOv5 ISL model with {len(isl_class_names)} classes")
    
    print("Loading YOLOv8 model for ASL...")
    asl_model_path = Path(r"C:\Users\mujah\FYP\Detection\asl1.pt")
    asl_model = YOLO(str(asl_model_path))
    
    print("Both models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def generate_audio(text):
    """Generate audio file from text and return the filename"""
    if not text:
        return None
    
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        audio_filename = os.path.join(tempfile.gettempdir(), f"audio_{uuid.uuid4()}.mp3")
        tts.save(audio_filename)
        return audio_filename
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

def process_detections(detection_string):
    """Convert raw detections to meaningful text"""
    # This is where you'd integrate your GenAI API
    # For now, we'll use some simple mappings
    
    simple_mapping = {
        "A": "Hello",
        "B": "How are you",
        "C": "Thank you",
        "HELLO": "Hello, nice to meet you",
        "THANK": "Thank you very much",
        "HELP": "I need some help please"
    }
    
    # Check if the detection string matches any key in our mapping
    for key, value in simple_mapping.items():
        if key in detection_string:
            return value
    
    # If no match and we have at least 3 characters, return something generic
    if len(detection_string) >= 3:
        return f"I'm signing: {detection_string}"
    
    return detection_string

def video_detection_thread():
    global cap, is_running, current_detection, detected_chars, last_audio_time
    
    print("Starting video detection thread")
    
    while is_running:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break
        
        display_frame = frame.copy()
        current_frame_detections = []
        
        # ISL detection
        if detection_mode in ["isl"]:
            img_isl = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_isl = cv2.resize(img_isl, (640, 640))
            img_isl = img_isl.transpose(2, 0, 1)
            img_isl = torch.from_numpy(img_isl).float().div(255.0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred_isl = isl_model(img_isl)[0]
                pred_isl = non_max_suppression(pred_isl, 0.25, 0.45)
            
            for det in pred_isl:
                if len(det):
                    det[:, :4] = scale_coords(img_isl.shape[2:], det[:, :4], frame.shape).round()
                    for *xyxy, conf, cls_id in det:
                        cls_id = int(cls_id)
                        cls_name = isl_class_names[cls_id] if cls_id < len(isl_class_names) else f"ISL_{cls_id}"
                        
                        if conf > 0.5:  # Only consider high confidence detections
                            current_frame_detections.append(cls_name)
                        
                        # Draw on display frame
                        label = f"ISL: {cls_name} {conf:.2f}"
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # ASL detection
        if detection_mode in ["asl"]:
            results = asl_model(frame)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get coordinates, confidence and class
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())
                    
                    # Get class name
                    try:
                        cls_name = result.names[cls_id]
                    except:
                        cls_name = f"ASL_{cls_id}"
                    
                    if conf > 0.5:  # Only consider high confidence detections
                        current_frame_detections.append(cls_name)
                    
                    # Format label
                    label = f"ASL: {cls_name} {conf:.2f}"
                    
                    # Draw rectangle and text
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Update the display frame with mode information
        mode_text = f"Mode: {detection_mode.upper()}"
        cv2.putText(display_frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Process current frame detections
        if current_frame_detections:
            # Take the first detection with highest confidence
            latest_detection = current_frame_detections[0]
            
            # Add to our sequence of detected characters if it's new
            if not detected_chars or detected_chars[-1] != latest_detection:
                detected_chars.append(latest_detection)
                
                # Keep only the last 5 detections to avoid too many
                if len(detected_chars) > 5:
                    detected_chars = detected_chars[-5:]
                
                # Update current detection string
                current_detection = " ".join(detected_chars)
                
                # Process the detection text
                processed_text = process_detections(current_detection)
                
                # Check if we should play audio (with cooldown)
                current_time = time.time()
                if current_time - last_audio_time > audio_cooldown:
                    audio_file = generate_audio(processed_text)
                    if audio_file:
                        last_audio_time = current_time
                        # Emit event with audio file
                        with open(audio_file, "rb") as audio_data:
                            audio_base64 = base64.b64encode(audio_data.read()).decode('utf-8')
                            socketio.emit('audio_update', {'audio_data': audio_base64})
                        # Clean up the temporary file
                        try:
                            os.remove(audio_file)
                        except:
                            pass
                
                # Emit the detection updates
                socketio.emit('detection_update', {
                    'raw_detection': current_detection,
                    'processed_text': processed_text,
                    'confidence': 0.85  # Placeholder - could calculate actual average confidence
                })
        
        # Convert the frame to JPEG for streaming
        ret, jpeg = cv2.imencode('.jpg', display_frame)
        frame_bytes = jpeg.tobytes()
        
        # Emit the frame
        socketio.emit('video_frame', {'frame': base64.b64encode(frame_bytes).decode('utf-8')})
        
        # Short sleep to reduce CPU usage
        time.sleep(0.03)

def start_video_capture():
    global cap, is_running
    
    # Initialize video capture
    if cap is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Unable to open webcam")
            return False
    
    # Start processing thread
    if not is_running:
        is_running = True
        thread = threading.Thread(target=video_detection_thread)
        thread.daemon = True
        thread.start()
    
    return True

def stop_video_capture():
    global is_running, cap
    
    is_running = False
    time.sleep(0.5)  # Give the thread time to stop
    
    if cap is not None:
        cap.release()
        cap = None

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    global is_running
    print('Client disconnected')
    if is_running:
        stop_video_capture()

@socketio.on('start_detection')
def handle_start_detection():
    print('Starting detection')
    success = start_video_capture()
    emit('status', {'message': 'Detection started' if success else 'Failed to start detection'})

@socketio.on('stop_detection')
def handle_stop_detection():
    print('Stopping detection')
    stop_video_capture()
    emit('status', {'message': 'Detection stopped'})

@socketio.on('set_mode')
def handle_set_mode(data):
    global detection_mode
    mode = data.get('mode', 'asl')
    if mode in ['asl', 'isl']:
        detection_mode = mode
        emit('status', {'message': f'Detection mode set to {mode.upper()}'})
    else:
        emit('status', {'message': 'Invalid detection mode'})

@socketio.on('reset_detection')
def handle_reset_detection():
    global detected_chars, current_detection
    detected_chars = []
    current_detection = ""
    emit('detection_update', {
        'raw_detection': "",
        'processed_text': "",
        'confidence': 0.0
    })
    emit('status', {'message': 'Detection reset'})

if __name__ == '__main__':
    # Ensure the templates directory exists
    os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
    
    # Make sure the static directory exists for CSS and JS
    os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'static', 'css'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'static', 'js'), exist_ok=True)
    
    print("Starting server...")
    socketio.run(app, debug=True)