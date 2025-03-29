from flask import Flask, render_template, Response
import torch
import cv2
import numpy as np
from pathlib import Path
import pathlib
from ultralytics import YOLO
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
from yolov5.models.experimental import attempt_load

app = Flask(__name__)

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def scale_coords(img1_shape, coords, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clip(min=0)
    return coords

# Load models
device = select_device('cpu')
isl_model = attempt_load("C:/Users/mujah/FYP/Detection/isl.pt", device)
asl_model = YOLO("C:/Users/mujah/FYP/Detection/asl1.pt")

def generate_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        display_frame = frame.copy()
        
        # ISL detection
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
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # ASL detection
        results = asl_model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        _, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)