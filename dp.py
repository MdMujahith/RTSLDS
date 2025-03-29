import torch
import cv2
import numpy as np
from pathlib import Path
import pathlib
import sys
import os
from ultralytics import YOLO
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
from yolov5.models.experimental import attempt_load

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

try:
    # Load YOLOv5 model for ISL
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
    
    # Load YOLOv8 model for ASL
    print("Loading YOLOv8 model for ASL...")
    asl_model_path = Path(r"C:\Users\mujah\FYP\Detection\asl1.pt")  # Update with correct path
    asl_model = YOLO(str(asl_model_path))
    
    print("Both models loaded successfully!")
    
    # Open webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Unable to open webcam")
        sys.exit()
    
    print("Webcam opened successfully! Press 'q' to quit, 'i' for ISL only, 'a' for ASL only, 'b' for both")
    
    
    detection_mode = "both"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        
        display_frame = frame.copy()
        
        
        if detection_mode in ["isl", "both"]:
            
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
                        
                        
                        label = f"ISL: {cls_name} {conf:.2f}"
                        
            
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Process with ASL model (YOLOv8)
        if detection_mode in ["asl", "both"]:
            # YOLOv8 processes the original frame directly
            results = asl_model(frame)
            
            # Draw ASL bounding boxes
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
                    
                    # Format label
                    label = f"ASL: {cls_name} {conf:.2f}"
                    
                    # Draw rectangle and text (blue for ASL)
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Display information on the frame
        mode_text = f"Mode: {detection_mode.upper()}"
        cv2.putText(display_frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_frame, "Press 'q'-quit, 'i'-ISL only, 'a'-ASL only, 'b'-both", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow("Combined Sign Language Detection", display_frame)
        
        # Check for key commands
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('i'):
            detection_mode = "isl"
            print("Switched to ISL detection only")
        elif key == ord('a'):
            detection_mode = "asl"
            print("Switched to ASL detection only")
        elif key == ord('b'):
            detection_mode = "both"
            print("Switched to combined ISL and ASL detection")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Clean up
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    print("Cleanup complete")