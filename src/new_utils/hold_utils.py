import torch
import math
import numpy as np
import cv
from ultralytics import YOLO

def predict_and_detect(model, img_path):
    results = chosen_model.predict(img_path, conf=0.25, imgsz=(640,640), iou=0.45, agnostic_nms=True)
    holds = []
    img = Image.open(img_path)
    img = np.array(img)
    boxes = []

    # Extracting hold images
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]),int(box.xyxy[0][3])
            holds.append(img[y1:y2,x1:x2].copy())
            boxes.append((x1, y1, x2, y2))
            
    # Drawing the bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]),int(box.xyxy[0][3])
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
            
    return img, holds, boxes

def get_holds(model_path, img_path):
    model = YOLO(model_path)
    out, holds = predict_and_detect(model, img_path) #output image, list of hold images
    display(Image.fromarray(out))



get_holds('best.pt', '')
    
