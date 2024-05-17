import os 
import random

import cv2 
from ultralytics import YOLO

from tracker import Tracker 


video_path = '/home/jorge/kalman_env/test.mp4'

cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()

model = YOLO("yolov8n.pt")

tracker = Tracker()

colores = [(random.randint(0, 255), random.radint(0,255)) for j in range(10)]


while ret: 

    obj = model(frame) 

    for objeto in obj:
        detections =[]
        for r in obj.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            detections.append([int(x1), int(y1), int(x2), int(y2), 
                               int(class_id), score])
        
        tracker.update(frame, detections)
        
        for track in tracker.tracks:
            bbox = track,bbox 
            track_id = track.track_id
            x1, y1, x2, y2 = bbox
            
            cv2.rectangle(frame, (x1, y1), (x2,y2), 
                          (colores[track_id % len(colores)]),3)
    
    
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(25)

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()