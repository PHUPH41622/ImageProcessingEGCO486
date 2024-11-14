import os
import time
from utils import map_result_to_price
from ultralytics import YOLO
import cv2
import numpy as np

VIDEOS_DIR = 'test_data/videos'

video_path = os.path.join(VIDEOS_DIR, 'test1.mov')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

# Fix resolution to width x height (W, H)
H, W, _ = frame.shape

model_path = os.path.join('.', 'weights', 'yolov11s_aug.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

while ret:
    results = model(frame)[0]
    result_list = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        
        if class_id == 0:
            item_name = "Snack"
        elif class_id == 1:
            item_name = "Water"
        elif class_id == 2:
            item_name = "Milk"
        elif class_id == 3:
            item_name = "Crackers"
        elif class_id == 4:
            item_name = "Candy"
                
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, item_name.upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        result_list.append(int(class_id))

    # Display the frame with predictions
    cv2.putText(frame, "Total Price: {}".format(map_result_to_price(result_list)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Predictions', frame)
    
    # Wait for a short time to display frames; press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
