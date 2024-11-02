import os
import time

from ultralytics import YOLO
import cv2
import numpy as np

VIDEOS_DIR = 'validate_data/videos'

video_path = os.path.join(VIDEOS_DIR, 'test1.mov')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

# Fix resolution to width x height (W, H)
H, W, _ = frame.shape

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

while ret:
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Display the frame with predictions
    cv2.imshow('Predictions', frame)
    
    # Wait for a short time to display frames; press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
