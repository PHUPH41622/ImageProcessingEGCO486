import os
import cv2
from ultralytics import YOLO

# Set paths
IMAGES_DIR = 'validate_data/images'
image_path = os.path.join(IMAGES_DIR, 'test6.jpg')
output_image_path = '{}_out.jpg'.format(image_path)

# Load the image
image = cv2.imread(image_path)

# Load a model
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')
model = YOLO(model_path)  # load a custom model

threshold = 0.5

# Predict on the image
results = model(image)[0]

# Draw bounding boxes on the image
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
        cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# Save the output image
cv2.imshow("output (press any key to quit)", image)
cv2.waitKey(0) # Wait for a key press
