import os
import cv2
from ultralytics import YOLO
from utils import map_result_to_price

# Set paths
IMAGES_DIR = 'validate_data/images'
image_path = os.path.join(IMAGES_DIR, 'test7.jpg')
output_image_path = '{}_out.jpg'.format(image_path)

# Load the image
image = cv2.imread(image_path)

# Load a model and weights
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')
model = YOLO(model_path)  # load a custom model

threshold = 0.5

# Predict on the image
results = model(image)[0]
result_list = []

# Draw bounding boxes on the image
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)
        cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)
    result_list.append(int(class_id))

cv2.putText(image, "Total Price: {}".format(map_result_to_price(result_list)), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 8, cv2.LINE_AA)

# show output image
cv2.imshow("output (press any key to quit)", image)
cv2.waitKey(0) # Wait for a key press
