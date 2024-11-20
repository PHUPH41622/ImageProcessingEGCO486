import subprocess
import ast
import os
import sys
import cv2
from ultralytics import YOLO
from utils import map_result_to_price

def evaluate(testing_image):
    # Set paths
    IMAGES_DIR = 'val/images/'
    image_path = os.path.join(IMAGES_DIR, testing_image)
    output_image_path = '{}_out.jpg'.format(image_path)

    # Load the image
    image = cv2.imread(image_path)

    # Load a model and weights
    model_path = os.path.join('.', 'weights', 'yolov11n_aug.pt')
    model = YOLO(model_path)  # load a custom model

    threshold = 0.5

    # Predict on the image
    results = model(image)[0]
    result_list = []

    # Draw bounding boxes on the image
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
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)
            cv2.putText(image, item_name.upper(), (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)
        result_list.append(int(class_id))

    return result_list

def main():
    test_images = sorted(os.listdir('val/images'))
    labels = sorted(os.listdir('val/labels'))
    amount = len(test_images)
    result_list = []

    labels_content = []
    for label_file in labels:
        with open(os.path.join('val/labels', label_file), 'r') as file:
            lines = file.read().strip().splitlines()
            first_column = [int(line.split()[0]) for line in lines]
            first_column.sort()
            labels_content.append(first_column)

    for index, image in enumerate(test_images):
        output = evaluate(image)
        output = str(output)
        
        try:
            result_list = ast.literal_eval(output[output.find('['):output.find(']')+1])
            result_list.sort()
        
        except (ValueError, SyntaxError):
            print("Error: Could not parse output as a list")

        print(f"{result_list} : {labels_content[index]}")
        if result_list != labels_content[index]:
            amount =  amount - 1

    accuracy = (100 * amount) / 44
    print(f'accuracy of this model is {accuracy}')

if __name__ == '__main__':
    main()







