import cv2
import os
from shutil import copy2

def rotate_image(folder):
    
    # Loop through each file in the source folder
    for idx, filename in enumerate(os.listdir(folder)):
        # Check if the file is an image (you may adjust the extensions as needed)
        img = cv2.imread(folder+filename, 1)
        print(img.shape)
        height, width = img.shape[:2]
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and height > width:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(folder+filename, img)
            


def resize_image(folder):
    
    # Loop through each file in the source folder
    for idx, filename in enumerate(os.listdir(folder)):
        # Check if the file is an image (you may adjust the extensions as needed)
        img = cv2.imread(folder+filename, 1)
        print(img.shape)
        height, width = img.shape[:2]
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.resize(img, (int(height*0.8), int(width*0.8)))
            cv2.imwrite(folder+filename, img)

            

if __name__ == "__main__":
    rotate_image("preprocessed_data/random/")
    # print(os.walk("preprocessed_data/"))