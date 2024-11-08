# ImageProcessingEGCO486
# Introduction
- train YOLOv11 on custom datasets for detecting and simple classifying 5 objects such as water, snacks, crackers, candy and milk.

## :zap: Usage
Write about how to use this project.

###  :electric_plug: Installation
- I use conda environment for this project or you can use virtual environment
```
$ conda create --name co486 python=3.11
& conda activate co486
& pip install uv
& uv pip install -r requirements.txt
```

##  :wrench: Development
goal is to reach this detection model on local website and use to predict summation price of the product


###  :file_folder: File Structure
basic details about files, below.

```
.
├── runs
│   ├── detect
│   │   ├── train
│   │   │   ├── weights
│   │   │   │   ├── best.pt
│   │   │   │   └── last.pt
│   │   │   └── other result from training...
├── src
│   ├── predict.py                                     // python file for predicting a simgle sample image (writing bounding box and classes).
│   ├── preprocess.py                                  // python file for preprocess the datasets before train.
│   ├── rename.py                                      // python file for rename the entire datasets.
│   ├── train.py                                       // python file for training yolov8.
│   └── video_predict.py                               // python file for predicting a video (writing bounding box and classes).
├── validate_data                                      // use for validate or test the model
│   ├── images
│   └── videos
├── README.md                                          // Overview and instructions for this project.
├── config.yaml                                        // config file for the training session.
├── main.py                                            // main python file to run this project
├── obj.data                                           // obj result from training.
├── obj.names                                          // obj result from training.
├── receipts.csv                                       // receipt history csv file
├── requirements.txt                                   // requirements packages installation for this project.
└── train.txt                                          // text file result from training.
```


##  :camera: Gallery

!["Screenshot 2567-11-05 at 15 55 20"](https://github.com/user-attachments/assets/dfac185a-3bbd-4314-a3d8-2aa3e15c9af4)

## Annotation
```
snack = 10
water = 7
milk = 12
crackers = 20
candy = 15
```

##  :electric_plug: model performance annotations
Box(P, R, mAP50, mAP50-95): This metric provides insights into the model's performance in detecting objects:

- P (Precision): The accuracy of the detected objects, indicating how many detections were correct.

- R (Recall): The ability of the model to identify all instances of objects in the images.

- mAP50: Mean average precision calculated at an intersection over union (IoU) threshold of 0.50. It's a measure of the model's accuracy considering only the "easy" detections.

- mAP50-95: The average of the mean average precision calculated at varying IoU thresholds, ranging from 0.50 to 0.95. It gives a comprehensive view of the model's performance across different levels of detection difficulty.

## :star2: Credit/Acknowledgment

```
Napasrapee Satittham
Sorawit Phattarakuldilok
Thinnaphat Phumphotingam
Wish Semangern
```
