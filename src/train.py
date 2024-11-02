from ultralytics import YOLO

# load a model
model = YOLO('yolov8n.yaml')

# train the model
results = model.train(data='config.yaml', epochs=100)