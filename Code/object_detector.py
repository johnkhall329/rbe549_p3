import cv2
import torch

import matplotlib.pyplot as plt
import os

from ultralytics import YOLO

class ObjectDetectorYolo():
    # model_type should be 
    def __init__(self, model_name="yolo26n.pt"):
        # Define the target directory
        model_dir = "Models"
        
        # Create the directory if it doesn't exist yet
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        # Construct the full path: "Models/yolo11n.pt"
        model_path = os.path.join(model_dir, model_name)
        self.model = YOLO(model_path)
        
    def predict(self, image, format="BGR"):
        if format == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.model(image)

        return results[0]

    def gen_bounded_image(self, image, format="BGR"):
        if format == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.model(image)

        # Get annotated image (with boxes, labels, confidence)
        annotated_img = results[0].plot()

        # Convert back to BGR for saving with cv2
        # annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

        # cv2.imwrite('Output/test.jpg', annotated_img)

        return annotated_img, results[0]

