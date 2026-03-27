import cv2
import torch

import matplotlib.pyplot as plt
import os

from ultralytics import YOLO

class ObjectDetector():
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

        lisa_path = os.path.join(model_dir, 'last.pt')
        self.lisa_model = YOLO(lisa_path)

        light_path = os.path.join(model_dir, 'best_traffic_small_yolo.pt')
        self.light_model = YOLO(light_path)
        
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

        return annotated_img
    
    def predict_all(self, image, format="BGR"):
        if format == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        main_results = self.model(image)[0]

        # Get annotated image (with boxes, labels, confidence)
        # annotated_img = results[0].plot()

        lisa_results = self.lisa_model(image)[0]
        # lisa_annotated = lisa_results[0].plot()

        light_results = self.light_model(image)[0]
        # light_annotated = light_results[0].plot()

        
        combined_res = main_results.new() 
        combined_res.path = main_results.path
        combined_res.orig_img = main_results.orig_img


        combined_res.boxes = main_results.boxes
        if len(lisa_results.boxes) > 0:
            combined_res.boxes.data = torch.cat([combined_res.boxes.data, lisa_results.boxes.data], dim=0)
        if len(light_results.boxes) > 0:
            combined_res.boxes.data = torch.cat([combined_res.boxes.data, light_results.boxes.data], dim=0)

        fused_img = combined_res.plot()

        return {'yolo26': main_results, 'lisa': lisa_results, 'lights': light_results}, fused_img

