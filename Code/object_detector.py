import cv2
import torch

import matplotlib.pyplot as plt
import os

from ultralytics import YOLO

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from accelerate import Accelerator

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
    

class ObjectDetectorGroundedDINO():
    def __init__(self, device="cpu"):

        self.device = device

        dino_model_id = "IDEA-Research/grounding-dino-tiny"

        # self.dino_label_list = [["car", "person", "traffic light", "truck", "fire hydrant", "stop sign", "stop", "speed limit sign", "garbage bin", "bicycle", "traffic cone", "motorcycle", "trash can"]]
        self.dino_labels = "sedan . " \
        "hatchback . " \
        "person . " \
        "traffic light . " \
        "pickup truck . " \
        "box truck . " \
        "SUV . " \
        "fire hydrant . " \
        "stop sign . " \
        "stop . " \
        "garbage bin . " \
        "bicycle . " \
        "cone . " \
        "motorcycle . " \
        "road sign . " \
        
        self.dino_traffic_labels = "circular light . asymmetric light . green . yellow . red . triangular light . left arrow light . diamond light . 3 . non-circular light . chevron light . left leaning light ."

        self.processor = AutoProcessor.from_pretrained(dino_model_id)
        self.grounded_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(self.device)

    
    def predict_traffic(self, image):

        upscaled_light = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)

        height, width = upscaled_light.shape[:2]

        torch.cuda.empty_cache()
        dino_inputs = self.processor(images=upscaled_light, text=self.dino_traffic_labels, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.grounded_dino_model(**dino_inputs)

        dino_result = self.processor.post_process_grounded_object_detection(
            outputs,
            dino_inputs.input_ids,
            threshold=0.4,
            text_threshold=0.3,
            target_sizes=[(height, width)]
        )[0]


        # use some logic to parse this
        return dino_result["labels"][0]

    def predict(self, image, format="BGR"):
        if format == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        height, width = image.shape[:2]

        torch.cuda.empty_cache()
        dino_inputs = self.processor(images=image, text=self.dino_labels, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.grounded_dino_model(**dino_inputs)

        dino_result = self.processor.post_process_grounded_object_detection(
            outputs,
            dino_inputs.input_ids,
            threshold=0.4,
            text_threshold=0.3,
            target_sizes=[(height, width)]
        )[0]   

        dino_img = image.copy()
        details = []
        for box, score, label in zip(dino_result["boxes"], dino_result["scores"], dino_result["labels"]):
            # Convert box to integers
            xmin, ymin, xmax, ymax = map(int, box.tolist())
            
            # Draw rectangle (Green)
            cv2.rectangle(dino_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Add label text
            label_text = f"{label}: {score:.2f}"
            cv2.putText(dino_img, label_text, (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            image_crop = image[ymin:ymax, xmin:xmax]

            # detail = self.analyze_details(image_crop, label)
            # details.append(detail)

        # dino_result["details"] = details
        return dino_result, dino_img

    def analyze_details(self, image, label):
        if label == "traffic light":
            return self.predict_traffic()
        elif label == "person":
            pass
        elif label == "car":
            pass
        return ''