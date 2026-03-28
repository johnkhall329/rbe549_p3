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

        dino_model_id = "IDEA-Research/grounding-dino-tiny"

        self.dino_labels = [["red light", " car", "truck", "pedestrian", "stop sign", "yield sign", "green arrow", "stop light", "garbage bin"]]

        self.processor = AutoProcessor.from_pretrained(dino_model_id)
        self.grounded_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id) # .to(device)

        
        
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
        
        height, width = image.shape[:2]

        main_results = self.model(image)[0]

        # Get annotated image (with boxes, labels, confidence)
        # annotated_img = results[0].plot()

        lisa_results = self.lisa_model(image)[0]
        # lisa_annotated = lisa_results[0].plot()

        light_results = self.light_model(image)[0]
        # light_annotated = light_results[0].plot()

        dino_inputs = self.processor(images=image, text=self.dino_labels, return_tensors="pt")# .to(model.device)
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
        for box, score, label in zip(dino_result["boxes"], dino_result["scores"], dino_result["labels"]):
            # Convert box to integers
            xmin, ymin, xmax, ymax = map(int, box.tolist())
            
            # Draw rectangle (Green)
            cv2.rectangle(dino_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Add label text
            label_text = f"{label}: {score:.2f}"
            cv2.putText(dino_img, label_text, (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        combined_res = main_results.new() 
        combined_res.path = main_results.path
        combined_res.orig_img = main_results.orig_img


        combined_res.boxes = main_results.boxes
        if len(lisa_results.boxes) > 0:
            combined_res.boxes.data = torch.cat([combined_res.boxes.data, lisa_results.boxes.data], dim=0)
        if len(light_results.boxes) > 0:
            combined_res.boxes.data = torch.cat([combined_res.boxes.data, light_results.boxes.data], dim=0)

        fused_img = combined_res.plot()

        return {'yolo26': main_results, 'lisa': lisa_results, 'lights': light_results}, fused_img, dino_img

