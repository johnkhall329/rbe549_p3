import cv2
import torch
import torchvision
import re

import easyocr
import matplotlib.pyplot as plt
import os
import numpy as np
import glob

from ultralytics import YOLO

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from accelerate import Accelerator

from orientation_detection import detect3d
from traffic_light_classification import classify_light
from orient_anything_detection import OrientAnythingModel
from pedestrian_pose import HumanDetector
from car_signal_detection import detect_signals

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
        "garbage bin . " \
        "bicycle . " \
        "cone . " \
        "motorcycle . " \
        "road sign . " \
        
        # "stopsign . " \
        # "stop . " \
        # self.dino_traffic_labels = "circular light . asymmetric light . green . yellow . red . triangular light . left arrow light . diamond light . 3 . non-circular light . chevron light . left leaning light ."
        
        self.processor = AutoProcessor.from_pretrained(dino_model_id)
        self.grounded_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(self.device)

        self.human_detector = HumanDetector()

        self.reader = easyocr.Reader(['en'])

        # lisa_path = os.path.join(model_dir, 'last.pt')
        self.lisa_model = YOLO("./Models/last.pt")
        self.lisa_model.eval()

        # Orient Anything
        self.orient_anything_model = OrientAnythingModel(device='cpu')
        
        # Supplemental Car Detection
        self.yolo = YOLO("./Models/yolo26n.pt")
        self.yolo.eval()

        self.daylight_thresh = 100.0
    
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
        human_objs = glob.glob("./Output/humans/*.obj") # clear previous run of human predictions
        for obj_file in human_objs:
            if os.path.isfile(obj_file) or os.path.islink(obj_file): os.remove(obj_file)

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
        if any(l == 'road sign' for l in dino_result["text_labels"]):
            lisa_results = self.lisa_model(image)[0]
        else:
            lisa_results = None
        check_overlap = any('motorcycle' in l or 'bicycle' in l for l in dino_result["text_labels"])
        new_labels = []
        new_boxes = []
        new_scores = []            
        for box, score, label in zip(dino_result["boxes"], dino_result["scores"], dino_result["text_labels"]):
            # parse labels that G-DINO says is two words
            if label.split()[0] in {"sedan", "hatchback", "suv", "pickup", "box", "motorcycle", "bicycle", "truck"}:
                label = label.split()[0]
            new_labels.append(label)
            new_boxes.append(box)
            new_scores.append(score)
            xmin, ymin, xmax, ymax = map(int, box.tolist())
            
            # Draw rectangle (Green)
            cv2.rectangle(dino_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Add label text
            label_text = f"{label}: {score:.2f}"
            cv2.putText(dino_img, label_text, (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detail = self.analyze_details(image, box, label, lisa_results)
            if check_overlap and label == 'person':
                best_iou = 0.0
                overlap_i = None
                for i, box2, label2 in zip(range(len(dino_result["boxes"])), dino_result["boxes"], dino_result["text_labels"]):
                    if 'motorcycle' in label2 or 'bicycle' in label2:
                        iou = torchvision.ops.box_iou(box.detach().cpu()[None,:], box2.detach().cpu()[None,:])
                        if iou > 0.25 and iou>best_iou:
                            best_iou = iou
                            overlap_i = i
                if overlap_i is not None: detail.append(overlap_i)
            details.append(detail)

        yolo_results = self.yolo(image)
        for yolo_box in yolo_results[0].boxes:
            yolo_label = yolo_results[0].names[int(yolo_box.cls[0])]
            if yolo_label in ["car"]:
                yolo_label = "sedan"
                for dino_box, dino_label in zip(dino_result["boxes"], new_labels):
                    overlap = False
                    if dino_label in {"sedan", "hatchback", "suv", "pickup"}:
                        iou = torchvision.ops.box_iou(yolo_box.xyxy.detach().cpu(), dino_box.detach().cpu()[None, :])
                        if iou > 0.25:
                            overlap = True
                            break
                if not overlap:
                    new_boxes.append(yolo_box.xyxy[0])
                    new_scores.append(yolo_box.conf)
                    new_labels.append(yolo_label)

                    xmin, ymin, xmax, ymax = map(int, yolo_box.xyxy[0].tolist())
                    cv2.rectangle(dino_img, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
                
                    # Add label text
                    label_text = f"{yolo_label}: {score:.2f}"
                    cv2.putText(dino_img, label_text, (xmin, ymin - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
                    detail = self.analyze_details(image, yolo_box.xyxy[0], yolo_label, None)
                    details.append(detail)

        dino_result["details"] = details
        dino_result["new_labels"] = new_labels
        dino_result["new_boxes"] = new_boxes
        dino_result["new_scores"] = new_scores
        return dino_result, dino_img
    

    def analyze_details(self, image, box, label, lisa_results):
        if label == "traffic light":
            return classify_light(image, box)
        elif label == "person":
            return self.human_detector.detect_humans(image, box)
        elif label in {"sedan", "hatchback", "suv", "pickup", "truck", "box", "motorcycle", "bicycle"}:
            xmin, ymin, xmax, ymax = map(int, box.tolist())
            bounds = [(xmin, ymin), (xmax, ymax)]
            signals = detect_signals(image, bounds, self.daylight_thresh)
            # return f"orientation: {detect3d(image, bounds, label)}"
            return f"orientation: {self.orient_anything_model.predict(image[ymin:ymax, xmin:xmax])}"
        elif label == 'road sign':
            xmin, ymin, xmax, ymax = map(int, box.tolist())
            crop = cv2.cvtColor(image[ymin:ymax, xmin:xmax], cv2.COLOR_RGB2BGR)
            text = self.reader.readtext(crop, detail=0)
            for lisa_box in lisa_results.boxes:
                iou = torchvision.ops.box_iou(box.detach().cpu().unsqueeze(0), lisa_box.xyxy.detach().cpu())
                if iou > 0.5:
                    lisa_cls = lisa_results.names[int(lisa_box.cls[0])]
                    if "speedLimit" in lisa_cls:
                        integer_list = []
                        for t in text:
                            numbers = re.findall(r'\d+', t)
                            integer_list += [int(n) for n in numbers]
                        if len(integer_list) >0:
                            speed = np.max(integer_list)
                            return {"type": "speed limit", "speed": str(speed)}
                        
                    if "stop" in lisa_cls.lower():
                        if any('stop' in t.lower() for t in text):
                            return {"type": "stop"}
            return {}
        return ''
