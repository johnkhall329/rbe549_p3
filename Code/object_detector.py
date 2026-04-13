import cv2
import torch
import torchvision
import re

import easyocr
from rapidfuzz import process, fuzz
import matplotlib.pyplot as plt
import os
import numpy as np
import glob

from ultralytics import YOLO

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from accelerate import Accelerator

from traffic_light_classification import classify_light
from orient_anything_detection import OrientAnythingModel

import supervision as sv
from supervision.draw.color import ColorPalette

from pedestrian_pose import HumanDetector
from car_signal_detection import detect_signals

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CUSTOM_COLOR_MAP = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]

SAM_2_CKPT = "./Modules/Grounded-SAM-2/checkpoints/sam2.1_hiera_small.pt"
SAM_2_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

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
    def __init__(self, camera_calib, device="cpu"):

        self.device = device
        self.K = camera_calib

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

        # Segmentation
        self.sam2_model = build_sam2(SAM_2_CFG, SAM_2_CKPT, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        self.daylight_thresh = 100.0
    
    # def predict_traffic(self, image):

    #     upscaled_light = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)

    #     height, width = upscaled_light.shape[:2]

    #     torch.cuda.empty_cache()
    #     dino_inputs = self.processor(images=upscaled_light, text=self.dino_traffic_labels, return_tensors="pt").to(self.device)

    #     with torch.no_grad():
    #         outputs = self.grounded_dino_model(**dino_inputs)

    #     dino_result = self.processor.post_process_grounded_object_detection(
    #         outputs,
    #         dino_inputs.input_ids,
    #         threshold=0.4,
    #         text_threshold=0.3,
    #         target_sizes=[(height, width)]
    #     )[0]


    #     # use some logic to parse this
    #     return dino_result["labels"][0]

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
            # might save memory?
            box, score = box.cpu(), score.cpu().numpy()

            # parse labels that G-DINO says is two words
            if label.split()[0] in {"sedan", "hatchback", "suv", "pickup", "box", "motorcycle", "bicycle", "truck"}:
                label = label.split()[0]
            new_labels.append(label)
            new_boxes.append(box)
            new_scores.append(score)
            xmin, ymin, xmax, ymax = map(int, box.tolist())
            
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

        # Add yolo cars that G-DINO misses
        yolo_results = self.yolo(image)
        yolo_cars, yolo_conf = self.sort_yolo(yolo_results[0])
        for yolo_box, conf in zip(yolo_cars, yolo_conf):
            for dino_box, dino_label in zip(dino_result["boxes"], new_labels):
                overlap = False
                if dino_label in {"sedan", "hatchback", "suv", "pickup"}:
                    iou = torchvision.ops.box_iou(yolo_box, dino_box.detach().cpu()[None, :])
                    if iou > 0.25:
                        overlap = True
                        break
            if not overlap:
                new_boxes.append(yolo_box[0])
                new_scores.append(conf)
                new_labels.append("sedan")

                xmin, ymin, xmax, ymax = map(int, yolo_box[0].tolist())
                cv2.rectangle(dino_img, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)
            
                # Add label text
                label_text = f"sedan: {conf:.2f}"
                cv2.putText(dino_img, label_text, (xmin, ymin - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                detail = self.analyze_details(image, yolo_box[0], "sedan", None)
                details.append(detail)             

        # Segmentation
        self.sam2_predictor.set_image(image)

        input_boxes = np.stack(new_boxes)
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # comes out as (c, 1, h, w)
        if len(masks.shape) > 3: masks = masks.squeeze(1)

        # Plotting
        class_ids = np.array(list(range(len(new_labels))))

        plot_labels = [
            f"{class_name} {float(confidence):.2f}"
            for class_name, confidence
            in zip(new_labels, new_scores)
        ]

        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = box_annotator.annotate(scene=dino_img, detections=detections)

        label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=plot_labels)

        mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

        dino_result["masks"] = list(masks)
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
            signals = tuple(map(bool, signals))

            raw_orientation = self.orient_anything_model.predict(image[ymin:ymax, xmin:xmax])

            center_offset = (xmax + xmin)/2 - self.K[0, 2]

            theta_ray = np.rad2deg(np.arctan2(center_offset, self.K[0,0]))

            rot_val = raw_orientation - theta_ray

            # round close angles
            for degree in range(-180, 181, 90):
                if abs(rot_val - degree) < 8:
                    rot_val = degree

            return {"orientation": rot_val, "signals": signals}
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
                        if len(integer_list) > 0:
                            speed = np.max(integer_list)
                            return {"type": "speed limit", "speed": str(speed)}
                        
                    if "stop" in lisa_cls.lower():
                        if any('stop' in t.lower() for t in text):
                            return {"type": "stop"}
            return self.check_ocr_signs(text)
        return ''
    
    def check_ocr_signs(self, ocr_text):
        if len(ocr_text) < 1: return {}
        speed_match = process.extractOne("SPEED", ocr_text, scorer=fuzz.WRatio)
        stop_match = process.extractOne("STOP", ocr_text, scorer=fuzz.WRatio)
        limit_match = process.extractOne("LIMIT", ocr_text, scorer=fuzz.WRatio)
        bump_match = process.extractOne("BUMP", ocr_text, scorer=fuzz.WRatio)
        hump_match = process.extractOne("HUMP", ocr_text, scorer=fuzz.WRatio)

        if speed_match[1] > 80 and limit_match[1] > 80:
            integer_list = []
            for t in ocr_text:
                numbers = re.findall(r'\d+', t)
                integer_list += [int(n) for n in numbers]
            if len(integer_list) >0:
                speed = np.max(integer_list)
                return {"type": "speed limit", "speed": str(speed)}
        elif speed_match[1] > 80 and (bump_match[1] > 80 or hump_match[1] > 80):
            return {"type": "speed bump"}
        elif stop_match[1] > 80:
            return {"type": "stop"}
        else:
            return {}

    def sort_yolo(self, results, iou_thresh=0.2):
        yolo_cars = []
        yolo_conf = []
        for box in results.boxes:
            if results.names[int(box.cls[0])] == "car":
                conf = float(box.conf.detach().cpu())
                if conf > 0.4:
                    yolo_cars.append(box.xyxy.detach().cpu())
                    yolo_conf.append(float(box.conf.detach().cpu()))
        sorted_cars = []
        sorted_conf = []
        if len(yolo_cars) > 2:
            merged_idxs = []
            for i, i_box in enumerate(yolo_cars[:-1]):
                max_iou = iou_thresh
                iou_idx = None
                for j, j_box in enumerate(yolo_cars[i+1:]):
                    iou = torchvision.ops.box_iou(i_box, j_box)
                    if iou > iou_thresh and iou > max_iou and i+j+1 not in merged_idxs:
                        iou_idx = i+j+1
                        max_iou = float(iou)
                merged_idxs.append((max_iou, i, iou_idx))
            merged_idxs.append((iou_thresh, len(yolo_cars)-1, None))
            merged = []
            for _, i_car, j_car in sorted(merged_idxs, key=lambda item:item[0], reverse=True):
                if i_car not in merged and j_car is not None:
                    i_box = yolo_cars[i_car]
                    j_box = yolo_cars[j_car]
                    combined = torch.cat((i_box, j_box))
                    x_min, y_min = combined[:,:2].min(axis=0)[0]
                    x_max, y_max = combined[:,2:].max(axis=0)[0]
                    new_box = torch.tensor([[x_min, y_min, x_max, y_max]])
                    merged.append(i_car)
                    merged.append(j_car)
                    sorted_cars.append(new_box)
                    max_conf = float(max(yolo_conf[i_car], yolo_conf[j_car]))
                    sorted_conf.append(max_conf)
                elif i_car not in merged and j_car is None:
                    sorted_cars.append(yolo_cars[i_car])
                    sorted_conf.append(yolo_conf[i_car])
                else:
                    continue
        else:
            sorted_cars = yolo_cars
            sorted_conf = yolo_conf
        return sorted_cars, sorted_conf