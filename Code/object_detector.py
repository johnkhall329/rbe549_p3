import cv2
import torch

import matplotlib.pyplot as plt
import os

from ultralytics import YOLO
import sys
import numpy as np
from argparse import Namespace


detic_repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Modules", "Detic"))
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Modules", "Detic", "third_party", "CenterNet2"))
sys.path.append(detic_repo_path)
sys.path.append(repo_path)

from detic.config import add_detic_config
from centernet.config import add_centernet_config
from detic.predictor import VisualizationDemo
from detectron2.config import get_cfg

class ObjectDetectorDETIC:
    def __init__(self, config_path="Modules/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml", weights_path="Models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth", confidence_threshold=0.5, device="cuda"):
        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        
        detic_root = "Modules/Detic"
        cfg.merge_from_file(config_path)
        cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = os.path.join(detic_root, "datasets/metadata/lvis_v1_train_cat_info.json")
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1203
        cfg.MODEL.CENTERNET.NUM_CLASSES = 1203
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = os.path.join(detic_root, "datasets/metadata/lvis_v1_clip_a+cname.npy")
        cfg.MODEL.RESET_CLS_TESTS = False

        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.DEVICE = device
        cfg.freeze()
        
        # The demo object handles the vocabulary and visualization logic
        # Define the settings Detic expects from the 'args' variable
        args = Namespace(
            vocabulary="lvis",          # or "custom", "coco", etc.
            custom_vocabulary="",        # only used if vocabulary="custom"
            confidence_threshold=0.5,
            opts=[],                     # any extra config overrides
            cpu=False                    # set to True if you aren't using your GPU
        )
        self.demo = VisualizationDemo(cfg, args)

    def gen_bounded_image(self, image):
        """
        Takes a BGR image (OpenCV format) and returns predictions and visualized image.
        """
        # predictions is a dict containing 'instances' (Boxes, Masks, Scores)
        predictions, visualized_output = self.demo.run_on_image(image)
        return predictions, visualized_output.get_image()

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

