import cv2
import torch
import os
import sys

import matplotlib.pyplot as plt

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Modules", "Depth-Anything-V2"))
sys.path.append(repo_path)

# Now you can import directly from the inner folder name
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}



class DepthPredictor():
    def __init__(self):
        encoder = 'vitl' # or 'vits', 'vitb'
        dataset = 'vkitti' # 'hypersim' for indoor model, 'vkitti' for outdoor model
        max_depth = 40 # 20 for indoor model, 80 for outdoor model

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth}).to(self.device)

        self.model.load_state_dict(torch.load(f'Models/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=self.device))
        self.model.eval()

        
    def predict(self, image, format="BGR"):
        if format == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        torch.cuda.empty_cache()
        with torch.no_grad():
            depth = self.model.infer_image(image) #

            return depth