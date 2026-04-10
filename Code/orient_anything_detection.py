import os
import sys

from transformers import AutoImageProcessor
import torch
from PIL import Image

from huggingface_hub import hf_hub_download

import numpy as np
import torch
import tempfile
import torch.nn.functional as F

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Modules", "Orient-Anything"))
sys.path.append(repo_path)

from paths import *
from vision_tower import DINOv2_MLP

from utils import *
from inference import *


# repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Modules", "Orient-Anything-V2"))
# sys.path.append(repo_path)

# from vision_tower import VGGT_OriAny_Ref
# from utils.app_utils import *

# HF_CKPT_PATH = "demo_ckpts/rotmod_realrotaug_best.pt"

class OrientAnythingModel():
    def __init__(self, device='cpu'):
        ckpt_path = hf_hub_download(repo_id="Viglong/Orient-Anything", filename="croplargeEX2/dino_weight.pt", repo_type="model", cache_dir='./Models/', local_dir='./Models/', resume_download=True)

        if not device:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # In the library itself, you can change where this saves to from './' to './Models/'
        dino = DINOv2_MLP(
                            dino_mode   = 'large',
                            in_dim      = 1024, # 1024 for large, 384 for small
                            out_dim     = 360+180+180+2,
                            evaluate    = True,
                            mask_dino   = False,
                            frozen_back = False
                        )

        dino.eval()
        print('model create')
        dino.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.dino = dino.to(self.device)
        print('weight loaded')
        self.val_preprocess = AutoImageProcessor.from_pretrained(DINO_LARGE, save_path='./Models/',cache_dir='./Models/')

    def predict(self, image):
        # image_path = 'P3Data/Sequences/test/cropped_ims/cropped_truck.jpg'
        # origin_image = Image.open(image_path).convert('RGB')
        origin_image = Image.fromarray(image.astype('uint8'))
        angles = get_3angle(origin_image, self.dino, self.val_preprocess, self.device)
        azimuth     = float(angles[0])
        # polar       = float(angles[1])
        # rotation    = float(angles[2])
        # confidence  = float(angles[3])

        rot_val = -(azimuth - 180)
        return rot_val
    
# TAKEN AND HOPEFULLY OVERWRITTEN FROM INFERENCE, IF THERE ARE TORCH NUMPY CONVERSION ISSUES CHECK THIS AND THE CORRESPONDING INFERENCE FUNCTION
def get_3angle(image, dino, val_preprocess, device):
    
    # image = Image.open(image_path).convert('RGB')
    image_inputs = val_preprocess(images = image)
    image_inputs['pixel_values'] = torch.stack(image_inputs['pixel_values']).to(device)
    with torch.no_grad():
        dino_pred = dino(image_inputs)

    gaus_ax_pred   = torch.argmax(dino_pred[:, 0:360], dim=-1)
    gaus_pl_pred   = torch.argmax(dino_pred[:, 360:360+180], dim=-1)
    gaus_ro_pred   = torch.argmax(dino_pred[:, 360+180:360+180+360], dim=-1)
    confidence     = F.softmax(dino_pred[:, -2:], dim=-1)[0][0]
    angles = torch.zeros(4)
    angles[0]  = gaus_ax_pred
    angles[1]  = gaus_pl_pred - 90
    angles[2]  = gaus_ro_pred - 180
    angles[3]  = confidence
    return angles



# model = OrientAnythingModel()

# print(model.predict(None))


# class OrientAnythingV2Model():
#     def __init__(self, device='cpu'):
        
#         mark_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

#         if not device:
#             self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         else:
#             self.device = device

#         ckpt_path = hf_hub_download(repo_id="Viglong/Orient-Anything-V2", filename=HF_CKPT_PATH, repo_type="model", cache_dir='./Models/', local_dir='./Models/', resume_download=True)


#         model = VGGT_OriAny_Ref(out_dim=900, dtype=mark_dtype, nopretrain=True)
#         model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
#         model.eval()
#         self.model = model.to(device)
#         print('Orient-Anything-V2 weights loaded')

#     def predict(self, image, mask):
#         image_path = 'P3Data/Sequences/test/cropped_ims/cropped_truck.jpg'
#         pathed_image = Image.open(image_path).convert('RGB')

#         pil_ref = self.process_to_pil(pathed_image, mask)

#         pil_ref_pre = background_preprocess(pil_ref, do_remove_background=True)

#         try:
#             results = inf_single_case(self.model, pil_ref_pre, None)
            
#             # Parse the results
#             print(f"--- Inference Results ---")
#             print(f"Reference - Azimuth: {results['ref_az_pred'].item():.2f}°")
#             print(f"Reference - Elevation: {results['ref_el_pred'].item():.2f}°")
#             print(f"Reference - Roll: {results['ref_ro_pred'].item():.2f}°")
#             print(f"Symmetry Alpha: {results['ref_alpha_pred'].item()}")
#             return results
        
#         except Exception as e:
#             print(f"Inference failed: {e}")
#             return None
        

#     def process_to_pil(self, img_arr, mask_arr):
#         # Combine Image + Mask into RGBA for the background_preprocess logic
#         if mask_arr is not None:
#             # Ensure mask is 0-255 uint8
#             if mask_arr.max() <= 1.0: mask_arr = (mask_arr * 255).astype(np.uint8)
#             rgba = np.dstack((img_arr, mask_arr))
#             return Image.fromarray(rgba, 'RGBA')
#         return Image.fromarray(img_arr).convert('RGB')
    

# model = OrientAnythingV2Model(
# )

# prediction = model.predict(None, None)
# print(prediction)