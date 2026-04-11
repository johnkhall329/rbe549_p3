import os
import sys

from loguru import logger as loguru_logger
import numpy as np
import torch
import random
from PIL import Image

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Modules", "MemFlow"))
sys.path.append(repo_path)

from core.Networks import build_network
from core.utils import flow_viz
from core.utils import frame_utils
from core.utils.utils import InputPadder, forward_interpolate
from configs.kitti_memflownet_t import get_cfg

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Modules", "MemFlow", "inference"))
sys.path.append(repo_path)

import inference_core_skflow as inference_core

class FlowDetector():
    def __init__(self):
        cfg = get_cfg()
        self.model = build_network(cfg).cuda()

        ckpt = torch.load('./Models/MemFlowNet_T_kitti.pth', map_location='cpu')
        ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt

        if 'module' in list(ckpt_model.keys())[0]:
            for key in ckpt_model.keys():
                ckpt_model[key.replace('module.', '', 1)] = ckpt_model.pop(key)
            self.model.load_state_dict(ckpt_model, strict=True)
        else:
            self.model.load_state_dict(ckpt_model, strict=True)

        self.model.eval()