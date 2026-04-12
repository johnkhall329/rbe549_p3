import os
import sys
import cv2

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
    def __init__(self, device='cpu'):
        self.cfg = get_cfg()
        self.device = device
        if device == 'cpu':
            self.model = build_network(self.cfg)
        elif device == 'cuda':
            self.model = build_network(self.cfg).cuda()

        ckpt = torch.load('./Models/MemFlowNet_T_kitti.pth', map_location='cpu', weights_only=True)
        ckpt_model = ckpt['model'] if 'model' in ckpt else ckpt

        if 'module' in list(ckpt_model.keys())[0]:
            for key in ckpt_model.keys():
                ckpt_model[key.replace('module.', '', 1)] = ckpt_model.pop(key)
            self.model.load_state_dict(ckpt_model, strict=True)
        else:
            self.model.load_state_dict(ckpt_model, strict=True)

        self.model.eval()

    def predict(self, image_list, save=False):
        init_height, init_width = image_list[0].shape[:2]
        image_list = [cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) for img in image_list]

        imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in image_list]
        imgs = [np.array(img).astype(np.uint8) for img in imgs]
        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
        images = torch.stack(imgs).to(self.device)

        processor = inference_core.InferenceCore(self.model, config=self.cfg)
        # 1, T, C, H, W
        if self.device == 'cuda':
            images = images.cuda().unsqueeze(0)
        elif self.device == 'cpu':
            images = images.unsqueeze(0)

        padder = InputPadder(images.shape)
        images = padder.pad(images)

        images = 2 * (images / 255.0) - 1.0
        flow_prev = None
        results = []
        print(f"start inference...")
        with torch.no_grad():
            for ti in range(images.shape[1] - 1):
                print('processer step')
                flow_low, flow_pre = processor.step(images[:, ti:ti + 2], end=(ti == images.shape[1] - 2),
                                                    add_pe=('rope' in self.cfg and self.cfg.rope), flow_init=flow_prev)
                print('unpad')
                flow_pre = padder.unpad(flow_pre[0]).cpu()
                print('flow pre')
                results.append(flow_pre.permute(1, 2, 0).detach().numpy())

        flow_np = results[0]

        flow_np = cv2.resize(flow_np, (init_width, init_height), interpolation=cv2.INTER_LINEAR)
        
        flow_img = None
        
        if save:
            flow_img = flow_viz.flow_to_image(flow_np)
            image = Image.fromarray(flow_img)
            image.save('Output/flow.jpg')

        return flow_np, flow_img




if __name__ == "__main__":
    im_1 = "optical_test_4frame1.png"
    im_2 = "optical_test_4frame2.png"
    test_dir = "P3Data/Sequences/test/optical_flow_test/"
    image_list = [cv2.imread(f"{test_dir}{im_1}", cv2.IMREAD_COLOR), cv2.imread(f"{test_dir}{im_2}", cv2.IMREAD_COLOR)]
    
    flow_det = FlowDetector(device='cpu')

    flow_out = flow_det.predict(image_list, save=True)
    print(flow_out.shape)