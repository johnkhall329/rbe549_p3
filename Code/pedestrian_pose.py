import torch
import numpy as np
import os
import glob

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full

class HumanDetector():
    def __init__(self, scene_name):
        download_models(CACHE_DIR_4DHUMANS)
        self.hmr2, self.hmr2_cfg = load_hmr2()
        self.hmr2.eval()
        self.hmr2_renderer = Renderer(self.hmr2_cfg, self.hmr2.smpl.faces)
        os.makedirs(f"./Output/{scene_name}/humans", exist_ok=True)

        human_objs = glob.glob("./Output/humans/*.obj") # clear previous run of human predictions
        for obj_file in human_objs:
            if os.path.isfile(obj_file) or os.path.islink(obj_file): os.remove(obj_file)

    def detect_humans(self, image, box):
            dataset = ViTDetDataset(self.hmr2_cfg, image, box[None,:].numpy())
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

            all_verts = []
            all_keypoints = []
            all_3d_keypoints = []
            for batch in dataloader:
                patch_size = batch['img'].shape[-1] 
                with torch.no_grad():
                    out = self.hmr2(batch)
                batch_size = batch['img'].shape[0]

                for n in range(batch_size):               
                    verts = out['pred_vertices'][n].detach().cpu().numpy()
                    k_pts = out['pred_keypoints_2d'][n]
                    k_pts_3d = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                    # keypoints_patch = k_pts * (patch_size / 2.0)
                    center = batch['box_center'][n] # [N, 2]
                    size = batch['box_size'][n]     # [N] (this is the bbox_size from your code)
                    # scale_factor = (size / patch_size).unsqueeze(-1).unsqueeze(-1)
                    keypoints_full = k_pts*size + center

                    all_verts.append(verts)
                    all_keypoints.append(keypoints_full.detach().cpu().numpy())
                    all_3d_keypoints.append(k_pts_3d)

            all_verts = np.vstack(all_verts)
            all_keypoints = np.vstack(all_keypoints)
            all_3d_keypoints = np.vstack(all_3d_keypoints)
            min_y = all_verts.max(axis=0)[1]
            tmesh = self.hmr2_renderer.vertices_to_trimesh(all_verts, np.array([0,-min_y,0]))
            mesh_low_poly = tmesh.simplify_quadric_decimation(0.8)
            return [mesh_low_poly, all_keypoints]