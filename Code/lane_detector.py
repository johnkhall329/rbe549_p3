import torch
import torchvision
import cv2
import argparse
import numpy as np
import torch.nn as nn
import glob
import os
from skimage.morphology import skeletonize

from PIL import Image
# from infer_utils import draw_segmentation_map, get_outputs
from torchvision.transforms import transforms as transforms
# from CLASS_NAMES import INSTANCE_CATEGORY_NAMES as CLASS_NAMES

CLASS_NAMES = [
    '__background__', 
    'divider-line',
    'dotted-line',
    'double-line',
    'random-line',
    'road-sign-line',
    'solid-line'
]

COLORS = np.random.uniform(0, 255, size=(len(CLASS_NAMES), 3))

class LaneDetector():
    def __init__(self, device=None):
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
            pretrained=False, num_classes=91
        )

        self.model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=len(CLASS_NAMES), bias=True)
        self.model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=len(CLASS_NAMES)*4, bias=True)
        self.model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(256, len(CLASS_NAMES), kernel_size=(1, 1), stride=(1, 1))

        # initialize the model
        ckpt = torch.load("./Models/model_15.pth", weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        # set the computation device
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        # load the modle on to the computation device and set to eval mode
        self.model.to(self.device).eval()

        self.yolop = torch.jit.load("./Models/yolopv2.pt")
        self.yolop.to(self.device).eval()

        self.max_blob_size = 500
        self.yellow_thresh = 140

        os.makedirs('./Output/road_signs', exist_ok=True)

    def detect(self, image, K, extrinsics):
        clear_road_signs()
        orig_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w,_ = image.shape
        image = self.transform(orig_image.copy())

        image = image.unsqueeze(0).to(self.device)
        masks, boxes, labels = self.get_outputs(image, 0.5)
    
        # result = self.draw_segmentation_map(orig_image, masks, boxes, labels, no_boxes=True)

        image = self.transform(cv2.resize(orig_image, (640,640)))
        image = image.unsqueeze(0).to(self.device)
        [pred, anchor_grid], seg, ll = self.yolop(image)
        ll_mask = self.lane_line_mask(ll)


        filtered_mask = cv2.dilate(cv2.erode(ll_mask, np.ones((5,5))),np.ones((3,3)))
        num_lanes, labels_im = cv2.connectedComponents(filtered_mask)

        fused_viz = cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR)

        results = []
        for lane_id in range(1,num_lanes):
            lane_blob = (labels_im == lane_id).astype(np.uint8)

            skel_lane = skeletonize(cv2.dilate(lane_blob, np.ones((5,5)))).astype(np.uint8)*255
        
            # Track which Mask R-CNN label is most common for THIS lane
            class_votes = {name: 0 for name in CLASS_NAMES}
            mask_map = {name: np.zeros((h,w), dtype=np.uint8) for name in CLASS_NAMES}

            for m_idx, mask in enumerate(masks):
                # Calculate overlap pixels
                intersection = cv2.bitwise_and(cv2.dilate(lane_blob, np.ones((5,5))), mask.astype(np.uint8))
                mask_map[labels[m_idx]] = cv2.bitwise_or(mask_map[labels[m_idx]], intersection)
                vote_count = np.sum(intersection)
                class_votes[labels[m_idx]] += vote_count
                class_votes[labels[m_idx]]
                
            # Determine winning class (Majority Vote)
            winner_class = max(class_votes, key=class_votes.get)
            
            # Only draw if we actually got a match, otherwise default to a generic class
            if class_votes[winner_class] > 0 and np.sum(lane_blob)>self.max_blob_size:
                if winner_class in [CLASS_NAMES[0], CLASS_NAMES[4], CLASS_NAMES[5]]: continue
                winner_mask = mask_map[winner_class]*255
                if winner_class == CLASS_NAMES[1]: winner_class = CLASS_NAMES[-1]
                
                lane_color = self.get_lane_color(cv2.cvtColor(orig_image, cv2.COLOR_RGB2BGR), winner_mask)
                # print(f'{winner_class}:{lane_color}')

                color = COLORS[CLASS_NAMES.index(winner_class)]
                # Color the YOLOP lane with the Mask R-CNN winner color
                colored_lane = np.zeros_like(fused_viz)
                # colored_lane[lane_blob == 1] = color
                colored_lane[cv2.dilate(skel_lane, np.ones((3,3)))==255] = color
                # if lane_color == 'yellow':
                #     print("sent yellow")
                fused_viz = cv2.addWeighted(fused_viz, 1.0, colored_lane, 0.8, 0)

                world_points = self.convert_to_3D(skel_lane, K, extrinsics)
                curve_model, in_idxs, x_dir = ransac_curve(world_points)
                if curve_model is not None:
                    blender_points = sample_curve(curve_model, world_points[in_idxs], x_dir)
                    results.append({'type': winner_class, 'color': lane_color, 'curve_points': blender_points.tolist()})

        i=0
        for mask_lane, box, label in zip(masks, boxes, labels):
            if label == CLASS_NAMES[5]:
                mask = mask_lane.astype(np.uint8)*255
                box.append((box[0][0],box[1][1]))
                box.append((box[1][0], box[0][1]))
                box = np.array(box, dtype=np.float32)
                w, h = box[1] - box[0]
                mask_3d = self.convert_to_3D(mask, K, extrinsics)
                if len(mask_3d) == 0: continue
                min_x, min_y, _ = mask_3d.min(axis=0)
                max_x, max_y, _ = mask_3d.max(axis=0)

                if np.hypot(min_x, min_y) > 15: # ignore ground arrows too far away or else they appear very messed up
                    continue

                box_3d = np.array([[max_x, max_y, 0],
                                   [min_x, min_y, 0], 
                                   [min_x, max_y, 0],
                                   [max_x, min_y, 0]])
                rvec, _ = cv2.Rodrigues(extrinsics[:,:3].T)
                t = -extrinsics[:,:3].T@extrinsics[:,3]
                reproj_box, _ = cv2.projectPoints(box_3d, rvec, t, K, np.zeros(5))
                reproj_box = reproj_box.reshape(-1,2).astype(np.float32)
                aspect_ratio = (max_x-min_x)/(max_y-min_y)

                new_h = w*aspect_ratio
                img_box = np.array([[0,0],
                                    [int(w), int(new_h)],
                                    [0, int(new_h)],
                                    [int(w), 0]], dtype=np.float32)
                
                M = cv2.getPerspectiveTransform(reproj_box, img_box)
                warped_img = cv2.warpPerspective(mask, M, (int(w), int(new_h)))

                trans_idx = np.where(warped_img!=255) 
                warped_img = cv2.cvtColor(warped_img, cv2.COLOR_GRAY2BGRA)
                warped_img[trans_idx[0], trans_idx[1], 3] = 0

                save_name = f'./Output/road_signs/road_sign_{i}.png'
                cv2.imwrite(save_name, warped_img)
                patch_info = {'type': 'road-sign-line', 'box': box_3d.tolist(), 'file_loc': save_name}
                results.append(patch_info)
                i+=1
        # result = np.array(result)
        # blank = np.zeros_like(orig_image)
        # for mask in masks:
        #     overlap = cv2.bitwise_and(mask, mask, mask=ll_mask)
        #     blank = cv2.bitwise_or(blank, overlap)

        # output = cv2.addWeighted(result, 0.5, ll_mask, 0.5, 0)
        # cv2.imshow('Segmented Image', cv2.bitwise_and(result, result, mask=ll_mask))
        # cv2.imshow('Overlap', blank)
        # cv2.imshow('Fused', fused_viz)
        # cv2.waitKey(1)
        return fused_viz, results


    def convert_to_3D(self, mask, K, extrinsics, max_dist=35.0):
        points_2d = np.argwhere(mask > 0)
        img_points = np.stack((points_2d[:, 1], points_2d[:, 0], np.ones_like(points_2d[:, 0]))).T
        norm_points = (np.linalg.inv(K)@img_points.T).T
        rays = (extrinsics[:3,:3]@norm_points.T).T

        safe_rays = np.where(rays[:,2] < -1e-6)
        ts = -extrinsics[2,3]/rays[safe_rays][:,[2]] # assuming Z = 0
        world_points = extrinsics[:3,3] + ts*rays[safe_rays]

        return world_points[world_points[:,0]<=max_dist]


    def get_lane_color(self, image, lane_mask):
        """
        Determines if a lane is 'yellow' or 'white' based on Lab color space.
        """
        # Convert BGR to Lab
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        
        # Extract the 'b' channel (Yellow-Blue)
        # In OpenCV Lab, 128 is neutral. > 128 is Yellowish.
        b_channel = lab[:, :, 2]
        
        # Mask the b_channel to only look at the lane pixels
        lane_pixels = b_channel[lane_mask == 255]

        outside_mask = cv2.dilate(lane_mask, np.ones((7,7)), iterations=5)

        outside_mask = cv2.bitwise_and(outside_mask, cv2.bitwise_not(cv2.dilate(lane_mask, np.ones((7,7)), iterations=2)))

        other_mask = b_channel[outside_mask == 255]
        
        if len(lane_pixels) == 0:
            return "unknown"

        avg_yellow_score = np.mean(lane_pixels)
        
        # Thresholding 128 (neutral) + a small buffer for safety
        if avg_yellow_score > self.yellow_thresh and avg_yellow_score-other_mask.mean()>5.0: 
            return "yellow"
        else:
            return "white"

    def get_outputs(self, image, threshold):
        with torch.no_grad():
            # forward pass of the image through the model.
            outputs = self.model(image)
        
        # get all the scores
        scores = list(outputs[0]['scores'].detach().cpu().numpy())
        # index of those scores which are above a certain threshold
        thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
        thresholded_preds_count = len(thresholded_preds_inidices)
        # get the masks
        masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
        # discard masks for objects which are below threshold
        masks = masks[:thresholded_preds_count]

        # get the bounding boxes, in (x1, y1), (x2, y2) format
        boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
        # discard bounding boxes below threshold value
        boxes = boxes[:thresholded_preds_count]
        # get the classes labels
        labels = [CLASS_NAMES[i] for i in outputs[0]['labels']]
        return masks, boxes, labels    

    def draw_segmentation_map(self, image, masks, boxes, labels, no_boxes=True):
        alpha = 1.0
        beta = 1.0 # transparency for the segmentation map
        gamma = 0.0 # scalar added to each sum
        #convert the original PIL image into NumPy format
        image = np.array(image)
        # convert from RGN to OpenCV BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        blank = np.zeros_like(image)
        for i in range(len(masks)):
            # apply a randon color mask to each object
            color = COLORS[CLASS_NAMES.index(labels[i])]
            if masks[i].any() == True:
                red_map = np.zeros_like(masks[i]).astype(np.uint8)
                green_map = np.zeros_like(masks[i]).astype(np.uint8)
                blue_map = np.zeros_like(masks[i]).astype(np.uint8)
                red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1] = color
                # combine all the masks into a single image
                segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
                # apply mask on the image
                cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)

                lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # Line width.
                tf = max(lw - 1, 1) # Font thickness.
                p1, p2 = boxes[i][0], boxes[i][1]
                if not no_boxes:
                    # draw the bounding boxes around the objects
                    cv2.rectangle(
                        image, 
                        p1, p2, 
                        color=color, 
                        thickness=lw,
                        lineType=cv2.LINE_AA
                    )
                    w, h = cv2.getTextSize(
                        labels[i], 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=lw / 3, 
                        thickness=tf
                    )[0]  # text width, height
                    w = int(w - (0.20 * w))
                    outside = p1[1] - h >= 3
                    p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                    # put the label text above the objects
                    cv2.rectangle(
                        image, 
                        p1, 
                        p2, 
                        color=color, 
                        thickness=-1, 
                        lineType=cv2.LINE_AA
                    )
                    cv2.putText(
                        image, 
                        labels[i], 
                        (p1[0], p1[1] - 5 if outside else p1[1] + h + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=lw / 3.8, 
                        color=(255, 255, 255), 
                        thickness=tf, 
                        lineType=cv2.LINE_AA
                    )
        return image
    
    def lane_line_mask(self, ll = None):
        # ll_predict = ll[:, :, 12:372,:]
        ll_seg_mask = torch.nn.functional.interpolate(ll, scale_factor=2, mode='bilinear')
        ll_seg_mask = torch.round(ll_seg_mask).squeeze(1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        color_mask = np.zeros((ll_seg_mask.shape[0], ll_seg_mask.shape[1]), dtype=np.uint8)
        color_mask[ll_seg_mask==1] = 255
        color_mask = cv2.resize(color_mask, (1280, 960))
        return color_mask
    
def sample_curve(curve_model, in_points, x_dir = True, n_samples = 10):
    a,b,c = curve_model
    min_pts = np.min(in_points,axis=0)[0]
    max_pts = np.max(in_points,axis=0)[0]

    samples = np.linspace(min_pts, max_pts, num=n_samples, endpoint=True)
    if min_pts < 5.0 and x_dir:
        samples = np.insert(samples, 0, 0.0)
    result = a*samples**2 + b*samples + c
    z = np.zeros_like(samples)
    output = np.column_stack([samples, result, z]) if x_dir else np.column_stack([result, samples, z])
    return output



def ransac_curve(world_points, max_iter = 100, threshold = 0.15, early_exit = 0.9, min_inliers=10):
    best_inliers = []
    best_model = None
    x_dir = True
    
    x_pts = world_points[:, 0]
    y_pts = world_points[:, 1]
    n_points = len(x_pts)

    if n_points < 3:
        return None, [], True
    
    for i in range(max_iter):
        sample_idxs = np.random.choice(n_points, 3, replace=False)
        xs = x_pts[sample_idxs]
        ys = y_pts[sample_idxs]

        try:
            Ax = np.column_stack([xs**2, xs, np.ones(3)])
            x_curve_model = np.linalg.solve(Ax, ys)

            Ay = np.column_stack([ys**2, ys, np.ones(3)])
            y_curve_model = np.linalg.solve(Ay, xs)


        except np.linalg.LinAlgError:
            continue

        a,b,c = x_curve_model
        y_pred = a * (x_pts**2) + b * x_pts + c

        distances = np.abs(y_pts-y_pred)
        inlier_idxs = np.where(distances<threshold)[0]

        if len(inlier_idxs) > len(best_inliers):
            best_inliers = inlier_idxs
            best_model = x_curve_model
            x_dir = True
            if len(inlier_idxs)/n_points >= early_exit:
                break

        a,b,c = y_curve_model
        x_pred = a * (y_pts**2) + b * y_pts + c

        distances = np.abs(x_pts-x_pred)
        inlier_idxs = np.where(distances<threshold)[0]

        if len(inlier_idxs) > len(best_inliers):
            best_inliers = inlier_idxs
            best_model = y_curve_model
            x_dir = False
            if len(inlier_idxs)/n_points >= early_exit:
                break

    if len(best_inliers) >= min_inliers:
        final_x = x_pts[best_inliers]
        final_y = y_pts[best_inliers]
        if x_dir: 
            A_final = np.column_stack([final_x**2, final_x, np.ones(len(final_x))])
            final_model, _, _, _ = np.linalg.lstsq(A_final, final_y, rcond=None)
        else:
            A_final = np.column_stack([final_y**2, final_y, np.ones(len(final_y))])
            final_model, _, _, _ = np.linalg.lstsq(A_final, final_x, rcond=None)
        return final_model, best_inliers, x_dir
    
    
    return None, [], True

def clear_road_signs():
    road_imgs = glob.glob("./Output/road_signs/*") # clear previous run of human predictions
    for img_file in road_imgs:
        if os.path.isfile(img_file) or os.path.islink(img_file): os.remove(img_file)