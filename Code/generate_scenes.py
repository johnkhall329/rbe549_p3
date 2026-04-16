import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

from parse_video import *
from parse_results import save_dino_results_to_json
from depth_predictor import DepthPredictor
from object_detector import ObjectDetector, ObjectDetectorGroundedDINO
from lane_detector import LaneDetector
from flow_detection import FlowDetector

def main(args):
    image_gen = get_images_from_scene(args)

    # Camera Calib
    K = np.load(os.path.join(args.data_path, 'Calib', 'calibration.npy'))
    extrinsics = np.array([[0,0,1.0,0], # camera to world of front camera
                            [-1.0,0,0,0], 
                            [0,-1.0,0,1.25]]) 
    pitch = 0.01
    r = np.array([[1, 0, 0],[0, np.cos(pitch), -np.sin(pitch)],[0,np.sin(pitch), np.cos(pitch)]])
    extrinsics[:3,:3] = extrinsics[:3,:3] @ r

    # Initialize Models
    depth_predictor = DepthPredictor()

    object_detector = ObjectDetectorGroundedDINO(camera_calib=K, scene_name=args.sequence, device='cpu')

    lane_detector = LaneDetector(scene_name=args.sequence, device='cpu')

    flow_detector = FlowDetector(device='cpu')

    os.makedirs(f"./Output/{args.sequence}", exist_ok=True)

    for frame_i, frames in enumerate(image_gen):
        prev_frame, frame = frames

        object_results, annotated_img = object_detector.predict(frame)
        depth_im = depth_predictor.predict(frame)

        lanes_im, lane_results = lane_detector.detect(frame, K, extrinsics)

        if prev_frame is not None:
            motion, flow_im = flow_detector.predict([prev_frame, frame], save=True)
        else:
            motion = np.zeros((frame.shape[0], frame.shape[1], 2), dtype=np.float32)
            flow_im = np.zeros_like(frame)

        scene_objects = save_dino_results_to_json(frame, object_results, depth_im, lane_results, motion, args, K, extrinsics)

        with open(f"./Output/{args.sequence}/{frame_i}_scene.json", "w") as f:
            json.dump(scene_objects, f, indent=4)

        plt.imsave(f'Output/{args.sequence}/{frame_i}_bounded.jpg', annotated_img)
        plt.imsave(f'Output/{args.sequence}/output{frame_i}_gdino.jpg', annotated_img)
        plt.imsave(f'Output/{args.sequence}/output{frame_i}_depth.jpg', depth_im)
        plt.imsave(f'Output/{args.sequence}/output{frame_i}_lanes.jpg', cv2.cvtColor(lanes_im, cv2.COLOR_BGR2RGB))
        plt.imsave(f'Output/{args.sequence}/output{frame_i}_flow.jpg', flow_im)
        # cv2.imshow('frame', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
        # cv2.imshow('frame', frame)
        # cv2.waitKey(1)
        # do detections



def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default="./P3Data/",help="dataset path")
    parser.add_argument('--sequence',default='trimmed', help="Select which sequence to generate visuals for")
    parser.add_argument('--stride', default=1575, help="How many frames to skip in video")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)