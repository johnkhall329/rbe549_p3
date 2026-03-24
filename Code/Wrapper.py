import os
import json
import subprocess
import argparse
import cv2
import socket
import time
import matplotlib.pyplot as plt


from parse_video import *
from depth_predictor import DepthPredictor
from object_detector import ObjectDetector
from lane_detector import LaneDetector

CLEAR = "clear\n"
CLOSE = "close\n"

def connect_to_blender(asset_path, args, host, port, retry_limit=10):
    attempt=0
    cmd = [os.path.expanduser("~")+args.blender_path, 
           args.base_blender_scene, "-P", 
           "Code/blender_py.py", "--",  asset_path]
    if args.headless: cmd.insert(1, '-b')
    # process = subprocess.Popen(cmd)

    exists = False
    
    while True:
        if attempt > retry_limit:
            raise Exception("Unable to Connect to Blender")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # s.settimeout(3.0)

        try:
            print(f"Attempting to connect to Blender (Attempt {attempt + 1})...")
            s.connect((host, port))
            print("Successfully connected to Blender!")
            return s  # Return the active socket
            
        except (socket.timeout, ConnectionRefusedError):
            attempt += 1
            if not exists:
                process = subprocess.Popen(cmd)
                print("Initializing Blender Socket")
                exists = True
                s.close()
                time.sleep(0.5)
            else:
                print("Blender not reached. Retrying in 0.5 seconds...")
                s.close()
            time.sleep(0.5)  # Wait a bit before the next 3-second attempt
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            s.close()
            break

def main(args):
    if isinstance(args.headless, str): args.headless = args.headless == "True"
    image_gen = get_images_from_scene(args)

    depth_predictor = DepthPredictor("MiDaS_small")

    object_detector = ObjectDetector()

    lane_detector = LaneDetector()
    os.makedirs("./Output", exist_ok=True)
    asset_path = os.path.abspath(os.path.join(args.data_path, "Assets/"))
    # cmd = [os.path.expanduser("~")+args.blender_path, 
    #        args.base_blender_scene, "-P", 
    #        "Code/blender_py.py", "--",  asset_path]
    # if args.headless: cmd.insert(1, '-b')
    # process = subprocess.Popen(cmd)
    
    # s = connect_to_blender('127.0.0.1', 65432, 10)
    # time.sleep(3)
    s = connect_to_blender(asset_path, args, '127.0.0.1', 65432, 10)
    time.sleep(1)
    s.sendall(CLEAR.encode('utf-8'))
    # time.sleep(1)
    for frame_i, frame in enumerate(image_gen):
        bounded_im = object_detector.gen_bounded_image(frame)
        object_result = object_detector.predict(frame)
        depth_im = depth_predictor.predict(frame)
        lanes = lane_detector.detect(frame)

        save_results_to_json(object_result, depth_im)

        plt.imsave(f'Output/output{frame_i}_bounded.jpg', bounded_im)
        plt.imsave(f'Output/output{frame_i}_depth.jpg', depth_im)
        plt.imsave(f'Output/output{frame_i}_lanes.jpg', cv2.cvtColor(lanes, cv2.COLOR_BGR2RGB))

        # cv2.imshow('frame', frame)
        # cv2.waitKey(1)
        # do detections

        # save to json
        # run blender to render scene from json        
        # s.sendall(CLEAR.encode('utf-8'))
        # time.sleep(1)
        # s.sendall("load_new ./Code/temp_scene.json\n".encode('utf-8'))
        # time.sleep(1)
        # s.sendall(f"render ./Output/{args.sequence}\n".encode('utf-8'))
        # time.sleep(1)
        # s.sendall("spawn SUV\n".encode('utf-8'))
        # time.sleep(2)
        # s.sendall("spawn Trashbin\n".encode('utf-8'))
    s.sendall(CLOSE.encode('utf-8'))
    # time.sleep(2)
    return

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default="./P3Data/",help="dataset path")
    parser.add_argument('--sequence',default='scene3', help="Select which sequence to generate visuals for")
    parser.add_argument('--stride', default=30, help="How many frames to skip in video")
    parser.add_argument('--blender_path', default="/Downloads/blender-5.1.0-linux-x64/blender")
    parser.add_argument('--base_blender_scene', default="./Blender/road_scene.blend")
    parser.add_argument('--headless', default=True)
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)