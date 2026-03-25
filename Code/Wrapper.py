import os
import json
import subprocess
import argparse
import cv2
import socket
import time
import matplotlib.pyplot as plt
import numpy as np


from parse_video import *
from parse_results import save_yolo_results_to_json
from depth_predictor import DepthPredictor
from object_detector import ObjectDetector

CLEAR = "clear\n"
CLOSE = "close\n"
FPS = 8

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

def send_and_wait(sock, message):
    sock.sendall(message.encode('utf-8'))
    # This blocks until Blender sends b"DONE\n"
    response = sock.recv(1024).decode('utf-8')
    return response

def main(args):
    if isinstance(args.headless, str): args.headless = args.headless == "True"
    image_gen = get_images_from_scene(args)

    depth_predictor = DepthPredictor()

    object_detector = ObjectDetector()
    os.makedirs("./Output", exist_ok=True)
    asset_path = os.path.abspath(os.path.join(args.data_path, "Assets/"))
    cmd = [os.path.expanduser("~")+args.blender_path, 
           args.base_blender_scene, "-P", 
           "Code/blender_py.py", "--",  asset_path]
    if args.headless: cmd.insert(1, '-b')

    process = None
    s = None

    try:
        process = subprocess.Popen(cmd)
        
        s = connect_to_blender('127.0.0.1', 65432, 10)

        time.sleep(3)
        # time.sleep(1)

        fps = FPS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
        video_writer = None

        for frame_i, frame in enumerate(image_gen):
            bounded_im = object_detector.gen_bounded_image(frame)
            object_result = object_detector.predict(frame)
            depth_im = depth_predictor.predict(frame)

            save_yolo_results_to_json(object_result, depth_im, args)

            # plt.imsave(f'Output/output{frame_i}_bounded.jpg', bounded_im)
            # plt.imsave(f'Output/output{frame_i}_depth.jpg', depth_im)

            # cv2.imshow('frame', frame)
            # cv2.waitKey(1)
            # do detections

            # save to json
            # run blender to render scene from json        
            send_and_wait(s, CLEAR)
            send_and_wait(s, "load_new ./Code/temp_scene.json\n")
            send_and_wait(s, f"render ./Output/{args.sequence}\n")

            blender_frame = cv2.imread(f"./Output/{args.sequence}.png")


            bounded_bgr = cv2.cvtColor(bounded_im, cv2.COLOR_RGB2BGR)
            bounded_h, bounded_w = bounded_bgr.shape[:2]

            blender_resized = cv2.resize(blender_frame, (bounded_w, bounded_h), interpolation=cv2.INTER_AREA)

            combined_im = np.concatenate([bounded_bgr, blender_resized], axis=1)

            if video_writer is None:
                height, width, _ = combined_im.shape
                video_writer = cv2.VideoWriter(f'Output/{args.sequence}.mp4', fourcc, fps, (width, height))

            video_writer.write(combined_im)

        if video_writer:
            video_writer.release() 

    finally:
        if s is not None:
            try:
                s.sendall(CLOSE.encode('utf-8'))
                s.close()
            except Exception:
                pass

        # Kill Blender process
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                process.kill()

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default="../P3Data/",help="dataset path")
    parser.add_argument('--sequence',default='scene5', help="Select which sequence to generate visuals for")
    parser.add_argument('--stride', default=30, help="How many frames to skip in video")
    parser.add_argument('--blender_path', default="/Downloads/blender-5.1.0-linux-x64/blender")
    parser.add_argument('--base_blender_scene', default="./Blender/road_scene.blend")
    parser.add_argument('--headless', default=True)
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)