import os
import json
import subprocess
import argparse
import cv2
import socket
import time

from parse_video import *

CLEAR = "clear\n"
CLOSE = "close\n"

def connect_to_blender(host, port, retry_limit=10):
    attempt=0
    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # s.settimeout(3.0)

        try:
            print(f"Attempting to connect to Blender (Attempt {attempt + 1})...")
            s.connect((host, port))
            print("Successfully connected to Blender!")
            return s  # Return the active socket
            
        except (socket.timeout, ConnectionRefusedError):
            attempt += 1
            print("Blender not reached. Retrying in 0.5 seconds...")
            s.close()
            time.sleep(0.5)  # Wait a bit before the next 3-second attempt
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            s.close()
            break

def main(args):
    image_gen = get_images_from_scene(args)
    os.makedirs("./Output", exist_ok=True)
    cmd = [os.path.expanduser("~")+args.blender_path, args.base_blender_scene, "-P", "Code/blender_socket2.py"]
    process = subprocess.Popen(cmd)
    
    # time.sleep(5)
    s = connect_to_blender('127.0.0.1', 65432, 10)
    s.sendall(CLEAR.encode('utf-8'))
    for frame_i, frame in enumerate(image_gen):
        # print(frame_i)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        # do detections

        # save to json
        # run blender to render scene from json
        # s.sendall("json\n".encode('utf-8'))
        
    s.sendall(CLOSE.encode('utf-8'))
    return

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default="./P3Data/Sequences",help="dataset path")
    parser.add_argument('--sequence',default='scene2', help="Select which sequence to generate visuals for")
    parser.add_argument('--stride', default=4, help="How many frames to skip in video")
    parser.add_argument('--blender_path', default="/Downloads/blender-5.1.0-linux-x64/blender")
    parser.add_argument('--base_blender_scene', default="./Blender/test.blend")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)