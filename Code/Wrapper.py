import os
import json
import subprocess
import argparse
import cv2

from parse_video import *

def main(args):
    image_gen = get_images_from_scene(args)

    for frame_i, frame in enumerate(image_gen):
        print(frame_i)
        cv2.imshow('frame', frame)
        cv2.waitKey(27)
        # do detections

        # save to json

        # run blender to render scene from json
    pass

def configParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',default="./P3Data/Sequences",help="dataset path")
    parser.add_argument('--sequence',default='scene2', help="Select which sequence to generate visuals for")
    parser.add_argument('--stride', default=1, help="How many frames to skip in video")
    return parser

if __name__ == "__main__":
    parser = configParser()
    args = parser.parse_args()
    main(args)