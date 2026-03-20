import cv2
import os
from glob import glob

# Generator function to save memory
def get_images_from_scene(args):
    undist_videos_path = os.path.abspath(os.path.join(args.data_path+'Sequences/', args.sequence, "Undist"))
    
    front_vid_path = glob(undist_videos_path+"/*front*")[0]

    cap = cv2.VideoCapture(front_vid_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {front_vid_path}")
        return

    i = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % args.stride == 0:
                yield frame
            i += 1
    finally:
        cap.release()