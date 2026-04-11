import cv2
import os
from glob import glob
import json

FRAMES_BACK = 4

# Generator function to save memory
def get_images_from_scene(args):
    if args.sequence == 'test':
        undist_videos_path = os.path.abspath(os.path.join(args.data_path+'Sequences/', args.sequence))

        image_paths = glob(undist_videos_path + "/*.jpg")
        image_paths += glob(undist_videos_path + '/*.png')


        for path in image_paths:
            im = cv2.imread(path, cv2.IMREAD_COLOR)
            yield im, None

    else:
        undist_videos_path = os.path.abspath(os.path.join(args.data_path+'Sequences/', args.sequence, "Undist"))
        
        front_vid_path = glob(undist_videos_path+"/*front*")[0]

        cap = cv2.VideoCapture(front_vid_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {front_vid_path}")
            return

        i = 0
        try:
            prev_frame = None
            while True:
                ret, frame = cap.read()
                    
                if not ret:
                    break

                if i % args.stride == 0:
                    yield prev_frame, frame
                elif (i + FRAMES_BACK) % args.stride == 0:
                    prev_frame = frame
                
                i += 1
        finally:
            cap.release()

