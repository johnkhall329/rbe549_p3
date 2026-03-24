import cv2
import os
from glob import glob
import json

label_map = {
    "car": "SedanAndHatchback",
    "person": "Pedestrain",
    "traffic light": "TrafficSignal",
    "truck": "PickupTruck",
    "fire hydrant": "fire",
    "stop sign": "StopSign"
}

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

def save_results_to_json(object_detection_results, depth_results):
    scene_objects = {}

    h, w = depth_results.shape
    depth_z_scale = 300 # calbrated be somewhat greater than depth_results.mean() in an outdoor image?

    blender_x_max, blender_y_max, blender_z_max = 15, 25, 1

    for box in object_detection_results.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x_center, y_center, _, _ = box.xywh[0]
        cls = int(box.cls[0])
        label = object_detection_results.names[cls]

        # find depth
        z_depth = depth_results[y1:y2, x1:x2].mean()

        # scale
        blender_y = -2*((x_center - w/2)/w)*blender_x_max

        blender_x = (z_depth/depth_z_scale)*blender_y_max

        blender_z = -2*((y_center - h/2)/h)*blender_z_max

        # store
        obj_dict = {
            "location": [float(blender_x), float(blender_y), float(blender_z)],
            "rotation": [0.0, 0.0, 0.0],  # placeholder
        }

        if label in label_map.keys():
            real_label = label_map[label]

            if real_label not in scene_objects.keys():
                scene_objects[real_label] = []

            scene_objects[real_label].append(obj_dict)

    
    with open("Code/temp_scene.json", "w") as f:
        json.dump(scene_objects, f, indent=4)

