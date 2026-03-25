import json


from scipy.io import loadmat, savemat
from scipy.io.matlab._mio5 import varmats_from_mat
from io import BytesIO
import os
import numpy as np

label_map = {
    "car": "SedanAndHatchback",
    "person": "Pedestrain",
    "traffic light": "TrafficSignal",
    "truck": "PickupTruck",
    "fire hydrant": "fire",
    "stop sign": "StopSign"
}


def save_yolo_results_to_json(object_detection_results, depth_results, args):

    scene_objects = {}

    for box in object_detection_results.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x_center, y_center, _, _ = map(int, box.xywh[0])
        cls = int(box.cls[0])
        label = object_detection_results.names[cls]

        # find depth
        z_depth = depth_results[y1:y2, x1:x2].mean()

        x, y, z = locate_3D_point(z_depth, x_center, y_center)
        blender_y, blender_z, blender_x = -x/2, y*0.0, z/2

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

def locate_3D_point(depth, u, v):
    K = np.array([[1594.7, 0.0, 655.3], [0, 1607.7, 414.4], [0, 0, 1]]) # Found by viewing the .mat file in matlav CV toolbox

    K_inv = np.linalg.inv(K)
    
    # Create homogeneous pixel vector
    pixel_coords = np.array([u, v, 1.0])
    
    # Back-project to normalized coordinates (z=1)
    normalized_coords = K_inv @ pixel_coords
    
    # Scale by depth to get coordinates in meters
    world_coords_m = normalized_coords * depth
    
    return world_coords_m

