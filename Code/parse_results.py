import json

import numpy as np
import cv2
import os
import math

import glob

LABEL_MAP_YOLO = {
    "car": "SedanAndHatchback",
    "person": "Pedestrain",
    "traffic light": "TrafficSignal",
    "truck": "PickupTruck",
    "fire hydrant": "fire",
    "stop sign": "StopSign",
    "stop": "StopSign",
    "speedLimit": "SpeedLimitSign",
    "red": "RED_ON",
    "yellow": "YELLOW_ON",
    "green": "GREEN_ON",
    "off": "OFF"
}

LABEL_MAP_DINO = {
    "sedan": "SedanAndHatchback",
    "hatchback": "SedanAndHatchback",
    "suv": "SUV",
    "person": "Pedestrain",
    "traffic light": "TrafficSignal",
    "pickup": "PickupTruck",
    "truck": "Truck",
    "box": "Truck",
    "fire hydrant": "fire",
    "stop sign": "StopSign",
    "stop": "StopSign",
    "speed limit": "SpeedLimitSign",
    "garbage bin":"trashbin",
    "bicycle": "Bicycle",
    "motorcycle": "Motorcycle",
    "cone": "TrafficConeAndCylinder"
}


def save_dino_results_to_json(image, object_detection_results, depth_results, lane_results, args, K, extrinsics):
    scene_objects = {}

    for box, mask, score, label, detail in zip(object_detection_results["new_boxes"], 
                                               object_detection_results["masks"], 
                                               object_detection_results["new_scores"], 
                                               object_detection_results["new_labels"], 
                                               object_detection_results["details"]):
        
        xmin, ymin, xmax, ymax = map(int, box.tolist())

        y_coords, x_coords = np.where(mask == 1)

        if len(x_coords) > 0:
            x_center = x_coords.mean()
            y_center = y_coords.mean()
            depth_results_masked = depth_results[y_coords, x_coords]

            # extra filtering for depth
            if len(x_coords) > 500:
                margins = len(x_coords)//10
                depth_results_sorted = np.sort(depth_results_masked)
                depth_results_filtered = depth_results_sorted[margins:-margins]
                z_depth = depth_results_filtered.mean()

                # Increase depth for thick vehicles
                if label in {"sedan", "hatchback", "suv", "pickup", "truck", "box"}:
                    close_depths = depth_results_filtered[:(2*margins)]
                    far_depths = depth_results_filtered[-(2*margins):]

                    mean_close = close_depths.mean()
                    mean_far = far_depths.mean()

                    depth_range = mean_far - mean_close

                    if depth_range < 0.5:
                        z_depth = mean_close
                        z_depth += 4 if label == "box" else 2.5
                    else:
                        z_depth = (mean_close + mean_far)/2

            else:
                # This should not happen
                print('WARNING: empty mask on object')
                z_depth = depth_results_masked.mean()


        else:
            # Using bounding box center if there is an error. This shouldn't come up ideally
            x_center, y_center = ((xmax + xmin)//2), ((ymax + ymin)//2)
            z_depth = depth_results[ymin:ymax, xmin:xmax].mean()
            print("ERROR: Mask is empty!")
        
        blender_x, blender_y, blender_z = locate_3D_point(z_depth, x_center, y_center, K, extrinsics)
        blender_z = 0 # not using this right now
        if label=="person":
            if len(detail) == 2:
                kpts = detail[1].astype(np.int64)
                x_center, y_center = kpts[8]
            else:
                other_box_idx = detail[2]
                other_box = object_detection_results["new_boxes"][other_box_idx]
                xmin, ymin, xmax, ymax = map(int, other_box.tolist())

                x_center, y_center = ((xmax + xmin)//2), ((ymax + ymin)//2)
                
                z_depth = depth_results[ymin:ymax, xmin:xmax].mean()

            blender_x, blender_y, blender_z = locate_3D_point(z_depth, x_center, y_center, K, extrinsics)
            blender_z = 0
            print('person')
 
        if "road sign" in label:
            sign_type = detail.get("type", None)
            if sign_type == 'stop':
                label = 'stop'
            elif sign_type == 'speed limit':
                label = 'speed limit'

        contin = True
        if abs(blender_x) > 50:
            if label == "traffic light":
                blender_x = 50
            else:
                contin = False

        if abs(blender_y) > 25:
            contin = False

        if contin:
            if label not in LABEL_MAP_DINO.keys():
                continue

            obj_dict = {"location": [float(blender_x), float(blender_y), float(blender_z)]}

            if label == "speed limit": obj_dict["speed"] = detail.get("speed","")
            # Pedestrian Pose Parsing
            if "person" in label:
                # detail.apply_translation([bx, by, bz])
                tmesh, k_pts = detail[:2]
                prev_humans = glob.glob("./Output/humans/*.obj")
                id = len(prev_humans)
                file_name = f'./Output/humans/{id}.obj'
                tmesh.export(file_name)
                obj_dict["file location"] = file_name

            # Orientation Parsing
            orientation = detail.get("orientation", False) if isinstance(detail, dict) else False
            if orientation:
                rot_val = float(orientation)
                # rot_val = math.degrees(rot_val)
                obj_dict["rotation"] = [0.0, 0.0, rot_val]
            else:
                obj_dict["rotation"] = [0.0, 0.0, 0.0]

            signals = detail.get("signals", False) if isinstance(detail, dict) else False
            if signals:
                obj_dict["signals"] = signals
            
            # Traffic Light Parsing
            if label == 'traffic light':
                if detail['qt'] == 0:
                    continue
                elif detail['qt'] == 1:
                    color = detail['light_0']['color']
                    shape = detail['light_0']['shape']
                    if color == "unknown/off":
                        continue
                        
                    if 'arrow' in shape:
                        shape_name = "ARROW"
                        if 'up' in shape:
                            shape_name += '_U'
                        elif 'down' in shape:
                            shape_name += '_D'
                        elif 'right' in shape:
                            shape_name += '_R'
                        elif 'left' in shape:
                            shape_name += '_L'
                        else:
                            print('WARNING: direction labeling is incorrect in traffic signal')

                    elif 'circle' in shape:
                        shape_name = "ON"
                    else:
                        continue
                    
                    obj_dict["material"] = color + shape_name
                elif detail['qt'] == 2:
                    real_label = LABEL_MAP_DINO[label]

                    if real_label not in scene_objects.keys():
                        scene_objects[real_label] = []

                    for i in range(2):
                        y_offset = -0.32 + 0.64*i
                        color = detail[f'light_{i}']['color']
                        shape = detail[f'light_{i}']['shape']

                        if color == "unknown/off":
                            continue
                        
                        if 'arrow' in shape:
                            shape_name = "ARROW"
                            if 'up' in shape:
                                shape_name += '_U'
                            elif 'down' in shape:
                                shape_name += '_D'
                            elif 'right' in shape:
                                shape_name += '_R'
                            elif 'left' in shape:
                                shape_name += '_L'
                            else:
                                print('WARNING: direction labeling is incorrect in traffic signal')


                        elif 'circle' in shape:
                            shape_name = "ON"
                        else:
                            continue

                        obj_dict["material"] = color + shape_name
                        obj_dict["location"] = [float(blender_x), float(blender_y - y_offset), float(blender_z)]
                        scene_objects[real_label].append(obj_dict.copy())

                    continue

            real_label = LABEL_MAP_DINO[label]

            if real_label not in scene_objects.keys():
                scene_objects[real_label] = []

            scene_objects[real_label].append(obj_dict)

    if len(lane_results) > 0:
        scene_objects["Lanes"] = lane_results

    with open("Code/temp_scene.json", "w") as f:
        json.dump(scene_objects, f, indent=4)


def locate_3D_point(depth, u, v, K, extrinsics, max_depth=20):
    K_inv = np.linalg.inv(K)
    
    # Create homogeneous pixel vector
    pixel_coords = np.array([u, v, 1.0])
    
    # Back-project to normalized coordinates (z=1)
    normalized_coords = K_inv @ pixel_coords

    # rotate according to extrinsics
    rn_coords = extrinsics[:3,:3] @ normalized_coords
    
    # Scale by depth to get coordinates in meters
    world_coords_m = (rn_coords * depth) + extrinsics[:3, 3]
    
    return world_coords_m

def locate_3D_point_old(depth, u, v, K, max_depth=20):
    K_inv = np.linalg.inv(K)
    
    # Create homogeneous pixel vector
    pixel_coords = np.array([u, v, 1.0])
    
    # Back-project to normalized coordinates (z=1)
    normalized_coords = K_inv @ pixel_coords
    
    # Scale by depth to get coordinates in meters
    world_coords_m = normalized_coords * depth
    
    return world_coords_m


# def light_seg(resutl, box):
#     green = [63,  18, 146, 158, 255, 255]

