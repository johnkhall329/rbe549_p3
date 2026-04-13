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
    "cone": "TrafficConeAndCylinder", 
    "speed bump": "SpeedBump"
}


def save_dino_results_to_json(image, object_detection_results, depth_results, lane_results, motion_results, args, K, extrinsics, frame_num):
    scene_objects = {}
    im_h, im_w = image.shape[:2]

    # calc scene motion
    motion_h, motion_w = motion_results.shape[:2]

    v_edge = motion_h//10
    bottom_edge = motion_results[-v_edge:, :]

    avg_scene_direction = np.average(bottom_edge, axis=(0,1))

    avg_scene_norms = np.linalg.norm(bottom_edge, axis=2)

    user_dir = np.zeros(2)
    if np.mean(avg_scene_norms > 4) > 0.4:
        user_dir = np.array([avg_scene_direction[1], -avg_scene_direction[0]])
        mag = np.linalg.norm(user_dir)
        if mag > 21:
            user_dir = 1.5*user_dir/mag
        elif mag > 14:
            user_dir = 1*user_dir/mag
        elif mag > 7:
            user_dir = 0.67*user_dir/mag
        elif mag > 2:
            user_dir = 0.33*user_dir/mag
    

    for box, mask, score, label, detail in zip(object_detection_results["new_boxes"], 
                                               object_detection_results["masks"], 
                                               object_detection_results["new_scores"], 
                                               object_detection_results["new_labels"], 
                                               object_detection_results["details"]):
        
        xmin, ymin, xmax, ymax = map(int, box.tolist())

        cropped_mask = mask[ymin:ymax, xmin:xmax]
        cropped_depth = depth_results[ymin:ymax, xmin:xmax]

        y_coords, x_coords = np.where(cropped_mask == 1)

        global_y_coords, global_x_coords = np.where(mask == 1)

        motion = None

        if len(x_coords) > 0:
            x_center = global_x_coords.mean()
            y_center = global_y_coords.mean()
            depth_results_masked = cropped_depth[y_coords, x_coords]

            # extra filtering for depth
            if len(x_coords) > 500:
                margins = len(x_coords)//10
                depth_results_sorted = np.sort(depth_results_masked)
                depth_results_filtered = depth_results_sorted[margins:-margins]
                z_depth = depth_results_filtered.mean()

                # Increase depth for thick vehicles
                # Also find the optical flow of mask
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
                    
                    # Motion finding
                    motion_results_cropped = motion_results[ymin:ymax, xmin:xmax]
                    motion = motion_results_cropped[y_coords, x_coords].mean(axis=0)

                    # World frame calcs instead

                    bx, by, bz = locate_3D_point(z_depth, x_center, y_center, K, extrinsics)

                    future_pix2 = np.array([x_center, y_center]) + motion

                    fbx2, fby2, fbz2 = locate_3D_point_given_world_height(bz, future_pix2[0], future_pix2[1], K, extrinsics)

                    delta_bx2 = fbx2 - bx
                    delta_by2 = fby2 - by

                    world_dir_isolated = np.array([delta_bx2, delta_by2])

                    magnitude = np.linalg.norm(world_dir_isolated)

                    if magnitude > 0.4:
                        true_dir = 3*(world_dir_isolated/magnitude)
                    elif magnitude > 0.1:
                        true_dir = 2*(world_dir_isolated/magnitude)
                    elif magnitude > 0.04:
                        true_dir = world_dir_isolated/magnitude
                    else:
                        true_dir = np.zeros(2)

                    world_true_dir = true_dir + user_dir



            else:
                # This should happen with smaller/further objects
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
                other_mask = object_detection_results["masks"][other_box_idx]

                y_coords, x_coords = np.where(other_mask == 1)
                
                x_center = x_coords.mean()
                y_center = y_coords.mean()
                depth_results_masked = depth_results[y_coords, x_coords]

                # extra filtering for depth
                if len(x_coords) > 500:
                    margins = len(x_coords)//10
                    depth_results_sorted = np.sort(depth_results_masked)
                    depth_results_filtered = depth_results_sorted[margins:-margins]
                    z_depth = depth_results_filtered.mean()

            blender_x, blender_y, blender_z = locate_3D_point(z_depth, x_center, y_center, K, extrinsics)
            blender_z = 0
            print('person')
 
        if "road sign" in label:
            sign_type = detail.get("type", None)
            if sign_type == 'stop':
                label = 'stop'
            elif sign_type == 'speed limit':
                label = 'speed limit'
            elif sign_type == 'speed bump':
                label = 'speed bump'

        contin = True
        if abs(blender_x) > 50:
            if label == "traffic light":
                blender_x = 50
            else:
                contin = False

        if abs(blender_y) > 40:
            contin = False

        if contin:
            if label not in LABEL_MAP_DINO.keys():
                continue

            obj_dict = {"location": [float(blender_x), float(blender_y), float(blender_z)]}
            if motion is not None:
                obj_dict["motion"] = motion.tolist()
                obj_dict["world_vec_isolated"] = [delta_bx2, delta_by2]


                if np.linalg.norm(world_true_dir) < 1.1:
                    obj_dict["parked"] = True
                else:
                    obj_dict["parked"] = False

                
                obj_dict["iso_direction"] = true_dir.tolist()
                obj_dict["direction"] = world_true_dir.tolist()

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

    scene_objects["SceneDir"] = user_dir.tolist()
    scene_objects["SceneDirPx"] = avg_scene_direction.tolist()

    if len(lane_results) > 0:
        scene_objects["Lanes"] = lane_results

    with open("Code/temp_scene.json", "w") as f:
        json.dump(scene_objects, f, indent=4)

    with open(f"Code/temp_scene_{frame_num}.json", "w") as f:
        json.dump(scene_objects, f, indent=4)


def locate_3D_point(depth, u, v, K, extrinsics):
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

def locate_3D_point_given_world_height(world_height, u, v, K, extrinsics):
    K_inv = np.linalg.inv(K)
    
    # Create homogeneous pixel vector
    pixel_coords = np.array([u, v, 1.0])
    
    # Back-project to normalized coordinates (z=1)
    normalized_coords = K_inv @ pixel_coords

    # rotate according to extrinsics
    rn_coords = extrinsics[:3,:3] @ normalized_coords
    
    # scale the vector so that it reaches the given z value
    height_diff = (world_height - extrinsics[2,3])
    scale = height_diff/rn_coords[2] 

    # Scale by depth to get coordinates in meters
    world_coords_m = (rn_coords*scale) + extrinsics[:3, 3]
    
    return world_coords_m

def locate_3D_point_old(depth, u, v, K):
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

