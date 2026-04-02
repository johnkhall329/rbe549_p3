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
    "truck": "PickupTruck",
    "fire hydrant": "fire",
    "stop sign": "StopSign",
    "stop": "StopSign",
    "speed limit sign": "SpeedLimitSign",
    "garbage bin":"trashbin",
    "bicycle": "Bicycle",
    "motorcycle": "Motorcycle",
    "cone": "TrafficConeAndCylinder"
}


def save_dino_results_to_json(image, object_detection_results, depth_results, lane_results, args, K, extrinsics):
    scene_objects = {}

    for box, score, label, detail in zip(object_detection_results["boxes"], object_detection_results["scores"], object_detection_results["labels"], object_detection_results["details"]):
        xmin, ymin, xmax, ymax = map(int, box.tolist())

        x_center, y_center = ((xmax + xmin)//2), ((ymax + ymin)//2)
        
        z_depth = depth_results[ymin:ymax, xmin:xmax].mean()

        blender_x, blender_y, blender_z = locate_3D_point(z_depth, x_center, y_center, K, extrinsics)
        blender_z = 0 # not using this right now
        if label=="person":
            kpts = detail[1].astype(np.int64)
        #     draw_img = image.copy()
        #     for pt in kpts:
        #         cv2.circle(draw_img, pt, 2, (0,0,255), -1)
        #     cv2.imshow('person', draw_img)
        #     cv2.waitKey(1)
            x_center, y_center = kpts[8]
            blender_x, blender_y, blender_z = locate_3D_point(z_depth, x_center, y_center, K, extrinsics)
            blender_z = 0
            print('person')
 

        contin = True
        if abs(blender_x) > 30:
            if label == "traffic light":
                blender_x = 30
            else:
                contin = False

        if abs(blender_y) > 25:
            contin = False

        if contin:
            if label not in LABEL_MAP_DINO.keys():
                continue

            obj_dict = {"location": [float(blender_x), float(blender_y), float(blender_z)]}

            if "speedLimit" in label:
                label = label[:label.find(label.split("speedLimit")[-1])]

            if "person" in label:
                # detail.apply_translation([bx, by, bz])
                tmesh, k_pts = detail
                prev_humans = glob.glob("'./Output/humans/*.obj")
                id = len(prev_humans)
                file_name = f'./Output/humans/{id}.obj'
                tmesh.export(file_name)
                obj_dict["file location"] = file_name

            if detail != '' and isinstance(detail,str) and detail.split()[0] == 'orientation:':
                rot_val = float(detail.split()[1])
                rot_val = math.degrees(rot_val)
                for degree in range(-360, 361, 90):
                    if abs(rot_val - degree) < 10:
                        rot_val = degree
                rot_val = -(90 + rot_val)
                obj_dict["rotation"] = [0.0, 0.0, rot_val]
            else:
                obj_dict["rotation"] = [0.0, 0.0, 0.0]

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


# def hsv_slider(result):
#     image = cv2.cvtColor(result.orig_img, cv2.COLOR_RGB2HSV)
#     cv2.namedWindow('Threshold Adjustment',cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('Threshold Adjustment', 640,480)

#     def callback(x):
#         pass

#     thresh = np.array([63,  18, 146, 158, 255, 255])
#     cv2.createTrackbar('lowH','Threshold Adjustment',thresh[0],255,callback)
#     cv2.createTrackbar('highH','Threshold Adjustment',thresh[3],255,callback)

#     cv2.createTrackbar('lowS','Threshold Adjustment',thresh[1],255,callback)
#     cv2.createTrackbar('highS','Threshold Adjustment',thresh[4],255,callback)

#     cv2.createTrackbar('lowV','Threshold Adjustment',thresh[2],255,callback)
#     cv2.createTrackbar('highV','Threshold Adjustment',thresh[5],255,callback)

#     sel = True
#     while sel:
#         thresh = np.array([cv2.getTrackbarPos('lowH', 'Threshold Adjustment'),cv2.getTrackbarPos('lowS', 'Threshold Adjustment'),cv2.getTrackbarPos('lowV', 'Threshold Adjustment'),
#                            cv2.getTrackbarPos('highH', 'Threshold Adjustment'),cv2.getTrackbarPos('highS', 'Threshold Adjustment'),cv2.getTrackbarPos('highV', 'Threshold Adjustment')])
#         frame_threshold = cv2.inRange(image, thresh[:3], thresh[3:])
#         # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
#         # dilated = cv2.dilate(frame_threshold, kernel,iterations=3)
#         # erode = cv2.erode(dilated,kernel,iterations=2)
#         # dilated2 = cv2.dilate(erode, kernel,iterations=3)
#         cv2.imshow('Threshold Adjustment', frame_threshold)
#         key = cv2.waitKey(10)
#         sel = (key != 13 and key != 27)
#     print(thresh)
#     cv2.destroyWindow('Threshold Adjustment')
#     # print('h')
#     pass
#     # image = result.

