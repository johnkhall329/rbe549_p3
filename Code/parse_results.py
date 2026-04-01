import json

import numpy as np
import cv2
import os
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
            save_result_to_dict(scene_objects, label.strip(), detail, blender_x, blender_y, blender_z, label_map=LABEL_MAP_DINO)

    if len(lane_results) > 0:
        scene_objects["Lanes"] = lane_results

    with open("Code/temp_scene.json", "w") as f:
        json.dump(scene_objects, f, indent=4)


def save_result_to_dict(scene_dict, label, detail, bx, by, bz, label_map):
    obj_dict = {
            "location": [float(bx), float(by), float(bz)],
            "rotation": [0.0, 0.0, 0.0],  # placeholder
        }

    if "speedLimit" in label:
        label = label[:label.find(label.split("speedLimit")[-1])]
    # if model and model == 'lights': # WILL NEED TO UPDATE WITH NEW LIGHT DETECTION
    #     color = label
    #     label = 'traffic light'

    #     print(f'\n\n{color}\n\n')
    #     obj_dict["material"] = label_map[color]
    if "person" in label:
        # detail.apply_translation([bx, by, bz])
        tmesh, k_pts = detail
        prev_humans = glob.glob("'./Output/humans/*.obj")
        id = len(prev_humans)
        file_name = f'./Output/humans/{id}.obj'
        tmesh.export(file_name)
        obj_dict["file location"] = file_name
    if label in label_map.keys():
        real_label = label_map[label]

        if real_label not in scene_dict.keys():
            scene_dict[real_label] = []

        scene_dict[real_label].append(obj_dict)



def save_yolo_results_to_json(object_detection_results, depth_results, lane_results, args, K, extrinsics):

    scene_objects = {}
    extra_classes = ['stop sign', 'traffic light'] # add other classes to skip in yolo26

    for model, model_results in object_detection_results.items():
        for box in model_results.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x_center, y_center, _, _ = map(int, box.xywh[0])
            cls = int(box.cls[0])
            label = model_results.names[cls]
            if model == 'yolo26' and label in extra_classes: 
                continue
            # elif model == 'lisa':
            #     print(f'\n\n{label}\n\n')
            

            # if label == 'traffic light':
            #     hsv_slider(model_results)

            # find depth
            z_depth = depth_results[y1:y2, x1:x2].mean()

            x, y, z = locate_3D_point_old(z_depth, x_center, y_center, K)
            x, y, z = locate_3D_point(z_depth, x_center, y_center, K, extrinsics)

            blender_y, blender_z, blender_x = -x/2, y*0.0, z/2

            # store if in bounds
            contin = True
            if abs(blender_x) > 35:
                if label == "traffic light":
                    blender_x = 35
                else:
                    contin = False

            if abs(blender_y) > 30:
                contin = False

            if contin:
                save_result_to_dict(scene_objects, label, blender_x, blender_y, blender_z, label_map=LABEL_MAP_YOLO, model=model)
                
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

