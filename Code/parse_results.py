import json

import numpy as np
import cv2

label_map = {
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


def save_yolo_results_to_json(object_detection_results, depth_results, lane_results, args):

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

            x, y, z = locate_3D_point(z_depth, x_center, y_center)
            blender_y, blender_z, blender_x = -x/2, y*0.0, z/2

            # store
            obj_dict = {
                "location": [float(blender_x), float(blender_y), float(blender_z)],
                "rotation": [0.0, 0.0, 0.0],  # placeholder
            }

            if "speedLimit" in label:
                label = label[:label.find(label.split("speedLimit")[-1])]
            if model == 'lights':
                color = label
                label = 'traffic light'

                print(f'\n\n{color}\n\n')
                obj_dict["material"] = label_map[color]
            if label in label_map.keys():
                real_label = label_map[label]

                if real_label not in scene_objects.keys():
                    scene_objects[real_label] = []

                scene_objects[real_label].append(obj_dict)

    if len(lane_results) > 0:
        scene_objects["Lanes"] = lane_results

    
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

