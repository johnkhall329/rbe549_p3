import cv2
import numpy as np
import torchvision
import torch

def detect_signals(image, bounds, daylight_thresh):
    daylight = predict_daylight(image, daylight_thresh)
    # hsv_slider(image[bounds[0][1]:bounds[1][1], bounds[0][0]:bounds[1][0]], daylight)

    hsv_img = cv2.cvtColor(image[bounds[0][1]:bounds[1][1], bounds[0][0]:bounds[1][0]], cv2.COLOR_RGB2HSV)
    brake_thresh = np.array([0,  75, 90, 19, 255, 255]) if daylight else np.array([0, 0, 206, 73, 56, 255])
    turn_thresh = np.array([20, 70, 127, 255, 255, 255]) if daylight else np.array([26, 0, 237, 106, 43, 255])
    img_filter = np.array([1,1])

    hsv_img = cv2.GaussianBlur(hsv_img, (3,3), 0)
    brake_res = cv2.inRange(hsv_img, brake_thresh[:3], brake_thresh[3:])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    erode = cv2.erode(brake_res ,kernel,iterations=img_filter[0])
    brake_res = cv2.dilate(erode, kernel,iterations=img_filter[1])

    brake_on, brake_stats, _ = analyze_thresh(brake_res)

    turn_res = cv2.inRange(hsv_img, turn_thresh[:3], turn_thresh[3:])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    erode = cv2.erode(turn_res ,kernel,iterations=img_filter[0])
    turn_res = cv2.dilate(erode, kernel,iterations=img_filter[1])
    
    turn_on, turn_stats, turn_center = analyze_thresh(turn_res, turn=True)
    # if brake_on and turn_on:

    #     brake_box = torch.tensor(brake_stats[1+np.argmax(brake_stats[1:, 4]),:4])
    #     turn_box = torch.tensor(turn_stats[1+np.argmax(turn_stats[1:, 4]),:4])
    #     if torchvision.ops.box_iou(brake_box[None,:], turn_box[None,:], fmt="xywh") > 0.5:
    #         brake_on = False
    left = False if not turn_on else turn_center[0] < turn_res.shape[1]/2

    return brake_on, turn_on, left


def predict_daylight(image, daylight_thresh = 100.0, top_pcnt = 0.5):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h,w = gray.shape
    daylight = gray[:int(h*top_pcnt),:].mean() > daylight_thresh
    return daylight


def hsv_slider(image, daylight):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    cv2.namedWindow('Threshold Adjustment',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Threshold Adjustment', 640,480)

    def callback(x):
        pass

    thresh = np.array([0,  75, 90, 19, 255, 255]) if daylight else np.array([0, 0, 206, 73, 56, 255])
    filter = np.array([1,1])
    cv2.createTrackbar('lowH','Threshold Adjustment',thresh[0],255,callback)
    cv2.createTrackbar('highH','Threshold Adjustment',thresh[3],255,callback)

    cv2.createTrackbar('lowS','Threshold Adjustment',thresh[1],255,callback)
    cv2.createTrackbar('highS','Threshold Adjustment',thresh[4],255,callback)

    cv2.createTrackbar('lowV','Threshold Adjustment',thresh[2],255,callback)
    cv2.createTrackbar('highV','Threshold Adjustment',thresh[5],255,callback)

    cv2.createTrackbar('Erode','Threshold Adjustment',filter[0],5,callback)
    cv2.createTrackbar('Dilate','Threshold Adjustment',filter[1],5,callback)

    sel = True
    image = cv2.GaussianBlur(image, (3,3), 0)
    while sel:
        new_thresh = np.array([cv2.getTrackbarPos('lowH', 'Threshold Adjustment'),cv2.getTrackbarPos('lowS', 'Threshold Adjustment'),cv2.getTrackbarPos('lowV', 'Threshold Adjustment'),
                           cv2.getTrackbarPos('highH', 'Threshold Adjustment'),cv2.getTrackbarPos('highS', 'Threshold Adjustment'),cv2.getTrackbarPos('highV', 'Threshold Adjustment')])
        filter = np.array([cv2.getTrackbarPos('Erode', 'Threshold Adjustment'),cv2.getTrackbarPos('Dilate', 'Threshold Adjustment')])
        if not np.allclose(thresh, new_thresh): print(new_thresh)
        thresh = new_thresh
        frame_threshold = cv2.inRange(image, thresh[:3], thresh[3:])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        erode = cv2.erode(frame_threshold,kernel,iterations=filter[0])
        dilated = cv2.dilate(erode, kernel,iterations=filter[1])
        # dilated2 = cv2.dilate(erode, kernel,iterations=3)
        cv2.imshow('Threshold Adjustment', dilated)
        key = cv2.waitKey(10)
        sel = (key != 13 and key != 27)
    print(thresh)
    print(filter)
    passed, _, _ = analyze_thresh(dilated)
    print(passed)

    cv2.destroyWindow('Threshold Adjustment')

def analyze_thresh(image, turn=False):
    all_stats = cv2.connectedComponentsWithStats(image)
    stats = all_stats[2]
    if stats.shape[0] < 2: return False, None, None
    total_area = stats[0,4]
    thresh_area = np.sum(stats[1:,4])

    total_pcnt = thresh_area/total_area
    
    largest_pcnt = np.max(stats[1:,4])/total_area

    thresh = 0.003 if turn else 0.005
    passed = total_pcnt > thresh and largest_pcnt < 0.25
    max_centroid = all_stats[3][1+np.argmax(stats[1:, 4])]
    return passed, stats, max_centroid