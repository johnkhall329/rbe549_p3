import cv2
import numpy as np

def detect_signals(image, bounds, daylight_thresh):
    daylight = predict_daylight(image, daylight_thresh)
    hsv_slider(image[bounds[0][1]:bounds[1][1], bounds[0][0]:bounds[1][0]])

def predict_daylight(image, daylight_thresh = 100.0, top_pcnt = 0.5):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h,w = gray.shape
    daylight = gray[:int(h*top_pcnt),:].mean() > daylight_thresh
    return daylight


def hsv_slider(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    cv2.namedWindow('Threshold Adjustment',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Threshold Adjustment', 640,480)

    def callback(x):
        pass

    thresh = np.array([0,  0, 0, 255, 255, 255])
    filter = np.array([0,0])
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
    pcnt = np.nonzero(dilated)[0].shape[0]/(image.shape[0]*image.shape[1])
    print(pcnt)
    cv2.destroyWindow('Threshold Adjustment')