
import os
import sys

import cv2
import torch
import glob

import numpy as np

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Modules"))
sys.path.append(repo_path)

# Now you can import directly from the inner folder name
from YOLO3D.inference import detect3d, plot3d
from YOLO3D.script.Model import ResNet, ResNet18, VGG11
from YOLO3D.script import Model, ClassAverages
from YOLO3D.library.Plotting import *
from YOLO3D.script.Dataset import generate_bins, DetectedObject
from torchvision.models import resnet18, vgg11

# detect3d(
    #     reg_weights=opt.reg_weights,
    #     model_select=opt.model_select,
    #     source=opt.source,
    #     calib_file=opt.calib_file,
    #     show_result=opt.show_result,
    #     save_result=opt.save_result,
    #     output_path=opt.output_path
    # )


def detect3D_deprecated(imgs_path):
    detect = detect3d(
        reg_weights='Modules/YOLO3D/weights/resnet18.pkl',
        model_select='resnet18',
        source='P3Data/Sequences/test',
        calib_file='P3Data/Calib/front/calibration.txt',
        coco_config_file='Modules/YOLO3D/data/coco128.yaml',
        show_result=False,
        save_result=True,
        output_path='Output/')[0]
        
    return detect



model_factory = {
    'resnet': resnet18(pretrained=True),
    'resnet18': resnet18(pretrained=True),
    # 'vgg11': vgg11(pretrained=True)
}
regressor_factory = {
    'resnet': ResNet,
    'resnet18': ResNet18,
    'vgg11': VGG11
}

AVERAGES_LABEL_MAP = {
    "sedan": "car", 
    "hatchback": "car", 
    "suv": "car", 
    "box": "truck", 
    "pickup": "truck",
    "bicycle": "cyclist",
    "motorcycle": "cyclist"
}


def detect3d(
        img,
        box_2d,
        label,
        reg_weights='Modules/YOLO3D/weights/resnet18.pkl',
        model_select='resnet18',
        calib_file='P3Data/Calib/front/calibration.txt',
        output_path='Output/',
        show_result=False,
        save_result=True
    ):

    label = AVERAGES_LABEL_MAP[label]
    calib = str(calib_file)

    # load model
    base_model = model_factory[model_select]
    regressor = regressor_factory[model_select](model=base_model).cuda()

    # load weight
    checkpoint = torch.load(reg_weights)
    regressor.load_state_dict(checkpoint['model_state_dict'])
    regressor.eval()

    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(2)

    if not averages.recognized_class(label):
        return 'unrecognized class, orientation detection fail'
    try: 
        detectedObject = DetectedObject(img, label, box_2d, calib)
    except:
        return 'invalid bounds or something lol idk'

    theta_ray = detectedObject.theta_ray
    input_img = detectedObject.img
    proj_matrix = detectedObject.proj_matrix
    detected_class = label
    input_tensor = torch.zeros([1,3,224,224]).cuda()
    input_tensor[0,:,:,:] = input_img

    # predict orient, conf, and dim
    [orient, conf, dim] = regressor(input_tensor)
    orient = orient.cpu().data.numpy()[0, :, :]
    conf = conf.cpu().data.numpy()[0, :]
    dim = dim.cpu().data.numpy()[0, :]

    dim += averages.get_item(detected_class)

    argmax = np.argmax(conf)
    orient = orient[argmax, :]
    cos = orient[0]
    sin = orient[1]
    alpha = np.arctan2(sin, cos)
    alpha += angle_bins[argmax]
    alpha -= np.pi

    # plot 3d detection
    plot3d(img, proj_matrix, box_2d, dim, alpha, theta_ray)

    if show_result:
        cv2.imshow('3d detection', img)
        cv2.waitKey(1)

    if save_result and output_path is not None:
        try:
            os.mkdir(output_path)
        except:
            pass
        cv2.imwrite(f'{output_path}/orientation_disp.png', img)


    return alpha + theta_ray