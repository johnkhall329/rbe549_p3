To run this project, the following projects must be installed in the Modules folder:

https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth#pre-trained-models


and the following models must be stored in the checkpoints folder:

https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth?download=true

Mask-R-CNN found here, extract the .pth from outputs/training/road_line:
https://debuggercafe.com/lane-detection-using-mask-rcnn/

YOLOPv2 can be downloaded here and put in Models:
https://github.com/CAIC-AD/YOLOPv2?tab=readme-ov-file

YOLOv8 trained on the GLARE and LISA datasets is used for sign detection:
https://github.com/NicholasCG/GLARE_Dataset 

YOLOv8 for traffic light detection Phase 1:
https://github.com/Syazvinski/Traffic-Light-Detection-Color-Classification/ 