# Phase 1 References

## YOLO For Object Detection
**Paper**:J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look
once: Unified, real-time object detection,” 2016. [Online]. Available:
https://arxiv.org/abs/1506.02640

**Usage**: We utilize several different YOLO models in this phase to detect various objects. We utilize YOLO26 for general detection of cars, pedestrians, and other random objects. YOLOv8 is used for traffic sign detection and traffic light color detection.

**Datasets**:
- YOLO26 is pre-trained on the COCO dataset: 
    - T.-Y. Lin, M. Maire, S. Belongie, L. Bourdev, R. Girshick, J. Hays,
P. Perona, D. Ramanan, C. L. Zitnick, and P. Doll´ar, “Microsoft
coco: Common objects in context,” 2015. [Online]. Available: https:
//arxiv.org/abs/1405.0312

- YOLOv8 for traffic sign detection is trained on the combined GLARE+LISA dataset:
    - N. Gray, M. Moraes, J. Bian, A. Wang, A. Tian, K. Wilson, Y. Huang,
H. Xiong, and Z. Guo, “Glare: A dataset for traffic sign detection in sun
glare,” 2023. [Online]. Available: https://arxiv.org/abs/2209.08716

- YOLOv8 for traffic light detection is trained on an unknown dataset, but we plan to train our own for light direction using LISA:
    - https://github.com/Syazvinski/Traffic-Light-Detection-Color-Classification/


## YOLOPv2 For Lane Detection

**Paper**:C. Han, Q. Zhao, S. Zhang, Y. Chen, Z. Zhang, and J. Yuan, “Yolopv2:
Better, faster, stronger for panoptic driving perception,” 2022. [Online].
Available: https://arxiv.org/abs/2208.11434

**Usage**: YOLOPv2 is utilized to find the general shape of the lanes for fitting lanes. Because it does not do classification, it is supported by Mask R-CNN.

**Datasets**:
YOLOPv2 is pre-trained on the BDD100k dataset:
- F. Yu, H. Chen, X. Wang, W. Xian, Y. Chen, F. Liu, V. Madhavan,
and T. Darrell, “Bdd100k: A diverse driving dataset for heterogeneous
multitask learning,” 2020. [Online]. Available: https://arxiv.org/abs/1805.
04687

## Mask R-CNN For Lane Detection

**Paper**:K. He, G. Gkioxari, P. Doll´ar, and R. Girshick, “Mask r-cnn,” 2018.
[Online]. Available: https://arxiv.org/abs/1703.06870

**Usage**: Mask R-CNN is used to classify any lanes in the scene, however it doesn't have a good understanding of where the lanes are going, especially for dotted lines. Color classification is done classically.

**Datasets**:
We used a pre-trained model from [this implementation](https://debuggercafe.com/lane-detection-using-mask-rcnn/). The author trained the model on the JPJ lane dataset
- https://www.kaggle.com/datasets/sovitrath/road-lane-instance-segmentation 



## Depth Anything V2 For Monocular Depth
**Paper**:L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and
H. Zhao, “Depth anything v2,” 2024. [Online]. Available: https:
//arxiv.org/abs/2406.09414

**Usage**: Depth Anything V2 is used to estimate the monocular depth of the scene to place objects within the Blender scene.

**Datasets**: 
We utilized a pre-trained Depth Anything V2 model that was trained on the VKITTI2 outdoor dataset
- Y. Cabon, N. Murray, and M. Humenberger, “Virtual kitti 2,” 2020.
[Online]. Available: https://arxiv.org/abs/2001.1077
