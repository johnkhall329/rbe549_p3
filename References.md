# Phase 2 References

## Grounding DINO for Object Detection
**Paper**:S. Liu, Z. Zeng, T. Ren, F. Li, H. Zhang, J. Yang, Q. Jiang, C. Li,
J. Yang, H. Su, J. Zhu, and L. Zhang, “Grounding dino: Marrying
dino with grounded pre-training for open-set object detection,” 2024.
[Online]. Available: https://arxiv.org/abs/2303.05499

**Usage**: We use Grounding DINO for general object detection purposes. Because of Grounding DINOs capabilities, we are able to input different text prompts to detect and classify (including vehicle subclasses) frames. Detections can be further refined for better results (humans, traffic lights, road signs).

**Datasets**:
Grounding DINO is trained on a variety of large image datasets such as COCO, O365, and OpenImage. It also uses Grounding datasets GoldG and RefC. We used a pre-trained model, Grounding DINO tiny.


## YOLO For Object Detection
**Paper**:J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look
once: Unified, real-time object detection,” 2016. [Online]. Available:
https://arxiv.org/abs/1506.02640

**Usage**: In contrast to last phase, we transitioned away from using YOLO for object detection. Now we use YOLO to augment Grounding DINO by detected any missed cars and classifying traffic signs.

**Datasets**:

- YOLO26 for car supplementary detection is pre-trained on the COCO dataset: 
    - T.-Y. Lin, M. Maire, S. Belongie, L. Bourdev, R. Girshick, J. Hays,
P. Perona, D. Ramanan, C. L. Zitnick, and P. Doll´ar, “Microsoft
coco: Common objects in context,” 2015. [Online]. Available: https://arxiv.org/abs/1405.0312
- YOLOv8 for traffic sign detection is trained on the combined GLARE+LISA dataset:
    - N. Gray, M. Moraes, J. Bian, A. Wang, A. Tian, K. Wilson, Y. Huang,
H. Xiong, and Z. Guo, “Glare: A dataset for traffic sign detection in sun
glare,” 2023. [Online]. Available: https://arxiv.org/abs/2209.08716


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

**Usage**: Mask R-CNN is used to classify any lanes in the scene, however it doesn't have a good understanding of where the lanes are going, especially for dotted lines. Color classification is done classically. Mask R-CNN is also used to detect and display any arrows on the road.

**Datasets**:
We used a pre-trained model from [this implementation](https://debuggercafe.com/lane-detection-using-mask-rcnn/). The author trained the model on the JPJ lane dataset
- https://www.kaggle.com/datasets/sovitrath/road-lane-instance-segmentation 



## Depth Anything V2 For Monocular Depth
**Paper**:L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, and
H. Zhao, “Depth anything v2,” 2024. [Online]. Available: https://arxiv.org/abs/2406.09414

**Usage**: Depth Anything V2 is used to estimate the monocular depth of the scene to place objects within the Blender scene.

**Datasets**: 
We utilized a pre-trained Depth Anything V2 model that was trained on the VKITTI2 outdoor dataset
- Y. Cabon, N. Murray, and M. Humenberger, “Virtual kitti 2,” 2020.
[Online]. Available: https://arxiv.org/abs/2001.1077


## 4D-Humans:
**Paper**:S. Goel, G. Pavlakos, J. Rajasegaran, A. Kanazawa, and J. Malik,
“Humans in 4d: Reconstructing and tracking humans with transformers,”
2023. [Online]. Available: https://arxiv.org/abs/2305.20091

**Usage**: We utilize 4D-Humans to estimate the pose of any humans seen in the scene. The pretrained model will predict 2D/3D keypoints along with generating an SMPL mesh of the human to load into blender as a .obj file.

**Datasets**:
We use 4D-Humans released pre-trained model that was trained on numerous datsets: Human3.6M, MPI-INF-3DHP, COCO, MPII, InstaVariety, AVA and AI Challenger.

## EasyOCR

**Repo**: https://github.com/JaidedAI/EasyOCR

**Usage**: EasyOCR is used to augment our traffic sign detection to ensure that YOLO is detecting the correct sign and to extract the speed limit information.

## Orient-Anything:
**Paper**: Z. Wang, Z. Zhang, T. Pang, C. Du, H. Zhao, and Z. Zhao, “Orient Anything: Learning Robust Object Orientation Estimation from Rendering 3D Models,” 2024. [Online]. Available: https://arxiv.org/abs/2412.18605

**Repo**: https://github.com/SpatialVision/Orient-Anything

**Usage**: We utilize Orient-Anything to estimate the rotation of objects detected in the scene. Given an input image, the model predicts object orientation in a category-agnostic manner. This allows us to use the same model for various different vehicle types

**Datasets**: The pretrained Orient-Anything model is trained on a large synthetic dataset of ~2 million rendered 3D object images with precise 6D orientation annotations, created by rendering 3D models from random views to provide orientation supervision.