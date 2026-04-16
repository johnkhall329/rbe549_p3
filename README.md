# Folder Structure:

To run this project, you must have also create a Output, Modules, Videos, and Models folder. It is expected that you have the P3Data in the same directory as well.

You can download our custom assets and other data that is necessary to run the visualization from this [Google Drive Folder](https://drive.google.com/drive/folders/1lv6-ocuUh5Z2NXijecUbw2u6x0qbk63E?usp=sharing). Put the Blender folder at the base directory, Assets and Calib go inside the P3Data folder. 

# Modules:

Clone these repos and place them in your modules folder
- https://github.com/DepthAnything/Depth-Anything-V2/tree/main
- https://github.com/shubham-goel/4D-Humans
- https://github.com/DQiaole/MemFlow.git
- https://github.com/IDEA-Research/Grounded-SAM-2
- https://github.com/SpatialVision/Orient-Anything

## Orient Anything edits

You may need to edit line 245 to be:
`axis_model = Model("./Modules/Orient-Anything/assets/axis.obj", texture_filename="./Modules/Orient-Anything/assets/axis.png")`

In lines 105, 107, 109, and 111 change the path to be:
`self.dinov2 = FLIP_DINOv2.from_pretrained(DINO_BASE, cache_dir='./Models/')`

## MemFlow

To prevent import errors, You will need to edit the files within the Inference folder to change the import statements.
Just delete `inference` from `from inference.memory_manager_skflow import MemoryManager` in each file. 

## 4D-Humans

You will need to add `weights_only=False` to line 84 in hmr2/models/__init__.py

Also `pip install -e .` in the base directory

## Grounded-SAM2

Execute `pip install -e .` in the base directory.


# Models:

- DepthAnythingV2
    - https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-Large/resolve/main/depth_anything_v2_metric_vkitti_vitl.pth?download=true
- MemFlow
    - https://github.com/DQiaole/MemFlow/releases/tag/v1.0.0
- YOLO26
    - https://docs.ultralytics.com/models/yolo26/#supported-tasks-and-modes 
- YOLOPv2
    - https://github.com/CAIC-AD/YOLOPv2?tab=readme-ov-file
- YOLO-GLARE
    - https://github.com/NicholasCG/GLARE_Dataset 
- Mask-R-CNN
    - https://debuggercafe.com/lane-detection-using-mask-rcnn/

You will also need to pip install the requirements.txt file. This was tested on python versions 3.10 and 3.12.

# Inference:

To run our code, you can either directly use `Wrapper.py` to run both perception and visualiztion. Or you can split the task up and run `generate_scenes.py` for perception and `render_scenes.py` for visualization. 

