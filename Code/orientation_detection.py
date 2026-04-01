
import os
import sys

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Modules"))
sys.path.append(repo_path)

# Now you can import directly from the inner folder name
from YOLO3D.inference import detect3d

# detect3d(
    #     reg_weights=opt.reg_weights,
    #     model_select=opt.model_select,
    #     source=opt.source,
    #     calib_file=opt.calib_file,
    #     show_result=opt.show_result,
    #     save_result=opt.save_result,
    #     output_path=opt.output_path
    # )


detect = detect3d(
    reg_weights='Modules/YOLO3D/weights/resnet18.pkl',
    model_select='resnet18',
    source='Modules/YOLO3D/eval/image_2',
    calib_file='Modules/YOLO3D/eval/camera_cal/calib_cam_to_cam.txt',
    coco_config_file='Modules/YOLO3D/data/coco128.yaml',
    show_result=False,
    save_result=True,
    output_path='Output/')

print(detect)