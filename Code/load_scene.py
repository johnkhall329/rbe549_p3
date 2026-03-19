import bpy
import numpy as np
import json
import os

# LOAD SCENE FROM JSON FILE
# EXECUTED BY BLENDER PYTHON SCRIPT
# IF EDITING IN BLENDER SAVE SCRIPT AFTER EDITING

def load_from_json(file_path):
    bpy.ops.mesh.primitive_cube_add(enter_editmode=False, align='WORLD', location=(-2.89929, 1.5, 0.05), scale=(1, 1, 1))
    print(os.getcwd())
    pass


if __name__ == '__main__':
    print("rendering")
    load_from_json("")
    