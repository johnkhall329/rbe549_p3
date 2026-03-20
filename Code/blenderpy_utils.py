import bpy
import os
from glob import glob

def clear_scene(protected_assests):
    for obj in bpy.data.objects:
        if obj.name in protected_assests:
            continue
        if any(col.name == "AssetLibrary" for col in obj.users_collection):
            continue
        bpy.data.objects.remove(obj, do_unlink=True)
    print("Scene Cleared")

def preload_assets(asset_folder, asset_info):
    master_assets = {}
    if "AssetLibrary" not in bpy.data.collections:
        lib_col = bpy.data.collections.new("AssetLibrary")
        bpy.context.scene.collection.children.link(lib_col)
    else:
        lib_col = bpy.data.collections["AssetLibrary"]
    
    lib_col.hide_viewport = lib_col.hide_render = True

    to_load = glob(f"{asset_folder}/**/*.blend",recursive=True)

    for file_path in to_load:
        file_split = file_path.split('/')
        obj_name = file_split[-1].split('.')[0]
        obj_info = asset_info[obj_name]
        master_assets[obj_name] = {}
        for model in obj_info["models"].keys():
            path = os.path.join(file_path, "Object", model)
            directory = os.path.join(file_path, "Object")
            try:
                bpy.ops.wm.append(filepath=path, directory=directory, filename=model)
                obj = bpy.data.objects[model]
                for col in obj.users_collection: col.objects.unlink(obj)
                lib_col.objects.link(obj)
                master_assets[obj_name][model] = obj
                print(f"Preloaded: {obj_name}/{model}")
            except Exception as e:
                print(f"Error loading {obj_name}/{model}: {e}")

            if len(master_assets[obj_name])==0: master_assets.pop(obj_name)
    
    return master_assets

def create_instance(asset_name, blender_assets):
    if asset_name in blender_assets:
        for model_name, obj in blender_assets[asset_name].items():
            print(model_name)
            new_inst = bpy.data.objects.new(name=f"Instance_{asset_name}/{model_name}", object_data=obj.data)
            bpy.context.scene.collection.objects.link(new_inst)
            new_inst.location = (0,0,0)
        # return new_inst