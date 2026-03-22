import bpy
import os
from glob import glob
import math

bpy.context.scene.render.engine = 'BLENDER_EEVEE'

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
    master_collections = {}
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
        for model_type, model_info in obj_info.items():
            if model_type == "Object": master_assets[obj_name] = {}
            for model, model_values in model_info.items():
                path = os.path.join(file_path, model_type, model)
                directory = os.path.join(file_path, model_type)
                try:
                    bpy.ops.wm.append(filepath=path, directory=directory, filename=model)
                    if model_type == "Collection":
                        new_col = bpy.data.collections.get(model)
                        # Move the collection under our AssetLibrary so it's organized
                        # and unhide it from the 'Master' scene list if needed
                        if new_col:
                            master_collections[model] = {k:v for k,v in model_values.items()}
                            if new_col.name in bpy.context.scene.collection.children:
                                bpy.context.scene.collection.children.unlink(new_col)
                            if "offset" in model_values:
                                new_col.instance_offset = model_values["offset"]
                            for parent_col in bpy.data.collections:
                                if new_col.name in parent_col.children:
                                    parent_col.children.unlink(new_col)
                            
                            if new_col.name not in lib_col.children:
                                lib_col.children.link(new_col)
                        else:
                            print(f"didn't get model {model}")
                    else:
                        obj = bpy.data.objects[model]
                        for col in obj.users_collection: col.objects.unlink(obj)
                        lib_col.objects.link(obj)
                        master_assets[obj_name][model] = {k:v for k,v in model_values.items()}
                        master_assets[obj_name][model]["model"] = obj

                    print(f"Preloaded: {obj_name}/{model}")
                except Exception as e:
                    print(f"Error loading {obj_name}/{model}: {e}")

        if  model_type=="Object" and len(master_assets[obj_name])==0: master_assets.pop(obj_name)
    
    return master_assets, master_collections

def create_instance(asset_name, location, rotation, blender_assets, blender_collections):
    if asset_name in blender_assets:
        for model_name, model_info in blender_assets[asset_name].items():
            obj = model_info["model"]
            new_inst = bpy.data.objects.new(name=f"Instance_{asset_name}/{model_name}", object_data=obj.data)
            bpy.context.scene.collection.objects.link(new_inst)
            new_inst.location = [pos+offset for pos,offset in zip(location, model_info["offset"])]
            new_inst.rotation_euler = [math.radians(rot)+math.radians(offset) for rot,offset in zip(rotation, model_info["rotation"])]
            if isinstance(model_info["scale"], (int,float)):
                new_inst.scale = (model_info["scale"], model_info["scale"], model_info["scale"])
            else:
                new_inst.scale = model_info["scale"]
    elif asset_name in blender_collections:
        master_col = bpy.data.collections.get(asset_name)
        if not master_col:
            print(f"{asset_name} not found in collection")
            return None

        # 1. Create an 'Empty' object
        instance_name = f"Instance_{asset_name}"
        instance_empty = bpy.data.objects.new(instance_name, None)
        
        # 2. Tell the Empty to 'Instance' the collection
        instance_empty.instance_type = 'COLLECTION'
        instance_empty.instance_collection = master_col
        
        # 3. Place it in the scene
        bpy.context.scene.collection.objects.link(instance_empty)
        
        # instance_empty.location = [pos+offset for pos,offset in zip(location, blender_collections[asset_name]["offset"])]   
        instance_empty.location = location       
        instance_empty.rotation_euler = [math.radians(rot)+math.radians(offset) for rot,offset in zip(rotation, blender_collections[asset_name]["rotation"])]
        if isinstance(blender_collections[asset_name]["scale"], (int,float)):
            instance_empty.scale = (blender_collections[asset_name]["scale"], blender_collections[asset_name]["scale"], blender_collections[asset_name]["scale"])
        else:
            instance_empty.scale = blender_collections[asset_name]["scale"]
        # return instance_empty
        # return new_inst
    else:
        print(f'{asset_name} not found')

def render_scene(output_path):
    """
    Renders the current scene and saves it as a PNG.
    """
    scene = bpy.context.scene
    
    # 1. Set File Format to PNG
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA' # Use 'RGB' if no transparency is needed
    
    # 2. Set Output Path
    # Blender automatically adds .png to the end if not provided
    # full_path = os.path.join(output_path, filename)
    scene.render.filepath = output_path
    
    # 3. Optional: Set Resolution (e.g., 1080p)
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    
    # 4. Trigger the Render
    # write_still=True tells Blender to actually save the file to disk
    print(f"Rendering to: {output_path}...")
    bpy.ops.render.render(write_still=True)
    print("Render Complete.")