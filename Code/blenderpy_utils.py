import bpy
import os
from glob import glob
import math

bpy.context.scene.render.engine = 'BLENDER_EEVEE'

traffic_library = {}

def clear_scene(protected_assets):
    """
    Only deletes objects created during the simulation loop,
    leaving the AssetLibrary and infrastructure intact.
    """
    # We iterate over a copy of the list [:] to avoid index errors while deleting
    for obj in bpy.data.objects[:]:
        
        # 1. Protect specific names (Camera, Light, etc.)
        if obj.name in protected_assets:
            continue
            
        # 2. ONLY delete objects we specifically spawned as instances
        if obj.name.startswith("Instance_") or obj.name.startswith("Lane_"):
            # do_unlink=True removes it from all collections
            bpy.data.objects.remove(obj, do_unlink=True)
            
    # Optional: Clean up orphaned mesh data to save RAM
    # This removes the "mesh" data-blocks that no longer have an object using them
    # But it won't touch your Master assets because they are linked to your Library
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
            
    print("Scene Cleared: Removed all instances.")

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
    
    # Preload your dictionary
    global traffic_library
    global off
    off = create_traffic_material("Lens_Off", "grey.png", asset_folder)
    traffic_library = {
        "RED_ON": [create_traffic_material("Red_Circle", "red.png", asset_folder),off,off],
        "YELLOW_ON": [off, create_traffic_material("Yellow_Circle", "yellow.png", asset_folder),off,],
        "GREEN_ON": [off,off,create_traffic_material("Green_Circle", "green.png", asset_folder)],
        "OFF": [off,off,off]
        # ... etc for all 9
    }
    
    colors = ["Red_Arrow", "Yellow_Arrow", "Green_Arrow"]
    directions = ["_L", "_U", "_R", "_D"]
    rotation_nums = [0, 90, 180, 270]
    for color in colors:
        for dir, degrees in zip(directions, rotation_nums):
            lab = color.upper() + dir
            mat = create_traffic_material((color + dir), (color.lower() + ".png"), asset_folder, rotation_deg=degrees)
            if 'Red' in color:
                traffic_library[lab] = [mat, off, off]
            elif 'Yellow' in color:
                traffic_library[lab] = [off, mat, off]
            elif 'Green' in color:
                traffic_library[lab] = [off, off, mat]
            else:
                print("WARNING: color naming is not working properly")
            
                
    return master_assets, master_collections

def create_traffic_material(name, image_name, asset_folder, rotation_deg=0):
    bg_color = (0.467, 0.463, 0.482)
    # Create a new material
    mat = bpy.data.materials.new(name=name)
    mat.use_fake_user = True
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    if 'arrow' in image_name:
        node_tex = nodes.new('ShaderNodeTexImage')
        node_mix = nodes.new('ShaderNodeMix') 
        node_mix.data_type = 'RGBA'

        # --- NEW NODE: Boosts the Color Saturated ---
        node_hsv = nodes.new('ShaderNodeHueSaturation')
        node_hsv.inputs['Saturation'].default_value = 2.0 # Double the color intensity
        node_hsv.inputs['Value'].default_value = 0.5      # Keep brightness steady
        
        node_bright = nodes.new('ShaderNodeMath')
        node_bright.operation = 'MULTIPLY'
        node_bright.inputs[1].default_value = 2.0 # Lower this slightly so color stays visible

        node_bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        node_out = nodes.new('ShaderNodeOutputMaterial')
        
        # Load image (standard code)
        image_path = os.path.join(asset_folder, image_name)
        if os.path.exists(image_path):
            node_tex.image = bpy.data.images.load(image_path)
        
        # APPLY ROTATION
        # --- NEW: COORDINATE NODE ---
        node_coords = nodes.new('ShaderNodeTexCoord')
        
        # --- MAPPING NODE ---
        mapping_node = nodes.new('ShaderNodeMapping')
        mapping_node.vector_type = 'POINT' # Ensure it's in Point mode

        # Apply the rotation
        mapping_node.inputs['Rotation'].default_value[2] = math.radians(rotation_deg)

        # --- LINKING ---
        # Link UVs -> Mapping -> Texture
        links.new(node_coords.outputs['UV'], mapping_node.inputs['Vector'])
        links.new(mapping_node.outputs['Vector'], node_tex.inputs['Vector'])

        # 1. MIXING THE COLORS
        links.new(node_tex.outputs['Alpha'], node_mix.inputs['Factor'])
        node_mix.inputs['A'].default_value = (*bg_color, 1.0)
        links.new(node_tex.outputs['Color'], node_mix.inputs['B']) 

        # 2. PROCESSING THE COLOR (The "Pronounced" Fix)
        # We send the mixed result through the HSV node to "deepen" the color
        links.new(node_mix.outputs['Result'], node_hsv.inputs['Color'])

        # 3. GLOW LOGIC
        links.new(node_tex.outputs['Alpha'], node_bright.inputs[0])
        links.new(node_bright.outputs['Value'], node_bsdf.inputs['Emission Strength'])

        # 4. FINAL CONNECTIONS
        # Use the "Deepened" color for Base and Emission
        links.new(node_hsv.outputs['Color'], node_bsdf.inputs['Base Color'])
        links.new(node_hsv.outputs['Color'], node_bsdf.inputs['Emission Color'])
        
        links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])
    else:
        node_tex = nodes.new('ShaderNodeTexImage')
        node_emit = nodes.new('ShaderNodeBsdfPrincipled') # Or ShaderNodeEmission
        node_out = nodes.new('ShaderNodeOutputMaterial')

        image_path = os.path.join(asset_folder, image_name)
        if os.path.exists(image_path):
            node_tex.image = bpy.data.images.load(image_path)
        else:
            print("no img")

        links.new(node_tex.outputs['Color'], node_emit.inputs['Base Color'])
        links.new(node_emit.outputs['BSDF'], node_out.inputs['Surface'])


    # Link them
    # links.new(node_tex.outputs['Color'], node_bsdf.inputs['Base Color'])
    
    # print(image_name)
    # if "arrow" in image_name:
    # # Link texture to Emission Color (this makes it a light source)
    #     links.new(node_tex.outputs['Color'], node_bsdf.inputs['Emission Color'])
        
    #     # 4. SET BRIGHTNESS:
    #     # Increase this value (e.g., 5.0 to 20.0) to make it brighter
    #     node_bsdf.inputs['Emission Strength'].default_value = 15.0
        
    # links.new(node_bsdf.outputs['BSDF'], node_out.inputs['Surface'])

    
    return mat

def set_light_state(instance_obj, state_key):
    """
    instance_obj: The mesh object of the traffic light
    slot_index: 0 for Red, 1 for Yellow, 2 for Green
    state_key: e.g., "RED_ARROW" or "OFF"
    """
    new_mat = traffic_library.get(state_key)
    if new_mat:
        # This is the "magic" line:
        # It tells Blender: "Only change the material for THIS instance"
        for i, slot_mat in enumerate(new_mat):
            instance_obj.material_slots[i].link = 'OBJECT'
            instance_obj.material_slots[i].material = slot_mat

def create_instance(asset_name, location, rotation, blender_assets, blender_collections, material = None):
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
            if asset_name == "TrafficSignal":
                if material is None: material = "OFF"
                print(material)
                set_light_state(new_inst, material)    
                # bpy.ops.object.constraint_add(type='TRACK_TO')
                # bpy.context.object.constraints["Track To"].target = bpy.data.objects["Camera"]
                # bpy.context.object.constraints["Track To"].up_axis = 'UP_Y'
                # bpy.context.object.constraints["Track To"].track_axis = 'TRACK_X'
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

def insert_human(location, rotation, model_path, blender_assets, material=None):
    for model_name, model_info in blender_assets["Pedestrain"].items():
        if os.path.exists(model_path):
            bpy.ops.wm.obj_import(filepath=model_path)
            new_inst = bpy.context.selected_objects[0]
            new_inst.name = f"Instance_Pedestrain/{model_name}"
            new_inst.location = [pos+offset for pos,offset in zip(location, model_info["offset"])]
            new_inst.rotation_euler = [math.radians(rot)+math.radians(offset) for rot,offset in zip(rotation, model_info["rotation"])]
            # if isinstance(model_info["scale"], (int,float)):
            #     new_inst.scale = (model_info["scale"], model_info["scale"], model_info["scale"])
            # else:
            #     new_inst.scale = model_info["scale"]
        else:
            print(f"Model path not found: {model_path}")

def insert_lane(lane_num, lane_type, lane_color, lane_points, blender_collections):
    name = f'Lane_{lane_type}_{lane_num}'
    curve_data = bpy.data.curves.new(name, type='CURVE')
    curve_data.dimensions = '3D'
    polyline = curve_data.splines.new('POLY')

    polyline.points.add(len(lane_points) - 1)
    for i, point in enumerate(lane_points):
        polyline.points[i].co = (point[0], point[1], point[2], 1.0) #X, Y, Z, Weight

    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(obj)

    color_rgb = (1, 1, 0) if lane_color == "yellow" else (1, 1, 1)
    is_dashed = "dotted" in lane_type or "dashed" in lane_type
    
    mat_name = f"Mat_{lane_color}_{'Dotted' if is_dashed else 'Solid'}"
    
    # Reuse material if it exists, otherwise create
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
    else:
        mat = create_lane_material(mat_name, color_rgb, is_dashed)
    
    obj.data.materials.append(mat)
    curve_data.bevel_depth = 0.04 # Thickness in meters
    curve_data.use_fill_caps = True

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

def create_lane_material(name, color, is_dashed=False):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    
    # Compatibility check for Blender 4.2+
    if hasattr(mat, "blend_method"):
        mat.blend_method = 'HASHED'
    
    # In 4.2+, transparency is often handled by the 'Render Method'
    # but for simple viewport display, HASHED is still a good fallback
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Create Nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (*color, 1.0)
    emission.inputs['Strength'].default_value = 3.0

    if is_dashed:
        transparent = nodes.new('ShaderNodeBsdfTransparent')
        mix_shader = nodes.new('ShaderNodeMixShader')
        tex_coord = nodes.new('ShaderNodeTexCoord')
        sep_xyz = nodes.new('ShaderNodeSeparateXYZ')
        math_mod = nodes.new('ShaderNodeMath')
        math_greater = nodes.new('ShaderNodeMath')
        
        # Logic for the dashes
        math_mod.operation = 'MODULO'
        math_mod.inputs[1].default_value = 4.0 # 4m total cycle
        math_greater.operation = 'GREATER_THAN'
        math_greater.inputs[1].default_value = 2.0 # 2m dash
        
        # Connections
        # Using 'Object' coordinates for real-world meter scaling
        links.new(tex_coord.outputs['Object'], sep_xyz.inputs[0])
        links.new(sep_xyz.outputs['X'], math_mod.inputs[0])
        links.new(math_mod.outputs[0], math_greater.inputs[0])
        
        links.new(math_greater.outputs[0], mix_shader.inputs[0])
        links.new(transparent.outputs[0], mix_shader.inputs[1])
        links.new(emission.outputs[0], mix_shader.inputs[2])
        links.new(mix_shader.outputs[0], output.inputs[0])
    else:
        links.new(emission.outputs[0], output.inputs[0])

    return mat