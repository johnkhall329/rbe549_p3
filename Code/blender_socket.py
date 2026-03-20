import bpy
import socket
import json
import os
import sys
import glob

# EXECUTED BY BLENDER PYTHON SCRIPT
# IF EDITING IN BLENDER SAVE SCRIPT AFTER EDITING

# Setup the Server
HOST = '127.0.0.1' 
PORT = 65432

# Use a global to ensure the socket doesn't get garbage collected
if "server_sock" not in globals():
    global server_sock
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(5)
    server_sock.setblocking(False)
    
    # Track open connections and their individual buffers
    global active_connections
    active_connections = [] 

print(f"Blender listening on {PORT}...")

def socket_listener():
    global active_connections
    
    # 1. Check for NEW connections
    try:
        conn, addr = server_sock.accept()
        conn.setblocking(False)
        # Store connection and a string buffer for it
        active_connections.append({"conn": conn, "buffer": ""})
        print(f"Connected by {addr}")
    except (BlockingIOError, socket.error):
        pass

    # 2. Read from EXISTING connections
    # We iterate backwards so we can safely remove closed connections
    for i in range(len(active_connections) - 1, -1, -1):
        item = active_connections[i]
        c = item["conn"]
        
        try:
            chunk = c.recv(4096).decode('utf-8')
            if not chunk:
                # Connection closed by client
                c.close()
                active_connections.pop(i)
                continue
            
            item["buffer"] += chunk
            
            # 3. Process the buffer for complete messages
            while "\n" in item["buffer"]:
                line, item["buffer"] = item["buffer"].split("\n", 1)
                line = line.strip()
                if line:
                    print(f"Received Command: {line}")
                    handle_command(line)
                    
        except BlockingIOError:
            # No data available right now on this specific socket
            pass
        except Exception as e:
            print(f"Socket error: {e}")
            c.close()
            active_connections.pop(i)

    return 0.01

# Dictionary to store references to our "Master" objects
master_assets = {}

def preload_assets(asset_folder):
    """
    Run this once at the start. 
    It appends the masters into a hidden collection.
    """
    # Create a hidden collection for the masters
    if "AssetLibrary" not in bpy.data.collections:
        lib_col = bpy.data.collections.new("AssetLibrary")
        bpy.context.scene.collection.children.link(lib_col)
    else:
        lib_col = bpy.data.collections["AssetLibrary"]
    
    # Hide it from the render and viewport
    lib_col.hide_viewport = True
    lib_col.hide_render = True

    # List of assets to load from your .blend files
    # Format: {"ObjectName": "FileName.blend"}
    to_load = {
        "Sedan_Model": "vehicles.blend",
        "Truck_Model": "vehicles.blend",
        "Pedestrian_A": "characters.blend"
    }

    
    for obj_name, file_name in to_load.items():
        path = os.path.join(asset_folder, file_name, "Object", obj_name)
        directory = os.path.join(asset_folder, file_name, "Object")
        
        try:
            bpy.ops.wm.append(filepath=path, directory=directory, filename=obj_name)
            obj = bpy.data.objects[obj_name]
            
            # Move it to the hidden library collection
            for col in obj.users_collection:
                col.objects.unlink(obj)
            lib_col.objects.link(obj)
            
            master_assets[obj_name] = obj
            print(f"Preloaded: {obj_name}")
        except Exception as e:
            print(f"Error loading {obj_name}: {e}")

def create_instance(asset_name, location):
    """Creates a lightweight instance of a preloaded asset."""
    if asset_name not in master_assets:
        print(f"Asset {asset_name} not found in library!")
        return None
    
    master_obj = master_assets[asset_name]
    
    # Create a new object that SHARES the mesh data of the master
    new_instance = bpy.data.objects.new(name=f"Instance_{asset_name}", object_data=master_obj.data)
    
    # Link it to the active scene collection so we can see it
    bpy.context.scene.collection.objects.link(new_instance)
    
    new_instance.location = location
    return new_instance

def handle_command(cmd):
    if cmd == 'clear':
        clear_scene()
    elif cmd == 'close':
        print("Closing Blender...")
        os._exit(0)
    else:
        # Assume it's a path or other data
        load_from_json(cmd)

def load_from_json(file_path):
    # For now, just a visual indicator it worked
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
    print(f"Command processed: {file_path}")

def clear_scene():
    for obj in bpy.data.objects:
        if obj.name not in ['Camera', 'Light', 'Ego_Car', 'Empty', 'Plane', 'Sun']:
            bpy.data.objects.remove(obj, do_unlink=True)
    print("Scene Cleared")


def get_custom_args():
    # sys.argv contains the full command line call
    # We only want what comes after '--'
    if "--" in sys.argv:
        index = sys.argv.index("--") + 1
        return sys.argv[index:]
    return []

# --- Logic to use the path ---
args = get_custom_args()

if args:
    asset_path = args[0] # The first item after --
    print(f"Loading assets from: {asset_path}")
    
    # Now call your preloader (from the previous step)
    preload_assets(asset_path)
else:
    print("No asset path provided!")

# Ensure timer is registered
if not bpy.app.timers.is_registered(socket_listener):
    bpy.app.timers.register(socket_listener)