import bpy
import os
import sys
import time
import json
import numpy as np

# 1. ADD CURRENT DIRECTORY TO PATH SO BLENDER CAN IMPORT THE OTHER FILES
script_dir = os.path.dirname(os.path.realpath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

import blenderpy_utils
import socket_manager

# --- INIT ---
HOST, PORT = '127.0.0.1', 65432
active_connections = []
server_sock = socket_manager.setup_server(HOST, PORT)


with open('./Code/asset_info.json', 'r') as f:
    global asset_info
    asset_info = json.load(f)

def handle_command(cmd, client_conn):
    global asset_info
    if 'close' in cmd:
        print(f"Closing Blender ...")
        os._exit(0)
    elif 'clear' in cmd:
        # print(asset_info)
        blenderpy_utils.clear_scene(asset_info["protected_assets"])
    elif 'load_new' in cmd:
        json_file = cmd.split('load_new')[-1].strip()
        with open(json_file, 'r') as f:
            new_scene = json.load(f)
            for asset_name, asset_instances in new_scene.items():
                if asset_name == 'Lanes':
                    for i, lane_info in enumerate(asset_instances):
                        lane_type = lane_info.get("type", "solid-line")
                        if lane_type == "road-sign-line":
                            box = np.array(lane_info.get("box", np.zeros((4,2))))
                            file_path = lane_info.get("file_loc", "")
                            blenderpy_utils.insert_road_sign(box, file_path)
                        else:
                            lane_color = lane_info.get("color", "white")
                            lane_points = lane_info.get("curve_points", [])
                            blenderpy_utils.insert_lane(i, lane_type, lane_color, lane_points, blender_collections)
                elif asset_name == "Pedestrain":
                    for i, human_info in enumerate(asset_instances):
                        loc = human_info.get("location", [0.0,0.0,0.0])
                        rot = human_info.get("rotation", [0.0,0.0,0.0])
                        model_path = human_info.get("file location", "")
                        mat = human_info.get("material", None)
                        blenderpy_utils.insert_human(loc, rot, model_path, blender_assets, material=mat)
                else:
                    for info in asset_instances:
                        loc = info.get("location", [0.0,0.0,0.0])
                        rot = info.get("rotation", [0.0,0.0,0.0])
                        mat = info.get("material", None)
                        blenderpy_utils.create_instance(asset_name, loc, rot, blender_assets, blender_collections, material=mat)
    elif 'spawn' in cmd: # Simple example: "spawn Sedan_Model 1,2,0"
        spawned_asset = cmd.split('spawn')[-1].strip()
        blenderpy_utils.create_instance(spawned_asset, (0,0,0), (0,0,0), blender_assets, blender_collections)
    elif 'render' in cmd:
        img_name = cmd.split('render')[-1].strip()
        blenderpy_utils.render_scene(img_name)
    else:
        print(f"Generic command: {cmd}")

    # AFTER the command is finished, send a confirmation
    try:
        client_conn.sendall(b"DONE\n")
    except Exception as e:
        print(f"Failed to send ACK: {e}")

def socket_tick():
    socket_manager.read_socket(server_sock, active_connections, handle_command)
    return 0.01

# --- STARTUP ---
if "--" in sys.argv:
    asset_path = sys.argv[sys.argv.index("--") + 1]
    blender_assets, blender_collections = blenderpy_utils.preload_assets(asset_path, asset_info)

if bpy.app.background:
    print("Headless mode detected. Entering main loop...")
    try:
        while True:
            # Manually trigger the socket check
            socket_tick()
            
            # Sleep for a tiny amount to prevent 100% CPU usage
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Manual shutdown requested.")
        os._exit(0)
else:
    # GUI mode: Register the timer normally
    if not bpy.app.timers.is_registered(socket_tick):
        bpy.app.timers.register(socket_tick)