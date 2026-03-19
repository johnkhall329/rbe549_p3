import bpy
import socket
import json
import os

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
        if obj.type not in ['CAMERA', 'LIGHT']:
            bpy.data.objects.remove(obj, do_unlink=True)
    print("Scene Cleared")

# Ensure timer is registered
if not bpy.app.timers.is_registered(socket_listener):
    bpy.app.timers.register(socket_listener)