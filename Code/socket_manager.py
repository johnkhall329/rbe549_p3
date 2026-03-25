import socket
import json

def setup_server(host, port):
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(5)
    server_sock.setblocking(False)
    return server_sock

def read_socket(server_sock, active_connections, handle_callback):
    try:
        conn, addr = server_sock.accept()
        conn.setblocking(False)
        active_connections.append({"conn": conn, "buffer": ""})
    except (BlockingIOError, socket.error):
        conn = None

    for i in range(len(active_connections) - 1, -1, -1):
        item = active_connections[i]
        try:
            chunk = item["conn"].recv(4096).decode('utf-8')
            if not chunk:
                item["conn"].close()
                active_connections.pop(i)
                continue
            
            item["buffer"] += chunk
            while "\n" in item["buffer"]:
                line, item["buffer"] = item["buffer"].split("\n", 1)
                handle_callback(line.strip(), item["conn"])
        except (BlockingIOError, socket.error):
            pass