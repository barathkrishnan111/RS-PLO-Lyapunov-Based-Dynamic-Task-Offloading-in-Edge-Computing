"""
RS-PLO Live Dashboard — Real-Time Web Visualization

Starts:
  1. HTTP server on port 8080 to serve dashboard.html
  2. SSE endpoint on /events for real-time data push
  3. 3 Edge servers (9999, 10000, 10001)
  4. Interactive CLI (same as demo.py)

Open http://localhost:8080 in your browser to see live charts!

Usage:  python dashboard.py
"""

import http.server
import socketserver
import threading
import json
import time
import sys
import os
import queue

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from demo import LiveDemo, C


# ─────────────────────────────────────────────
#  SSE (Server-Sent Events) Manager
# ─────────────────────────────────────────────

class SSEManager:
    """Manages Server-Sent Events connections and broadcasts."""
    def __init__(self):
        self.clients = []
        self.lock = threading.Lock()

    def add_client(self, client_queue):
        with self.lock:
            self.clients.append(client_queue)

    def remove_client(self, client_queue):
        with self.lock:
            if client_queue in self.clients:
                self.clients.remove(client_queue)

    def broadcast(self, event_type, data):
        """Send an event to all connected clients."""
        message = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        with self.lock:
            dead = []
            for q in self.clients:
                try:
                    q.put_nowait(message)
                except:
                    dead.append(q)
            for q in dead:
                self.clients.remove(q)


sse_manager = SSEManager()


# ─────────────────────────────────────────────
#  HTTP Request Handler (serves HTML + SSE)
# ─────────────────────────────────────────────

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler that serves the dashboard and SSE events."""

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.path = "/dashboard.html"
            return super().do_GET()

        elif self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            client_queue = queue.Queue()
            sse_manager.add_client(client_queue)

            try:
                # Send initial state
                self.wfile.write(b"event: connected\ndata: {\"status\": \"ok\"}\n\n")
                self.wfile.flush()

                while True:
                    try:
                        message = client_queue.get(timeout=15)
                        self.wfile.write(message.encode())
                        self.wfile.flush()
                    except queue.Empty:
                        # Send keepalive
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            finally:
                sse_manager.remove_client(client_queue)
            return

        else:
            return super().do_GET()

    def log_message(self, format, *args):
        pass  # Suppress HTTP logs


# ─────────────────────────────────────────────
#  Dashboard Server
# ─────────────────────────────────────────────

def start_http_server(port=8080):
    """Start the HTTP + SSE server in a background thread."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    handler = DashboardHandler
    httpd = socketserver.TCPServer(("", port), handler)
    httpd.allow_reuse_address = True

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    print(f"""
{C.BOLD}{C.CYAN}{'='*70}{C.RESET}
{C.BOLD}{C.WHITE}  RS-PLO LIVE DASHBOARD — Real-Time Multi-Edge Visualization{C.RESET}
{C.BOLD}{C.CYAN}{'='*70}{C.RESET}
""")

    # Start HTTP + SSE server
    http_port = 8080
    httpd = start_http_server(http_port)
    print(f"  {C.GREEN}Dashboard running at: http://localhost:{http_port}{C.RESET}")
    print(f"  {C.DIM}Open this URL in your browser to see live charts!{C.RESET}")
    print()

    # Create the demo engine
    demo = LiveDemo()

    # Hook into task completion to broadcast SSE events
    def on_task_complete(task_entry):
        sse_manager.broadcast("task", task_entry)
        # Also send current state
        state = {
            "Q": demo.user.Q,
            "Z": demo.user.Z,
            "V": demo.engine.compute_V(demo.user.Z),
            "distance": demo.user.distance,
            "task_count": demo.task_count,
        }
        sse_manager.broadcast("state", state)

    demo.on_task_complete = on_task_complete

    # Start edge servers
    demo.start_servers()
    demo.show_banner()

    # Send initial server info to dashboard
    servers_info = []
    for server in demo.params.edge_servers:
        servers_info.append({
            "name": server.name,
            "port": server.port,
            "distance": server.distance,
            "compute_multiplier": server.compute_multiplier,
            "description": server.description,
        })

    # Delay to let browser connect, then send
    def send_init():
        time.sleep(2)
        sse_manager.broadcast("init", {
            "servers": servers_info,
            "params": {
                "V_max": demo.params.V_max,
                "beta": demo.params.beta,
                "gamma": demo.params.gamma,
            }
        })
    threading.Thread(target=send_init, daemon=True).start()

    demo.show_help()
    demo.show_status()

    print(f"{C.GREEN}  Ready! Type a command (or 'help'):{C.RESET}")
    print(f"{C.DIM}  Dashboard: http://localhost:{http_port}{C.RESET}")
    print()

    running = True
    while running:
        try:
            prompt = f"{C.BOLD}{C.CYAN}  rsplo>{C.RESET} "
            raw = input(prompt)
            running = demo.handle_command(raw)
        except (KeyboardInterrupt, EOFError):
            print()
            running = False

    print(f"\n{C.DIM}  Shutting down...{C.RESET}")
    demo.stop_servers()
    httpd.shutdown()

    if demo.task_log:
        demo.show_history()
    print(f"  {C.GREEN}Goodbye!{C.RESET}\n")


if __name__ == "__main__":
    main()
