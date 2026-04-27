#!/usr/bin/env python3
"""
Active Inference Dashboard — Backend Server
============================================
Serves the dashboard UI and REST API reading simulation JSON outputs.
Includes QR code detection state and camera frame serving.

Usage:
    python3 server.py [--port 8060]
    Open http://localhost:8060
"""

import argparse
import json
import mimetypes
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = "/tmp/aif_state.json"
HISTORY_PATH = "/tmp/aif_history.json"
QR_STATE_PATH = "/tmp/qr_state.json"
CAMERA_FRAMES_DIR = "/tmp/camera_frames"


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves static files from dashboard_aif/ and JSON API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/api/state":
            self._serve_json(STATE_PATH)
        elif self.path == "/api/history":
            self._serve_json(HISTORY_PATH)
        elif self.path == "/api/qr_state":
            self._serve_json(QR_STATE_PATH)
        elif self.path.startswith("/api/frame/"):
            self._serve_frame(self.path)
        else:
            super().do_GET()

    def _serve_json(self, filepath: str):
        try:
            with open(filepath, "r") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data.encode())
        except FileNotFoundError:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"data not available yet"}')

    def _serve_frame(self, path: str):
        """Serve camera frame images.
        Routes:
            /api/frame/latest_drone_0.jpg        — latest raw frame
            /api/frame/annotated_drone_0.jpg     — annotated frame with QR bbox
        """
        filename = path.split("/api/frame/")[-1]
        # Strip query string (e.g. ?t=1234 cache-busting param)
        if "?" in filename:
            filename = filename.split("?")[0]
        # Sanitize: only allow safe filenames
        if "/" in filename or ".." in filename:
            self.send_response(400)
            self.end_headers()
            return

        filepath = os.path.join(CAMERA_FRAMES_DIR, filename)
        if not os.path.isfile(filepath):
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"frame not found"}')
            return

        mime, _ = mimetypes.guess_type(filepath)
        mime = mime or "image/jpeg"
        try:
            with open(filepath, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            self.send_response(500)
            self.end_headers()

    def log_message(self, format, *args):
        # suppress per-request log noise; only log errors
        if args and "200" not in str(args[0]):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="AIF Dashboard Server")
    parser.add_argument("--port", type=int, default=8060)
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"[Dashboard] http://localhost:{args.port}")
    print(f"[Dashboard] Reading {STATE_PATH}")
    print(f"[Dashboard] QR state: {QR_STATE_PATH}")
    print(f"[Dashboard] Camera frames: {CAMERA_FRAMES_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Dashboard] Stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
