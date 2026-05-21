#!/usr/bin/env python3
"""
Active Inference Dashboard — Backend Server
============================================
Serves the dashboard UI and REST API reading simulation JSON outputs.
Includes QR code detection state and camera frame serving.

New endpoints (Tâche 5 du PROMPT_CLAUDE_CODE.md) :
    /api/ns3                          → parse /tmp/ns3_output.csv ou
                                         /tmp/drone_latency_ns3.csv (5G)
                                         + état réseau lu depuis aif_state.json
    /api/runs                         → liste des dossiers logs/runs/* avec
                                         leur config.json
    /api/run/<tag>/img/<name>         → sert un PNG du run (belief_map_final.png,
                                         trajectories.png, ...)

Usage:
    python3 server.py [--port 8060] [--runs-dir <path>]
    Open http://localhost:8060
"""

import argparse
import csv
import json
import mimetypes
import os
import re
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import unquote

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_PATH = "/tmp/aif_state.json"
HISTORY_PATH = "/tmp/aif_history.json"
QR_STATE_PATH = "/tmp/qr_state.json"
CAMERA_FRAMES_DIR = "/tmp/camera_frames"

# ── NS-3 sources possibles ──
NS3_WIFI_CSV = "/tmp/ns3_output.csv"
NS3_LTE_CSV  = "/tmp/drone_latency_ns3.csv"
NS3_5G_METRICS = "/tmp/drone_5g_metrics.csv"

# ── Dossier des runs ── (override via --runs-dir ou env AIF_RUNS_DIR)
_WORKSPACE_ROOT = os.path.dirname(DASHBOARD_DIR)  # parent of dashboard_aif/
DEFAULT_RUNS_DIR = os.environ.get(
    "AIF_RUNS_DIR",
    os.path.join(_WORKSPACE_ROOT, "logs", "runs"),
)
RUNS_DIR = DEFAULT_RUNS_DIR  # peut être surchargé via CLI

# Whitelist images servies par /api/run/<tag>/img/<name>
SAFE_IMG_RE = re.compile(r"^[a-zA-Z0-9_.\-]+\.(png|jpg|jpeg|svg)$")
SAFE_TAG_RE = re.compile(r"^[a-zA-Z0-9_.\-]+$")


def _read_ns3_csv(path: str):
    """Lit un CSV NS-3 (formats wifi / 5g) → liste de dicts {a, b, latency_ms, jitter_ms, rx_packets}.
    Tolère les fichiers en cours d'écriture (lignes malformées ignorées)."""
    if not os.path.isfile(path):
        return []
    out = []
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    i_key = "drone_i" if "drone_i" in row else "drone_a"
                    j_key = "drone_j" if "drone_j" in row else "drone_b"
                    a = int(row[i_key])
                    b = int(row[j_key])
                    lat = float(row.get("latency_ms", 0.0))
                    jitter = float(row.get("jitter_ms", 0.0)) if "jitter_ms" in row else 0.0
                    rx = int(row.get("rx_packets", 0)) if "rx_packets" in row else 0
                    out.append({
                        "a": min(a, b), "b": max(a, b),
                        "latency_ms": round(lat, 3),
                        "jitter_ms": round(jitter, 3),
                        "rx_packets": rx,
                    })
                except (ValueError, KeyError):
                    continue
    except (IOError, OSError):
        return out
    # déduplication : garder la dernière mesure par paire
    by_pair = {}
    for row in out:
        by_pair[(row["a"], row["b"])] = row
    return list(by_pair.values())


def _ns3_state() -> dict:
    """Construit la réponse /api/ns3 : pairs (latencies) + état réseau."""
    pairs = []
    # essai wifi puis 5g
    if os.path.exists(NS3_WIFI_CSV):
        pairs = _read_ns3_csv(NS3_WIFI_CSV)
    if not pairs and os.path.exists(NS3_LTE_CSV):
        pairs = _read_ns3_csv(NS3_LTE_CSV)
    if not pairs and os.path.exists(NS3_5G_METRICS):
        pairs = _read_ns3_csv(NS3_5G_METRICS)

    network = {}
    ns3_mode = "none"
    cloud_link = True
    queue_size = 0
    dropped = 0
    sent = 0
    delivered = 0
    cut_pairs = []
    all_cut = False
    try:
        with open(STATE_PATH) as f:
            state = json.load(f)
        ns3_mode = state.get("ns3_mode", "none")
        net = state.get("network", {}) or {}
        cloud_link = bool(net.get("cloud_link_active", True))
        queue_size = int(net.get("queue_size", 0))
        dropped    = int(net.get("msg_dropped", 0))
        sent       = int(net.get("msg_sent", 0))
        delivered  = int(net.get("msg_delivered", 0))
        cut_pairs  = list(net.get("cut_pairs", []))
        all_cut    = bool(net.get("all_drone_links_cut", False))
        # si aif_state.json contient déjà les pairs, on les utilise (plus à jour)
        if not pairs and isinstance(net.get("ns3_pairs"), list):
            pairs = net["ns3_pairs"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass

    return {
        "ns3_mode": ns3_mode,
        "cloud_link": "up" if cloud_link else "down",
        "pairs": pairs,
        "queue_size": queue_size,
        "dropped": dropped,
        "sent": sent,
        "delivered": delivered,
        "cut_pairs": cut_pairs,
        "all_drone_links_cut": all_cut,
    }


def _runs_list() -> dict:
    """Liste les sous-dossiers run_* dans RUNS_DIR avec leur config.json."""
    runs = []
    if os.path.isdir(RUNS_DIR):
        for name in sorted(os.listdir(RUNS_DIR), reverse=True):
            run_path = os.path.join(RUNS_DIR, name)
            if not os.path.isdir(run_path) or not name.startswith("run_"):
                continue
            cfg = {}
            cfg_path = os.path.join(run_path, "config.json")
            try:
                if os.path.isfile(cfg_path):
                    with open(cfg_path) as f:
                        cfg = json.load(f)
            except (IOError, json.JSONDecodeError):
                cfg = {}
            # Images disponibles
            images = []
            try:
                for fn in sorted(os.listdir(run_path)):
                    if SAFE_IMG_RE.match(fn):
                        images.append(fn)
            except OSError:
                pass
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(run_path)).isoformat(timespec="seconds")
            except OSError:
                mtime = ""
            runs.append({
                "tag": name,
                "config": cfg,
                "images": images,
                "mtime": mtime,
            })
    return {"runs_dir": RUNS_DIR, "runs": runs}


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves static files from dashboard_aif/ and JSON API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def do_GET(self):
        # routes JSON
        if self.path == "/api/state":
            self._serve_json_file(STATE_PATH)
        elif self.path == "/api/history":
            self._serve_json_file(HISTORY_PATH)
        elif self.path == "/api/qr_state":
            self._serve_json_file(QR_STATE_PATH)
        elif self.path == "/api/ns3":
            self._serve_json_obj(_ns3_state())
        elif self.path == "/api/runs":
            self._serve_json_obj(_runs_list())
        elif self.path.startswith("/api/frame/"):
            self._serve_frame(self.path)
        elif self.path.startswith("/api/run/"):
            self._serve_run_img(self.path)
        else:
            super().do_GET()

    # ── JSON helpers ──
    def _serve_json_obj(self, obj):
        data = json.dumps(obj, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_json_file(self, filepath: str):
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
        if "?" in filename:
            filename = filename.split("?")[0]
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

    def _serve_run_img(self, path: str):
        """Route: /api/run/<tag>/img/<name>  — sert un PNG d'un run."""
        # On strippe d'éventuels query strings (cache-busting éventuel)
        clean = path.split("?", 1)[0]
        m = re.match(r"^/api/run/([^/]+)/img/(.+)$", clean)
        if not m:
            self.send_response(404)
            self.end_headers()
            return
        tag = unquote(m.group(1))
        name = unquote(m.group(2))
        if not SAFE_TAG_RE.match(tag) or not SAFE_IMG_RE.match(name):
            self.send_response(400)
            self.end_headers()
            return
        filepath = os.path.join(RUNS_DIR, tag, name)
        # Sanity check : le path résolu doit rester dans RUNS_DIR
        try:
            real = os.path.realpath(filepath)
            real_root = os.path.realpath(RUNS_DIR)
            if not real.startswith(real_root + os.sep):
                self.send_response(400)
                self.end_headers()
                return
        except OSError:
            self.send_response(500)
            self.end_headers()
            return
        if not os.path.isfile(filepath):
            self.send_response(404)
            self.end_headers()
            return
        mime, _ = mimetypes.guess_type(filepath)
        mime = mime or "image/png"
        try:
            with open(filepath, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except OSError:
            self.send_response(500)
            self.end_headers()

    def log_message(self, format, *args):
        # suppress per-request log noise; only log errors
        if args and "200" not in str(args[0]):
            super().log_message(format, *args)


def main():
    global RUNS_DIR
    parser = argparse.ArgumentParser(description="AIF Dashboard Server")
    parser.add_argument("--port", type=int, default=8060)
    parser.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR,
                        help="Path to logs/runs/ (default: <workspace>/logs/runs)")
    args = parser.parse_args()
    RUNS_DIR = os.path.abspath(args.runs_dir)

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"[Dashboard] http://localhost:{args.port}")
    print(f"[Dashboard] Reading {STATE_PATH}")
    print(f"[Dashboard] QR state: {QR_STATE_PATH}")
    print(f"[Dashboard] Camera frames: {CAMERA_FRAMES_DIR}")
    print(f"[Dashboard] Runs dir   : {RUNS_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Dashboard] Stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
