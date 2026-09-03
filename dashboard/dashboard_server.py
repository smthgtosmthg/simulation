#!/usr/bin/env python3
"""
Dashboard Backend Server
========================
Reads the CSV files produced by:
  - 08_wifi_bridge.py   → WiFi RSSI + latency
  - 09_5g_lena_bridge.py → 5G NR RSSI + latency

Serves a REST API consumed by the HTML/JS dashboard.

Usage:
    python3 dashboard_server.py [--port 8050]

Then open http://localhost:8050 in your browser.
"""

import argparse
import csv
import json
import math
import os
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

# ─── CSV paths (same as in the bridge scripts) ───
WIFI_LOG_CSV     = "/tmp/drone_bridge_log.csv"
WIFI_RSSI_CSV    = "/tmp/drone_rssi_latency.csv"
NS3_OUTPUT_CSV   = "/tmp/ns3_output.csv"

FIVEG_LOG_CSV    = "/tmp/drone_5g_log.csv"
FIVEG_RSSI_CSV   = "/tmp/drone_rssi_sionna.csv"
FIVEG_LAT_CSV    = "/tmp/drone_latency_ns3.csv"
FIVEG_METRICS    = "/tmp/drone_5g_metrics.csv"

POS_CSV          = "/tmp/drone_positions.csv"
EXPLORE_JSON     = "/tmp/exploration_state.json"
AIF_STATE_JSON   = "/tmp/aif_state.json"

GNB_POSITION = (0.0, 0.0, 6.0)

# ─── Utility readers ───

def read_positions(path=POS_CSV):
    """Read drone positions → {drone_id: {x, y, z}}."""
    pos = {}
    try:
        with open(path) as f:
            for row in csv.reader(f):
                if not row or row[0].startswith("#"):
                    continue
                if row[0].strip() in ("drone_id", "timestamp"):
                    continue
                try:
                    if len(row) >= 5:
                        did, x, y, z = int(row[1]), float(row[2]), float(row[3]), float(row[4])
                    else:
                        did, x, y, z = int(row[0]), float(row[1]), float(row[2]), float(row[3])
                    pos[did] = {"x": x, "y": y, "z": z}
                except (ValueError, IndexError):
                    continue
    except FileNotFoundError:
        pass
    return pos


def read_wifi_rssi_csv(path=WIFI_RSSI_CSV):
    """Read the WiFi RSSI snapshot CSV (overwritten each tick).
    Format: drone_a,drone_b,rssi_dBm,latency_ms,distance_m,latency_source."""
    pairs = []
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append({
                    "drone_a": int(row["drone_a"]),
                    "drone_b": int(row["drone_b"]),
                    "rssi": row["rssi_dBm"],
                    "latency": row["latency_ms"],
                    "distance": row["distance_m"],
                    "source": row.get("latency_source", ""),
                })
    except (FileNotFoundError, KeyError):
        pass
    return pairs


def read_wifi_log_tail(path=WIFI_LOG_CSV, max_lines=50):
    """Read the last N lines of the WiFi log."""
    lines = []
    try:
        with open(path) as f:
            all_lines = f.readlines()
            lines = all_lines[-max_lines:] if len(all_lines) > max_lines else all_lines
    except FileNotFoundError:
        pass
    return lines


def read_5g_rssi_csv(path=FIVEG_RSSI_CSV):
    """Read the 5G RSSI CSV (appended each tick).
    Keep only the latest entry per drone.
    Format: timestamp,tick,drone_id,x,y,z,dist_gnb_m,rssi_gnb_dBm,delay_gnb_ns,status."""
    latest = {}
    max_tick = -1
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    tick = int(row["tick"])
                    if tick > max_tick:
                        max_tick = tick
                        latest.clear()
                    if tick == max_tick:
                        did = int(row["drone_id"])
                        rssi = None if row["rssi_gnb_dBm"] == "BLOCKED" else float(row["rssi_gnb_dBm"])
                        delay = None if row["delay_gnb_ns"] == "BLOCKED" else int(row["delay_gnb_ns"])
                        latest[did] = {
                            "id": did,
                            "x": float(row["x"]),
                            "y": float(row["y"]),
                            "z": float(row["z"]),
                            "dist_gnb": float(row["dist_gnb_m"]),
                            "rssi": rssi,
                            "delay_ns": delay,
                            "status": row["status"],
                        }
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        pass
    return list(latest.values()), max_tick


def read_5g_latency_csv(path=FIVEG_LAT_CSV):
    """Read the 5G latency CSV (appended each tick).
    Keep only the latest entry per pair.
    Format: timestamp,tick,drone_a,drone_b,latency_ms,jitter_ms,dist_ab_m,rx_packets."""
    latest = {}
    max_tick = -1
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    tick = int(row["tick"])
                    if tick > max_tick:
                        max_tick = tick
                        latest.clear()
                    if tick == max_tick:
                        a, b = int(row["drone_a"]), int(row["drone_b"])
                        latest[(a, b)] = {
                            "drone_a": a,
                            "drone_b": b,
                            "latency": float(row["latency_ms"]),
                            "jitter": float(row["jitter_ms"]),
                            "distance": float(row["dist_ab_m"]),
                            "rx": int(row["rx_packets"]),
                        }
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        pass
    return list(latest.values()), max_tick


def read_5g_metrics_csv(path=FIVEG_METRICS):
    """Read the NS-3 5G metrics CSV produced by drone-5g-nr-scenario.
    Format: drone_a,drone_b,rssi_a_dBm,rssi_b_dBm,latency_ms,jitter_ms,dist_ab_m,rx_packets."""
    pairs = []
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pairs.append({
                    "drone_a": int(row["drone_a"]),
                    "drone_b": int(row["drone_b"]),
                    "rssi_a": float(row["rssi_a_dBm"]),
                    "rssi_b": float(row["rssi_b_dBm"]),
                    "latency": float(row["latency_ms"]),
                    "jitter": float(row["jitter_ms"]),
                    "distance": float(row["dist_ab_m"]),
                    "rx": int(row["rx_packets"]),
                })
    except (FileNotFoundError, KeyError):
        pass
    return pairs


def read_ns3_wifi_output(path=NS3_OUTPUT_CSV):
    """Read the NS-3 WiFi scenario output as backup.
    Format: time_s,drone_i,drone_j,distance_m,rssi_dbm,latency_ms,..."""
    latest = {}
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    i, j = int(row["drone_i"]), int(row["drone_j"])
                    key = (min(i, j), max(i, j))
                    latest[key] = {
                        "drone_a": key[0],
                        "drone_b": key[1],
                        "rssi": float(row["rssi_dbm"]),
                        "latency": float(row["latency_ms"]),
                        "distance": float(row["distance_m"]),
                    }
                except (ValueError, KeyError):
                    continue
    except FileNotFoundError:
        pass
    return list(latest.values())


def read_exploration_state(path=EXPLORE_JSON):
    """Read exploration data.

    Priority:
      1) New AIF format from /tmp/aif_state.json (script 12)
      2) Legacy exploration format from /tmp/exploration_state.json
    """
    aif_state = _read_json_file(AIF_STATE_JSON)
    if aif_state:
        converted = _convert_aif_state_to_exploration(aif_state)
        if converted:
            return converted

    legacy = _read_json_file(path)
    if legacy:
        return legacy
    return None


def _read_json_file(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _convert_aif_state_to_exploration(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert /tmp/aif_state.json payload into dashboard exploration format."""
    env = state.get("environment") or {}
    fused = state.get("fused_belief") or []
    drones = state.get("drones") or []
    metrics = state.get("metrics") or {}

    if not isinstance(fused, list) or not fused:
        return None

    height = len(fused)
    width = len(fused[0]) if isinstance(fused[0], list) else 0
    if width <= 0:
        return None

    flat_probs: List[float] = []
    visited: List[int] = []
    for row in fused:
        if not isinstance(row, list):
            return None
        for p in row:
            try:
                pv = float(p)
            except (TypeError, ValueError):
                pv = 0.5
            pv = max(0.0, min(1.0, pv))
            flat_probs.append(pv)
            visited.append(1 if abs(pv - 0.5) > 0.02 else 0)

    if len(flat_probs) != width * height:
        return None

    resolution = float(env.get("grid_resolution", 0.5))
    origin_x = float(env.get("origin_x", 0.0))
    origin_y = float(env.get("origin_y", 0.0))

    expl_drones: List[Dict[str, Any]] = []
    trajectories: Dict[str, List[List[float]]] = {}

    for d in drones:
        if not isinstance(d, dict):
            continue
        did = int(d.get("id", len(expl_drones)))
        trail = d.get("trail") if isinstance(d.get("trail"), list) else []

        sx = 0.0
        if len(trail) >= 2:
            try:
                x0, y0 = float(trail[-2][0]), float(trail[-2][1])
                x1, y1 = float(trail[-1][0]), float(trail[-1][1])
                sx = math.hypot(x1 - x0, y1 - y0)
            except (TypeError, ValueError, IndexError):
                sx = 0.0

        x = float(d.get("x", 0.0))
        y = float(d.get("y", 0.0))

        expl_drones.append({
            "id": did,
            "x": x,
            "y": y,
            "z": float(d.get("z", 2.0)),
            "speed": sx,
            "yaw": float(d.get("heading", 0.0)),
            "status": "flying",
            "target_x": None,
            "target_y": None,
            "cells_discovered": 0,
            "distance_traveled": float(d.get("total_distance", 0.0)),
        })

        if trail:
            clean_trail: List[List[float]] = []
            for pt in trail:
                try:
                    clean_trail.append([float(pt[0]), float(pt[1])])
                except (TypeError, ValueError, IndexError):
                    continue
            trajectories[str(did)] = clean_trail

    explored_cells = int(sum(visited))
    total_cells = int(len(visited))

    # Use metric when provided; fallback to visited ratio.
    if "exploration_pct" in metrics:
        explored_pct = float(metrics.get("exploration_pct", 0.0))
    else:
        explored_pct = (100.0 * explored_cells / total_cells) if total_cells else 0.0

    return {
        "step": int(state.get("step", 0)),
        "explored_pct": explored_pct,
        "explored_cells": explored_cells,
        "total_cells": total_cells,
        "grid": {
            "width": width,
            "height": height,
            "resolution": resolution,
            "origin_x": origin_x,
            "origin_y": origin_y,
            "data": flat_probs,
            "visited": visited,
        },
        "frontiers": [],
        "drones": expl_drones,
        "trajectories": trajectories,
        "events": [],
        "environment": {
            "width": float(env.get("width", width * resolution)),
            "height": float(env.get("height", height * resolution)),
            "obstacles": env.get("obstacles", []),
            "lidar_max_range": float(env.get("lidar_max_range", 8.0)),
        },
    }


# ─── Build API response ───

def build_response():
    """Aggregate all data sources into a single JSON response."""

    # WiFi data
    wifi_pairs = read_wifi_rssi_csv()
    positions = read_positions()

    # Enrich WiFi pairs with positions
    for p in wifi_pairs:
        a, b = p["drone_a"], p["drone_b"]
        p["pos_a"] = positions.get(a)
        p["pos_b"] = positions.get(b)

    # If no WiFi RSSI CSV, fallback to NS-3 output
    if not wifi_pairs:
        ns3_pairs = read_ns3_wifi_output()
        for p in ns3_pairs:
            a, b = p["drone_a"], p["drone_b"]
            p["pos_a"] = positions.get(a)
            p["pos_b"] = positions.get(b)
            p["source"] = "ns3"
        wifi_pairs = ns3_pairs

    wifi_tick = 0
    # Try to get tick from log
    log_lines = read_wifi_log_tail(max_lines=1)
    if log_lines and len(log_lines) > 0:
        try:
            parts = log_lines[-1].strip().split(",")
            wifi_tick = int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, IndexError):
            pass

    # 5G data
    drones_5g, tick_5g_rssi = read_5g_rssi_csv()
    pairs_5g, tick_5g_lat   = read_5g_latency_csv()

    # If no Sionna RSSI CSV, try the NS-3 metrics CSV directly
    if not drones_5g:
        metrics = read_5g_metrics_csv()
        if metrics:
            # Build drone list from metrics
            drone_set = {}
            for m in metrics:
                for side in ["a", "b"]:
                    did = m[f"drone_{side}"]
                    if did not in drone_set:
                        pos = positions.get(did, {"x": 0, "y": 0, "z": 0})
                        dist_gnb = math.sqrt(sum((pos[k] - g) ** 2 for k, g in zip(["x", "y", "z"], GNB_POSITION)))
                        drone_set[did] = {
                            "id": did,
                            "x": pos["x"], "y": pos["y"], "z": pos["z"],
                            "dist_gnb": round(dist_gnb, 2),
                            "rssi": m[f"rssi_{side}"],
                            "delay_ns": None,
                            "status": "LOS",
                        }
            drones_5g = sorted(drone_set.values(), key=lambda d: d["id"])
            if not pairs_5g:
                pairs_5g = [{
                    "drone_a": m["drone_a"],
                    "drone_b": m["drone_b"],
                    "latency": m["latency"],
                    "jitter": m["jitter"],
                    "distance": m["distance"],
                    "rx": m["rx"],
                } for m in metrics]

    # Always update 5G drone positions from the live positions CSV
    # (RSSI CSV positions may be stale if the bridge hasn't run recently)
    for d in drones_5g:
        live_pos = positions.get(d["id"])
        if live_pos:
            d["x"] = live_pos["x"]
            d["y"] = live_pos["y"]
            d["z"] = live_pos["z"]
            d["dist_gnb"] = round(math.sqrt(
                sum((live_pos[k] - g) ** 2 for k, g in zip(["x", "y", "z"], GNB_POSITION))
            ), 2)

    # Also update pair distances from live positions
    for p in pairs_5g:
        pa = positions.get(p["drone_a"])
        pb = positions.get(p["drone_b"])
        if pa and pb:
            p["distance"] = round(math.sqrt(
                sum((pa[k] - pb[k]) ** 2 for k in ["x", "y", "z"])
            ), 2)

    tick = max(wifi_tick, tick_5g_rssi, tick_5g_lat, 0)

    # Exploration data
    exploration = read_exploration_state()
    if exploration:
        tick = max(tick, exploration.get("step", 0))

    return {
        "tick": tick,
        "timestamp": time.strftime("%H:%M:%S"),
        "positions": positions,
        "wifi": {
            "pairs": wifi_pairs,
        } if wifi_pairs else None,
        "fiveg": {
            "drones": drones_5g,
            "pairs": pairs_5g,
            "gnb": {"x": GNB_POSITION[0], "y": GNB_POSITION[1], "z": GNB_POSITION[2]},
        } if (drones_5g or pairs_5g) else None,
        "exploration": exploration,
    }


# ─── HTTP Server ───

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves static files from dashboard/ and the /api/data endpoint."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/api/data':
            self.send_json(build_response())
        elif parsed.path == '/api/health':
            self.send_json({"status": "ok", "time": time.time()})
        else:
            # Serve static files
            super().do_GET()

    def send_json(self, data):
        body = json.dumps(data, ensure_ascii=False, default=str).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        # Quieter logging — only log API calls, not static assets
        if '/api/' in str(args[0]) if args else False:
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(description="Dashboard Server for Drone Simulation")
    parser.add_argument("--port", type=int, default=8050, help="HTTP port (default: 8050)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), DashboardHandler)

    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║   DRONE SIMULATION DASHBOARD                    ║")
    print("  ╠══════════════════════════════════════════════════╣")
    print(f"  ║   URL: http://localhost:{args.port}                 ║")
    print("  ║                                                  ║")
    print("  ║   Fichiers surveillés:                           ║")
    print(f"  ║   WiFi  : {WIFI_RSSI_CSV:<38} ║")
    print(f"  ║   5G    : {FIVEG_RSSI_CSV:<38} ║")
    print(f"  ║   Pos   : {POS_CSV:<38} ║")
    print(f"  ║   AIF   : {AIF_STATE_JSON:<38} ║")
    print("  ║                                                  ║")
    print("  ║   Lancez les bridges (08/09) dans un autre       ║")
    print("  ║   terminal pour voir les données en temps réel.  ║")
    print("  ║                                                  ║")
    print("  ║   Ctrl+C pour arrêter                            ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard arrêté.")
        server.shutdown()


if __name__ == "__main__":
    main()
