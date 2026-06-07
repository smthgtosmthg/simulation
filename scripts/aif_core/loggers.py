"""
Loggers — Écriture JSON pour le dashboard + log diagnostique humain.

DataLogger        : écrit /tmp/aif_state.json + /tmp/aif_history.json (atomique)
DiagnosticLogger  : log texte lisible (LiDAR par drone, table des candidats AIF…)
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ════════════════════════════════════════════════════════════════════
# DataLogger — JSON pour le dashboard
# ════════════════════════════════════════════════════════════════════


class DataLogger:
    def __init__(self, output_dir: str = "/tmp"):
        self.state_path = os.path.join(output_dir, "aif_state.json")
        self.history_path = os.path.join(output_dir, "aif_history.json")
        self._write(self.history_path, [])

    def log(self, state: Dict, history: List[Dict]) -> None:
        self._write(self.state_path, state)
        self._write(self.history_path, history)

    @staticmethod
    def _write(path: str, data: Any) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        os.replace(tmp, path)


# ════════════════════════════════════════════════════════════════════
# DiagnosticLogger — Log texte lisible (debug)
# ════════════════════════════════════════════════════════════════════


class DiagnosticLogger:
    """Trace humaine de chaque step : LiDAR, candidats AIF, belief, collisions."""

    def __init__(self, output_dir: str, cfg):
        self.cfg = cfg
        self.log_path = os.path.join(output_dir, "aif_diagnostic.log")
        with open(self.log_path, "w") as f:
            f.write("=" * 90 + "\n")
            f.write("  AIF DIAGNOSTIC LOG\n")
            f.write(f"  Started : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Drones  : {cfg.num_drones}\n")
            f.write(f"  Env     : {cfg.env_width} x {cfg.env_height} m  "
                    f"(grid {cfg.grid_width} x {cfg.grid_height}, res {cfg.grid_resolution} m)\n")
            f.write(f"  Arch    : {cfg.arch}  Planner: {cfg.planner}  "
                    f"NS-3: {cfg.ns3_mode}\n")
            f.write(f"  Weights : epistemic={cfg.w_epistemic}  pragmatic={cfg.w_pragmatic}  "
                    f"movement={cfg.w_movement}  collision={cfg.w_collision}\n")
            f.write(f"  Softmax : temp={cfg.softmax_temp}  fusion_mix={cfg.fusion_mix}\n")
            f.write(f"  Occ thr : {cfg.occ_threshold}   step_size={cfg.step_size} m\n")
            f.write(f"  LiDAR   : rays={cfg.num_rays}  range=[{cfg.lidar_min_range}, "
                    f"{cfg.lidar_max_range}] m  hz={cfg.lidar_hz}\n")
            f.write("=" * 90 + "\n\n")
        print(f"[INFO] Diagnostic log → {self.log_path}")

    def _w(self, text: str) -> None:
        with open(self.log_path, "a") as f:
            f.write(text)

    def log_step_header(self, step: int) -> None:
        self._w(f"\n{'━' * 90}\n")
        self._w(f"  STEP {step}   ({time.strftime('%H:%M:%S')})\n")
        self._w(f"{'━' * 90}\n")

    def log_lidar(self, drone_id: int, pos_x: float, pos_y: float,
                  angles: np.ndarray, ranges: np.ndarray, hits: np.ndarray) -> None:
        n_total = len(hits)
        n_hits = int(hits.sum())
        pct = n_hits / max(n_total, 1) * 100.0

        self._w(f"\n  ┌─ DRONE {drone_id}  LiDAR  pos=({pos_x:.2f}, {pos_y:.2f})\n")
        self._w(f"  │  Rays total={n_total}  hits={n_hits} ({pct:.1f}%)\n")

        if n_hits > 0:
            hr = ranges[hits]
            ha = angles[hits]
            self._w(f"  │  Hit ranges : min={hr.min():.2f} m  max={hr.max():.2f} m  "
                    f"mean={hr.mean():.2f} m\n")
            order = np.argsort(hr)[:5]
            self._w(f"  │  Closest obstacles:\n")
            for k in order:
                adeg = math.degrees(ha[k]) % 360
                self._w(f"  │    angle={adeg:6.1f}°  dist={hr[k]:.2f} m\n")
            for label, lo, hi in [("FRONT 315-45°", 315, 45),
                                   ("RIGHT 45-135°", 45, 135),
                                   ("BACK 135-225°", 135, 225),
                                   ("LEFT 225-315°", 225, 315)]:
                adeg = np.degrees(ha) % 360
                if lo > hi:
                    mask = (adeg >= lo) | (adeg < hi)
                else:
                    mask = (adeg >= lo) & (adeg < hi)
                cnt = int(mask.sum())
                if cnt > 0:
                    mn = hr[mask].min()
                    self._w(f"  │    {label:16s}  hits={cnt:3d}  closest={mn:.2f} m\n")
                else:
                    self._w(f"  │    {label:16s}  hits=  0\n")
        else:
            self._w(f"  │  ⚠  NO obstacles detected by LiDAR\n")
        self._w(f"  └{'─' * 60}\n")

    def log_candidates(self, drone_id: int, pos_x: float, pos_y: float,
                       others: List[Tuple[float, float]],
                       cand: List[Dict], selected_idx: int) -> None:
        self._w(f"\n  ┌─ DRONE {drone_id}  Action Selection  pos=({pos_x:.2f}, {pos_y:.2f})\n")
        if others:
            self._w(f"  │  Other drones:\n")
            for ox, oy in others:
                d = math.hypot(pos_x - ox, pos_y - oy)
                tag = " ⚠ DANGER" if d < 1.5 else " ⚠ CLOSE" if d < 3.0 else ""
                self._w(f"  │    ({ox:.2f}, {oy:.2f})  dist={d:.2f} m{tag}\n")
        self._w(f"  │\n")
        self._w(f"  │  {'#':>2} {'Act':<5} {'Valid':<6} {'Reason':<14} "
                f"{'nx':>6} {'ny':>6}  "
                f"{'IG':>8} {'Front':>7} {'Move':>5} {'Coll':>7} "
                f"{'G_total':>9}\n")
        self._w(f"  │  {'-' * 88}\n")
        for c in cand:
            sel = " ◄" if c["idx"] == selected_idx else ""
            self._w(f"  │  {c['idx']:2d} {c['name']:<5} "
                    f"{str(c['valid']):<6} {c['reason']:<14} "
                    f"{c['nx']:6.2f} {c['ny']:6.2f}  "
                    f"{c.get('ig', 0):8.3f} {c.get('frontier', 0):7.3f} "
                    f"{c.get('move', 0):5.2f} {c.get('coll', 0):7.3f} "
                    f"{c.get('G', 9999):9.3f}{sel}\n")
        sel_c = cand[selected_idx] if 0 <= selected_idx < len(cand) else cand[0]
        self._w(f"  │\n")
        self._w(f"  │  ➜ SELECTED: {sel_c['name']}  target=({sel_c['nx']:.2f}, {sel_c['ny']:.2f})  "
                f"G={sel_c.get('G', 0):.4f}\n")
        self._w(f"  └{'─' * 60}\n")

    def log_belief(self, drone_id: int, belief, fused=None) -> None:
        total = belief.width * belief.height
        occ = int((belief.probability >= self.cfg.occ_threshold).sum())
        free = int((belief.probability < 0.3).sum())
        unc = total - occ - free
        self._w(f"\n  ┌─ DRONE {drone_id}  Belief Grid\n")
        self._w(f"  │  Entropy  : {belief.mean_entropy():.4f}\n")
        self._w(f"  │  Coverage : {belief.exploration_ratio() * 100:.1f}%\n")
        self._w(f"  │  Cells    : occupied={occ} ({occ/total*100:.1f}%)  "
                f"free={free} ({free/total*100:.1f}%)  "
                f"uncertain={unc} ({unc/total*100:.1f}%)\n")
        if fused is not None:
            f_occ = int((fused.probability >= self.cfg.occ_threshold).sum())
            self._w(f"  │  Fused    : entropy={fused.mean_entropy():.4f}  "
                    f"coverage={fused.exploration_ratio() * 100:.1f}%  "
                    f"occ_cells={f_occ}\n")
        self._w(f"  └{'─' * 60}\n")

    def log_step_summary(self, step: int, agents, fused) -> None:
        self._w(f"\n  ── Step {step} Summary ──\n")
        for a in agents:
            arrived = ""
            if a.controller is not None:
                arrived = "  arrived=True" if a.controller.is_at_target() else "  arrived=False"
            self._w(f"    D{a.id} pos=({a.x:.2f}, {a.y:.2f})  "
                    f"action={a.last_action}  src={a.last_decision_source}  "
                    f"dist_total={a.total_dist:.2f} m{arrived}\n")
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                d = math.hypot(agents[i].x - agents[j].x, agents[i].y - agents[j].y)
                tag = " ⚠ COLLISION-RISK" if d < 1.5 else " ⚠ CLOSE" if d < 3.0 else ""
                self._w(f"    D{agents[i].id}↔D{agents[j].id} = {d:.2f} m{tag}\n")
        for a in agents:
            warns = []
            if a.x < 1.0:
                warns.append(f"LEFT wall x={a.x:.2f}")
            if a.x > self.cfg.env_width - 1.0:
                warns.append(f"RIGHT wall x={a.x:.2f}")
            if a.y < 1.0:
                warns.append(f"BOTTOM wall y={a.y:.2f}")
            if a.y > self.cfg.env_height - 1.0:
                warns.append(f"TOP wall y={a.y:.2f}")
            if warns:
                self._w(f"    ⚠ D{a.id} WALL: {', '.join(warns)}\n")
        cells = {}
        for a in agents:
            gx, gy = fused.world_to_grid(a.x, a.y)
            key = (gx, gy)
            cells.setdefault(key, []).append(a.id)
        for key, ids in cells.items():
            if len(ids) > 1:
                self._w(f"    ⚠ OVERLAP: drones {ids} in same grid cell {key}\n")
        self._w(f"    Fused coverage={fused.exploration_ratio() * 100:.1f}%  "
                f"entropy={fused.mean_entropy():.4f}\n")
