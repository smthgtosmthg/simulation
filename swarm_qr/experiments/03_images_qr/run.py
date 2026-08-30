"""Test 3 — Les QR sont-ils nets ? Et le drone vole-t-il vraiment ?

Deux sorties. Une planche : la vue de la caméra latérale devant un QR, à plusieurs distances.
Une vidéo : le drone longe un rack, avec sa vue caméra et la vue de dessus côte à côte.

  python run.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--speed", type=float, default=0.5)
parser.add_argument("--standoff", type=float, default=1.2)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import traceback  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402

from swarm_qr.env import scene as scene_mod  # noqa: E402
from swarm_qr.env.config import CAMERAS, SIM_DT, RenderClock  # noqa: E402
from swarm_qr.env.layout import make_layout  # noqa: E402
from swarm_qr.experiments import _viz  # noqa: E402

DISTANCES = (0.5, 0.8, 1.1, 1.5, 2.0, 3.0)
YAW_FOR_LEFT_CAM = 90.0


def _quat_yaw(deg: float) -> list[float]:
    h = math.radians(deg) / 2.0
    return [math.cos(h), 0.0, 0.0, math.sin(h)]


def _pick_tag(tags):
    """Un QR tourné vers +X, à hauteur de vol confortable."""
    good = [t for t in tags if t.normal[0] > 0.9 and 1.0 < t.position[2] < 3.2]
    if not good:
        good = [t for t in tags if t.normal[0] > 0.9] or list(tags)
    return sorted(good, key=lambda t: abs(t.position[2] - 1.6))[0]


def _park(scene, pos, yaw_deg):
    state = scene.drones.data.default_root_state.clone()
    state[:, :] = 0.0
    state[0, 0:3] = torch.tensor(pos, device=state.device)
    state[0, 3:7] = torch.tensor(_quat_yaw(yaw_deg), device=state.device)
    for k in (1, 2):
        state[k, 0:3] = torch.tensor([pos[0], pos[1] - 6.0 * k, 0.4], device=state.device)
        state[k, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=state.device)
    scene.drones.write_root_state_to_sim(state)


def _decode(bgr) -> tuple[bool, str]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    det = cv2.QRCodeDetectorAruco() if hasattr(cv2, "QRCodeDetectorAruco") else cv2.QRCodeDetector()
    for img in (gray, cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)):
        try:
            ok, texts, _, _ = det.detectAndDecodeMulti(img)
        except cv2.error:
            continue
        if ok:
            found = [t for t in texts if t]
            if found:
                return True, found[0]
    return False, ""


def main() -> None:
    layout = make_layout(args.seed)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=SIM_DT, device="cuda:0"))
    scene = scene_mod.build(layout)
    overview = _viz.overview_camera()
    stage = omni.usd.get_context().get_stage()
    _viz.hide_roof(stage)
    clock = RenderClock(CAMERAS.update_hz)
    sim.reset()
    scene_mod.place_drones(scene)

    for _ in range(20):
        clock.step(sim)
        scene.update(SIM_DT)
        overview.update(SIM_DT)

    tag = _pick_tag(scene.tags)
    print(f"QR vise : {tag.tag_id} a {tuple(round(c,2) for c in tag.position)} "
          f"normale {tuple(round(c,2) for c in tag.normal)} taille {tag.size:.2f} m")

    # --- planche : le meme QR vu de plusieurs distances ---
    tiles, rows = [], []
    settle = max(3, int(round(1.0 / (CAMERAS.update_hz * SIM_DT))))
    for d in DISTANCES:
        pos = (tag.position[0] + d, tag.position[1], tag.position[2])
        _park(scene, pos, YAW_FOR_LEFT_CAM)
        for _ in range(settle):
            clock.step(sim)
            scene.update(SIM_DT)
        img = _viz.to_bgr(scene.rgb("left")[0])
        ok, text = _decode(img)
        rows.append({"distance": d, "decode": ok, "text": text})
        tiles.append(_viz.label(img, f"{d:.1f} m - {'DECODE ' + text if ok else 'non decode'}"))
        print(f"  {d:.1f} m : {'decode ' + text if ok else 'non decode'}")

    _viz.board(tiles, HERE / "planche_qr.jpg", cols=3, cell=460)

    # --- video : le drone longe le rack ---
    rack = min(layout.racks, key=lambda r: abs(r.x - (tag.position[0] - 0.7)))
    y0, y1 = rack.y_bounds
    start = (tag.position[0] + args.standoff, y0 - 1.0, 1.6)
    _park(scene, start, YAW_FOR_LEFT_CAM)
    for _ in range(10):
        clock.step(sim)
        scene.update(SIM_DT)

    vel = torch.zeros((3, 6), device=scene.drones.data.root_pos_w.device)
    vel[0, 1] = args.speed
    frames, decoded = [], set()
    steps = int((y1 - y0 + 2.0) / args.speed / SIM_DT)
    every = max(1, int(round(1.0 / (CAMERAS.update_hz * SIM_DT))))

    for i in range(steps):
        scene.command_velocity(vel)
        clock.step(sim)
        scene.update(SIM_DT)
        if i % every:
            continue
        overview.update(SIM_DT)
        cam = _viz.to_bgr(scene.rgb("left")[0])
        ok, text = _decode(cam)
        if ok:
            decoded.add(text)
        p = scene.positions()[0].tolist()
        cam = _viz.label(cam, f"camera gauche - {len(decoded)} QR lus")
        if ok:
            cam = _viz.label(cam, f"lu : {text}", bottom=True)
        top = _viz.label(_viz.to_bgr(overview.data.output["rgb"]),
                         f"vue de dessus - drone en x={p[0]:.1f} y={p[1]:.1f}")
        frames.append(cv2.resize(_viz.side_by_side(cam, top), (1280, 480)))

    scene.command_velocity(torch.zeros_like(vel))
    path = _viz.video(frames, HERE / "vol_le_long_du_rack.mp4", fps=12)
    print(f"video : {path.name}, {len(frames)} images, {len(decoded)} QR lus pendant le vol")

    decoded_ok = [r for r in rows if r["decode"]]
    (HERE / "resultat.json").write_text(
        json.dumps(
            {
                "tag": tag.tag_id,
                "tag_size_m": round(tag.size, 3),
                "distances": rows,
                "portee_max_m": max([r["distance"] for r in decoded_ok], default=0.0),
                "qr_lus_en_vol": len(decoded),
                "images_video": len(frames),
            },
            indent=2,
        )
    )


try:
    main()
except Exception:
    traceback.print_exc()
finally:
    simulation_app.close()
