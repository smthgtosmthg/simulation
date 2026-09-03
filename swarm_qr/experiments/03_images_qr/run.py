"""Test 3 — Les QR sont-ils nets, et le drone SITL les lit-il en volant ?

Sur le socle Pegasus, plus de téléportation : le drone décolle, vole vers chaque point de
mesure, s'y stabilise, et le décodage se fait sur l'image réelle de sa caméra latérale.

Deux sorties : la planche des distances, et la vidéo du vol le long d'un rack.

  DISPLAY=:1 ~/isaac5_env/bin/python run.py --seed 7
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

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--speed", type=float, default=0.5)
parser.add_argument("--standoff", type=float, default=1.2)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {"headless": True, "extra_args": ["--/rtx/verifyDriverVersion/enabled=false"]}
)

import traceback  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402

from swarm_qr.env import scene as scene_mod  # noqa: E402
from swarm_qr.env.layout import make_layout  # noqa: E402
from swarm_qr.env.pilot import Pilot  # noqa: E402
from swarm_qr.experiments import _viz  # noqa: E402

DISTANCES = (0.5, 0.8, 1.1, 1.5, 2.0, 3.0)
FLY_ALT = 1.6
CAM_BELOW = 0.11  # la caméra est sous le corps : on vise avec elle, pas avec le corps


def _pick_tag(tags):
    """Un QR tourné vers +X, à hauteur de vol confortable."""
    good = [t for t in tags if t.normal[0] > 0.9 and 1.0 < t.position[2] < 3.0]
    if not good:
        good = [t for t in tags if t.normal[0] > 0.9] or list(tags)
    return sorted(good, key=lambda t: abs(t.position[2] - FLY_ALT))[0]


def _yaw_for_left_cam(normal) -> float:
    """Cap du drone pour que la caméra GAUCHE regarde la face du tag."""
    return math.atan2(normal[1], normal[0]) + math.pi / 2.0


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


def _full(img) -> bool:
    return img is not None and getattr(img, "ndim", 0) == 3 and img.size > 0


def _capture(scene, overview=None):
    """Rendus jusqu'à obtenir des images pleines : après une longue phase sans rendu, les
    caméras renvoient None ou un tableau vide pendant quelques rendus."""
    for _ in range(3):
        scene.world.step(render=True)
    for _ in range(120):
        cam = scene.rgb("left", 0)
        top = overview.get_rgb() if overview is not None else None
        if _full(cam) and (overview is None or _full(top)):
            return _viz.to_bgr(cam), (_viz.to_bgr(top) if top is not None else None)
        scene.world.step(render=True)
    raise RuntimeError("camera vide apres 120 rendus")


def main() -> None:
    layout = make_layout(args.seed)
    scene = scene_mod.build(layout, with_sitl=True, n_drones=1)
    overview = _viz.overview_camera()
    _viz.hide_roof(omni.usd.get_context().get_stage())

    scene.world.reset()
    scene.finalize()
    _viz.overview_init(overview)
    omni.timeline.get_timeline_interface().play()

    def get_pos():
        return scene.drones[0].state.position

    def get_z():
        return float(get_pos()[2])

    def get_yaw():
        from scipy.spatial.transform import Rotation
        return float(Rotation.from_quat(scene.drones[0].state.attitude).as_euler("ZYX")[0])

    pilot = Pilot(scene.world, 0)
    if not pilot.ready(FLY_ALT, get_z):
        print("ECHEC : le drone n'a pas decolle")
        return

    tag = _pick_tag(scene.tags)
    yaw = _yaw_for_left_cam(tag.normal)
    n = np.array(tag.normal)
    print(f"QR vise : {tag.tag_id} a {tuple(round(c, 2) for c in tag.position)} "
          f"taille {tag.size:.2f} m")

    # --- planche : voler a chaque distance, se stabiliser, decoder ---
    tiles, rows = [], []
    for d in DISTANCES:
        target = np.array(tag.position) + n * d
        target[2] = max(0.8, tag.position[2]) + CAM_BELOW
        reached = pilot.goto(target, yaw, get_pos, tol=0.15, get_yaw=get_yaw)
        pilot.pump(1.0)
        img, _ = _capture(scene)
        ok, text = _decode(img)
        err = float(np.linalg.norm(target - np.array(get_pos())))
        yaw_err = math.degrees(math.atan2(math.sin(yaw - get_yaw()), math.cos(yaw - get_yaw())))
        rows.append({"distance": d, "atteint": bool(reached), "decode": ok, "text": text,
                     "err_pos_m": round(err, 3), "err_cap_deg": round(yaw_err, 1)})
        tiles.append(_viz.label(img, f"{d:.1f} m - {'DECODE ' + text if ok else 'non decode'}"))
        print(f"  {d:.1f} m : {'decode ' + text if ok else 'non decode'}"
              f"{'' if reached else ' (point non atteint)'}"
              f" | pos {err:.2f} m, cap {yaw_err:+.0f} deg")

    _viz.board(tiles, HERE / "planche_qr.jpg", cols=3, cell=460)

    # --- video : longer le rack en lisant ---
    rack = min(layout.racks, key=lambda r: abs(r.x - (tag.position[0] - 0.7)))
    y0, y1 = rack.y_bounds
    start = np.array([tag.position[0] + args.standoff, y0 - 1.0, tag.position[2] + CAM_BELOW])
    pilot.goto(start, yaw, get_pos, tol=0.2, get_yaw=get_yaw)

    R = pilot._ned_in_world
    v_w = np.array([0.0, args.speed, 0.0])
    v_ned = R.T @ v_w
    cw = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    yaw_ned = float(np.arctan2(np.dot(cw, R[:, 1]), np.dot(cw, R[:, 0])))
    frames, decoded = [], set()
    total = (y1 - y0 + 2.0) / args.speed
    t_sim = 0.0
    while t_sim < total:
        pilot.velocity(float(v_ned[0]), float(v_ned[1]), float(v_ned[2]), yaw_rad=yaw_ned)
        pilot.pump(0.4)
        t_sim += 0.4
        cam, top = _capture(scene, overview)
        ok, text = _decode(cam)
        if ok:
            decoded.add(text)
        p = get_pos()
        cam = _viz.label(cam, f"camera gauche - {len(decoded)} QR lus")
        if ok:
            cam = _viz.label(cam, f"lu : {text}", bottom=True)
        top = _viz.label(top, f"vue de dessus - drone x={p[0]:.1f} y={p[1]:.1f}")
        frames.append(cv2.resize(_viz.side_by_side(cam, top), (1280, 480)))

    pilot.velocity(0.0, 0.0, 0.0)
    pilot.pump(1.0)
    path = _viz.video(frames, HERE / "vol_le_long_du_rack.mp4", fps=8)
    print(f"video : {path.name}, {len(frames)} images, {len(decoded)} QR lus pendant le vol")

    decoded_rows = [r for r in rows if r["decode"]]
    (HERE / "resultat.json").write_text(
        json.dumps(
            {
                "socle": "pegasus+ardupilot_sitl",
                "tag": tag.tag_id,
                "tag_size_m": round(tag.size, 3),
                "distances": rows,
                "portee_max_m": max([r["distance"] for r in decoded_rows], default=0.0),
                "qr_lus_en_vol": sorted(decoded),
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
