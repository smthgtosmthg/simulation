"""Le drone sait-il où il est ?

Le test 3 a laissé un doute sérieux : la taille du QR dans l'image indiquait une distance
d'environ deux tiers de celle que le pilote croyait avoir. Ce test tranche, en comparant trois
estimations indépendantes de la même distance à chaque point de mesure :

  - ce que croit le pilote, c'est-à-dire la position rapportée par le drone ;
  - la pose de la caméra lue dans le simulateur ;
  - la distance optique, déduite de la taille du code dans l'image.

La distance optique fait référence : une caméra fixe placée à des distances connues la retrouve
à 1 % près.

  DISPLAY=:1 run.py --seed 7
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
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {"headless": True, "extra_args": ["--/rtx/verifyDriverVersion/enabled=false"]}
)

import traceback  # noqa: E402

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import zxingcpp  # noqa: E402

from swarm_qr.env import scene as scene_mod  # noqa: E402
from swarm_qr.env.layout import make_layout  # noqa: E402
from swarm_qr.env.pilot import Pilot  # noqa: E402
from swarm_qr.experiments import _img  # noqa: E402

DISTANCES = (1.0, 2.0, 3.0)
FLY_ALT = 1.6
CAM_BELOW = 0.11
CODE_FRACTION = 21.0 / 25.0   # le code occupe 21 modules sur les 25 du panneau


def capture(scene):
    return _img.to_bgr(scene.capture("left", 0))


def cote_du_code(bgr, texte_attendu):
    """Côté moyen du code voulu, en pixels, ou None s'il n'est pas lu."""
    for res in zxingcpp.read_barcodes(bgr[:, :, ::-1]):
        if res.text != texte_attendu:
            continue
        p = res.position
        q = np.array([[p.top_left.x, p.top_left.y], [p.top_right.x, p.top_right.y],
                      [p.bottom_right.x, p.bottom_right.y], [p.bottom_left.x, p.bottom_left.y]], float)
        return float(np.mean([np.linalg.norm(q[(k + 1) % 4] - q[k]) for k in range(4)]))
    return None


def main() -> None:
    scene = scene_mod.build(make_layout(args.seed), with_sitl=True, n_drones=1)
    scene.world.reset()
    scene.finalize()
    omni.timeline.get_timeline_interface().play()

    cam = scene.cameras[0]["left"]
    K = np.asarray(cam.get_intrinsics_matrix())
    fx = float(K[0, 0])

    def get_pos():
        return scene.drones[0].state.position

    def get_yaw():
        from scipy.spatial.transform import Rotation
        return float(Rotation.from_quat(scene.drones[0].state.attitude).as_euler("ZYX")[0])

    pilot = Pilot(scene.world, 0)
    if not pilot.ready(FLY_ALT, lambda: float(get_pos()[2])):
        print("ECHEC : pas de decollage")
        return

    tags = [t for t in scene.tags if t.normal[0] > 0.9 and 1.0 < t.position[2] < 3.0]
    tag = sorted(tags, key=lambda t: abs(t.position[2] - FLY_ALT))[0]
    taille_code = tag.size * CODE_FRACTION
    yaw = math.atan2(tag.normal[1], tag.normal[0]) + math.pi / 2.0
    n = np.array(tag.normal)
    print(f"cible {tag.tag_id} en {np.round(tag.position, 3)}, panneau {tag.size:.3f} m, "
          f"code {taille_code:.3f} m, fx {fx:.1f}")

    lignes = []
    print(f"\n{'vise':>6} {'pilote':>8} {'camera':>8} {'optique':>8}   {'ecart pilote':>13}")
    for d in DISTANCES:
        cible = np.array(tag.position) + n * d
        cible[2] = tag.position[2] + CAM_BELOW
        pilot.goto(cible, yaw, get_pos, tol=0.15, get_yaw=get_yaw)
        pilot.pump(1.0)
        img = capture(scene)
        _img.save(img, HERE / f"vue_{d:.1f}m.jpg")

        p_pilote = np.array(get_pos(), dtype=float)
        p_cam = np.asarray(cam.get_world_pose()[0], dtype=float)
        d_pilote = float(np.linalg.norm((p_pilote - np.array(tag.position))[:2]))
        d_cam = float(np.linalg.norm((p_cam - np.array(tag.position))[:2]))
        cote = cote_du_code(img, tag.tag_id)
        d_opt = fx * taille_code / cote if cote else float("nan")

        lignes.append({
            "vise_m": d, "pilote_m": round(d_pilote, 3), "camera_m": round(d_cam, 3),
            "optique_m": None if cote is None else round(d_opt, 3),
            "cote_px": None if cote is None else round(cote, 1),
            "pos_pilote": [round(v, 3) for v in p_pilote],
            "pos_camera": [round(v, 3) for v in p_cam],
        })
        opt = f"{d_opt:8.2f}" if cote else "   non lu"
        ecart = f"{100*(d_pilote-d_opt)/d_opt:+12.0f}%" if cote else "            -"
        print(f"{d:>5.1f}m {d_pilote:>7.2f}m {d_cam:>7.2f}m {opt}m {ecart}")

    pilot.velocity(0.0, 0.0, 0.0)
    pilot.pump(0.5)
    (HERE / "resultat.json").write_text(json.dumps(
        {"tag": tag.tag_id, "panneau_m": round(tag.size, 3), "code_m": round(taille_code, 3),
         "fx": round(fx, 1), "points": lignes}, indent=2))
    print("\nFINI")


try:
    main()
except Exception:
    traceback.print_exc()
finally:
    simulation_app.close()
