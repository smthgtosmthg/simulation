"""Étalonnage des caméras fixes : on place des caméras à des orientations connues, on rend, on
regarde. Les premières images en vol montraient le sol et le mur derrière chaque caméra ; la
convention d'orientation de la classe Camera se vérifie ici, sans drone ni autopilote.

    cameras_test.py --sortie /tmp/cams
"""
import argparse
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=9033)
ap.add_argument("--sortie", required=True)
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True, "extra_args": ["--/rtx/verifyDriverVersion/enabled=false"]})

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from swarm_qr.env import scene as scene_mod  # noqa: E402
from swarm_qr.env.layout import make_layout  # noqa: E402
from swarm_qr.experiments import _img  # noqa: E402

SORTIE = Path(args.sortie)
SORTIE.mkdir(parents=True, exist_ok=True)
scene = scene_mod.build(make_layout(args.seed), with_sitl=False, n_drones=1)
scene.world.reset()
scene.finalize()
omni.timeline.get_timeline_interface().play()

# Au milieu du couloir central, à 3,5 m : quatre lacets sans plongée, puis deux plongées
ESSAIS = {
    # dans la grande zone : est = face ouest du rack 2 a 4,4 m ; ouest = face est du rack 1 a 4,3 m ;
    # nord = espace libre puis le mur nord a 13 m ; sud = espace libre puis le mur sud a 16 m
    "zone_lacet_0_est": dict(position=(2.7, 4.0, 2.0), yaw=0.0, plongee=0.0, fov=80.0),
    "zone_lacet_90_nord": dict(position=(2.7, 4.0, 2.0), yaw=90.0, plongee=0.0, fov=80.0),
    "zone_lacet_180_ouest": dict(position=(2.7, 4.0, 2.0), yaw=180.0, plongee=0.0, fov=80.0),
    "zone_lacet_m90_sud": dict(position=(2.7, 4.0, 2.0), yaw=-90.0, plongee=0.0, fov=80.0),
    "zone_nord_plongee_30": dict(position=(2.7, 4.0, 5.5), yaw=90.0, plongee=30.0, fov=80.0),
    "zone_nord_plongee_m30": dict(position=(2.7, 4.0, 2.0), yaw=90.0, plongee=-30.0, fov=80.0),
    **{f"mur_{n}": c for n, c in scene_mod.CAMERAS_VIDEO.items()},
}
cams = scene_mod.cameras_fixes(ESSAIS)
for _ in range(scene_mod.RENDER_LAG + 5):
    scene.world.step(render=True)
for nom, cam in cams.items():
    img = None
    for _ in range(30):
        img = cam.get_rgb()
        if img is not None and getattr(img, "ndim", 0) == 3 and img.size:
            break
        scene.world.step(render=True)
    if img is None:
        print(f"  {nom}: pas d'image")
        continue
    bgr = _img.to_bgr(img)
    cv2.imwrite(str(SORTIE / f"{nom}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    p, q = cam.get_world_pose()
    print(f"  {nom:18s} pose lue (monde) : position {np.round(p, 2).tolist()} orientation {np.round(q, 3).tolist()}")
print("CAMERAS TEST FINI")
simulation_app.close()
