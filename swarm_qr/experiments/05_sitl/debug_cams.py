"""Débogage caméras natives : pourquoi la vue de dessus est-elle blanche ?

Capture la vue de dessus à 40 puis 240 pas, relit la pose réelle de la caméra, et capture la
caméra latérale du drone 0. Écrit tout dans debug_cams/.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
OUT = HERE / "debug_cams"

sys.stdout.reconfigure(line_buffering=True)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {"headless": True, "extra_args": ["--/rtx/verifyDriverVersion/enabled=false"]}
)

import traceback  # noqa: E402

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402

from swarm_qr.env import scene as scene_mod  # noqa: E402
from swarm_qr.env.layout import make_layout  # noqa: E402
from swarm_qr.experiments import _img, _viz  # noqa: E402


def dump(cam, name):
    rgb = cam.get_rgb()
    arr = np.asarray(rgb)
    print(f"[DBG] {name}: shape={arr.shape} dtype={arr.dtype} "
          f"min={arr.min() if arr.size else '-'} max={arr.max() if arr.size else '-'} "
          f"mean={arr.mean():.1f}" if arr.size else f"[DBG] {name}: VIDE")
    if arr.size:
        _img.save(_img.to_bgr(arr), OUT / f"{name}.jpg")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    layout = make_layout(7)
    scene = scene_mod.build(layout, with_sitl=False)
    overview = _viz.overview_camera()
    _viz.hide_roof(omni.usd.get_context().get_stage())

    scene.world.reset()
    scene.finalize()
    _viz.overview_init(overview)
    omni.timeline.get_timeline_interface().play()

    p, q = overview.get_world_pose(camera_axes="usd")
    print(f"[DBG] overview pose monde : pos={np.round(p,2)} quat(usd)={np.round(q,3)}")

    for _ in range(40):
        scene.world.step(render=True)
    dump(overview, "overview_40pas")
    dump(scene.cameras[0]["left"], "drone0_gauche_40pas")

    for _ in range(200):
        scene.world.step(render=True)
    dump(overview, "overview_240pas")
    dump(scene.cameras[0]["left"], "drone0_gauche_240pas")
    dump(scene.cameras[0]["front"], "drone0_face_240pas")

    p0 = scene.positions()[0]
    print(f"[DBG] drone 0 en {np.round(p0, 2)}")
    print("[DBG] FINI")


try:
    main()
except Exception:
    traceback.print_exc()
finally:
    simulation_app.close()
