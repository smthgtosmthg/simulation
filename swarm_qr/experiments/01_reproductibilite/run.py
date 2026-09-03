"""Test 1 — La même graine donne-t-elle exactement le même entrepôt ?

On construit deux fois l'entrepôt de la graine 7, dans deux processus séparés, et on compare
les vues de dessus pixel par pixel.

  python run.py --seed 7 --pass A
  python run.py --seed 7 --pass B
  python run.py --compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--pass", dest="run_pass", choices=["A", "B"], default="A")
parser.add_argument("--compare", action="store_true")


def compare() -> int:
    import cv2
    import numpy as np

    a = cv2.imread(str(HERE / "vue_A.jpg"))
    b = cv2.imread(str(HERE / "vue_B.jpg"))
    if a is None or b is None:
        print("il manque une des deux vues")
        return 1

    diff = cv2.absdiff(a, b).max(axis=2)
    mean_diff = float(diff.mean())
    structural = float((diff > 40).mean())
    layout_a = json.loads((HERE / "layout_A.json").read_text())
    layout_b = json.loads((HERE / "layout_B.json").read_text())
    same_layout = layout_a == layout_b

    from swarm_qr.experiments import _img

    heat = cv2.applyColorMap(cv2.convertScaleAbs(diff, alpha=8), cv2.COLORMAP_INFERNO)
    panel = np.hstack(
        [
            _img.label(a, "passage A"),
            np.full((a.shape[0], 8, 3), 40, np.uint8),
            _img.label(b, "passage B"),
            np.full((a.shape[0], 8, 3), 40, np.uint8),
            _img.label(heat, f"ecart moyen {mean_diff:.2f}/255 - structurel {structural:.3%}"),
        ]
    )
    cv2.imwrite(str(HERE / "comparaison.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # La preuve qui fait foi est la disposition ecrite en JSON. L'image confirme qu'aucune
    # geometrie n'a bouge de facon visible. L'ecart moyen est du bruit de rendu et de
    # stabilisation physique : on le rapporte, on ne juge pas dessus.
    ok = same_layout and structural < 0.001
    print(f"disposition identique  : {same_layout}   <- la preuve")
    print(f"ecart structurel       : {structural:.4%}  <- geometrie deplacee (seuil 0.1%)")
    print(f"ecart moyen des pixels : {mean_diff:.2f} / 255  (bruit de rendu, informatif)")
    print("VERDICT :", "REPRODUCTIBLE" if ok else "NON REPRODUCTIBLE")
    (HERE / "resultat.json").write_text(
        json.dumps(
            {"same_layout": same_layout, "mean_diff": mean_diff, "structural": structural, "ok": ok},
            indent=2,
        )
    )
    return 0 if ok else 2


args_pre, _ = parser.parse_known_args()
if args_pre.compare:
    raise SystemExit(compare())

args, _ = parser.parse_known_args()

import sys as _sys  # noqa: E402

_sys.stdout.reconfigure(line_buffering=True)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {"headless": True, "extra_args": ["--/rtx/verifyDriverVersion/enabled=false"]}
)

import traceback  # noqa: E402

import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402

from swarm_qr.env import scene as scene_mod  # noqa: E402
from swarm_qr.env.layout import make_layout  # noqa: E402
from swarm_qr.experiments import _viz  # noqa: E402


def main() -> None:
    layout = make_layout(args.seed)
    scene = scene_mod.build(layout, with_sitl=False)
    overview = _viz.overview_camera()
    _viz.hide_roof(omni.usd.get_context().get_stage())

    scene.world.reset()
    scene.finalize()
    _viz.overview_init(overview)
    omni.timeline.get_timeline_interface().play()

    for _ in range(40):
        scene.world.step(render=True)

    img = _viz.to_bgr(overview.get_rgb())
    _viz.save(img, HERE / f"vue_{args.run_pass}.jpg")
    (HERE / f"layout_{args.run_pass}.json").write_text(
        json.dumps(
            {
                "seed": layout.seed,
                "racks": [[r.prim, round(r.x, 6), round(r.y_min, 6)] for r in layout.racks],
                "fill": round(layout.fill_fraction, 6),
                "boxes": len(scene.kept_boxes),
                "tags": len(scene.tags),
                "spawns": [[round(c, 6) for c in s] for s in layout.spawns],
            },
            indent=2,
        )
    )
    print(f"passage {args.run_pass} : {len(scene.kept_boxes)} cartons, {len(scene.tags)} QR")


try:
    main()
except Exception:
    traceback.print_exc()
finally:
    simulation_app.close()
