"""Test 4 — Combien de pas par seconde tient le socle Pegasus + SITL ?

La physique ArduPilot tourne à 1/800 s : le temps réel correspond à 800 pas par seconde. On
mesure trois régimes sur la scène complète (3 drones, caméras créées) :
  - pas purs, sans rendu — le rythme du vol et des commandes ;
  - rendu à ~5 images/s — le régime de mission (lecture des QR) ;
  - rendu à chaque pas — le plafond, pour référence.

  python run.py            # les trois régimes, un seul lancement d'Isaac
  python run.py --plot     # le tableau final
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
OUT = HERE / "mesures.json"

sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=1600)
parser.add_argument("--plot", action="store_true")
args, _ = parser.parse_known_args()


def plot() -> int:
    rows = json.loads(OUT.read_text())
    print(f"{'régime':<28} {'pas/s':>8} {'x temps réel':>13} {'mission 10 min':>15}")
    for r in rows:
        rt = r["steps_per_s"] * r["phys_dt"]
        print(f"{r['regime']:<28} {r['steps_per_s']:>8.0f} {rt:>12.2f}x {600/max(rt,1e-9)/60:>12.1f} min")
    return 0


if args.plot:
    raise SystemExit(plot())

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {"headless": True, "extra_args": ["--/rtx/verifyDriverVersion/enabled=false"]}
)

import traceback  # noqa: E402

import omni.timeline  # noqa: E402

from swarm_qr.env import scene as scene_mod  # noqa: E402
from swarm_qr.env.layout import make_layout  # noqa: E402
from swarm_qr.env.pilot import PHYS_DT  # noqa: E402

RENDER_EVERY_5HZ = max(1, round(1.0 / (5.0 * PHYS_DT)))


def bench(world, steps: int, render_every: int | None) -> float:
    t0 = time.perf_counter()
    for i in range(steps):
        render = render_every is not None and i % render_every == 0
        world.step(render=render)
    return steps / (time.perf_counter() - t0)


def main() -> None:
    scene = scene_mod.build(make_layout(7), with_sitl=False)
    scene.world.reset()
    scene.finalize()
    omni.timeline.get_timeline_interface().play()

    for _ in range(30):
        scene.world.step(render=True)

    regimes = [
        ("sans rendu", None),
        ("rendu 5 images/s", RENDER_EVERY_5HZ),
        ("rendu chaque pas", 1),
    ]
    rows = []
    for name, every in regimes:
        n = args.steps if every != 1 else max(200, args.steps // 8)
        sps = bench(scene.world, n, every)
        rows.append({"regime": name, "steps_per_s": round(sps, 1), "phys_dt": PHYS_DT})
        print(f"[RESULTAT] {name:<20} : {sps:7.1f} pas/s ({sps * PHYS_DT:.2f}x le temps reel)")

    OUT.write_text(json.dumps(rows, indent=2))


try:
    main()
except Exception:
    traceback.print_exc()
finally:
    simulation_app.close()
