"""Test 4 — Combien de pas par seconde tient le simulateur ?

Une fréquence de caméra par exécution : Isaac ne sait pas reconstruire une scène en cours de
route. `run_all.sh` boucle sur les fréquences, puis `--plot` trace la courbe.

  python run.py --hz 5
  python run.py --plot
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
CSV = HERE / "mesures.json"

parser = argparse.ArgumentParser()
parser.add_argument("--hz", type=float, default=5.0)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--plot", action="store_true")


def plot() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(json.loads(CSV.read_text()), key=lambda r: r["hz"])
    hz = [r["hz"] for r in rows]
    sps = [r["steps_per_s"] for r in rows]
    ratio = [r["steps_per_s"] * (1.0 / 60.0) for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hz, sps, "o-", color="#0f766e")
    ax[0].set_xlabel("images demandees par seconde")
    ax[0].set_ylabel("pas de simulation par seconde")
    ax[0].set_title("Debit du simulateur")
    ax[0].grid(alpha=0.3)

    ax[1].axhline(1.0, color="#b45309", ls="--", lw=1, label="temps reel")
    ax[1].plot(hz, ratio, "o-", color="#0f766e")
    ax[1].set_xlabel("images demandees par seconde")
    ax[1].set_ylabel("temps simule / temps reel")
    ax[1].set_title("Vitesse par rapport au temps reel")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "courbe_debit.png", dpi=120)

    best = max(rows, key=lambda r: r["steps_per_s"])
    chosen = next((r for r in rows if abs(r["hz"] - 5.0) < 0.01), best)
    print(f"{'Hz':>6} {'pas/s':>9} {'x temps reel':>14} {'mission 5 min':>16}")
    for r in rows:
        rt = r["steps_per_s"] / 60.0
        print(f"{r['hz']:>6.0f} {r['steps_per_s']:>9.1f} {rt:>13.2f}x {300/rt/60:>13.1f} min")
    print(f"\nRetenu : {chosen['hz']:.0f} images/s -> {chosen['steps_per_s']:.1f} pas/s")
    return 0


args, _ = parser.parse_known_args()
if args.plot:
    raise SystemExit(plot())

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import traceback  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

from swarm_qr.env import config as cfg  # noqa: E402
from swarm_qr.env import scene as scene_mod  # noqa: E402
from swarm_qr.env.layout import make_layout  # noqa: E402


def main() -> None:
    object.__setattr__(cfg.CAMERAS, "update_hz", args.hz)
    layout = make_layout(7)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=cfg.SIM_DT, device="cuda:0"))
    scene = scene_mod.build(layout)
    clock = cfg.RenderClock(args.hz)
    sim.reset()
    scene_mod.place_drones(scene)

    vel = torch.zeros((cfg.DRONES.count, 6), device=scene.drones.data.root_pos_w.device)
    vel[:, 1] = 0.4

    for _ in range(30):
        scene.command_velocity(vel)
        clock.step(sim)
        scene.update(cfg.SIM_DT)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        scene.command_velocity(vel)
        clock.step(sim)
        scene.update(cfg.SIM_DT)
        _ = scene.rgb("left")
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    sps = args.steps / dt
    row = {"hz": args.hz, "steps_per_s": round(sps, 2), "images_per_s": round(sps * cfg.SIM_DT * args.hz * 9, 2)}
    rows = json.loads(CSV.read_text()) if CSV.exists() else []
    rows = [r for r in rows if abs(r["hz"] - args.hz) > 1e-6] + [row]
    CSV.write_text(json.dumps(rows, indent=2))
    print(f"[RESULTAT] {args.hz:.0f} images/s -> {sps:.1f} pas de simulation par seconde "
          f"({sps/60.0:.2f}x le temps reel)")


try:
    main()
except Exception:
    traceback.print_exc()
finally:
    simulation_app.close()
