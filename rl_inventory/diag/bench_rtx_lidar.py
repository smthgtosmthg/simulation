"""Benchmark RTX LiDAR (MESURE, ne modifie pas l'env du projet).

Charge le vrai entrepôt, crée N RTX LiDAR (capteur ray-tracé réaliste) en nombre
croissant, et mesure VRAM + vitesse (updates/s) pour chaque N → savoir combien de
drones-LiDAR réalistes tiennent sur 8 Go et si l'entraînement est viable.

Lancement :
  cd ~/IsaacLab
  ./isaaclab.sh -p ~/simulation_mc02/rl_inventory/bench_rtx_lidar.py --headless --enable_cameras
"""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Benchmark RTX LiDAR sur 8 Go")
parser.add_argument("--counts", type=str, default="1,2,4,8,16", help="nombres de lidars à tester")
parser.add_argument("--steps", type=int, default=60, help="updates chronométrés par palier")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import os  # noqa: E402

import torch  # noqa: E402
import omni.usd  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

# l'extension RTX LiDAR doit être activée pour enregistrer la commande de création
enable_extension("isaacsim.sensors.rtx")


def vram_used_gb():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9, total / 1e9


USD = os.getenv(
    "AIF_FACTORY_USD",
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/4.2/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
)

print(f"\n[bench] Chargement de l'entrepôt : {USD}")
omni.usd.get_context().open_stage(USD)
for _ in range(30):
    simulation_app.update()

u0, total = vram_used_gb()
print(f"[bench] VRAM totale = {total:.2f} Go | utilisée après chargement entrepôt (0 lidar) = {u0:.2f} Go\n")

def make_rtx_lidar(i):
    res, prim = omni.kit.commands.execute(
        "IsaacSensorCreateRtxLidar",
        path=f"/World/RtxLidar_{i}",
        parent=None,
        config="Example_Rotary",
    )
    if not res:
        raise RuntimeError("IsaacSensorCreateRtxLidar a renvoyé False")
    rp = rep.create.render_product(prim.GetPath(), [1, 1])
    annot = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
    annot.attach([rp])
    return annot


counts = [int(c) for c in args.counts.split(",")]
annots = []
created = 0

# vérif : créer le 1er lidar (révèle tout problème d'API tout de suite)
print("=== Vérification : création d'un RTX LiDAR ===")
try:
    annots.append(make_rtx_lidar(0))
    created = 1
    for _ in range(10):
        simulation_app.update()
    print("OK — RTX LiDAR créé et rendu.\n")
except Exception as e:
    print(f"ÉCHEC création RTX LiDAR : {type(e).__name__}: {e}")
    print("\n>> L'API RTX LiDAR diffère sur cette version. Colle-moi cette sortie, je corrige.")
    simulation_app.close()
    raise SystemExit(0)

print(f"{'lidars':>7} | {'VRAM (Go)':>10} | {'updates/s':>10} | {'ms/update':>10}")
print("-" * 46)

results = []
for target in counts:
    try:
        while created < target:
            annots.append(make_rtx_lidar(created))
            created += 1
        # warmup
        for _ in range(10):
            simulation_app.update()
        t0 = time.time()
        for _ in range(args.steps):
            simulation_app.update()
        dt = time.time() - t0
        ups = args.steps / dt
        used, _ = vram_used_gb()
        results.append((target, used, ups))
        print(f"{target:>7} | {used:>10.2f} | {ups:>10.1f} | {1000 / ups:>10.1f}")
    except Exception as e:
        print(f"{target:>7} | ÉCHEC : {type(e).__name__}: {str(e)[:60]}")
        break

print("-" * 46)
print(f"VRAM totale GPU : {total:.2f} Go | base (entrepôt seul) : {u0:.2f} Go")
if len(results) >= 2:
    per = (results[-1][1] - u0) / max(results[-1][0], 1)
    print(f"coût VRAM ~ {per*1000:.0f} Mo / lidar | vitesse à {results[-1][0]} lidars : {results[-1][2]:.1f} updates/s")
print("\n[bench] Terminé.\n")

simulation_app.close()
