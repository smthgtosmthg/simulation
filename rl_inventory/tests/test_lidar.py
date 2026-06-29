"""Test T1.3 — le LiDAR (MultiMeshRayCaster) voit-il le vrai entrepôt ?

Construit l'arène, place le drone, lit le LiDAR 360×5 et affiche des statistiques
de distances : si le LiDAR voit les étagères/murs, la distance min est petite et
une bonne partie des rayons touchent quelque chose.

Lancement (venv Isaac Sim 5.1) :
  source ~/isaac5_env/bin/activate
  python ~/simulation_mc02/rl_inventory/test_lidar.py --headless --num_envs 2 \
      --kit_args="--/rtx/verifyDriverVersion/enabled=false"
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test T1.3 — LiDAR MultiMeshRayCaster")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

simulation_app = AppLauncher(args).app

import torch  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from rl_inventory.config_rl import CFG  # noqa: E402
from rl_inventory.env import QRInventoryEnv, QRInventoryEnvCfg  # noqa: E402


def main():
    cfg = QRInventoryEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = QRInventoryEnv(cfg)
    env.reset()

    act = torch.zeros((env.num_envs, 4), device=env.device)
    obs = None
    for _ in range(args.steps):
        obs, *_ = env.step(act)

    R = CFG.lidar.num_rays
    md = CFG.lidar.max_distance_m
    lidar = obs["policy"][:, :R] * md                      # (N, R) en mètres
    grid = lidar.reshape(env.num_envs, CFG.lidar.num_azimuth, CFG.lidar.num_channels)
    hit = lidar < (md - 0.01)

    print("\n=============== TEST T1.3 — LiDAR ===============")
    print(f"num_envs={env.num_envs} | rayons/drone={R} (grille {CFG.lidar.num_azimuth}×{CFG.lidar.num_channels})")
    print(f"forme grille LiDAR : {tuple(grid.shape)}")
    print(f"distance min  = {lidar.min().item():.2f} m   (doit être PETIT : étagère/mur proche)")
    print(f"distance moy  = {lidar.mean().item():.2f} m")
    print(f"distance max  = {lidar.max().item():.2f} m   (plafond = {md} m)")
    pct = 100.0 * hit.float().mean().item()
    print(f"rayons qui touchent qqch (<{md} m) : {pct:.1f} %")
    ok = (lidar.min().item() < md - 0.5) and (pct > 5.0)
    print("RÉSULTAT :", "OK — le LiDAR voit le vrai entrepôt ✅" if ok else "À VÉRIFIER — le LiDAR ne voit que du max ❓")
    print("=================================================\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
