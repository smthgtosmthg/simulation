"""Test de plomberie T1.2.

Boote Isaac Sim, construit l'arène (entrepôt + drone), applique une vitesse
AVANT constante et vérifie que le drone se déplace bien selon la commande.

Lancement :
  cd ~/IsaacLab
  ./isaaclab.sh -p ~/simulation_mc02/rl_inventory/test_drone.py --num_envs 2 --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test T1.2 — arène + drone vitesse")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=120)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

# rendre le package rl_inventory importable (racine du repo)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from rl_inventory.env import QRInventoryEnv, QRInventoryEnvCfg  # noqa: E402


def main():
    cfg = QRInventoryEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = QRInventoryEnv(cfg)
    env.reset()

    start = env._robot.data.root_pos_w.clone()

    # commande : vx = +100 % (avant), reste à 0
    act = torch.zeros((env.num_envs, 4), device=env.device)
    act[:, 0] = 1.0
    for _ in range(args.steps):
        env.step(act)

    end = env._robot.data.root_pos_w.clone()
    disp = (end - start).mean(dim=0).cpu().numpy()

    print("\n================= TEST T1.2 =================")
    print(f"num_envs = {env.num_envs} | steps = {args.steps} | dt_ctrl = {env.step_dt*1000:.1f} ms")
    print(f"déplacement moyen (x, y, z) = {disp} m")
    print(f"altitude finale moyenne     = {end[:, 2].mean().item():.2f} m")
    ok = abs(float(disp[0])) > 0.3
    print("RÉSULTAT :", "OK — le drone vole selon la commande" if ok else "ÉCHEC — pas de déplacement")
    print("============================================\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
