"""T1.5b — vérifie le proxy de lecture QR (drone face à un QR, lent -> lu ; sinon non lu)."""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from rl_inventory.config_rl import CFG  # noqa: E402
from rl_inventory.env import QRInventoryEnv, QRInventoryEnvCfg  # noqa: E402


def main():
    cfg = QRInventoryEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = QRInventoryEnv(cfg)
    env.reset()

    zero = torch.zeros((env.num_envs, 4), device=env.device)
    for _ in range(6):  # charge la géométrie + calcule les poses QR
        env.step(zero)

    print(f"\nnb cartons (QR) par env : {env._n_cartons}")

    # place le drone de l'env 0 juste devant le QR 0, face à lui, immobile
    qr = env._qr_pos_local[0] + env.scene.env_origins[0]
    n = env._qr_normal[0]
    dpos = qr + n * 1.2
    yaw = math.atan2(float(-n[1]), float(-n[0]))
    quat = torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)], device=env.device)
    eid = torch.tensor([0], device=env.device)
    env._robot.write_root_pose_to_sim(torch.cat([dpos, quat]).unsqueeze(0), eid)
    env._robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), eid)

    for _ in range(CFG.qr.min_dwell_steps + 3):
        env.step(zero)

    print("\n=============== TEST T1.5b — lecture QR ===============")
    print(f"env 0 (drone placé face au QR 0, immobile) : QR lus = {int(env._read[0].sum())}  | fraction = {env._read_frac[0]:.3f}")
    print(f"env 1 (drone au départ, loin des QR)       : QR lus = {int(env._read[1].sum())}  | fraction = {env._read_frac[1]:.3f}")
    ok = int(env._read[0].sum()) >= 1 and int(env._read[1].sum()) == 0
    print("RÉSULTAT :", "OK — lit quand on vise de près/lentement, pas sinon ✅" if ok else "À VÉRIFIER")
    print("======================================================\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
