"""Entraîne l'essaim QR multi-drone (skrl). IPPO par défaut, MAPPO en option.

  bash launch.sh train_ppo.py --headless --algorithm IPPO --num_envs 16
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Entraîne l'essaim QR (skrl IPPO/MAPPO).")
parser.add_argument("--task", type=str, default="Isaac-QR-Inventory-Swarm-Direct-v0")
parser.add_argument("--algorithm", type=str, default="IPPO", choices=["IPPO", "MAPPO"])
parser.add_argument("--num_envs", type=int, default=None, help="nb d'entrepôts en parallèle")
parser.add_argument("--timesteps", type=int, default=None, help="override durée d'entraînement")
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import os
import sys

import gymnasium as gym
from skrl.utils.runner.torch import Runner

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import rl_inventory  # noqa: F401  enregistre la tâche gym
from rl_inventory.env import SwarmQREnvCfg  # noqa: E402


def main():
    entry = {"IPPO": "skrl_ippo_cfg_entry_point", "MAPPO": "skrl_mappo_cfg_entry_point"}[args.algorithm]
    agent_cfg = load_cfg_from_registry(args.task, entry)
    if args.timesteps is not None:
        agent_cfg["trainer"]["timesteps"] = args.timesteps
    if args.seed is not None:
        agent_cfg["seed"] = args.seed

    env_cfg = SwarmQREnvCfg()
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs

    env = gym.make(args.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    runner = Runner(env, agent_cfg)
    print(f"\n>>> Entraînement {args.algorithm} | envs={env_cfg.scene.num_envs} | "
          f"timesteps={agent_cfg['trainer']['timesteps']}\n")
    runner.run()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
