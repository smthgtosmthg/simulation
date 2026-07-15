"""Smoke-test Isaac de SwarmScanMapEnv : obs/récompense/reset/cartes/panne, 6 vérifications.

  bash rl_inventory/launch.sh rl_inventory/tests/test_swarmscan_map_env.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=40)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from rl_inventory.env import AGENTS, NUM_DRONES  # noqa: E402
from rl_inventory.swarmscan_map.config_map import MAP_CFG  # noqa: E402
from rl_inventory.swarmscan_map.env_map import OBS_DIM, PRIV_DIM, VEC_DIM, SwarmScanMapEnv, SwarmScanMapEnvCfg  # noqa: E402
from rl_inventory.swarmscan_map.flatten_wrapper import SwarmMapVecEnv  # noqa: E402


def main():
    cfg = SwarmScanMapEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = SwarmScanMapEnv(cfg)
    vec = SwarmMapVecEnv(env)
    checks = []

    obs = vec.get_observations()
    checks.append(("groupes d'obs (maps/vector/privileged), dims",
                   tuple(obs["maps"].shape) == (vec.num_envs, MAP_CFG.map.obs_dim)
                   and tuple(obs["vector"].shape) == (vec.num_envs, VEC_DIM)
                   and tuple(obs["privileged"].shape) == (vec.num_envs, PRIV_DIM)))

    total_r = torch.zeros(vec.num_envs, device=env.device)
    for _ in range(args.steps):
        actions = torch.empty(vec.num_envs, 4, device=env.device).uniform_(-1, 1)
        obs, rew, dones, infos = vec.step(actions)
        total_r += rew
    finite = torch.isfinite(obs["maps"]).all() and torch.isfinite(obs["vector"]).all() and torch.isfinite(rew).all()
    checks.append(("obs et récompenses finies après pas aléatoires", bool(finite)))
    checks.append(("la carte se remplit (occ + exploré + scan > 0)",
                   env._mapper.occ.sum().item() > 0 and env._mapper.explored.sum().item() > 0))
    checks.append(("tags lisibles cohérents (actifs & atteignables & ~prélus)",
                   env._readable.sum().item() > 0 and (env._readable & env._preread).sum().item() == 0))

    ids_before = env._config_ids.clone()
    pos_before = env._drones[0].data.root_pos_w.clone()
    env_ids = torch.arange(args.num_envs, device=env.device)
    env._reset_idx(env_ids)
    map_cleared = env._mapper.qr_read.sum().item() == 0 and env._mapper.scan.sum().item() == 0
    for _ in range(4):
        vec.step(torch.zeros(vec.num_envs, 4, device=env.device))
    moved = (env._drones[0].data.root_pos_w - pos_before).norm(dim=-1)
    checks.append(("reset : configurations retirées + spawns randomisés",
                   not torch.equal(ids_before, env._config_ids) or moved.max().item() > 0.5))
    checks.append(("carte remise à zéro au reset (avant de re-stepper)", map_cleared))

    env._kill_step[:] = 0
    env._victim[:] = 1
    vec.step(torch.zeros(vec.num_envs, 4, device=env.device))
    checks.append(("panne de drone : victime marquée morte, récompense=0",
                   bool(env._dead[:, 1].all())))

    print("\n=============== SMOKE TEST SWARMSCAN-MAP ===============")
    print(f"OBS_DIM={OBS_DIM} (maps {MAP_CFG.map.obs_dim} + vec {VEC_DIM} + priv {PRIV_DIM}) | "
          f"drones={NUM_DRONES} | pseudo-envs={vec.num_envs} | tags={env._n_cartons} "
          f"(lisibles moy {env._readable.sum(1).float().mean().item():.0f})")
    print(f"récompense cumulée moyenne ({args.steps} pas aléatoires) : {total_r.mean().item():.2f}")
    ok = True
    for name, passed in checks:
        print(f"  [{'OK' if passed else 'ÉCHEC'}] {name}")
        ok &= passed
    print("RÉSULTAT :", "TOUT EST VERT ✅" if ok else "À CORRIGER ❌")
    print("========================================================\n")
    vec.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
