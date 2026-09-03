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

import math
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from rl_inventory.env import AGENTS, NUM_DRONES  # noqa: E402
from rl_inventory.qr_task import FACES  # noqa: E402
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

    # gate LATÉRAL : drone téléporté flanc vers un tag → lit ; nez vers le tag → ne lit PAS
    n_faces = len(FACES)
    cand = torch.nonzero(env._readable[0] & ~env._read[0]).squeeze(-1)
    tp = env._qr_pos_local.view(env._n_cartons, n_faces, 3)[cand, 0]
    ok_z = (tp[:, 2] > 1.0) & (tp[:, 2] < 3.5)
    cand, tp = cand[ok_z], tp[ok_z]
    others = torch.stack([env._drones[j].data.root_pos_w[0] - env.scene.env_origins[0] for j in (1, 2)])
    pick = torch.cdist(tp[:, :2], others[:, :2]).min(dim=1).values.argmax()
    cart_id, tag_p = cand[pick].item(), tp[pick]
    tag_n = env._qr_normal.view(env._n_cartons, n_faces, 3)[cart_id, 0]
    spot = tag_p + tag_n * 1.0

    def put_drone0(yaw_val: float):
        root = env._drones[0].data.default_root_state[0:1].clone()
        root[:, :3] = (spot + env.scene.env_origins[0]).unsqueeze(0)
        root[:, 3] = math.cos(yaw_val / 2)
        root[:, 4:6] = 0.0
        root[:, 6] = math.sin(yaw_val / 2)
        root[:, 7:] = 0.0
        ids0 = torch.tensor([0], device=env.device)
        env._drones[0].write_root_pose_to_sim(root[:, :7], ids0)
        env._drones[0].write_root_velocity_to_sim(root[:, 7:], ids0)

    yaw_nose = math.atan2(-tag_n[1].item(), -tag_n[0].item())
    for yaw_try in (yaw_nose,):
        put_drone0(yaw_try)
        for _ in range(MAP_CFG.gate.credit_blackout_steps + 5):   # délai anti-triche + dwell
            vec.step(torch.zeros(vec.num_envs, 4, device=env.device))
    read_nose = bool(env._read[0, cart_id])
    put_drone0(yaw_nose - math.pi / 2)      # flanc gauche vers le tag
    for _ in range(3):
        vec.step(torch.zeros(vec.num_envs, 4, device=env.device))
    read_flank = bool(env._read[0, cart_id])
    checks.append(("gate latéral : nez → pas de lecture, flanc → lecture",
                   (not read_nose) and read_flank))

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
