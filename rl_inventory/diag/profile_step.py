"""Chronomètre chaque partie d'un pas de l'essaim pour localiser le goulot de débit."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from rl_inventory.env import AGENTS, NUM_DRONES, SwarmQREnv, SwarmQREnvCfg  # noqa: E402


def timeit(fn, n=30):
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000.0  # ms


def main():
    cfg = SwarmQREnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = SwarmQREnv(cfg)

    def act():
        return {a: torch.empty((env.num_envs, 4), device=env.device).uniform_(-1.0, 1.0) for a in AGENTS}

    for _ in range(20):  # warmup
        env.step(act())

    dt = env.cfg.sim.dt
    decim = env.cfg.decimation

    t_step = timeit(lambda: env.step(act()))
    t_phys = timeit(lambda: env.sim.step(render=False))
    t_scene = timeit(lambda: env.scene.update(dt))

    def read_lidar():
        for k in range(NUM_DRONES):
            env._lidar_ranges(env._lidars[k])

    t_read = timeit(read_lidar)
    env._stepping = True
    t_qr = timeit(lambda: env._update_qr())

    def obs():
        env._refreshed = False
        env._get_observations()

    t_obs = timeit(obs)

    # vrai raycast forcé (le scene.update isolé est trompeur car caché)
    def force_lidar():
        for k in range(NUM_DRONES):
            env._lidars[k].update(dt, force_recompute=True)

    t_lidar_force = timeit(force_lidar, 15)
    t_apply = timeit(lambda: env._apply_action())
    t_write = timeit(lambda: env.scene.write_data_to_sim())

    print("\n===== PROFIL D'UN PAS (ms) =====")
    print(f"num_envs            : {env.num_envs}")
    print(f"env.step() complet  : {t_step:7.1f} ms   ({1000 / t_step:.2f} steps/s)")
    print(f"  sim.step (1 phys) : {t_phys:7.1f} ms   (×{decim} décimation = {t_phys * decim:.1f} ms)")
    print(f"  scene.update      : {t_scene:7.1f} ms   (caché/trompeur)")
    print(f"  RAYCAST FORCÉ ×3  : {t_lidar_force:7.1f} ms   <-- vrai coût LiDAR (×{decim} si par pas phys)")
    print(f"  _apply_action     : {t_apply:7.1f} ms")
    print(f"  write_data_to_sim : {t_write:7.1f} ms")
    print(f"  lecture LiDAR     : {t_read:7.1f} ms")
    print(f"  _update_qr        : {t_qr:7.1f} ms")
    print(f"  _get_observations : {t_obs:7.1f} ms")
    print("================================\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
