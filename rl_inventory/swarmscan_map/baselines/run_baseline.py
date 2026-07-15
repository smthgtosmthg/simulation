"""Évalue un contrôleur (pore | aif | policy) dans l'arène, au gate choisi, mêmes métriques pour tous.

  PYTHONUNBUFFERED=1 ~/isaac5_env/bin/python rl_inventory/swarmscan_map/baselines/run_baseline.py \
      --controller pore --episodes 20 --headless --kit_args="--/rtx/verifyDriverVersion/enabled=false"

Sortie : un CSV par run dans results/ + résumé (moyenne, écart-type, IQM) imprimé.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--controller", type=str, required=True, choices=["pore", "aif", "policy"])
parser.add_argument("--ckpt", type=str, default=None, help="checkpoint .pt (requis pour --controller policy)")
parser.add_argument("--episodes", type=int, default=20, help="épisodes TERMINÉS à collecter")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--level", type=int, default=12, help="niveau du gate (12 = conditions finales calibrées)")
parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import csv
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from rl_inventory.env import AGENTS
from rl_inventory.swarmscan_map.config_map import MAP_CFG
from rl_inventory.swarmscan_map.env_map import SwarmScanMapEnv, SwarmScanMapEnvCfg
from rl_inventory.swarmscan_map.baselines.pore_planner import PorePlanner
from rl_inventory.swarmscan_map.baselines.aif_controller import ActiveInferenceController

D = len(AGENTS)
MILESTONES = (0.5, 0.9, 0.95)


def make_controller(env):
    if args.controller == "pore":
        return PorePlanner(env)
    if args.controller == "aif":
        return ActiveInferenceController(env)
    from rl_inventory.swarmscan_map.flatten_wrapper import SwarmMapVecEnv
    from rl_inventory.swarmscan_map.models import MapActorCritic

    class PolicyController:
        def __init__(self, env):
            self.vec = SwarmMapVecEnv(env)
            obs = self.vec.get_observations()
            self.policy = MapActorCritic(
                obs, {"policy": ["maps", "vector"], "critic": ["maps", "vector", "privileged"]}, 4,
                map_channels=len(MAP_CFG.map.crop_spans_m) * MAP_CFG.map.n_channels,
                map_px=MAP_CFG.map.crop_px,
            ).to(env.device)
            ckpt = torch.load(args.ckpt, map_location=env.device, weights_only=False)
            self.policy.load_state_dict(ckpt["model_state_dict"])
            self.policy.eval()
            self._obs = obs

        def act(self):
            with torch.no_grad():
                flat = self.policy.act_inference(self._obs)
            return {a: flat[i * len(flat) // D:(i + 1) * len(flat) // D] for i, a in enumerate(AGENTS)}

        def refresh_obs(self, obs_dict):
            self._obs = self.vec._pack(obs_dict)

        def on_reset(self, env_ids):
            pass

    return PolicyController(env)


def main():
    cfg = SwarmScanMapEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = min(
        MAP_CFG.train.episode_length_s + MAP_CFG.curriculum.episode_s_per_notch * args.level,
        MAP_CFG.curriculum.episode_s_max,
    )
    env = SwarmScanMapEnv(cfg)
    env._split = args.split
    env._curr.level = args.level
    env._curr.cfg.min_episodes_per_notch = 10**9   # gate figé pendant toute l'éval
    env._curr.cfg.min_episodes_down = 10**9
    env._curr.cfg.spawn_near_prob = (0.0, 0.0)     # éval : spawns génériques, pas d'aide

    obs, _ = env.reset()
    zero = {a: torch.zeros(args.num_envs, 4, device=env.device) for a in AGENTS}
    obs, *_ = env.step(zero)                        # initialise QR/mapper (chargement paresseux)
    all_ids = torch.arange(args.num_envs, device=env.device)
    env._reset_idx(all_ids)                         # vrai départ : layouts du split + spawns propres
    obs = env._get_observations()

    ctl = make_controller(env)
    ctl.on_reset(all_ids)

    B = args.num_envs
    rf_max = torch.zeros(B, device=env.device)   # l'env se reset tout seul au done : on suit le max en continu
    t50 = torch.full((B,), -1.0, device=env.device)
    t90 = torch.full((B,), -1.0, device=env.device)
    t95 = torch.full((B,), -1.0, device=env.device)
    coll = torch.zeros(B, device=env.device)
    sep_min = torch.full((B,), 99.0, device=env.device)
    energy = torch.zeros(B, device=env.device)
    rows = []

    while len(rows) < args.episodes:
        if hasattr(ctl, "refresh_obs"):
            ctl.refresh_obs(obs)
        actions = ctl.act()
        obs, rew, term, trunc, _ = env.step(actions)

        rf = env._read_frac_readable
        done_now = term[AGENTS[0]] | trunc[AGENTS[0]]
        rf_max = torch.where(done_now, rf_max, torch.maximum(rf_max, rf))
        t = env.episode_length_buf.float() / 30.0
        for buf, ms in ((t50, 0.5), (t90, 0.9), (t95, 0.95)):
            hit = (rf >= ms) & (buf < 0)
            buf[hit] = t[hit]
        coll += (env._raw_min.min(dim=1).values < MAP_CFG.reward.collision_distance_m).float()
        pos = env._pos_local[..., :2]
        for i in range(D):
            for j in range(i + 1, D):
                sep_min = torch.minimum(sep_min, torch.norm(pos[:, i] - pos[:, j], dim=-1))
        for a in AGENTS:
            energy += actions[a].abs().sum(dim=-1)

        done = done_now
        if done.any():
            for b in torch.nonzero(done).squeeze(-1).tolist():
                final = max(float(rf_max[b]), 0.95 if float(t95[b]) > 0 else 0.0)
                rows.append({
                    "read_frac": round(final, 4),
                    "t50_s": round(float(t50[b]), 1), "t90_s": round(float(t90[b]), 1),
                    "t95_s": round(float(t95[b]), 1),
                    "collision_steps": int(coll[b]),
                    "min_separation_m": round(float(sep_min[b]), 2),
                    "energy": round(float(energy[b]), 1),
                })
                t50[b] = t90[b] = t95[b] = -1.0
                coll[b] = energy[b] = 0.0
                sep_min[b] = 99.0
            ids = torch.nonzero(done).squeeze(-1)
            ctl.on_reset(ids)
            if len(rows) % 5 == 0:
                print(f"  {len(rows)}/{args.episodes} épisodes collectés")

    out = args.out or os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "results",
        f"baseline_{args.controller}_L{args.level}_{args.split}.csv"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows[: args.episodes])

    vals = torch.tensor([r["read_frac"] for r in rows[: args.episodes]])
    q = vals.sort().values
    iqm = q[len(q) // 4: max(len(q) // 4 + 1, 3 * len(q) // 4)].mean()
    t90v = torch.tensor([r["t90_s"] for r in rows[: args.episodes] if r["t90_s"] > 0])
    print(f"\n===== {args.controller.upper()} | gate niveau {args.level} | split {args.split} | {len(rows[:args.episodes])} épisodes =====")
    print(f"  couverture : moyenne={vals.mean():.3f} ± {vals.std():.3f} | IQM={iqm:.3f} | min={vals.min():.3f} max={vals.max():.3f}")
    print(f"  temps-à-90 % : atteint dans {len(t90v)}/{len(vals)} épisodes" + (f", médiane={t90v.median():.0f} s" if len(t90v) else ""))
    print(f"  collisions (pas/épisode) : {sum(r['collision_steps'] for r in rows[:args.episodes])/len(rows[:args.episodes]):.1f}")
    print(f"  CSV : {out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
