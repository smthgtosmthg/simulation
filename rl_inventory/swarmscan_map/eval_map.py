"""Mesure un checkpoint SwarmScan-Map à plusieurs niveaux de bruit σ (0 = politique déterministe).

Répond à LA question du transfert : à quel bruit la politique lit-elle le mieux en conditions
NOMINALES (le gate calibré, la vraie tâche) ? Les runs frontaux ont montré un optimum vers
σ ≈ 0,20-0,35 et un effondrement sous 0,15 ; ce script mesure la courbe sur le modèle latéral.

Biais corrigés (chacun faussait une mesure précédente) :
  - terminaison anticipée sur mission : désactivée, sinon seuls les entrepôts les plus rapides
    sont comptés et le score du mode bruité est le seuil de mission, pas une performance ;
  - _ensure_qr forcé avant le premier reset, sinon les configurations et les spawns dirigés ne
    sont pas tirés à la première vague (tous les entrepôts partaient de la pose par défaut) ;
  - le bruit des actions est tiré d'un générateur DÉDIÉ : le générateur global n'est pas
    consommé, donc tous les modes voient exactement les mêmes entrepôts et les mêmes spawns.

  bash rl_inventory/launch.sh rl_inventory/swarmscan_map/eval_map.py --headless \
       --checkpoint swarmscan_runs/<run>/model_XXXX.pt --level 7 --sigmas 0,0.20,0.32,0.42
"""

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Courbe performance/bruit d'un checkpoint SwarmScan-Map.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--level", type=int, default=-1, help="niveau de gate évalué (-1 : nominal)")
parser.add_argument("--sigmas", type=str, default="0,0.20,0.32,0.42", help="0 = déterministe")
parser.add_argument("--waves", type=int, default=1, help="vagues d'épisodes complets par σ")
parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
parser.add_argument("--spawn_help", type=int, default=1, help="1 : spawns dirigés (comme à l'entraînement)")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from rl_inventory.swarmscan_map.config_map import MAP_CFG
from rl_inventory.swarmscan_map.env_map import SwarmScanMapEnv, SwarmScanMapEnvCfg
from rl_inventory.swarmscan_map.flatten_wrapper import SwarmMapVecEnv
from rl_inventory.swarmscan_map.models import MapActorCritic

N_SCALES = len(MAP_CFG.map.crop_spans_m)
OBS_GROUPS = {"policy": ["maps", "vector"], "critic": ["maps", "vector", "privileged"]}


def build_policy(obs, num_actions, device):
    policy = MapActorCritic(
        obs, OBS_GROUPS, num_actions,
        map_channels=N_SCALES * MAP_CFG.map.n_channels, map_px=MAP_CFG.map.crop_px,
        init_noise_std=MAP_CFG.train.init_noise_std,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()
    return policy, float(ckpt["model_state_dict"]["std"].mean())


def run_sigma(vec, env, policy, sigma: float, horizon: int) -> dict:
    """Une ou plusieurs vagues d'épisodes COMPLETS ; σ=0 → actions = moyenne de la politique."""
    tag = "det" if sigma == 0 else f"σ={sigma:.2f}"
    torch.manual_seed(args.seed)                 # mêmes entrepôts et mêmes spawns à chaque σ
    gen = torch.Generator(device=env.device)     # bruit tiré à part : le tirage global reste intact
    gen.manual_seed(args.seed)
    policy.std.data.fill_(max(sigma, 1e-6))
    vec.reset()
    env.map_log = {}
    gates, noms, rews = [], [], []
    total = torch.zeros(vec.num_envs, device=env.device)
    steps = 0
    while len(gates) < args.waves and steps < horizon * (args.waves + 1):
        obs = vec.get_observations()
        with torch.no_grad():
            policy.update_distribution(policy.actor_obs_normalizer(policy.get_actor_obs(obs)))
            mu = policy.distribution.mean
            act = mu if sigma == 0 else mu + policy.distribution.stddev * torch.randn(
                mu.shape, generator=gen, device=mu.device)
        _, rew, _, infos = vec.step(act)
        total += rew
        steps += 1
        if "log" in infos:
            gates.append(infos["log"]["episode/read_frac_gate"])
            noms.append(infos["log"]["episode/read_frac_nominal"])
            rews.append(total.mean().item())
            total[:] = 0.0
            print(f"    [{tag}] vague {len(gates)}/{args.waves} (pas {steps}) : gate {gates[-1]:.3f} | "
                  f"nominal {noms[-1]:.3f} | récompense {rews[-1]:.0f}", flush=True)
        elif steps % 1000 == 0:
            print(f"    [{tag}] pas {steps}/{horizon}", flush=True)
    n = max(1, len(gates))
    return {"gate": sum(gates) / n, "nominal": sum(noms) / n, "reward": sum(rews) / max(1, len(rews)),
            "episodes": len(gates) * env.num_envs}


def main():
    torch.manual_seed(args.seed)
    cfg = SwarmScanMapEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    env = SwarmScanMapEnv(cfg)

    lvl = MAP_CFG.curriculum.notches if args.level < 0 else args.level
    env._curr.level = lvl
    env._curr.cfg.min_episodes_per_notch = 10**9        # gate figé
    env._curr.cfg.min_episodes_down = 10**9
    if not args.spawn_help:
        env._curr.cfg.spawn_near_prob = (0.0, 0.0)
    MAP_CFG.train.mission_target = 2.0                  # pas de fin anticipée : envs synchronisés
    env._split = args.split
    vec = SwarmMapVecEnv(env)

    obs = vec.get_observations()
    env._ensure_qr()                                    # _qr_ready avant le 1er reset des mesures
    policy, std_ckpt = build_policy(obs, vec.num_actions, env.device)
    horizon = int(env.max_episode_length)
    th = env._curr.thresholds()
    print(f"\n>>> COURBE PERFORMANCE / BRUIT | {os.path.basename(args.checkpoint)} | niveau {lvl} | "
          f"split={args.split} | {args.num_envs} entrepôts × 3 drones | σ du checkpoint {std_ckpt:.3f}\n"
          f"    gate courant : {th['read_distance_m']:.2f} m / {th['view_angle_deg']:.1f}° / "
          f"{th['max_speed_mps']:.2f} m/s / dwell {int(th['dwell_steps'])}\n"
          f"    gate NOMINAL : {MAP_CFG.gate.read_distance_m} m / {MAP_CFG.gate.view_angle_deg}° / "
          f"{MAP_CFG.gate.max_speed_mps} m/s / dwell {MAP_CFG.gate.dwell_steps}\n"
          f"    épisodes complets de {horizon} pas, spawns dirigés={'oui' if args.spawn_help else 'non'}\n",
          flush=True)

    sigmas = [float(s) for s in args.sigmas.split(",")]
    res = {s: run_sigma(vec, env, policy, s, horizon) for s in sigmas}

    print("\n============= LECTURE EN FONCTION DU BRUIT =============")
    print(f"{'σ':>8}{'épisodes':>10}{'lu (gate)':>12}{'lu (NOMINAL)':>15}{'récompense':>13}")
    for s, r in res.items():
        label = "0 (det)" if s == 0 else f"{s:.2f}"
        print(f"{label:>8}{r['episodes']:>10}{r['gate']:>12.3f}{r['nominal']:>15.3f}{r['reward']:>13.1f}")
    best = max(res.items(), key=lambda kv: kv[1]["nominal"])
    print(f"\nmeilleur σ pour la tâche NOMINALE : {best[0]:.2f} → {best[1]['nominal']:.3f}")
    print("========================================================\n")
    vec.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
