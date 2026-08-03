"""Validation de l'ENVIRONNEMENT avant tout entraînement.

Un environnement sain doit être INEXPLOITABLE : une politique qui ne sait rien ne doit
obtenir aucune lecture. Si le hasard lit des QR, l'entraînement apprendra à tricher au lieu
d'apprendre la tâche — c'est ce qui a coûté quatre semaines sur ce projet.

  bash rl_inventory/launch.sh rl_inventory/swarmscan_map/verify_env.py --headless \
       --policy random --level 7

Politiques : random (uniforme ±1) | zero (immobile) | forward (plein gaz tout droit) |
             checkpoint (--checkpoint chemin.pt, avec --sigma)
"""

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Batterie de validation de l'environnement.")
parser.add_argument("--policy", type=str, default="random",
                    choices=["random", "zero", "forward", "checkpoint"])
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--sigma", type=float, default=-1.0, help="-1 : σ du checkpoint ; 0 : déterministe")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--level", type=int, default=-1, help="-1 : gate nominal (défaut)")
parser.add_argument("--spawn_help", type=int, default=1, help="0 : aucun spawn dirigé près d'un tag")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from rl_inventory.config_rl import CFG
from rl_inventory.env import AGENTS
from rl_inventory.swarmscan_map.config_map import MAP_CFG
from rl_inventory.swarmscan_map.env_map import SwarmScanMapEnv, SwarmScanMapEnvCfg
from rl_inventory.swarmscan_map.flatten_wrapper import SwarmMapVecEnv
from rl_inventory.swarmscan_map.models import MapActorCritic

D = len(AGENTS)
OBS_GROUPS = {"policy": ["maps", "vector"], "critic": ["maps", "vector", "privileged"]}


def make_policy(vec, env):
    """Renvoie une fonction obs → actions (num_envs*D, 4)."""
    n, dim = vec.num_envs, vec.num_actions
    if args.policy == "random":
        return lambda obs: torch.empty(n, dim, device=env.device).uniform_(-1.0, 1.0)
    if args.policy == "zero":
        return lambda obs: torch.zeros(n, dim, device=env.device)
    if args.policy == "forward":
        a = torch.zeros(n, dim, device=env.device)
        a[:, 0] = 1.0
        return lambda obs: a
    net = MapActorCritic(
        vec.get_observations(), OBS_GROUPS, dim,
        map_channels=len(MAP_CFG.map.crop_spans_m) * MAP_CFG.map.n_channels,
        map_px=MAP_CFG.map.crop_px, init_noise_std=MAP_CFG.train.init_noise_std,
    ).to(env.device)
    net.load_state_dict(torch.load(args.checkpoint, map_location=env.device,
                                   weights_only=False)["model_state_dict"])
    net.eval()
    if args.sigma >= 0:
        net.std.data.fill_(max(args.sigma, 1e-6))
    det = args.sigma == 0.0

    def policy(obs):
        with torch.no_grad():
            net.update_distribution(net.actor_obs_normalizer(net.get_actor_obs(obs)))
            return net.distribution.mean if det else net.distribution.sample()
    return policy


def main():
    torch.manual_seed(args.seed)
    cfg = SwarmScanMapEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    env = SwarmScanMapEnv(cfg)
    env._curr.level = MAP_CFG.curriculum.notches if args.level < 0 else args.level
    env._curr.cfg.min_episodes_per_notch = 10**9
    env._curr.cfg.min_episodes_down = 10**9
    if not args.spawn_help:
        env._curr.cfg.spawn_near_prob = (0.0, 0.0)
    MAP_CFG.train.mission_target = 2.0            # épisodes complets, envs synchronisés
    vec = SwarmMapVecEnv(env)
    vec.get_observations()
    env._ensure_qr()      # AVANT le reset de mesure : sinon les spawns dirigés ne sont pas tirés
    vec.reset()           # et tous les entrepôts démarrent à la pose par défaut, identiques
    env.map_log = {}
    policy = make_policy(vec, env)

    th = env._curr.thresholds()
    cap_cur, cap_nom = th["max_speed_mps"], MAP_CFG.gate.max_speed_mps
    horizon = int(env.max_episode_length)
    dt = CFG.train.physics_dt * CFG.train.decimation

    reads_tot = torch.zeros(1, device=env.device)
    reads_apres_survitesse = torch.zeros(1, device=env.device)
    prev_lin = torch.zeros(env.num_envs, D, device=env.device)
    accels = []
    slow_run = torch.zeros(env.num_envs, D, device=env.device)   # pas consécutifs sous le cap nominal
    run_lengths, gate_frac, nom_frac, snapshot = [], None, None, None
    v_sum, v_n = 0.0, 0

    for step in range(horizon):
        obs = vec.get_observations()
        snapshot = (env._read_frac_readable.clone(),      # AVANT le pas : le reset de fin
                    {k: v.clone() for k, v in env._rw_acc.items()},        # d'épisode remet
                    {k: v.clone() for k, v in env._beh_acc.items()},       # tout à zéro
                    env._mapper.scan.amax(dim=1).flatten(1).sum(-1).clone())
        _, _, _, infos = vec.step(policy(obs))
        lin = env._lin_k                                          # (B,D) vitesse planaire réelle
        new = env._new_reads.transpose(0, 1)                      # (B,D) lectures créditées ce pas
        reads_tot += new.sum()
        reads_apres_survitesse += (new * (prev_lin > cap_cur).float()).sum()
        # vol LIBRE seulement : un choc contre un rack est une décélération brutale réelle,
        # pas un défaut du modèle d'actionneur. On ne mesure que les drones loin des obstacles.
        libre = env._raw_min > 0.6
        if libre.any():
            accels.extend((((lin - prev_lin).abs() / dt)[libre]).tolist())
        under = lin <= cap_cur          # le cap du niveau évalué, pas le nominal
        finished = slow_run[~under & (slow_run > 0)]
        if finished.numel():
            run_lengths.extend(finished.tolist())
        slow_run = torch.where(under, slow_run + 1, torch.zeros_like(slow_run))
        prev_lin = lin.clone()
        v_sum += float(lin.mean()); v_n += 1
        if (step + 1) % 1000 == 0:
            print(f"    pas {step + 1}/{horizon} | lectures {int(reads_tot)}", flush=True)
        if "log" in infos:     # fin d'épisode : on s'arrête, sinon on compte les lectures
            gate_frac = infos["log"]["episode/read_frac_gate"]      # offertes aux spawns suivants
            nom_frac = infos["log"]["episode/read_frac_nominal"]
            break

    rt = float(reads_tot)
    ratio = float(reads_apres_survitesse) / rt if rt > 0 else 0.0
    run_lengths.extend(slow_run[slow_run > 0].tolist())   # séries encore en cours à la fin
    runs = torch.tensor(run_lengths) if run_lengths else torch.zeros(1)
    acc = torch.tensor(accels) if accels else torch.zeros(1)
    accel_max, accel_p99 = float(acc.max()), float(acc.quantile(0.99))
    name = args.policy if args.policy != "checkpoint" else f"checkpoint σ={args.sigma:.2f}"

    print("\n=============== VALIDATION DE L'ENVIRONNEMENT ===============")
    print(f"politique          : {name}")
    print(f"niveau / cap vitesse : {env._curr.level} / {cap_cur:.3f} m/s (nominal {cap_nom} m/s)")
    print(f"vitesse moyenne des drones          : {v_sum / max(1, v_n):.2f} m/s")
    print(f"accélération en vol libre : p99 {accel_p99:.2f} m/s² = {accel_p99 / 9.81:.2f} g | "
          f"max {accel_max:.2f} = {accel_max / 9.81:.2f} g   (limite du modèle "
          f"{CFG.action.accel_xy_mps2} = {CFG.action.accel_xy_mps2 / 9.81:.2f} g)")
    print(f"lectures totales créditées          : {int(rt)}")
    print(f"  dont juste après une SURVITESSE   : {ratio * 100:.1f} %   ← signature de la triche")
    print(f"durée de passage sous le cap (pas)  : médiane {float(runs.median()):.0f} | "
          f"moyenne {float(runs.float().mean()):.1f} | max {float(runs.max()):.0f}")
    if gate_frac is not None:
        print(f"fraction lue (gate niveau {env._curr.level})       : {gate_frac:.3f}")
        print(f"fraction lue (gate NOMINAL)         : {nom_frac:.3f}")
    if snapshot is not None:
        frac, rw_acc, beh, cover = snapshot
        pas = beh["pas"].clamp_min(1.0)
        ordre = torch.argsort(frac, descending=True)
        n3 = max(1, len(ordre) // 3)
        bons, mauvais = ordre[:n3], ordre[-n3:]
        print(f"\n--- CE QUI DISTINGUE UN BON ÉPISODE D'UN MAUVAIS ({n3} entrepôts de chaque côté) ---")
        print(f"{'grandeur':<26}{'bons':>12}{'mauvais':>12}{'écart':>10}")

        def cmp(nom, t, unite=""):
            b, m = float(t[bons].mean()), float(t[mauvais].mean())
            ratio = f"×{b / m:.1f}" if abs(m) > 1e-6 else "—"
            print(f"{nom:<26}{b:>12.2f}{m:>12.2f}{ratio:>10}{unite}")

        cmp("fraction lue", frac)
        cmp("distance parcourue (m)", beh["distance"])
        cmp("altitude moyenne (m)", beh["altitude"] / pas)
        cmp("vitesse moyenne (m/s)", beh["vitesse"] / pas)
        cmp("temps en contact (%)", beh["contacts"] / (pas * D) * 100)
        cmp("cellules couvertes", cover)
        print(f"{'':-<60}")
        for nom, t in rw_acc.items():
            cmp(f"récompense {nom}", t)

    verdicts = []
    if args.policy in ("random", "zero", "forward"):
        ok = (nom_frac is not None and nom_frac < 0.01)
        verdicts.append((f"une politique '{name}' ne doit RIEN lire au gate nominal (< 0.01)", ok))
    verdicts.append(("accélération réaliste hors chocs (p99 < 0.5 g)", accel_p99 / 9.81 < 0.5))
    verdicts.append(("lectures non issues de survitesses (< 20 %)", ratio < 0.20))
    print()
    for label, ok in verdicts:
        print(f"  [{'PASSE' if ok else 'ÉCHOUE'}] {label}")
    print("============================================================\n")
    vec.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
