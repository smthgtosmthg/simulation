"""Évaluation de l'essaim entraîné (déterministe) → métriques pour le rapport vs Pore.

Charge un checkpoint skrl, joue la politique en DÉTERMINISTE (action moyenne), et mesure
sur N épisodes parallèles : couverture, temps de mission, taux de succès, collisions,
distance mini inter-drones, sécurité (facteur de risque), efficacité d'exploration, effort.

Métriques différées (à ajouter ensuite) : vrai pyzbar (nécessite le rendu caméra),
latence NS3, erreur de trajectoire RMS (nécessite un chemin de référence).

Exemple :
  ~/isaac5_env/bin/python rl_inventory/eval.py --checkpoint .../best_agent.pt \
     --headless --num_envs 16 --kit_args="--/rtx/verifyDriverVersion/enabled=false"
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Évalue l'essaim QR (skrl, déterministe).")
parser.add_argument("--checkpoint", type=str, required=True, help="chemin vers best_agent.pt")
parser.add_argument("--task", type=str, default="Isaac-QR-Inventory-Swarm-Direct-v0")
parser.add_argument("--algorithm", type=str, default="IPPO", choices=["IPPO", "MAPPO"])
parser.add_argument("--num_envs", type=int, default=16, help="nb d'épisodes évalués en parallèle")
parser.add_argument("--max_steps", type=int, default=None, help="horizon (défaut = longueur d'épisode)")
parser.add_argument("--out", type=str, default=None, help="CSV de sortie (optionnel)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import os
import sys

import gymnasium as gym
import torch
from skrl.utils.runner.torch import Runner

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import rl_inventory  # noqa: F401  enregistre la tâche gym
from rl_inventory.config_rl import CFG  # noqa: E402
from rl_inventory.env import AGENTS, NUM_DRONES, SwarmQREnvCfg  # noqa: E402


def stat(x):
    """Renvoie (moyenne, écart-type) d'un tenseur 1D en floats."""
    return x.float().mean().item(), x.float().std(unbiased=False).item()


def main():
    entry = {"IPPO": "skrl_ippo_cfg_entry_point", "MAPPO": "skrl_mappo_cfg_entry_point"}[args.algorithm]
    agent_cfg = load_cfg_from_registry(args.task, entry)

    env_cfg = SwarmQREnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = gym.make(args.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    base = env.unwrapped

    runner = Runner(env, agent_cfg)
    runner.agent.load(args.checkpoint)
    runner.agent.enable_models_training_mode(False)  # mode évaluation

    n = args.num_envs
    dev = base.device
    horizon = args.max_steps or int(base.max_episode_length)
    target = CFG.qr.coverage_target
    r_safe = CFG.reward.safe_distance_m
    beta = 10.0

    # tampons par env (un épisode par env, figé dès qu'il se termine)
    done_mask = torch.zeros(n, dtype=torch.bool, device=dev)
    max_cov = torch.zeros(n, device=dev)
    mission_step = torch.full((n,), -1.0, device=dev)
    collided = torch.zeros(n, dtype=torch.bool, device=dev)
    min_inter = torch.full((n,), float("inf"), device=dev)
    safe_steps = torch.zeros(n, device=dev)
    alive_steps = torch.zeros(n, device=dev)
    path_len = torch.zeros(n, device=dev)
    effort = torch.zeros(n, device=dev)
    prev_pos = None

    obs, _ = env.reset()
    for step in range(horizon):
        upd = ~done_mask
        cov = base._read_frac
        pos = torch.stack([base._drones[k].data.root_pos_w for k in range(NUM_DRONES)], dim=0)  # (K,N,3)

        inter = torch.full((n,), float("inf"), device=dev)
        for i in range(NUM_DRONES):
            for j in range(i + 1, NUM_DRONES):
                inter = torch.minimum(inter, torch.norm(pos[i] - pos[j], dim=-1))

        risk = torch.sigmoid(beta * (r_safe - base._lidar_min.min(dim=0).values))  # (N,)

        max_cov = torch.where(upd, torch.maximum(max_cov, cov), max_cov)
        hit = upd & (cov >= target) & (mission_step < 0)
        mission_step = torch.where(hit, torch.full_like(mission_step, float(step)), mission_step)
        collided = collided | (upd & base._collision.any(dim=0))
        min_inter = torch.where(upd, torch.minimum(min_inter, inter), min_inter)
        safe_steps = safe_steps + (upd & (risk <= 0.4)).float()
        alive_steps = alive_steps + upd.float()
        if prev_pos is not None:
            disp = torch.norm(pos - prev_pos, dim=-1).sum(dim=0)  # somme sur les drones
            path_len = path_len + torch.where(upd, disp, torch.zeros_like(disp))
        prev_pos = pos.clone()

        with torch.no_grad():
            sampled, outputs = runner.agent.act(obs, env.state(), timestep=step, timesteps=horizon)
            actions = outputs.get("mean_actions", sampled)  # déterministe = moyenne de la politique

        obs, _, term, trunc, _ = env.step(actions)
        eff = torch.stack([base._actions[a].abs().mean(dim=-1) for a in AGENTS], dim=0).mean(dim=0)
        effort = effort + torch.where(upd, eff, torch.zeros_like(eff))
        done = (term[AGENTS[0]] | trunc[AGENTS[0]]) if isinstance(term, dict) else (term | trunc).reshape(n, -1).any(dim=-1)
        done_mask = done_mask | done.reshape(-1).bool()
        if bool(done_mask.all()):
            break

    # ---- agrégation ----
    alive = alive_steps.clamp_min(1.0)
    completed = max_cov >= target
    success = completed & (~collided)
    qr_read = max_cov * base._n_cartons
    expl_eff = qr_read / path_len.clamp_min(1e-6)
    pct_safe = 100.0 * safe_steps / alive
    eff_mean = effort / alive
    ctrl_dt = CFG.train.control_dt

    cov_m, cov_s = stat(max_cov * 100.0)
    inter_m, inter_s = stat(min_inter)
    safe_m, safe_s = stat(pct_safe)
    expl_m, expl_s = stat(expl_eff)
    eff_m, eff_s = stat(eff_mean)
    comp = completed.float().mean().item() * 100.0
    succ = success.float().mean().item() * 100.0
    coll = collided.float().mean().item() * 100.0
    if completed.any():
        mt = mission_step[completed]
        mt_m = mt.mean().item()
        mission_txt = f"{mt_m:.0f} pas ({mt_m * ctrl_dt:.1f} s)  [sur {int(completed.sum())} env complétés]"
    else:
        mission_txt = "jamais atteinte (couverture incomplète)"

    rows = [
        ("Couverture QR (%)", f"{cov_m:.1f} ± {cov_s:.1f}"),
        ("Mission complète (%)", f"{comp:.1f}"),
        ("Taux de succès (%)  [tout lu + 0 collision]", f"{succ:.1f}"),
        ("Temps de mission", mission_txt),
        ("Taux de collision (%)", f"{coll:.1f}"),
        ("Distance mini inter-drones (m)", f"{inter_m:.2f} ± {inter_s:.2f}"),
        ("% temps en zone sûre (risque<=0.4)", f"{safe_m:.1f} ± {safe_s:.1f}"),
        ("Efficacité exploration (QR/m)", f"{expl_m:.3f} ± {expl_s:.3f}"),
        ("Effort de contrôle moy (|action|)", f"{eff_m:.3f} ± {eff_s:.3f}"),
    ]
    print("\n================ ÉVALUATION ESSAIM (déterministe) ================")
    print(f"checkpoint : {os.path.basename(args.checkpoint)} | {args.num_envs} épisodes | horizon {horizon} pas")
    print(f"cartons QR : {base._n_cartons}\n")
    for name, val in rows:
        print(f"  {name:<44} {val}")
    print("\n  (à venir : précision pyzbar [rendu], latence [NS3], RMS trajectoire [vs Pore])")
    print("==================================================================\n")

    if args.out:
        import csv

        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for name, val in rows:
                w.writerow([name, val])
        print(f"métriques écrites dans {args.out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
