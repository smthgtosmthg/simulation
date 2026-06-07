"""
Sélection d'action — Active Inference (free energy minimization) + alternative
heuristique frontier-based.

Les deux fonctions ont la **même signature** pour qu'on puisse switcher
entre elles via `cfg.planner = "aif" | "heuristic"`.

Action space : 9 directions (stay + 8 voisinages de Moore), normalisées.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .belief import BeliefGrid, mix_beliefs
from .math_utils import bernoulli_entropy, softmax_sample


# ── Espace d'actions (normalisé en norme 1) ─────────────────────────

def _build_actions() -> List[Tuple[str, float, float]]:
    raw = [
        ("stay", 0.0, 0.0),
        ("N", 0.0, -1.0), ("NE", 1.0, -1.0), ("E", 1.0, 0.0),
        ("SE", 1.0, 1.0), ("S", 0.0, 1.0), ("SW", -1.0, 1.0),
        ("W", -1.0, 0.0), ("NW", -1.0, -1.0),
    ]
    out = []
    for name, dx, dy in raw:
        norm = math.hypot(dx, dy) or 1.0
        out.append((name, dx / norm, dy / norm))
    return out


ACTIONS: List[Tuple[str, float, float]] = _build_actions()


# ── Termes du free energy ───────────────────────────────────────────


def expected_info_gain(wx: float, wy: float, belief: BeliefGrid, cfg) -> float:
    """Score épistémique : combien d'incertitude on lèverait si on était là."""
    total = 0.0
    for angle in cfg.ray_angles:
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        p_reach = 1.0
        for step in range(1, cfg.max_range_cells + 1):
            cx = wx + step * cfg.grid_resolution * cos_a
            cy = wy + step * cfg.grid_resolution * sin_a
            gx, gy = belief.world_to_grid(cx, cy)
            if not belief.in_bounds(gx, gy):
                break
            p_occ = belief.probability[gy, gx]
            total += p_reach * bernoulli_entropy(p_occ)
            p_reach *= (1.0 - p_occ)
            if p_reach < 1e-4:
                break
    return total


def frontier_attraction(wx: float, wy: float, belief: BeliefGrid) -> float:
    """Score pragmatique : combien de cellules incertaines dans le voisinage 5x5."""
    gx, gy = belief.world_to_grid(wx, wy)
    window = 5
    total, count = 0.0, 0
    for dy in range(-window, window + 1):
        for dx in range(-window, window + 1):
            nx, ny = gx + dx, gy + dy
            if belief.in_bounds(nx, ny):
                p = belief.probability[ny, nx]
                total += (1.0 - abs(2.0 * p - 1.0)) / (math.hypot(dx, dy) + 1.0)
                count += 1
    return total / max(count, 1)


# ════════════════════════════════════════════════════════════════════
# Sélection d'action — AIF (Free Energy Minimization)
# ════════════════════════════════════════════════════════════════════


def select_action(pos_x: float, pos_y: float,
                  others: List[Tuple[float, float]],
                  belief: BeliefGrid,
                  fused: Optional[BeliefGrid],
                  cfg,
                  rng: np.random.Generator,
                  resilience_phase: str = "normal"
                  ) -> Tuple[Tuple[str, float, float], List[Dict], int]:
    """Minimise l'Expected Free Energy G sur l'horizon 1 step.

    G = - w_epistemic·IG - w_pragmatic·Frontier + w_movement·move + w_collision·coll
    (poids modifiés en phase recovery / durable)
    """
    plan_belief = mix_beliefs(belief, fused, cfg.fusion_mix) if fused else belief
    H = plan_belief.mean_entropy()
    n = len(ACTIONS)
    G = np.full(n, 1e6)
    valid = np.zeros(n, dtype=bool)
    cand_diag: List[Dict] = []

    for i, (name, dx, dy) in enumerate(ACTIONS):
        nx = pos_x + dx * cfg.step_size
        ny = pos_y + dy * cfg.step_size
        entry: Dict[str, Any] = {
            "idx": i, "name": name,
            "nx": round(nx, 3), "ny": round(ny, 3),
            "valid": False, "reason": "",
            "ig": 0.0, "frontier": 0.0,
            "move": 0.0, "coll": 0.0, "G": 1e6,
        }
        if not (0.5 <= nx < cfg.env_width - 0.5 and 0.5 <= ny < cfg.env_height - 0.5):
            entry["reason"] = "out-of-bounds"
            cand_diag.append(entry)
            continue
        gx, gy = plan_belief.world_to_grid(nx, ny)
        if plan_belief.probability[gy, gx] >= cfg.occ_threshold:
            entry["reason"] = f"occupied p={plan_belief.probability[gy, gx]:.2f}"
            cand_diag.append(entry)
            continue

        valid[i] = True
        ig = expected_info_gain(nx, ny, plan_belief, cfg)
        fr = frontier_attraction(nx, ny, plan_belief)
        move = 0.0 if name == "stay" else 1.0
        coll = sum(
            1.0 / (math.hypot(nx - ox, ny - oy) + 0.1)
            for ox, oy in others if math.hypot(nx - ox, ny - oy) < 8.0
        )

        if resilience_phase == "recovery":
            G[i] = (
                - cfg.w_entropy_recover * H
                - cfg.w_innov_recover * ig
                + cfg.w_movement * move
                + cfg.w_collision * coll
                - cfg.w_epistemic * ig
                - cfg.w_pragmatic * fr
            )
        elif resilience_phase == "durable":
            maintain_pen = 0.0
            if H > cfg.H_target:
                maintain_pen += (H - cfg.H_target)
            G[i] = (
                - cfg.w_entropy_durable * H
                - cfg.w_epistemic * ig
                - cfg.w_pragmatic * fr
                + cfg.w_movement * move
                + cfg.w_collision * coll
                + cfg.w_maintain * maintain_pen
            )
        else:
            G[i] = (
                - cfg.w_epistemic * ig
                - cfg.w_pragmatic * fr
                + cfg.w_movement * move
                + cfg.w_collision * coll
            )

        entry.update({
            "valid": True, "reason": "ok",
            "ig": round(ig, 4), "frontier": round(fr, 4),
            "move": round(move, 2), "coll": round(coll, 4),
            "G": round(G[i], 4),
        })
        cand_diag.append(entry)

    if not valid.any():
        valid[0] = True
        G[0] = 0.0
        if cand_diag:
            cand_diag[0]["valid"] = True
            cand_diag[0]["reason"] = "forced-stay"
            cand_diag[0]["G"] = 0.0
    idx = softmax_sample(G, cfg.softmax_temp, rng)
    return ACTIONS[idx], cand_diag, idx


# ════════════════════════════════════════════════════════════════════
# Sélection d'action — Heuristique (frontier-based, "drone idiot")
# ════════════════════════════════════════════════════════════════════

HEUR_FREE_THR = 0.4
HEUR_COLL_RADIUS = 1.5

# État per-drone indexé par id(rng) — chaque drone a sa propre instance rng.
_HEUR_STATE: Dict[int, Dict[str, Any]] = {}


def _count_unknown_neighbors(gx: int, gy: int, belief: BeliefGrid) -> int:
    n = 0
    p = belief.probability
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            nx, ny = gx + dx, gy + dy
            if belief.in_bounds(nx, ny):
                pc = p[ny, nx]
                if 0.4 <= pc <= 0.6:
                    n += 1
    return n


def select_action_heuristic(pos_x: float, pos_y: float,
                            others: List[Tuple[float, float]],
                            belief: BeliefGrid,
                            fused: Optional[BeliefGrid],
                            cfg,
                            rng: np.random.Generator,
                            resilience_phase: str = "normal"
                            ) -> Tuple[Tuple[str, float, float], List[Dict], int]:
    """Heuristique : commit-to-direction + confirmed-free + biais inconnu.

    Mêmes args/return que select_action() pour swap transparent.
    """
    del resilience_phase
    plan_belief = mix_beliefs(belief, fused, cfg.fusion_mix) if fused else belief

    key = id(rng)
    state = _HEUR_STATE.get(key)
    if state is None:
        state = {"current_idx": None}
        _HEUR_STATE[key] = state

    cand_diag: List[Dict] = []
    valid_indices: List[int] = []

    for i, (name, dx, dy) in enumerate(ACTIONS):
        nx = pos_x + dx * cfg.step_size
        ny = pos_y + dy * cfg.step_size
        entry: Dict[str, Any] = {
            "idx": i, "name": name,
            "nx": round(nx, 3), "ny": round(ny, 3),
            "valid": False, "reason": "",
            "ig": 0.0, "frontier": 0.0,
            "move": 0.0, "coll": 0.0, "G": 0.0,
        }
        if name == "stay":
            cand_diag.append(entry)
            continue
        if not (0.5 <= nx < cfg.env_width - 0.5 and 0.5 <= ny < cfg.env_height - 0.5):
            entry["reason"] = "out-of-bounds"
            cand_diag.append(entry)
            continue
        gx, gy = plan_belief.world_to_grid(nx, ny)
        p = float(plan_belief.probability[gy, gx])
        if p >= cfg.occ_threshold:
            entry["reason"] = f"occupied p={p:.2f}"
            cand_diag.append(entry)
            continue
        if p >= HEUR_FREE_THR:
            entry["reason"] = f"unknown p={p:.2f}"
            cand_diag.append(entry)
            continue
        min_d = math.inf
        for ox, oy in others:
            d = math.hypot(nx - ox, ny - oy)
            if d < min_d:
                min_d = d
        if min_d < HEUR_COLL_RADIUS:
            entry["reason"] = f"neighbor d={min_d:.2f}"
            entry["coll"] = round(1.0 / (min_d + 0.1), 3)
            cand_diag.append(entry)
            continue
        entry["valid"] = True
        entry["reason"] = "ok"
        entry["move"] = 1.0
        cand_diag.append(entry)
        valid_indices.append(i)

    if not valid_indices:
        cand_diag[0].update({"valid": True, "reason": "forced-stay (all blocked)"})
        state["current_idx"] = None
        return ACTIONS[0], cand_diag, 0

    if state["current_idx"] is not None and state["current_idx"] in valid_indices:
        idx = state["current_idx"]
        cand_diag[idx]["reason"] = "keep"
        return ACTIONS[idx], cand_diag, idx

    unknown_touching: List[int] = []
    for i in valid_indices:
        _, dx, dy = ACTIONS[i]
        nx = pos_x + dx * cfg.step_size
        ny = pos_y + dy * cfg.step_size
        gx, gy = plan_belief.world_to_grid(nx, ny)
        unk = _count_unknown_neighbors(gx, gy, plan_belief)
        cand_diag[i]["frontier"] = float(unk)
        if unk > 0:
            unknown_touching.append(i)

    pool = unknown_touching if unknown_touching else valid_indices
    idx = int(rng.choice(pool))
    state["current_idx"] = idx
    cand_diag[idx]["reason"] = "new-direction (unknown)" if unknown_touching else "new-direction (any)"
    return ACTIONS[idx], cand_diag, idx


def get_planner(name: str):
    """Renvoie la fonction de sélection d'action correspondante."""
    if name == "heuristic":
        return select_action_heuristic
    return select_action
