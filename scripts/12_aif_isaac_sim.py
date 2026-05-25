#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

sys.stdout.reconfigure(line_buffering=True)


#tus les paramaitres de la simulation sont dans cette classe
@dataclass
class SimConfig:

    # environment (metres)
    env_width: float = 30.0
    env_height: float = 20.0
    grid_resolution: float = 0.5 #taille d'une ceullule de la grille d'occupation
    fly_altitude: float = 2.0 #hauteur de vol des drones

    # drones
    num_drones: int = 3
    drone_spacing: float = 5.0 #espace de séparation entre les drones
    step_size: float = 1.0   # déplacement en mètres par action 

    # lidar 
    num_rays: int = 360 #nombre de rayon par scan 
    lidar_fov_h: float = 360.0      # champs de vision horizontal en degrés
    lidar_fov_v: float = 40.0       # champs de vision vertical en degrés (±20° pour détecter étagères)
    lidar_vert_res: float = 5.0     # résolution verticale en degrés (8 couches verticales)
    lidar_max_range: float = 8.0 # portée maximale du lidar en mètres
    lidar_min_range: float = 0.15 # portée minimale du lidar en mètres
    lidar_hz: float = 10.0 # fréquence de scan du lidar en Hz
    floor_filter_z: float = 0.25    # hauteur min pour filtrer les hits au sol

    # active inference
    prior_occupancy: float = 0.5 # probabilité a priori d'occupation d'une cellule
    lo_free: float = -0.55 #increment de log-odds pour cellule libre 
    lo_occ: float = 0.85
    lo_max: float = 30.0
    occ_threshold: float = 0.65# pour marquer un ceullule occupé 

    # AIF action selection weights
    w_epistemic: float = 2.5 #pour favoriser les zones de grande incertitude 
    w_pragmatic: float = 0.8 #pour favoriser les zones de frontiere incertaines
    w_movement: float = 0.1 #si on l'augumente le drone bouge moin 
    w_collision: float = 5.0 #pour éviter les collisions avec les autres drones
    softmax_temp: float = 0.3 #si on l'augumente on va aller vers des actions plus variées, sinon on va toujours choisir la même action
    fusion_mix: float = 0.3 #pour mélanger les croyances locales et fusionnées dans la planification

    # SITL (ArduPilot contrôle le vol, tolérance pour l'AIF)
    waypoint_tol: float = 0.3      # tolérance pour considérer qu'on est arrivé à un waypoint (en mètres)

    # resilience (adapté de AIF_controller)
    alpha: int = 30            # durée phase recovery (steps)
    beta: int = 60             # durée phase durable (steps)
    H_target: float = 0.44    # entropie cible pour considérer recovery OK
    innov_target: float = 0.16 # innovation cible
    k_sigma: float = 2.0      # seuil spike = EMA + k_σ·σ
    ema_alpha: float = 0.05   # lissage EMA
    # poids recovery
    w_entropy_recover: float = 3.0
    w_innov_recover: float = 1.2
    w_deadline: float = 12.0
    # poids durable
    w_entropy_durable: float = 1.2
    w_innov_durable: float = 0.8
    w_churn_durable: float = 2.2
    w_maintain: float = 10.0

    # simulation
    headless: bool = False
    max_steps: int = 500 #combien de fois on décide
    sim_steps_per_aif: int = 60   #combien de temps on laisse voler entre deux décisions.
    target_coverage: float = 93.0
    output_dir: str = "/tmp"

    # ── Planner (Tâche 1) ──
    planner: str = "aif"              # "aif" | "heuristic"

    # ── Architecture (Tâche 2) ──
    arch: str = "centralized"         # "centralized" | "distributed"
    neighbor_radius_m: float = 10.0   # rayon de fusion entre voisins (distribué)

    # ── NS-3 (Tâche 3) ──
    ns3_mode: str = "wifi"            # "none" | "wifi" | "5g"
    cloud_latency_ms: float = 20.0    # latence drone↔cloud (centralisé)
    ns3_sim_time: int = 600           # secondes que NS-3 simule
    physics_dt_s: float = 1.0 / 60.0  # Isaac Sim ~60 Hz par défaut

    # ── Cuts de liens (Tâche 4) ──
    cut_cloud_at_step: int = -1       # step où on coupe le lien cloud (-1 = jamais)
    cut_drone_link: str = ""          # "" | "0-1" | "all"
    cut_drone_link_at_step: int = -1  # step où on applique cut_drone_link

    # ── Run capture (Tâche 5) ──
    run_tag: str = "default"          # nom du dossier logs/runs/run_*_<tag>/
    runs_dir: str = ""                # racine où écrire (auto = <workspace>/logs/runs)

    world_origin_x: float = float("nan")
    world_origin_y: float = float("nan")
    factory_bounds_world: Optional[Tuple[float, float, float, float]] = None
    interior_inset_m: float = 2.5

    @property
    def origin_x(self) -> float:
        import math
        return -self.env_width  / 2 if math.isnan(self.world_origin_x) else self.world_origin_x

    @property
    def origin_y(self) -> float:
        import math
        return -self.env_height / 2 if math.isnan(self.world_origin_y) else self.world_origin_y

    def factory_bounds_grid(self) -> Optional[Tuple[int, int, int, int]]:

        if self.factory_bounds_world is None:
            return None
        fx0w, fy0w, fx1w, fy1w = self.factory_bounds_world
        gx0 = max(0, int((fx0w - self.origin_x) / self.grid_resolution))
        gy0 = max(0, int((fy0w - self.origin_y) / self.grid_resolution))
        gx1 = min(self.grid_width,
                  int(math.ceil((fx1w - self.origin_x) / self.grid_resolution)))
        gy1 = min(self.grid_height,
                  int(math.ceil((fy1w - self.origin_y) / self.grid_resolution)))
        return gx0, gy0, gx1, gy1

    def interior_area_cells(self) -> int:
        if self.factory_bounds_world is None:
            return self.grid_width * self.grid_height
        fx0w, fy0w, fx1w, fy1w = self.factory_bounds_world
        inset = float(self.interior_inset_m)
        interior_w = max(0.0, (fx1w - fx0w) - 2 * inset)
        interior_h = max(0.0, (fy1w - fy0w) - 2 * inset)
        cells = (interior_w * interior_h) / (self.grid_resolution ** 2)
        return max(1, int(round(cells)))

    @property #nombre de collone 
    def grid_width(self) -> int:
        return int(self.env_width / self.grid_resolution)

    @property #nombre de lignes
    def grid_height(self) -> int:
        return int(self.env_height / self.grid_resolution)

    @property
    def ray_angles(self) -> np.ndarray:
        return np.linspace(0, 2 * math.pi, self.num_rays, endpoint=False)

    @property #portée en nombre de ceullule du lidar
    def max_range_cells(self) -> int:
        return int(self.lidar_max_range / self.grid_resolution)


# ════════════════════════════════════════════════════════════════
# 2. Math Utilities
# ════════════════════════════════════════════════════════════════
#eviter les proba non valides 
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

#convertir une proba en log-odds
def logit(p: float) -> float:
    p = clamp(p, 1e-7, 1 - 1e-7)
    return math.log(p / (1 - p))

#log en proba 
def inv_logit(l: float) -> float:
    return 1.0 / (1.0 + math.exp(-l)) if l >= 0 else math.exp(l) / (1.0 + math.exp(l))

#log en proba version array
def inv_logit_v(arr: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -30, 30)))

#meusure l'incertitude d'une ceullule occupé ou libre
def bernoulli_entropy(p: float) -> float:
    p = clamp(p, 1e-7, 1 - 1e-7)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def bernoulli_entropy_v(arr: np.ndarray) -> np.ndarray:
    p = np.clip(arr, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

#choisir un action parmi les 9 candidates en fonction de leur score G, avec une temperature pour favoriser les actions les mieux notées ou pour favoriser la diversité des actions choisies
def softmax_sample(values: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    temperature = max(temperature, 1e-6)
    shifted = -(values - np.min(values)) / temperature 
    weights = np.exp(shifted - np.max(shifted))
    probs = weights / weights.sum()
    return int(rng.choice(len(values), p=probs))


# ════════════════════════════════════════════════════════════════
# 3. Belief Grid 
# ════════════════════════════════════════════════════════════════

class BeliefGrid:

    def __init__(self, cfg: SimConfig):
        self.width = cfg.grid_width
        self.height = cfg.grid_height
        self.resolution = cfg.grid_resolution
        self.lo_max = cfg.lo_max
        l0 = logit(cfg.prior_occupancy)
        self.logodds = np.full((self.height, self.width), l0, dtype=np.float64)#retourne une grille de log-odds initialisée à la valeur a priori d'occupation
        self.probability = np.full((self.height, self.width), cfg.prior_occupancy)

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        gx = max(0, min(int(wx / self.resolution), self.width - 1))
        gy = max(0, min(int(wy / self.resolution), self.height - 1))
        return gx, gy

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.width and 0 <= gy < self.height
    
#mettre ajour une ceullule 
    def update_cell(self, gx: int, gy: int, dl: float):
        if not self.in_bounds(gx, gy):
            return
        self.logodds[gy, gx] = np.clip(self.logodds[gy, gx] + dl, -self.lo_max, self.lo_max)
        self.probability[gy, gx] = inv_logit(float(self.logodds[gy, gx]))

    def update_from_lidar(
        self,
        ox: float, oy: float,
        angles: np.ndarray,#angle de chaque rayon du lidar
        ranges: np.ndarray,#distance mesurée par chaque rayon du lidar
        hits: np.ndarray,#boolean indiquant si chaque rayon a touché un obstacle ou a atteint la portée maximale
        max_range: float,#portée maximale du lidar en mètres
        lo_free: float,#increment de log-odds pour cellule libre
        lo_occ: float,#increment de log-odds pour cellule occupée
    ):
        ogx, ogy = self.world_to_grid(ox, oy)
        for i in range(len(angles)):
            cos_a = math.cos(angles[i])
            sin_a = math.sin(angles[i])
            d = min(float(ranges[i]), max_range) if hits[i] else max_range #distance à parcourir le long du rayon pour atteindre l'obstacle ou la portée maximale
            n_steps = int(d / self.resolution)
            for s in range(1, n_steps + 1): #pour chaque ceullule traversée par le rayon, on met à jour la croyance de la ceullule en libre ou occupé en fonction de si le rayon a touché un obstacle ou pas
                wx = ox + s * self.resolution * cos_a
                wy = oy + s * self.resolution * sin_a
                gx, gy = self.world_to_grid(wx, wy)
                if not self.in_bounds(gx, gy):
                    break
                if (gx, gy) != (ogx, ogy):
                    self.update_cell(gx, gy, lo_free)
            if hits[i] and ranges[i] <= max_range:
                hx = ox + float(ranges[i]) * cos_a
                hy = oy + float(ranges[i]) * sin_a
                hgx, hgy = self.world_to_grid(hx, hy)
                if self.in_bounds(hgx, hgy) and (hgx, hgy) != (ogx, ogy):
                    self.update_cell(hgx, hgy, lo_occ)

    def mean_entropy(self) -> float:
        return float(bernoulli_entropy_v(self.probability).mean())

    def exploration_ratio(self) -> float:
        known = (self.probability < 0.3) | (self.probability > 0.7)
        return float(known.sum() / known.size)

    def effective_bounds(self, occ_threshold: float = 0.65) -> Tuple[int, int, int, int]:
        """Bounding box (gx1, gy1, gx2, gy2) des cellules occupées (murs)."""
        occ = self.probability >= occ_threshold
        if not occ.any():
            return 0, 0, self.width, self.height
        rows = np.any(occ, axis=1)
        cols = np.any(occ, axis=0)
        rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
        cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
        return cmin, rmin, cmax + 1, rmax + 1

    def interior_exploration_ratio(
        self,
        drone_positions: List[Tuple[float, float]],
        occ_threshold: float = 0.65,
        bounds_grid: Optional[Tuple[int, int, int, int]] = None,
        interior_area_cells: int = 0,
    ) -> float:
        del drone_positions, occ_threshold
        if interior_area_cells <= 0:
            interior_area_cells = self.width * self.height
        if bounds_grid is None:
            bx0, by0, bx1, by1 = 0, 0, self.width, self.height
        else:
            bx0, by0, bx1, by1 = bounds_grid

        known = (self.probability < 0.3) | (self.probability > 0.7)
        known_in_bbox = int(known[by0:by1, bx0:bx1].sum())

        return float(min(1.0, known_in_bbox / interior_area_cells))

    def observed_inside_walls_ratio(
        self,
        occ_threshold: float = 0.65,
        bounds_grid: Optional[Tuple[int, int, int, int]] = None,
    ) -> float:

        if bounds_grid is None:
            bx0, by0, bx1, by1 = 0, 0, self.width, self.height
        else:
            bx0, by0, bx1, by1 = bounds_grid

        ex0, ey0, ex1, ey1 = self.effective_bounds(occ_threshold)
        walls_detected = (ex1 - ex0) < self.width and (ey1 - ey0) < self.height
        if not walls_detected:
            return 0.0

        wx0 = max(bx0, ex0); wy0 = max(by0, ey0)
        wx1 = min(bx1, ex1); wy1 = min(by1, ey1)
        walls_area = max(0, (wx1 - wx0) * (wy1 - wy0))
        if walls_area == 0:
            return 0.0

        observed = np.abs(self.logodds) > 0.01
        observed_in_walls = int(observed[wy0:wy1, wx0:wx1].sum())
        return float(min(1.0, observed_in_walls / walls_area))

    def copy(self) -> "BeliefGrid":
        new = BeliefGrid.__new__(BeliefGrid)
        new.width, new.height = self.width, self.height
        new.resolution, new.lo_max = self.resolution, self.lo_max
        new.logodds = self.logodds.copy()
        new.probability = self.probability.copy()
        return new

    def to_list(self) -> List[List[float]]:
        return np.round(self.probability, 2).tolist()


# ════════════════════════════════════════════════════════════════
# 4. Belief Fusion
# ════════════════════════════════════════════════════════════════

def fuse_beliefs_logodds(beliefs: List[BeliefGrid], prior_lo: float) -> BeliefGrid:
    """Independent Opinion Pool in log-odds: L_fused = L0 + mean(L_i - L0)."""
    ref = beliefs[0]
    fused = ref.copy()
    n = len(beliefs)
    fused.logodds[:] = prior_lo
    for b in beliefs:
        fused.logodds += (b.logodds - prior_lo) / n
    np.clip(fused.logodds, -fused.lo_max, fused.lo_max, out=fused.logodds) #Fusion faite par moyenne des écarts au prior
    fused.probability = inv_logit_v(fused.logodds)
    return fused


def mix_beliefs(local: BeliefGrid, fused: BeliefGrid, lam: float) -> BeliefGrid:
    """Blend local and fused: L = (1-λ)·L_local + λ·L_fused."""
    mixed = local.copy()
    mixed.logodds = (1 - lam) * local.logodds + lam * fused.logodds
    np.clip(mixed.logodds, -mixed.lo_max, mixed.lo_max, out=mixed.logodds)
    mixed.probability = inv_logit_v(mixed.logodds)
    return mixed


# ════════════════════════════════════════════════════════════════
# 4b. Resilience State (adapté de AIF_controller)
# ════════════════════════════════════════════════════════════════

@dataclass
class ResilienceState:
    """Tracks stress/recovery phases for the swarm."""
    stress_active: bool = False
    stress_t0: int = -1          # step quand le stress a commencé
    recovered_at: int = -1       # step quand H < H_target pour la 1ère fois
    durable_count: int = 0       # compteur de steps consécutifs post-recovery
    cause: str = ""              # "drone_lost" | "innovation_spike"
    events: List[Dict] = None    # log des événements pour le dashboard

    def __post_init__(self):
        if self.events is None:
            self.events = []

    def trigger(self, step: int, cause: str):
        self.stress_active = True
        self.stress_t0 = step
        self.recovered_at = -1
        self.durable_count = 0
        self.cause = cause
        self.events.append({
            "step": step, "type": "stress_start", "cause": cause,
        })
        print(f"  [RESILIENCE] ⚠ STRESS ACTIVATED at step {step}: {cause}")

    def update(self, step: int, H: float, innov: float, cfg: SimConfig):
        if not self.stress_active:
            return
        recovered_now = (H <= cfg.H_target) and (innov <= cfg.innov_target)
        if self.recovered_at < 0:
            if recovered_now:
                self.recovered_at = step
                self.durable_count = 0
                self.events.append({
                    "step": step, "type": "recovery_reached",
                    "entropy": round(H, 4), "innovation": round(innov, 4),
                })
                print(f"  [RESILIENCE] ✓ Recovery reached at step {step} (H={H:.4f})")
        else:
            if recovered_now:
                self.durable_count += 1
            else:
                self.durable_count = 0
            if self.durable_count >= cfg.beta:
                self.stress_active = False
                self.events.append({
                    "step": step, "type": "stress_resolved",
                    "duration": step - self.stress_t0,
                })
                print(f"  [RESILIENCE] ✓ STRESS RESOLVED at step {step} "
                      f"(duration={step - self.stress_t0} steps)")

    def phase(self, step: int, cfg: SimConfig) -> str:
        """'normal' | 'recovery' | 'durable'"""
        if not self.stress_active:
            return "normal"
        elapsed = step - self.stress_t0
        if elapsed <= cfg.alpha:
            return "recovery"
        return "durable"

    def to_dict(self) -> Dict:
        return {
            "stress_active": self.stress_active,
            "cause": self.cause,
            "phase": "normal" if not self.stress_active else self.cause,
            "stress_t0": self.stress_t0,
            "recovered_at": self.recovered_at,
            "durable_count": self.durable_count,
            "events": self.events[-20:],  # last 20 events
        }


# ════════════════════════════════════════════════════════════════
# 5. AIF: Expected Free Energy Minimization
# ════════════════════════════════════════════════════════════════

ACTIONS: List[Tuple[str, float, float]] = []
_RAW = [
    ("stay", 0.0, 0.0),
    ("N", 0.0, -1.0), ("NE", 1.0, -1.0), ("E", 1.0, 0.0),
    ("SE", 1.0, 1.0), ("S", 0.0, 1.0), ("SW", -1.0, 1.0),
    ("W", -1.0, 0.0), ("NW", -1.0, -1.0),
]
for _n, _dx, _dy in _RAW:
    _norm = math.hypot(_dx, _dy) or 1.0
    ACTIONS.append((_n, _dx / _norm, _dy / _norm))


def expected_info_gain(wx: float, wy: float, belief: BeliefGrid, cfg: SimConfig) -> float: #score epistémique
    total = 0.0
    for angle in cfg.ray_angles:
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        p_reach = 1.0 #
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
    return total #plus le score est élevé, plus la position est prometteuse pour réduire l'incertitude de la carte (en observant des zones très incertaines)


def frontier_attraction(wx: float, wy: float, belief: BeliefGrid) -> float: #Score élevé = zone avec beaucoup de cellules incertaines proches.
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
#Les deux favorisent l’incertain.
#Mais expected_info_gain = incertain visible par rayons.
#frontier_attraction = incertain local dans le voisinage.

def select_action(
    pos_x: float, pos_y: float,
    others: List[Tuple[float, float]],
    belief: BeliefGrid,
    fused: Optional[BeliefGrid],
    cfg: SimConfig,
    rng: np.random.Generator,
    resilience_phase: str = "normal",
) -> Tuple[Tuple[str, float, float], List[Dict], int]:
    """Returns (action_tuple, candidates_diagnostics, selected_index).
    resilience_phase: 'normal' | 'recovery' | 'durable'
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

        # ── 3-phase G computation (adapté de AIF_controller) ──
        if resilience_phase == "recovery":
            # Phase recovery: boost exploration pour re-mapper rapidement
            G[i] = (
                - cfg.w_entropy_recover * H
                - cfg.w_innov_recover * ig
                + cfg.w_movement * move
                + cfg.w_collision * coll
                - cfg.w_epistemic * ig
                - cfg.w_pragmatic * fr
            )
        elif resilience_phase == "durable":
            # Phase durable: maintenir la qualité, pénaliser si H > target
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
            # Phase normale
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


# ════════════════════════════════════════════════════════════════
# 5b. Heuristique naïve "drone idiot mais correct" — alternative à AIF
# ════════════════════════════════════════════════════════════════
# Règles :
#   1. Le drone ne bouge QUE vers une cellule confirmée libre dans la belief
#      (p < HEUR_FREE_THR). Une cellule inconnue (p ≈ 0.5) est refusée → pas
#      de fonçage à l'aveugle dans des murs non encore observés.
#   2. On garde la direction courante TANT QU'ELLE EST VALIDE (cellule devant
#      toujours confirmée libre, pas d'autre drone trop près). Pas de timer
#      de commit : on ne change que quand on est obligé.
#   3. Évitement collision : on refuse toute direction qui amènerait à moins
#      de HEUR_COLL_RADIUS d'un autre drone.
#   4. Re-choix de direction (uniquement quand la direction courante est
#      bloquée) : parmi les directions valides, on filtre celles dont la
#      cellule cible touche de l'inconnu (3×3) → on tire au sort uniformément
#      parmi celles-là. Si AUCUNE direction valide ne touche d'inconnu (on est
#      au milieu d'une zone déjà explorée), on tire au sort uniformément parmi
#      toutes les directions valides.
#   5. Si aucune direction n'est valide : on reste sur place et on attend que
#      le LiDAR mette à jour la belief au step suivant.
#
# Pas de softmax, pas d'expected_info_gain, pas de frontier_attraction. Juste
# des if/else et un rng.choice() uniforme.
#
# Même signature et même format de retour que select_action() pour switch
# transparent.

# Constantes algorithme
HEUR_FREE_THR = 0.4          # p < cette valeur ⇒ cellule confirmée libre
HEUR_COLL_RADIUS = 1.5       # m : on refuse une case avec un drone plus près

# État per-drone (direction courante). Indexé par id(rng) — chaque drone a une
# instance rng unique, chaque run est un process Python séparé (voir
# run_all_experiments.sh), donc pas de risque d'id réutilisé entre runs.
_HEUR_STATE: Dict[int, Dict[str, Any]] = {}


def _count_unknown_neighbors(gx: int, gy: int, belief: BeliefGrid) -> int:
    """Compte les cellules inconnues (0.4 ≤ p ≤ 0.6) dans le 3×3 centré
    sur (gx, gy). Utilisé pour biaiser le re-choix vers les zones non
    encore explorées. Pas de raisonnement probabiliste : juste un comptage."""
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


def select_action_heuristic(
    pos_x: float, pos_y: float,
    others: List[Tuple[float, float]],
    belief: BeliefGrid,
    fused: Optional[BeliefGrid],
    cfg: SimConfig,
    rng: np.random.Generator,
    resilience_phase: str = "normal",
) -> Tuple[Tuple[str, float, float], List[Dict], int]:
    """Heuristique naïve : commit-K + confirmed-free + bias inconnu.
    Signature identique à select_action() pour switch transparent."""
    del resilience_phase  # ignoré : l'heuristique n'a pas de phases
    plan_belief = mix_beliefs(belief, fused, cfg.fusion_mix) if fused else belief

    # État persistant par drone (direction courante) : indexé par id(rng)
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
            # "stay" n'est jamais préférée mais reste un fallback si tout est bloqué
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
            # cellule inconnue → on refuse (règle "confirmed-free only")
            entry["reason"] = f"unknown p={p:.2f}"
            cand_diag.append(entry)
            continue
        # collision avec autres drones
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
        # direction valide
        entry["valid"] = True
        entry["reason"] = "ok"
        entry["move"] = 1.0
        cand_diag.append(entry)
        valid_indices.append(i)

    # Aucune direction valide → on reste sur place (le LiDAR rafraîchira au step+1)
    if not valid_indices:
        cand_diag[0].update({"valid": True, "reason": "forced-stay (all blocked)"})
        state["current_idx"] = None
        return ACTIONS[0], cand_diag, 0

    # Direction courante encore valide → on la garde
    if state["current_idx"] is not None and state["current_idx"] in valid_indices:
        idx = state["current_idx"]
        cand_diag[idx]["reason"] = "keep"
        return ACTIONS[idx], cand_diag, idx

    # Sinon (bloqué ou premier coup) : on choisit une nouvelle direction.
    # Filtrer celles dont la cellule cible touche de l'inconnu (3×3).
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



_Backend = None
_Rotation = None


def _ensure_pegasus_imports():
    global _Backend, _Rotation
    if _Backend is None:
        from pegasus.simulator.logic.backends.backend import Backend
        from scipy.spatial.transform import Rotation
        _Backend, _Rotation = Backend, Rotation


# ── AifStateTracker : Backend léger qui expose la pose au contrôleur AIF ──

AifStateTracker = None  # peuplé par _create_state_tracker_class()


def _create_state_tracker_class():
    global AifStateTracker
    _ensure_pegasus_imports()

    class _AifStateTracker(_Backend):
        """Pegasus Backend passif : ne commande aucun moteur, stocke
        uniquement la position / attitude pour que le SitlController
        et le DroneAgent puissent lire la pose courante."""

        def __init__(self, drone_id: int):
            self.drone_id = drone_id
            self.p = np.zeros(3)
            self.v = np.zeros(3)
            self.R = np.eye(3)
            self.w = np.zeros(3)
            self.received_first_state = False
            self._vehicle = None

        # ── lectures utiles pour l'AIF ──
        def get_position(self) -> np.ndarray:
            return self.p.copy()

        def get_position_xy(self) -> Tuple[float, float]:
            return float(self.p[0]), float(self.p[1])

        def get_yaw(self) -> float:
            if not self.received_first_state:
                return 0.0
            return float(math.atan2(self.R[1, 0], self.R[0, 0]))

        # ── interface Pegasus Backend ──
        @property
        def vehicle(self):
            return self._vehicle

        def initialize(self, vehicle):
            self._vehicle = vehicle

        def update_state(self, state):
            self.p = np.array(state.position)
            self.v = np.array(state.linear_velocity)
            self.R = _Rotation.from_quat(state.attitude).as_matrix()
            self.w = np.array(state.angular_velocity)
            self.received_first_state = True

        def update_sensor(self, sensor_type: str, data):
            pass

        def update_graphical_sensor(self, sensor_type: str, data):
            pass

        def input_reference(self):
            return [0.0, 0.0, 0.0, 0.0]

        def update(self, dt: float):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def reset(self):
            self.received_first_state = False

    AifStateTracker = _AifStateTracker


# ── Patch des params SITL dans gazebo-iris.parm ──

def _patch_sitl_defaults():
    """Injecte les paramètres SITL nécessaires dans gazebo-iris.parm AVANT
    le lancement d'ArduPilot.  Sans ces params, le JSON backend Isaac Sim
    (qui tourne à ~250 Hz) provoque des PreArm impossibles à passer :
      - Main loop slow (250 < 400)
      - Gyro rate (250 < 720)
    """
    parm_file = os.path.expanduser(
        "~/ardupilot/Tools/autotest/default_params/gazebo-iris.parm"
    )
    if not os.path.isfile(parm_file):
        print(f"[WARN] gazebo-iris.parm introuvable : {parm_file}")
        return

    required = {
        "ARMING_CHECK":    "0",     # désactiver tous les PreArm checks
        "SCHED_LOOP_RATE": "50",    # permet au loop 250 Hz > 50*1.8 = 90 Hz
        "FS_THR_ENABLE":   "0",     # pas de failsafe throttle (pas de RC)
        "FS_GCS_ENABLE":   "0",     # pas de failsafe GCS
        "FS_CRASH_CHECK":  "0",     # pas de crash-detect disarm (physique SITL imparfaite)
    }

    with open(parm_file) as f:
        lines = f.readlines()

    # indexer les params existants
    idx_map: Dict[str, int] = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            parts = stripped.split()
            if len(parts) >= 2:
                idx_map[parts[0]] = i

    modified = False
    for param, value in required.items():
        if param in idx_map:
            i = idx_map[param]
            old = lines[i].strip().split()[1] if len(lines[i].strip().split()) >= 2 else ""
            if old != value:
                lines[i] = f"{param} {value}\n"
                modified = True
                print(f"  [SITL-PARM] {param}: {old} → {value}")
        else:
            lines.append(f"{param} {value}\n")
            modified = True
            print(f"  [SITL-PARM] ajouté {param} = {value}")

    if modified:
        with open(parm_file, "w") as f:
            f.writelines(lines)
        print(f"[INFO] Params SITL patchés dans {parm_file}")
    else:
        print(f"[INFO] {parm_file} — params SITL déjà corrects")


# ── SitlController : envoie les waypoints AIF à ArduPilot via MAVLink ──

class SitlController:
    """Interface entre la boucle AIF et ArduPilot SITL.

    Lit la pose depuis l'AifStateTracker et commande le drone via
    SET_POSITION_TARGET_LOCAL_NED sur la connexion MAVLink du
    ArduPilotMavlinkBackend (mode GUIDED).
    """

    # constantes MAVLink
    GUIDED_MODE = 4                     # ArduCopter GUIDED
    # type_mask : position + yaw, ignore velocity/accel/yaw_rate
    POS_YAW_MASK = 0b0000_1011_1111_1000  # = 0x0BF8

    def __init__(
        self,
        drone_id: int,
        mavlink_backend,          # ArduPilotMavlinkBackend
        state_tracker,            # AifStateTracker
        spawn_pos: np.ndarray,    # position Isaac Sim au spawn [x, y, z]
        cfg: SimConfig,
    ):
        self.drone_id = drone_id
        self._mav = mavlink_backend
        self._tracker = state_tracker
        self._spawn = spawn_pos.copy()
        self.cfg = cfg
        self.target = spawn_pos.copy()
        self.target_yaw = 0.0
        self.arrived = False
        self._armed = False
        self._cmd_count = 0

    # ── accès à la connexion pymavlink ──
    def _conn(self):
        return self._mav._connection

    def _drain(self):
        """Drainer tous les messages en attente sur la connexion."""
        conn = self._conn()
        if conn is None:
            return
        while conn.recv_match(blocking=False) is not None:
            pass

    # ── passer en mode GUIDED ──
    def set_guided(self):
        from pymavlink import mavutil as _mav
        conn = self._conn()
        if conn is None:
            return
        conn.target_system = self.drone_id + 1
        conn.target_component = 1
        conn.mav.set_mode_send(
            conn.target_system,
            _mav.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            self.GUIDED_MODE,
        )

    # ── forcer l'armement (bypass tous les checks) ──
    def force_arm(self):
        from pymavlink import mavutil as _mav
        conn = self._conn()
        if conn is None:
            return
        conn.target_system = self.drone_id + 1
        conn.target_component = 1
        self._drain()
        conn.mav.command_long_send(
            conn.target_system, conn.target_component,
            _mav.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1,      # param1 = arm
            21196,  # param2 = force arm magic
            0, 0, 0, 0, 0,
        )
        self._armed = True

    # ── vérifier si le drone vole (basé sur altitude du state tracker) ──
    def is_flying(self, min_alt: float = 0.5) -> bool:
        """True si le drone a décollé (altitude > min_alt).
        N'utilise PAS la connexion MAVLink (race condition avec le backend)."""
        if not self._tracker.received_first_state:
            return False
        return float(self._tracker.get_position()[2]) > min_alt

    # ── commande de décollage ──
    def takeoff(self, altitude: float):
        from pymavlink import mavutil as _mav
        conn = self._conn()
        if conn is None:
            return
        conn.mav.command_long_send(
            conn.target_system, conn.target_component,
            _mav.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0, 0, 0, altitude,
        )

    # ── envoyer les params SITL en backup (le parm file est la source primaire) ──
    def send_backup_params(self):
        from pymavlink import mavutil as _mav
        conn = self._conn()
        if conn is None:
            return
        conn.target_system = self.drone_id + 1
        conn.target_component = 1
        for name, value in [(b"ARMING_CHECK", 0), (b"SCHED_LOOP_RATE", 50),
                            (b"FS_THR_ENABLE", 0)]:
            conn.mav.param_set_send(
                conn.target_system, conn.target_component,
                name, float(value),
                _mav.mavlink.MAV_PARAM_TYPE_REAL32,
            )

    # ── envoi d'un waypoint en coordonnées Isaac Sim ──
    def set_target(self, x: float, y: float, z: float, yaw: float = 0.0):
        from pymavlink import mavutil as _mav
        self.target = np.array([x, y, z])
        self.target_yaw = yaw
        self.arrived = False

        conn = self._conn()
        if conn is None:
            return

        # conversion ENU (Isaac Sim) → NED (ArduPilot) relative au home
        ned_n = y - self._spawn[1]
        ned_e = x - self._spawn[0]
        ned_d = -(z - self._spawn[2])
        ned_yaw = math.pi / 2 - yaw

        conn.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms
            conn.target_system, conn.target_component,
            _mav.mavlink.MAV_FRAME_LOCAL_NED,
            self.POS_YAW_MASK,
            ned_n, ned_e, ned_d,
            0, 0, 0,   # velocity
            0, 0, 0,   # acceleration
            ned_yaw, 0, # yaw, yaw_rate
        )

        self._cmd_count += 1
        if self._cmd_count <= 3 or self._cmd_count % 100 == 0:
            print(f"  [SITL] D{self.drone_id} waypoint #{self._cmd_count} "
                  f"isaac=({x:.1f},{y:.1f},{z:.1f}) "
                  f"ned=({ned_n:.1f},{ned_e:.1f},{ned_d:.1f})")

    # ── lectures de pose (délégué au state tracker) ──
    def get_position(self) -> np.ndarray:
        return self._tracker.get_position()

    def get_position_xy(self) -> Tuple[float, float]:
        return self._tracker.get_position_xy()

    def get_yaw(self) -> float:
        return self._tracker.get_yaw()

    def is_at_target(self) -> bool:
        pos = self.get_position()
        self.arrived = np.linalg.norm(pos[:2] - self.target[:2]) < self.cfg.waypoint_tol
        return self.arrived


# ════════════════════════════════════════════════════════════════
# 9. LiDAR Reader (PhysX RotatingLidarPhysX — real raycasting)
# ════════════════════════════════════════════════════════════════

class LidarReader:
    """PhysX rotating LiDAR reader with scan accumulation.

    The PhysX RotatingLidarPhysX only returns a partial angular slice per
    physics tick (~60 rays out of 360).  We accumulate partial scans across
    multiple ticks so that `get_accumulated()` returns a full 360° scan.
    Each partial scan is rotated from body-frame to world-frame using the
    drone yaw supplied at accumulation time.
    """

    def __init__(self, drone_id: int, drone_prim_path: str, cfg: SimConfig):
        from isaacsim.sensors.physx import RotatingLidarPhysX

        self.prim_path = f"{drone_prim_path}/body/Lidar_{drone_id}"
        self.cfg = cfg
        self._read_count = 0

        self.sensor = RotatingLidarPhysX(
            prim_path=self.prim_path,
            rotation_frequency=cfg.lidar_hz,
            fov=(cfg.lidar_fov_h, cfg.lidar_fov_v),
            resolution=(cfg.lidar_fov_h / cfg.num_rays, cfg.lidar_vert_res),
            valid_range=(cfg.lidar_min_range, cfg.lidar_max_range),
        )
        self.sensor.add_linear_depth_data_to_frame()
        self.sensor.add_azimuth_data_to_frame()
        try:
            self.sensor.add_zenith_data_to_frame()
            self._has_zenith = True
        except Exception:
            self._has_zenith = False
            print(f"[WARN] Zenith data unavailable for {self.prim_path}, floor filtering disabled")
        self.sensor.enable_visualization(high_lod=False, draw_points=True, draw_lines=False)
        print(f"[INFO] PhysX LiDAR attached: {self.prim_path}")

        # Accumulation buffers (world-frame)
        self._acc_angles: List[np.ndarray] = []
        self._acc_depths: List[np.ndarray] = []
        self._acc_hits:   List[np.ndarray] = []

    def initialize(self):
        self.sensor.initialize()
        self.sensor.post_reset()
        print(f"[INFO] LiDAR initialized: {self.prim_path}")

    # ── internal: parse one partial frame ────────────────────────
    def _parse_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        frame = self.sensor.get_current_frame()
        depth   = frame.get("linear_depth")
        azimuth = frame.get("azimuth")
        if depth is None or azimuth is None or len(depth) == 0:
            return None
        depth   = np.asarray(depth,   dtype=np.float64).ravel()
        azimuth = np.asarray(azimuth, dtype=np.float64).ravel()
        # Isaac Sim may return degrees
        if azimuth.size > 0 and azimuth.max() > 2 * math.pi + 0.1:
            azimuth = np.deg2rad(azimuth)

        # ── Multi-layer: utiliser zenith pour projeter en 2D et filtrer le sol ──
        zenith = frame.get("zenith") if self._has_zenith else None
        if zenith is not None and len(zenith) > 0:
            zenith = np.asarray(zenith, dtype=np.float64).ravel()
            if zenith.size > 0 and zenith.max() > 2 * math.pi + 0.1:
                zenith = np.deg2rad(zenith)
            # Convention zenith: 0=up, pi/2=horizontal, pi=down
            # Si valeurs proches de 0 → c'est de l'élévation (0=horiz)
            if zenith.size > 0 and np.median(zenith) < math.pi / 4:
                # elevation convention: 0=horiz, >0=up, <0=down
                cos_elev = np.cos(zenith)
                sin_elev = np.sin(zenith)
                horizontal_range = depth * np.abs(cos_elev)
                hit_z = self.cfg.fly_altitude + depth * sin_elev
            else:
                # zenith convention: pi/2=horiz
                sin_zen = np.sin(zenith)
                cos_zen = np.cos(zenith)
                horizontal_range = depth * np.abs(sin_zen)
                hit_z = self.cfg.fly_altitude + depth * cos_zen
            # Filtrer les hits au sol
            valid_height = hit_z > self.cfg.floor_filter_z
            hits = (horizontal_range >= self.cfg.lidar_min_range) & \
                   (horizontal_range < (self.cfg.lidar_max_range - 0.05)) & valid_height
            depth = horizontal_range
        else:
            hits = (depth >= self.cfg.lidar_min_range) & (depth < (self.cfg.lidar_max_range - 0.05))

        return azimuth, depth, hits

    # ── called every physics tick ────────────────────────────────
    def accumulate(self, yaw: float = 0.0):
        """Read one partial LiDAR frame, rotate to world-frame, append to buffer."""
        parsed = self._parse_frame()
        if parsed is None:
            return
        angles, depths, hits = parsed
        # Body-frame → world-frame
        world_angles = (angles + yaw) % (2 * math.pi)
        self._acc_angles.append(world_angles)
        self._acc_depths.append(depths)
        self._acc_hits.append(hits)

    # ── called once per AIF step in perceive() ───────────────────
    def get_accumulated(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return all accumulated scans merged, then clear the buffer.
        Multi-layer scans are binned par azimuth: on garde le hit le plus
        proche par bin pour éviter de sur-compter les couches verticales."""
        if not self._acc_angles:
            n = self.cfg.num_rays
            return (
                np.linspace(0, 2 * math.pi, n, endpoint=False),
                np.full(n, self.cfg.lidar_max_range),
                np.zeros(n, dtype=bool),
            )
        angles = np.concatenate(self._acc_angles)
        depths = np.concatenate(self._acc_depths)
        hits   = np.concatenate(self._acc_hits)
        self._acc_angles.clear()
        self._acc_depths.clear()
        self._acc_hits.clear()

        # ── Bin par azimuth: un seul rayon par direction (le plus proche) ──
        n = self.cfg.num_rays
        bin_width = 2 * math.pi / n
        bins = (angles / bin_width).astype(int) % n

        out_angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
        out_depths = np.full(n, self.cfg.lidar_max_range)
        out_hits   = np.zeros(n, dtype=bool)

        for i in range(n):
            mask = bins == i
            if not mask.any():
                continue
            b_d = depths[mask]
            b_h = hits[mask]
            if b_h.any():
                hit_d = b_d[b_h]
                out_depths[i] = hit_d.min()  # obstacle le plus proche
                out_hits[i] = True
            else:
                out_depths[i] = b_d.max()  # portée libre la plus lointaine

        self._read_count += 1
        if self._read_count <= 5 or self._read_count % 50 == 0:
            n_hits = int(out_hits.sum())
            pct = n_hits / max(n, 1) * 100
            print(f"  [LiDAR] {self.prim_path} scan#{self._read_count}: "
                  f"raw_rays={len(angles)}  binned={n}  hits={n_hits} ({pct:.0f}%)")
        return out_angles, out_depths, out_hits


# ════════════════════════════════════════════════════════════════
# 10. Drone Agent (ties AIF logic to physical drone)
# ════════════════════════════════════════════════════════════════

class DroneAgent: 

    def __init__(self, drone_id: int, sx: float, sy: float, cfg: SimConfig):
        self.id = drone_id
        self.cfg = cfg
        self.belief = BeliefGrid(cfg)
        self.rng = np.random.default_rng(42 + drone_id * 1000)
        self.trail: List[Tuple[float, float]] = [(sx, sy)]
        self.last_action = "stay"
        self.last_G = 0.0
        self.last_ig = 0.0
        self.total_dist = 0.0
        self._prev_xy = (sx, sy)
        self.backend: Optional["SitlController"] = None
        self.lidar: Optional[LidarReader] = None
        # diagnostic data (populated each step for logging)
        self.lidar_diag: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self.candidates_diag: List[Dict] = []
        self.selected_idx: int = 0
        # anti-stagnation
        self._stag_pos = (sx, sy)
        self._stag_steps = 0
        self._STAG_THRESHOLD = 5  # steps stuck → force random move
        # resilience
        self.active: bool = True   # False = drone hors service (landed)
        self.last_innovation: float = 0.0  # innovation mesurée au dernier step
        # ── Tâche 2 (distributed) : belief fusionnée localement avec voisins ──
        self.local_fused: Optional[BeliefGrid] = None
        # ── Tâche 3 (NS-3) : cache des beliefs reçus avec latence ──
        self.last_received_belief: Dict[int, BeliefGrid] = {}

    def setup_physical(self, backend: "SitlController", lidar: LidarReader):
        self.backend = backend
        self.lidar = lidar

    @property
    def x(self) -> float:
        if self.backend:
            return self.backend.get_position_xy()[0] - self.cfg.origin_x
        return self.trail[-1][0]

    @property
    def y(self) -> float:
        if self.backend:
            return self.backend.get_position_xy()[1] - self.cfg.origin_y
        return self.trail[-1][1]

    def perceive(self):
        """Use accumulated LiDAR scans (full 360°) to update local belief grid.
        Also compute innovation = how much observations differ from predicted."""
        if not self.active or self.lidar is None:
            self.lidar_diag = None
            self.last_innovation = 0.0
            return
        angles, ranges, hits = self.lidar.get_accumulated()
        self.lidar_diag = (angles.copy(), ranges.copy(), hits.copy())

        # ── Compute innovation: mesure de surprise (adapté de AIF_controller) ──
        # Compare les hits LiDAR avec la belief actuelle
        n_compare = 0
        innov_sum = 0.0
        for i in range(len(angles)):
            if not hits[i]:
                continue
            cos_a = math.cos(angles[i])
            sin_a = math.sin(angles[i])
            hx = self.x + float(ranges[i]) * cos_a
            hy = self.y + float(ranges[i]) * sin_a
            gx, gy = self.belief.world_to_grid(hx, hy)
            if self.belief.in_bounds(gx, gy):
                predicted_occ = self.belief.probability[gy, gx]
                # Innovation = |observation(=1 hit) - prediction|
                innov_sum += abs(1.0 - predicted_occ)
                n_compare += 1
        self.last_innovation = innov_sum / max(n_compare, 1)

        self.belief.update_from_lidar(
            self.x, self.y, angles, ranges, hits,
            self.cfg.lidar_max_range, self.cfg.lo_free, self.cfg.lo_occ,
        )

    def accumulate_lidar(self):
        """Called every physics tick to gather a partial LiDAR scan."""
        if not self.active or self.lidar is None or self.backend is None:
            return
        self.lidar.accumulate(self.backend.get_yaw())

    def land(self):
        """Disable this drone (simulate out-of-service / landed)."""
        self.active = False
        self.last_action = "LANDED"
        if self.backend:
            # Set target to ground level at current position
            wx = self.x + self.cfg.origin_x
            wy = self.y + self.cfg.origin_y
            self.backend.set_target(wx, wy, 0.1)
        print(f"  [RESILIENCE] 🛬 Drone {self.id} LANDED (out of service)")

    def fuse_with_neighbors(self, neighbors: List["DroneAgent"],
                             prior_lo: float) -> BeliefGrid:
        """Tâche 2 (distribué) : fusionne self.belief avec celles des voisins
        (les beliefs sont prises dans le cache last_received_belief si la
        latence NS-3 force un délai ; sinon on prend le snapshot courant).

        Le résultat est stocké dans self.local_fused puis renvoyé pour
        que SwarmCoordinator l'utilise comme `plan_belief` à la place du
        fused_belief global (qui reste calculé pour le dashboard)."""
        beliefs = [self.belief]
        for n in neighbors:
            # Si on a un belief en cache (reçu via la queue) pour ce voisin,
            # on l'utilise (= belief retardée par NS-3). Sinon, snapshot direct.
            cached = self.last_received_belief.get(n.id)
            beliefs.append(cached if cached is not None else n.belief)
        self.local_fused = fuse_beliefs_logodds(beliefs, prior_lo)
        return self.local_fused

    def plan(self, others: List[Tuple[float, float]], fused: Optional[BeliefGrid],
             resilience_phase: str = "normal"):
        if not self.active:
            return
        cx, cy = self.x, self.y

        # ── Anti-stagnation: detect if drone hasn’t moved ──
        moved = math.hypot(cx - self._stag_pos[0], cy - self._stag_pos[1])
        if moved < 0.5:
            self._stag_steps += 1
        else:
            self._stag_steps = 0
            self._stag_pos = (cx, cy)

        if self._stag_steps >= self._STAG_THRESHOLD:
            # Force a random cardinal action to escape
            idx = int(self.rng.integers(1, len(ACTIONS)))  # skip “stay”
            name, dx, dy = ACTIONS[idx]
            # Build a minimal diagnostic entry so log_candidates doesn't crash
            self.candidates_diag = [{
                "idx": idx, "name": name,
                "nx": round(cx + dx * self.cfg.step_size, 3),
                "ny": round(cy + dy * self.cfg.step_size, 3),
                "valid": True, "reason": "forced-stag",
                "ig": 0.0, "frontier": 0.0, "move": 1.0, "coll": 0.0, "G": 0.0,
            }]
            self.selected_idx = 0  # index into the 1-element list
            self._stag_steps = 0
            print(f"  [STAG] D{self.id} stuck {self._STAG_THRESHOLD} steps → forced {name}")
        else:
            # ── Dispatch planner : AIF (free energy) ou heuristique frontier-based ──
            if self.cfg.planner == "heuristic":
                planner_fn = select_action_heuristic
            else:
                planner_fn = select_action
            (name, dx, dy), self.candidates_diag, self.selected_idx = planner_fn(
                cx, cy, others, self.belief, fused, self.cfg, self.rng,
                resilience_phase=resilience_phase,
            )
            # Update dashboard-exported diagnostics
            if self.candidates_diag and self.selected_idx < len(self.candidates_diag):
                sel = self.candidates_diag[self.selected_idx]
                self.last_G  = sel.get("G", 0.0)
                self.last_ig = sel.get("ig", 0.0)

        tx = clamp(cx + dx * self.cfg.step_size, 0.5, self.cfg.env_width - 0.5)
        ty = clamp(cy + dy * self.cfg.step_size, 0.5, self.cfg.env_height - 0.5)

        if self.backend:
            self.backend.set_target(
                tx + self.cfg.origin_x,
                ty + self.cfg.origin_y,
                self.cfg.fly_altitude,
            )

        self.total_dist += math.hypot(cx - self._prev_xy[0], cy - self._prev_xy[1])
        self._prev_xy = (cx, cy)
        self.trail.append((round(cx, 2), round(cy, 2)))
        if len(self.trail) > 200:
            self.trail = self.trail[-200:]
        self.last_action = name

    def get_state(self) -> Dict:
        heading = 0.0
        if len(self.trail) >= 2:
            dx = self.trail[-1][0] - self.trail[-2][0]
            dy = self.trail[-1][1] - self.trail[-2][1]
            if dx != 0 or dy != 0:
                heading = math.atan2(dy, dx)
        return {
            "id": self.id,
            "x": round(self.x, 3), "y": round(self.y, 3),
            "heading": round(heading, 3),
            "action": self.last_action,
            "free_energy": round(self.last_G, 4),
            "info_gain": round(self.last_ig, 4),
            "total_distance": round(self.total_dist, 2),
            "local_entropy": round(self.belief.mean_entropy(), 4),
            "trail": self.trail[-60:],
            "active": self.active,
            "innovation": round(self.last_innovation, 4),
        }



class NS3LatencyReader:
    """Lit les latences inter-drone calculées par NS-3 en arrière-plan."""

    # Fichiers que peut produire NS-3 (selon le scenario lancé par 12_ns3_bridge.py)
    WIFI_CSV = "/tmp/ns3_output.csv"
    LTE5G_CSV = "/tmp/drone_latency_ns3.csv"
    LTE5G_METRICS_CSV = "/tmp/drone_5g_metrics.csv"

    def __init__(self, mode: str = "none"):
        self.mode = mode  # "none" | "wifi" | "5g"
        self._warned_missing = False
        # cache pour exposer au dashboard
        self.last_pairs: Dict[Tuple[int, int], Dict[str, float]] = {}

    @property
    def active(self) -> bool:
        return self.mode in ("wifi", "5g")

    def _csv_path(self) -> str:
        if self.mode == "wifi":
            return self.WIFI_CSV
        if self.mode == "5g":
            # essayer drone_latency_ns3.csv d'abord, puis le format métrique
            if os.path.exists(self.LTE5G_CSV):
                return self.LTE5G_CSV
            return self.LTE5G_METRICS_CSV
        return ""

    def read_latencies(self) -> Dict[Tuple[int, int], float]:
        """Retourne dict {(i,j): latency_ms} avec i<j. Vide si NS-3 indispo."""
        if not self.active:
            return {}
        path = self._csv_path()
        if not path or not os.path.exists(path):
            if not self._warned_missing:
                print(f"  [NS-3] ⚠ Fichier latences absent ({path}) "
                      f"→ utilisation latence=0 (dégradation gracieuse)")
                self._warned_missing = True
            return {}
        import csv as _csv
        latencies: Dict[Tuple[int, int], float] = {}
        pairs_full: Dict[Tuple[int, int], Dict[str, float]] = {}
        try:
            with open(path) as f:
                # auto-detect dialect : drone_i/drone_j ou drone_a/drone_b
                reader = _csv.DictReader(f)
                for row in reader:
                    try:
                        i_key = "drone_i" if "drone_i" in row else "drone_a"
                        j_key = "drone_j" if "drone_j" in row else "drone_b"
                        i = int(row[i_key])
                        j = int(row[j_key])
                        lat = float(row.get("latency_ms", 0.0))
                        jitter = float(row.get("jitter_ms", 0.0)) if "jitter_ms" in row else 0.0
                        rx = int(row.get("rx_packets", 0)) if "rx_packets" in row else 0
                        key = (min(i, j), max(i, j))
                        latencies[key] = lat
                        pairs_full[key] = {
                            "latency_ms": round(lat, 3),
                            "jitter_ms": round(jitter, 3),
                            "rx_packets": rx,
                        }
                    except (ValueError, KeyError, TypeError):
                        continue
        except (IOError, OSError):
            return {}
        self.last_pairs = pairs_full
        return latencies

    def pair_latency(self, i: int, j: int, fallback: float = 0.0) -> float:
        """Retourne la latence pour la paire (i,j) ou fallback si absente."""
        key = (min(i, j), max(i, j))
        return self.last_pairs.get(key, {}).get("latency_ms", fallback)


# ════════════════════════════════════════════════════════════════
# 10c. Message Queue avec délais (Tâche 3.4)
# ════════════════════════════════════════════════════════════════
#
# File FIFO horodatée : un message envoyé au step t avec latency_ms est
# délivré au step t + ceil(latency_ms / step_dt_ms) - 1.
#   - step_dt_ms = sim_steps_per_aif * physics_dt_s * 1000
#     (≈ 1000 ms pour 60 ticks à 60 Hz)
#   - latency_ms None ou < 0 → drop (lien coupé)
# Les beliefs en attente restent dans le cache last_received_belief des
# drones, qui utilisent la dernière belief reçue pour la fusion locale.
# ════════════════════════════════════════════════════════════════


class MessageQueue:
    """File d'attente FIFO horodatée pour les broadcasts de beliefs.

    Un message envoyé au step t avec latency_ms arrive seulement au
    step t + ceil(latency_ms / step_dt_ms). Si latency_ms est None ou <0,
    le message est droppé (compté pour le dashboard).
    """

    def __init__(self):
        # entries : list of (arrival_step, src, dst, payload)
        self._pending: List[Tuple[int, int, int, Any]] = []
        self.dropped: int = 0       # cumul des messages droppés
        self.sent: int = 0          # cumul des envois acceptés
        self.delivered: int = 0     # cumul livrés

    def send(self, src: int, dst: int, payload: Any,
             current_step: int, latency_ms: Optional[float],
             step_dt_ms: float):
        if latency_ms is None or latency_ms < 0:
            self.dropped += 1
            return
        # 0 ms → délivré dès ce step (équivalent à pas de réseau)
        steps_delay = max(0, math.ceil(latency_ms / max(step_dt_ms, 1e-3)))
        arrival = current_step + steps_delay
        self._pending.append((arrival, src, dst, payload))
        self.sent += 1

    def deliver(self, current_step: int) -> List[Tuple[int, int, Any]]:
        """Retourne tous les messages dont arrival <= current_step et les
        retire de la queue. Liste de (src, dst, payload)."""
        ready: List[Tuple[int, int, Any]] = []
        remaining: List[Tuple[int, int, int, Any]] = []
        for entry in self._pending:
            arr, src, dst, payload = entry
            if arr <= current_step:
                ready.append((src, dst, payload))
            else:
                remaining.append(entry)
        self._pending = remaining
        self.delivered += len(ready)
        return ready

    @property
    def size(self) -> int:
        return len(self._pending)

    def stats(self) -> Dict[str, int]:
        return {
            "queue_size": self.size,
            "sent": self.sent,
            "delivered": self.delivered,
            "dropped": self.dropped,
        }


# ════════════════════════════════════════════════════════════════
# 11. Swarm Coordinator
# ════════════════════════════════════════════════════════════════

class SwarmCoordinator:
    """Coordonne perception/fusion/planification. Selon `cfg.arch`,
    la fusion peut être centralisée (un seul fused global) ou distribuée
    (chaque drone fusionne avec ses voisins dans un rayon).
    Intègre les latences NS-3 (Tâche 3) et les cuts de liens (Tâche 4)."""

    # sentinelles pour les broadcasts (utilisés dans MessageQueue)
    CLOUD_SRC = -99
    CLOUD_DST_ALL = -1

    def __init__(self, agents: List[DroneAgent], cfg: SimConfig,
                 diag_logger: Optional["DiagnosticLogger"] = None):
        self.agents = agents
        self.cfg = cfg
        self.prior_lo = logit(cfg.prior_occupancy)
        self.fused_belief = agents[0].belief.copy()
        self.step_count = 0
        self.history: List[Dict] = []
        self.diag = diag_logger
        # ── Resilience state ──
        self.resilience = ResilienceState()
        self.innov_ema = 0.0
        self.innov_var = 0.0
        # ── Tâche 3 : NS-3 + message queue ──
        self.ns3 = NS3LatencyReader(cfg.ns3_mode)
        self.msg_queue = MessageQueue()
        # belief reçue du cloud (centralisé) après latence
        self._cloud_belief: Optional[BeliefGrid] = None
        # ── Tâche 4 : état des cuts ──
        self.cloud_link_active: bool = True
        self.cut_pairs: Set[frozenset] = set()
        self.all_drone_links_cut: bool = False

        self.interior_area_cells: int = cfg.interior_area_cells()

    @property
    def active_agents(self) -> List[DroneAgent]:
        return [a for a in self.agents if a.active]

    def kill_drone(self, drone_id: int):
        """Disable a drone and trigger resilience recovery."""
        for a in self.agents:
            if a.id == drone_id and a.active:
                a.land()
                self.resilience.trigger(self.step_count, f"drone_{drone_id}_lost")
                return

    # ── Tâche 2 / 4 : architecture effective ──
    def effective_arch(self) -> str:
        """Si le cloud est coupé et qu'on était centralized → fallback distribué."""
        if self.cfg.arch == "centralized" and not self.cloud_link_active:
            return "distributed"
        return self.cfg.arch

    def _is_link_cut(self, a: int, b: int) -> bool:
        if self.all_drone_links_cut:
            return True
        return frozenset({a, b}) in self.cut_pairs

    def _step_dt_ms(self) -> float:
        return self.cfg.sim_steps_per_aif * self.cfg.physics_dt_s * 1000.0

    # ── Tâche 4 : appliquer les cuts programmés à temps ──
    def _apply_scheduled_cuts(self):
        if (self.cfg.cut_cloud_at_step >= 0 and
                self.step_count == self.cfg.cut_cloud_at_step and
                self.cloud_link_active):
            self.cloud_link_active = False
            self.resilience.trigger(self.step_count, "cloud_link_lost")
            print(f"\n  [CUT] ☁ Cloud link CUT at step {self.step_count}")
            if self.cfg.arch == "centralized":
                print(f"  [CUT] Auto-failover centralized → distributed")

        if (self.cfg.cut_drone_link and
                self.cfg.cut_drone_link_at_step >= 0 and
                self.step_count == self.cfg.cut_drone_link_at_step):
            spec = self.cfg.cut_drone_link.strip()
            if spec == "all":
                if not self.all_drone_links_cut:
                    self.all_drone_links_cut = True
                    self.resilience.trigger(self.step_count, "all_drone_links_lost")
                    print(f"\n  [CUT] ✂ ALL drone↔drone links CUT at step {self.step_count}")
            elif "-" in spec:
                try:
                    a, b = spec.split("-")
                    i, j = int(a), int(b)
                    pair = frozenset({i, j})
                    if pair not in self.cut_pairs:
                        self.cut_pairs.add(pair)
                        self.resilience.trigger(self.step_count, f"drone_link_{i}-{j}_lost")
                        print(f"\n  [CUT] ✂ Drone link {i}↔{j} CUT at step {self.step_count}")
                except ValueError:
                    print(f"  [WARN] Invalid --cut-drone-link spec: {spec!r}")

    # ── Tâche 3 : envoi des beliefs dans la queue ──
    def _broadcast_beliefs(self, active: List[DroneAgent]):
        step_dt_ms = self._step_dt_ms()
        eff = self.effective_arch()

        if eff == "centralized":
            if not self.cloud_link_active:
                return
            # Régression-free : sans NS-3, pas de latence cloud → bypass de la
            # queue, _cloud_belief = fused_belief instantanément. Identique
            # au comportement avant Tâche 3.
            if not self.ns3.active:
                self._cloud_belief = self.fused_belief
                return
            # Avec NS-3 actif : latence = cloud_latency_ms + médiane(latences NS-3)
            median_pair_lat = 0.0
            if self.ns3.last_pairs:
                lats = sorted(v["latency_ms"] for v in self.ns3.last_pairs.values())
                if lats:
                    median_pair_lat = lats[len(lats) // 2]
            total_lat = self.cfg.cloud_latency_ms + median_pair_lat
            self.msg_queue.send(
                src=self.CLOUD_SRC, dst=self.CLOUD_DST_ALL,
                payload=self.fused_belief.copy(),
                current_step=self.step_count, latency_ms=total_lat,
                step_dt_ms=step_dt_ms,
            )
        else:
            # distribué : chaque drone broadcast à ses voisins dans neighbor_radius_m.
            # On utilise la queue même sans NS-3 pour gérer proprement les cuts
            # (drop instantané sur lien coupé) et alimenter last_received_belief.
            R = self.cfg.neighbor_radius_m
            for src in active:
                for dst in active:
                    if dst.id == src.id:
                        continue
                    if math.hypot(src.x - dst.x, src.y - dst.y) > R:
                        continue
                    # drop si lien coupé
                    if self._is_link_cut(src.id, dst.id):
                        self.msg_queue.send(src.id, dst.id, src.belief,
                                            self.step_count, None, step_dt_ms)
                        continue
                    lat = self.ns3.pair_latency(src.id, dst.id, fallback=0.0) \
                        if self.ns3.active else 0.0
                    self.msg_queue.send(src.id, dst.id, src.belief.copy(),
                                        self.step_count, lat, step_dt_ms)

    def _deliver_and_cache(self):
        """Délivre les messages prêts et met à jour les caches des drones."""
        delivered = self.msg_queue.deliver(self.step_count)
        agents_by_id = {a.id: a for a in self.agents}
        for src, dst, payload in delivered:
            if src == self.CLOUD_SRC and dst == self.CLOUD_DST_ALL:
                # message du cloud : on stocke la dernière fused reçue
                self._cloud_belief = payload
            elif dst in agents_by_id:
                agents_by_id[dst].last_received_belief[src] = payload

    def step(self):
        h_before = self.fused_belief.mean_entropy()

        if self.diag:
            self.diag.log_step_header(self.step_count + 1)

        # ── Tâche 4 : déclencher les cuts si on a atteint leur step ──
        self._apply_scheduled_cuts()

        # ── 1) Perception ──
        for a in self.agents:
            a.perceive()
            if self.diag and a.active and a.lidar_diag is not None:
                angles, ranges, hits = a.lidar_diag
                self.diag.log_lidar(a.id, a.x, a.y, angles, ranges, hits)

        # ── 2) Innovation (EMA + spike) ──
        active = self.active_agents
        if active:
            innovations = [a.last_innovation for a in active]
            innov_mean = sum(innovations) / len(innovations)
        else:
            innov_mean = 0.0
        err = innov_mean - self.innov_ema
        self.innov_ema += self.cfg.ema_alpha * err
        self.innov_var += self.cfg.ema_alpha * ((err * err) - self.innov_var)
        sigma = math.sqrt(max(self.innov_var, 1e-8))
        spike = innov_mean > (self.innov_ema + self.cfg.k_sigma * sigma)
        if not self.resilience.stress_active and spike and self.step_count > 5:
            self.resilience.trigger(self.step_count, "innovation_spike")

        # ── 3) NS-3 : relire les latences inter-drones ──
        if self.ns3.active:
            self.ns3.read_latencies()

        # ── 4) Fused belief global (passive observer pour le dashboard) ──
        if active:
            self.fused_belief = fuse_beliefs_logodds(
                [a.belief for a in active], self.prior_lo,
            )
        # else: keep last fused belief
        h_after = self.fused_belief.mean_entropy()

        # ── 5) Resilience update ──
        self.resilience.update(self.step_count, h_after, innov_mean, self.cfg)
        phase = self.resilience.phase(self.step_count, self.cfg)

        # ── 6) Broadcast beliefs via MessageQueue + délivrer les arrivées ──
        self._broadcast_beliefs(active)
        self._deliver_and_cache()

        # ── 7) Planning : chaque drone choisit sa plan_belief selon l'arch ──
        eff_arch = self.effective_arch()
        for a in self.agents:
            if not a.active:
                continue
            others = [(o.x, o.y) for o in active if o.id != a.id]

            if eff_arch == "distributed":
                # voisins actifs dans le rayon, hors liens coupés
                neighbors = [
                    o for o in active
                    if o.id != a.id
                    and not self._is_link_cut(a.id, o.id)
                    and math.hypot(a.x - o.x, a.y - o.y) <= self.cfg.neighbor_radius_m
                ]
                plan_belief = a.fuse_with_neighbors(neighbors, self.prior_lo)
            else:
                # centralisé : on utilise la fused reçue du cloud si dispo,
                # sinon la fused actuelle (warmup avant 1ère arrivée).
                plan_belief = self._cloud_belief if self._cloud_belief is not None \
                              else self.fused_belief

            a.plan(others, plan_belief, resilience_phase=phase)
            if self.diag:
                self.diag.log_candidates(a.id, a.x, a.y, others,
                                         a.candidates_diag, a.selected_idx)
                self.diag.log_belief(a.id, a.belief, plan_belief)

        # ── 8) Historique ──
        self.step_count += 1
        drone_positions = [(a.x, a.y) for a in active]
        bounds_grid = self.cfg.factory_bounds_grid()

        coverage_pct = self.fused_belief.interior_exploration_ratio(
            drone_positions, self.cfg.occ_threshold,
            bounds_grid=bounds_grid,
            interior_area_cells=self.interior_area_cells,
        ) * 100 if drone_positions else 0.0

        # Métrique secondaire : % de la zone intérieure aux murs
        # détectés. Tend vers ~100 % quand le hangar est mappé.
        interior_pct = self.fused_belief.observed_inside_walls_ratio(
            self.cfg.occ_threshold, bounds_grid=bounds_grid,
        ) * 100 if drone_positions else 0.0
        q = self.msg_queue.stats()
        self.history.append({
            "step": self.step_count,
            "mean_entropy": round(h_after, 4),
            "exploration_pct": round(coverage_pct, 2),
            "exploration_pct_interior": round(interior_pct, 2),
            "exploration_pct_raw": round(self.fused_belief.exploration_ratio() * 100, 2),
            "step_info_gain": round(max(0.0, h_before - h_after), 4),
            "free_energies": [round(a.last_G, 4) for a in self.agents],
            "info_gains": [round(a.last_ig, 4) for a in self.agents],
            "innovation_mean": round(innov_mean, 4),
            "innovation_ema": round(self.innov_ema, 4),
            "resilience_phase": phase,
            "active_drones": len(active),
            # ── Réseau / arch (Tâches 2-3-4) ──
            "arch_effective": eff_arch,
            "cloud_link_active": self.cloud_link_active,
            "queue_size": q["queue_size"],
            "msg_dropped": q["dropped"],
            "msg_sent": q["sent"],
            "msg_delivered": q["delivered"],
        })

        if self.diag:
            self.diag.log_step_summary(self.step_count, self.agents,
                                       self.fused_belief)
            if phase != "normal":
                self.diag._w(f"    [RESILIENCE] phase={phase} "
                             f"cause={self.resilience.cause} "
                             f"innov_mean={innov_mean:.4f} "
                             f"innov_ema={self.innov_ema:.4f}\n")

    def get_full_state(self, obstacles: List[Dict]) -> Dict:
        eb = self.fused_belief.effective_bounds(self.cfg.occ_threshold)
        q = self.msg_queue.stats()
        # serialiser les pairs NS-3 pour le dashboard /api/ns3
        ns3_pairs = []
        for (i, j), info in sorted(self.ns3.last_pairs.items()):
            ns3_pairs.append({
                "a": i, "b": j,
                "latency_ms": info.get("latency_ms", 0.0),
                "jitter_ms": info.get("jitter_ms", 0.0),
                "rx_packets": info.get("rx_packets", 0),
            })
        cut_pairs_list = ["-".join(str(x) for x in sorted(p)) for p in self.cut_pairs]
        return {
            "step": self.step_count,
            "timestamp": time.time(),
            "environment": {
                "width": self.cfg.env_width, "height": self.cfg.env_height,
                "grid_resolution": self.cfg.grid_resolution,
                "grid_width": self.cfg.grid_width, "grid_height": self.cfg.grid_height,
                "obstacles": obstacles,
                "effective_bounds": {
                    "x1": eb[0], "y1": eb[1], "x2": eb[2], "y2": eb[3]
                },
            },
            "drones": [a.get_state() for a in self.agents],
            "fused_belief": self.fused_belief.to_list(),
            "metrics": self.history[-1] if self.history else {},
            "resilience": self.resilience.to_dict(),
            # ── Tâches 1-2-3 exposés au dashboard ──
            "planner": self.cfg.planner,
            "arch_configured": self.cfg.arch,
            "arch_effective": self.effective_arch(),
            "ns3_mode": self.cfg.ns3_mode,
            "neighbor_radius_m": self.cfg.neighbor_radius_m,
            "network": {
                "ns3_mode": self.cfg.ns3_mode,
                "cloud_link_active": self.cloud_link_active,
                "all_drone_links_cut": self.all_drone_links_cut,
                "cut_pairs": cut_pairs_list,
                "queue_size": q["queue_size"],
                "msg_sent": q["sent"],
                "msg_delivered": q["delivered"],
                "msg_dropped": q["dropped"],
                "ns3_pairs": ns3_pairs,
            },
        }


# ════════════════════════════════════════════════════════════════
# 12. Data Logger
# ════════════════════════════════════════════════════════════════

class DataLogger:
    def __init__(self, output_dir: str = "/tmp"):
        self.state_path = os.path.join(output_dir, "aif_state.json")
        self.history_path = os.path.join(output_dir, "aif_history.json")
        self._write(self.history_path, [])

    def log(self, state: Dict, history: List[Dict]):
        self._write(self.state_path, state)
        self._write(self.history_path, history)

    @staticmethod
    def _write(path: str, data: Any):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        os.replace(tmp, path)


# ════════════════════════════════════════════════════════════════
# 12b. Diagnostic Logger  — detailed per-drone log for debugging
# ════════════════════════════════════════════════════════════════

class DiagnosticLogger:
    """Writes a human-readable diagnostic log to help identify
    exploration, obstacle-detection, collision-avoidance and
    planning issues."""

    def __init__(self, output_dir: str, cfg: SimConfig):
        self.cfg = cfg
        self.log_path = os.path.join(output_dir, "aif_diagnostic.log")
        with open(self.log_path, "w") as f:
            f.write("=" * 90 + "\n")
            f.write("  AIF DIAGNOSTIC LOG\n")
            f.write(f"  Started : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Drones  : {cfg.num_drones}\n")
            f.write(f"  Env     : {cfg.env_width} x {cfg.env_height} m  "
                    f"(grid {cfg.grid_width} x {cfg.grid_height}, res {cfg.grid_resolution} m)\n")
            f.write(f"  Weights : epistemic={cfg.w_epistemic}  pragmatic={cfg.w_pragmatic}  "
                    f"movement={cfg.w_movement}  collision={cfg.w_collision}\n")
            f.write(f"  Softmax : temp={cfg.softmax_temp}  fusion_mix={cfg.fusion_mix}\n")
            f.write(f"  Occ thr : {cfg.occ_threshold}   step_size={cfg.step_size} m\n")
            f.write(f"  LiDAR   : rays={cfg.num_rays}  range=[{cfg.lidar_min_range}, "
                    f"{cfg.lidar_max_range}] m  hz={cfg.lidar_hz}\n")
            f.write("=" * 90 + "\n\n")
        print(f"[INFO] Diagnostic log → {self.log_path}")

    # ── helpers ──────────────────────────────────────────────────
    def _w(self, text: str):
        with open(self.log_path, "a") as f:
            f.write(text)

    # ── step header ──────────────────────────────────────────────
    def log_step_header(self, step: int):
        self._w(f"\n{'━' * 90}\n")
        self._w(f"  STEP {step}   ({time.strftime('%H:%M:%S')})\n")
        self._w(f"{'━' * 90}\n")

    # ── LiDAR diagnostics per drone ──────────────────────────────
    def log_lidar(self, drone_id: int, pos_x: float, pos_y: float,
                  angles: np.ndarray, ranges: np.ndarray, hits: np.ndarray):
        n_total = len(hits)
        n_hits = int(hits.sum())
        pct = n_hits / max(n_total, 1) * 100

        self._w(f"\n  ┌─ DRONE {drone_id}  LiDAR  pos=({pos_x:.2f}, {pos_y:.2f})\n")
        self._w(f"  │  Rays total={n_total}  hits={n_hits} ({pct:.1f}%)\n")

        if n_hits > 0:
            hr = ranges[hits]
            ha = angles[hits]
            self._w(f"  │  Hit ranges : min={hr.min():.2f} m  max={hr.max():.2f} m  "
                    f"mean={hr.mean():.2f} m\n")
            # 5 closest obstacles
            order = np.argsort(hr)[:5]
            self._w(f"  │  Closest obstacles:\n")
            for k in order:
                adeg = math.degrees(ha[k]) % 360
                self._w(f"  │    angle={adeg:6.1f}°  dist={hr[k]:.2f} m\n")
            # directional summary (4 quadrants)
            for label, lo, hi in [("FRONT 315-45°", 315, 45),
                                   ("RIGHT 45-135°", 45, 135),
                                   ("BACK 135-225°", 135, 225),
                                   ("LEFT 225-315°", 225, 315)]:
                adeg = np.degrees(ha) % 360
                if lo > hi:  # wraps around 0
                    mask = (adeg >= lo) | (adeg < hi)
                else:
                    mask = (adeg >= lo) & (adeg < hi)
                cnt = int(mask.sum())
                if cnt > 0:
                    mn = hr[mask].min()
                    self._w(f"  │    {label:16s}  hits={cnt:3d}  closest={mn:.2f} m\n")
                else:
                    self._w(f"  │    {label:16s}  hits=  0\n")
        else:
            self._w(f"  │  ⚠  NO obstacles detected by LiDAR\n")
        self._w(f"  └{'─' * 60}\n")

    # ── candidate actions table per drone ────────────────────────
    def log_candidates(self, drone_id: int, pos_x: float, pos_y: float,
                       others: List[Tuple[float, float]],
                       cand: List[Dict], selected_idx: int):
        self._w(f"\n  ┌─ DRONE {drone_id}  Action Selection  pos=({pos_x:.2f}, {pos_y:.2f})\n")

        # inter-drone distances
        if others:
            self._w(f"  │  Other drones:\n")
            for ox, oy in others:
                d = math.hypot(pos_x - ox, pos_y - oy)
                tag = " ⚠ DANGER" if d < 1.5 else " ⚠ CLOSE" if d < 3.0 else ""
                self._w(f"  │    ({ox:.2f}, {oy:.2f})  dist={d:.2f} m{tag}\n")

        # header
        self._w(f"  │\n")
        self._w(f"  │  {'#':>2} {'Act':<5} {'Valid':<6} {'Reason':<14} "
                f"{'nx':>6} {'ny':>6}  "
                f"{'IG':>8} {'Front':>7} {'Move':>5} {'Coll':>7} "
                f"{'G_total':>9}\n")
        self._w(f"  │  {'-' * 88}\n")

        for c in cand:
            sel = " ◄" if c["idx"] == selected_idx else ""
            self._w(f"  │  {c['idx']:2d} {c['name']:<5} "
                    f"{str(c['valid']):<6} {c['reason']:<14} "
                    f"{c['nx']:6.2f} {c['ny']:6.2f}  "
                    f"{c.get('ig', 0):8.3f} {c.get('frontier', 0):7.3f} "
                    f"{c.get('move', 0):5.2f} {c.get('coll', 0):7.3f} "
                    f"{c.get('G', 9999):9.3f}{sel}\n")

        sel_c = cand[selected_idx]
        self._w(f"  │\n")
        self._w(f"  │  ➜ SELECTED: {sel_c['name']}  target=({sel_c['nx']:.2f}, {sel_c['ny']:.2f})  "
                f"G={sel_c.get('G', 0):.4f}\n")
        self._w(f"  └{'─' * 60}\n")

    # ── belief grid stats per drone ──────────────────────────────
    def log_belief(self, drone_id: int, belief: BeliefGrid,
                   fused: Optional[BeliefGrid] = None):
        total = belief.width * belief.height
        occ = int((belief.probability >= self.cfg.occ_threshold).sum())
        free = int((belief.probability < 0.3).sum())
        unc = total - occ - free

        self._w(f"\n  ┌─ DRONE {drone_id}  Belief Grid\n")
        self._w(f"  │  Entropy  : {belief.mean_entropy():.4f}\n")
        self._w(f"  │  Coverage : {belief.exploration_ratio() * 100:.1f}%\n")
        self._w(f"  │  Cells    : occupied={occ} ({occ/total*100:.1f}%)  "
                f"free={free} ({free/total*100:.1f}%)  "
                f"uncertain={unc} ({unc/total*100:.1f}%)\n")
        if fused is not None:
            f_occ = int((fused.probability >= self.cfg.occ_threshold).sum())
            self._w(f"  │  Fused    : entropy={fused.mean_entropy():.4f}  "
                    f"coverage={fused.exploration_ratio() * 100:.1f}%  "
                    f"occ_cells={f_occ}\n")
        self._w(f"  └{'─' * 60}\n")

    # ── step summary (inter-drone + wall proximity) ──────────────
    def log_step_summary(self, step: int, agents, fused: BeliefGrid):
        self._w(f"\n  ── Step {step} Summary ──\n")
        # positions
        for a in agents:
            arrived = ""
            if a.backend is not None:
                arrived = "  arrived=True" if a.backend.is_at_target() else "  arrived=False"
            self._w(f"    D{a.id} pos=({a.x:.2f}, {a.y:.2f})  "
                    f"action={a.last_action}  dist_total={a.total_dist:.2f} m{arrived}\n")

        # pairwise distances
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                d = math.hypot(agents[i].x - agents[j].x, agents[i].y - agents[j].y)
                tag = " ⚠ COLLISION-RISK" if d < 1.5 else " ⚠ CLOSE" if d < 3.0 else ""
                self._w(f"    D{agents[i].id}↔D{agents[j].id} = {d:.2f} m{tag}\n")

        # wall proximity warnings
        for a in agents:
            warns = []
            if a.x < 1.0:
                warns.append(f"LEFT wall x={a.x:.2f}")
            if a.x > self.cfg.env_width - 1.0:
                warns.append(f"RIGHT wall x={a.x:.2f}")
            if a.y < 1.0:
                warns.append(f"BOTTOM wall y={a.y:.2f}")
            if a.y > self.cfg.env_height - 1.0:
                warns.append(f"TOP wall y={a.y:.2f}")
            if warns:
                self._w(f"    ⚠ D{a.id} WALL: {', '.join(warns)}\n")

        # overlap detection (same grid cell)
        cells = {}
        for a in agents:
            gx, gy = fused.world_to_grid(a.x, a.y)
            key = (gx, gy)
            cells.setdefault(key, []).append(a.id)
        for key, ids in cells.items():
            if len(ids) > 1:
                self._w(f"    ⚠ OVERLAP: drones {ids} in same grid cell {key}\n")

        self._w(f"    Fused coverage={fused.exploration_ratio() * 100:.1f}%  "
                f"entropy={fused.mean_entropy():.4f}\n")


# ════════════════════════════════════════════════════════════════
# 13. Main — Isaac Sim Setup & AIF Loop
# ════════════════════════════════════════════════════════════════

def _import_ns3_bridge():
    """Import du module scripts/12_ns3_bridge.py (nom commence par '12_'
    donc 'import' direct impossible → importlib)."""
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    bridge_path = os.path.join(here, "12_ns3_bridge.py")
    if not os.path.isfile(bridge_path):
        return None
    spec = importlib.util.spec_from_file_location("ns3_bridge", bridge_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_drone_positions(agents, cfg, path: str = "/tmp/drone_positions.csv"):
    """Écrit le CSV des positions courantes (lu par NS-3 — cf. 12_ns3_bridge.py).

    Le scénario drone-wifi-scenario.cc / drone-5g-nr-scenario.cc saute la
    1ʳᵉ ligne (`// Skip header`) → on DOIT inclure un en-tête `drone_id,x,y,z`
    sinon NS-3 perd le drone 0 et toutes les paires (0-i) sont calculées
    avec une position figée à l'origine."""
    try:
        with open(path, "w") as f:
            f.write("drone_id,x,y,z\n")
            for a in agents:
                if not a.active:
                    continue
                # Si le backend SITL est prêt, prendre la pose réelle.
                # Sinon (avant 1er state MAVLink) → fallback sur la pose
                # initiale connue de l'agent (sert au tout 1er write avant
                # le lancement de NS-3).
                if a.backend is not None:
                    p = a.backend.get_position()
                    x, y, z = float(p[0]), float(p[1]), float(p[2])
                else:
                    x, y, z = float(a.x), float(a.y), 2.0
                f.write(f"{a.id},{x:.4f},{y:.4f},{z:.4f}\n")
    except OSError:
        pass


def parse_args() -> SimConfig:
    p = argparse.ArgumentParser(description="Active Inference drones — Isaac Sim (real)")
    p.add_argument("--num-drones", type=int, default=3)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--env-width", type=float, default=30.0)
    p.add_argument("--env-height", type=float, default=20.0)
    # ── resilience demo events ──
    p.add_argument("--kill-drone-at-step", type=int, default=-1,
                   help="Step at which to land (disable) drone 0 to test resilience")
    # ── Tâche 1 : planner ──
    p.add_argument("--planner", choices=["aif", "heuristic"], default="aif",
                   help="Action selection strategy (default: aif)")
    # ── Tâche 2 : architecture ──
    p.add_argument("--arch", choices=["centralized", "distributed"], default="centralized",
                   help="Belief fusion architecture (default: centralized)")
    p.add_argument("--neighbor-radius-m", type=float, default=10.0,
                   help="Neighbor radius for distributed fusion (m)")
    # ── Tâche 3 : NS-3 ──
    p.add_argument("--ns3", choices=["none", "wifi", "5g"], default="wifi",
                   help="NS-3 network simulation mode (default: wifi)")
    p.add_argument("--cloud-latency-ms", type=float, default=20.0,
                   help="Extra latency for cloud link (centralized only)")
    p.add_argument("--ns3-sim-time", type=int, default=600,
                   help="NS-3 simulation duration (seconds)")
    # ── Tâche 4 : cuts ──
    p.add_argument("--cut-cloud-at-step", type=int, default=-1,
                   help="Step at which cloud link is cut (auto cent→dist)")
    p.add_argument("--cut-drone-link", default="",
                   help="Drone link to cut: 'i-j' (e.g. '0-1') or 'all'")
    p.add_argument("--cut-drone-link-at-step", type=int, default=-1,
                   help="Step at which drone link is cut")
    # ── Tâche 5 : run capture ──
    p.add_argument("--run-tag", default="default",
                   help="Tag suffix for logs/runs/run_<timestamp>_<tag>/")
    p.add_argument("--runs-dir", default="",
                   help="Override base directory for run artifacts (default: <workspace>/logs/runs)")
    a = p.parse_args()
    cfg = SimConfig(
        num_drones=a.num_drones, headless=a.headless,
        max_steps=a.max_steps, env_width=a.env_width, env_height=a.env_height,
        planner=a.planner, arch=a.arch, neighbor_radius_m=a.neighbor_radius_m,
        ns3_mode=a.ns3, cloud_latency_ms=a.cloud_latency_ms,
        ns3_sim_time=a.ns3_sim_time,
        cut_cloud_at_step=a.cut_cloud_at_step,
        cut_drone_link=a.cut_drone_link,
        cut_drone_link_at_step=a.cut_drone_link_at_step,
        run_tag=a.run_tag, runs_dir=a.runs_dir,
    )
    cfg._kill_drone_at = a.kill_drone_at_step
    return cfg


def create_sim_app(cfg: SimConfig):
    from isaacsim import SimulationApp
    app_cfg = {
        "headless": cfg.headless,
        "extra_args": ["--/rtx/verifyDriverVersion/enabled=false"],
    }
    if not cfg.headless:
        app_cfg["width"] = 1280
        app_cfg["height"] = 720
    return SimulationApp(app_cfg)


def setup_world():
    from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
    pif = PegasusInterface()
    pif.initialize_world()
    world = pif.world
    world.scene.add_default_ground_plane()
    factory_bounds = _load_factory_environment()
    _add_scene_lighting()
    return world, factory_bounds


def _load_factory_environment():
    """Load one forced factory USD path (no path search/fallback).
    Returns the XY bounding box (min_x, min_y, max_x, max_y) in metres."""
    try:
        from isaacsim.core.utils.stage import add_reference_to_stage
    except ImportError:
        from omni.isaac.core.utils.stage import add_reference_to_stage

    import omni.usd
    from pxr import UsdGeom, Gf

    forced_usd = os.getenv(
        "AIF_FACTORY_USD",
        "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
    ).strip()

    stage = omni.usd.get_context().get_stage()
    print(f"[INFO] Factory USD forced path: {forced_usd}")

    try:
        add_reference_to_stage(usd_path=forced_usd, prim_path="/World/Factory")
    except Exception as e:
        raise RuntimeError(f"Failed to load forced factory USD: {forced_usd} ({e})") from e

    # Verify that the prim actually resolved to scene content.
    prim = stage.GetPrimAtPath("/World/Factory")
    if not (prim.IsValid() and prim.GetChildren()):
        stage.RemovePrim("/World/Factory")
        raise RuntimeError(f"Forced factory USD is unreachable or empty: {forced_usd}")

    # ── Compute XY bounding box of the factory ──
    bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(prim)
    rng = bbox.GetRange()
    lo = rng.GetMin()
    hi = rng.GetMax()
    print(f"[INFO] Factory bounding box: "
          f"X=[{lo[0]:.2f}, {hi[0]:.2f}]  Y=[{lo[1]:.2f}, {hi[1]:.2f}]  "
          f"Z=[{lo[2]:.2f}, {hi[2]:.2f}]")
    print(f"[INFO] Factory environment loaded OK: {forced_usd}")
    return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))


def _add_scene_lighting():
    """Add dome + distant lights so the scene is well-lit."""
    from pxr import Gf, UsdGeom, UsdLux
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    stage.DefinePrim("/World/Lights", "Xform")

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(1000.0)
    dome.CreateColorAttr(Gf.Vec3f(0.85, 0.9, 1.0))  # slightly blue sky

    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(3000.0)
    sun.CreateAngleAttr(1.0)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))  # warm sunlight
    xf = UsdGeom.Xformable(sun.GetPrim())
    xf.ClearXformOpOrder()
    # Aim the sun downward at an angle
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 30.0, 0.0))

    print("[INFO] Scene lighting added (dome + sun)")


def setup_viewport_camera(cfg: SimConfig):

    if cfg.headless:
        return

    cx = cfg.env_width  / 2  
    cy = cfg.env_height / 2  
    tz = cfg.fly_altitude   

    cam_x = cx
    cam_y = cy - 18.0
    cam_z = 25.0

    try:
        import omni.usd
        from pxr import Gf, UsdGeom
        import omni.kit.viewport.utility as vp_utils

        viewport = vp_utils.get_active_viewport()
        if viewport is None:
            print("[WARN] setup_viewport_camera: no active viewport found, skipping.")
            return


        stage = omni.usd.get_context().get_stage()
        persp_prim = stage.GetPrimAtPath("/OmniverseKit_Persp")
        if persp_prim and persp_prim.IsValid():
            UsdGeom.Camera(persp_prim).CreateClippingRangeAttr(Gf.Vec2f(0.01, 500.0))

        try:
            from omni.kit.viewport.utility.camera_state import ViewportCameraState
            cs = ViewportCameraState(viewport)
            cs.set_position_world(Gf.Vec3d(cam_x, cam_y, cam_z), True)
            cs.set_target_world(Gf.Vec3d(cx, cy, tz), True)
            print(
                f"[INFO] Overview camera (ViewportCameraState) → "
                f"eye=({cam_x:.1f}, {cam_y:.1f}, {cam_z:.1f})  "
                f"target=({cx:.1f}, {cy:.1f}, {tz:.1f})"
            )
            return
        except Exception as e1:
            print(f"[INFO] ViewportCameraState unavailable ({e1}), trying USD camera…")

        import math
        cam_path = "/World/OverviewCamera"
        camera = UsdGeom.Camera.Define(stage, cam_path)

        camera.CreateFocalLengthAttr(18.0)
        camera.CreateHorizontalApertureAttr(36.0)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 500.0))

    
        dY = cy    - cam_y   
        dZ = tz    - cam_z  
        pitch_deg = math.degrees(math.atan2(-dZ, dY)) 

        xf = UsdGeom.Xformable(camera.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(cam_x, cam_y, cam_z))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(pitch_deg, 0.0, 0.0))

        viewport.set_active_camera(cam_path)
        print(
            f"[INFO] Overview camera (USD prim) → "
            f"eye=({cam_x:.1f}, {cam_y:.1f}, {cam_z:.1f})  "
            f"pitch={pitch_deg:.1f}°  path={cam_path}"
        )

    except Exception as e:
        print(f"[WARN] Could not configure viewport camera: {e}")


def create_physical_drones(agents: List[DroneAgent], cfg: SimConfig):
    """Crée les Multirotors Pegasus avec ArduPilot SITL + PhysX LiDAR."""
    from pegasus.simulator.params import ROBOTS
    from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
    from pegasus.simulator.logic.sensors.barometer import Barometer
    from pegasus.simulator.logic.sensors.imu import IMU
    from pegasus.simulator.logic.sensors.gps import GPS
    from pegasus.simulator.logic.backends.ardupilot_mavlink_backend import (
        ArduPilotMavlinkBackend, ArduPilotMavlinkBackendConfig,
    )
    from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

    # créer AifStateTracker comme sous-classe de Backend
    _create_state_tracker_class()

    ardupilot_dir = PegasusInterface().ardupilot_path

    multirotors = []
    controllers: List[SitlController] = []
    for agent in agents:
        gx = agent.trail[0][0]
        gy = agent.trail[0][1]

        sx = gx + cfg.origin_x
        sy = gy + cfg.origin_y
        spawn_pos = np.array([sx, sy, 0.1])

        # Backend 0 : pont capteurs/moteurs vers SITL
        mav_cfg = ArduPilotMavlinkBackendConfig({
            "vehicle_id": agent.id,
            "ardupilot_autolaunch": True,
            "ardupilot_dir": ardupilot_dir,
            "ardupilot_vehicle_model": "gazebo-iris",
        })
        mav_backend = ArduPilotMavlinkBackend(config=mav_cfg)

        # Backend 1 : lecture de pose pour la boucle AIF
        tracker = AifStateTracker(agent.id)

        mc = MultirotorConfig()
        mc.backends = [mav_backend, tracker]
        mc.sensors = [Barometer(), IMU(), GPS()]

        prim_path = f"/World/Drone_{agent.id:02d}"
        drone = Multirotor(
            prim_path, ROBOTS["Iris"], agent.id,
            [sx, sy, 0.1], [0.0, 0.0, 0.0, 1.0], mc,
        )
        multirotors.append(drone)

        # contrôleur SITL (envoie les waypoints AIF via MAVLink)
        controller = SitlController(agent.id, mav_backend, tracker, spawn_pos, cfg)
        controllers.append(controller)

        # LiDAR PhysX
        lidar = LidarReader(agent.id, prim_path, cfg)
        agent.setup_physical(controller, lidar)

        print(
            f"[INFO] Drone {agent.id} — grid({gx:.1f},{gy:.1f}) "
            f"→ world({sx:.1f},{sy:.1f}) — ArduPilot SITL + PhysX LiDAR"
        )

    return multirotors, controllers


def main():
    cfg = parse_args()

    running = True
    def _sig(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    # Récap de la config active (pour traçabilité dans les logs)
    print(f"[INFO] AIF config :")
    print(f"  planner={cfg.planner}  arch={cfg.arch}  "
          f"neighbor_radius_m={cfg.neighbor_radius_m}")
    print(f"  ns3={cfg.ns3_mode}  cloud_latency_ms={cfg.cloud_latency_ms}")
    print(f"  cut_cloud_at_step={cfg.cut_cloud_at_step}  "
          f"cut_drone_link={cfg.cut_drone_link!r} @ step={cfg.cut_drone_link_at_step}")
    print(f"  run_tag={cfg.run_tag!r}")

    # ── Tâche 3 : NS-3 importé tôt mais lancé APRÈS création des agents
    # (sinon NS-3 lit /tmp/drone_positions.csv stale/inexistant). ──
    ns3_bridge = None
    if cfg.ns3_mode != "none":
        ns3_bridge = _import_ns3_bridge()

    print(f"[INFO] Isaac Sim — {'headless' if cfg.headless else 'GUI'}")
    sim_app = create_sim_app(cfg)

    world, factory_bounds = setup_world()

    # ── Auto-adjust env dimensions from factory USD bounding box ──
    if factory_bounds:
        margin = 1.0  # 1 m padding around the factory (spawn safety only)
        bx0, by0, bx1, by1 = factory_bounds
        cfg.world_origin_x = bx0 - margin
        cfg.world_origin_y = by0 - margin
        cfg.env_width  = (bx1 - bx0) + 2 * margin
        cfg.env_height = (by1 - by0) + 2 * margin
        # round to grid resolution
        cfg.env_width  = math.ceil(cfg.env_width  / cfg.grid_resolution) * cfg.grid_resolution
        cfg.env_height = math.ceil(cfg.env_height / cfg.grid_resolution) * cfg.grid_resolution
        # Bbox réel SANS la marge — sert UNIQUEMENT pour le calcul de
        # coverage (cf. BeliefGrid.interior_exploration_ratio) afin
        # d'éviter les fuites de flood-fill par les trous des murs.
        cfg.factory_bounds_world = (float(bx0), float(by0),
                                    float(bx1), float(by1))
        print(f"[INFO] Env adapted to factory: "
              f"{cfg.env_width:.1f} × {cfg.env_height:.1f} m  "
              f"origin=({cfg.origin_x:.2f}, {cfg.origin_y:.2f})  "
              f"grid={cfg.grid_width}×{cfg.grid_height}")

    setup_viewport_camera(cfg)
    obstacles: List[Dict] = []

    agents: List[DroneAgent] = []
    for i in range(cfg.num_drones):
        gx = cfg.env_width  / 2 + (i - (cfg.num_drones - 1) / 2) * cfg.drone_spacing
        # Spawn well inside factory walls (USD bbox extends beyond walls due to floor)
        gy = max(8.0, cfg.env_height * 0.20)
        agents.append(DroneAgent(i, gx, gy, cfg))

    # ── Tâche 3 : positions initiales + lancement NS-3
    # On écrit le CSV (avec header `drone_id,x,y,z`) AVANT de lancer NS-3
    # pour que la 1ʳᵉ lecture du scénario ait des positions valides ; le
    # main loop le réécrit ensuite à chaque step. ──
    if ns3_bridge is not None:
        _write_drone_positions(agents, cfg)
        ns3_bridge.launch_ns3(
            n_drones=cfg.num_drones,
            sim_time=cfg.ns3_sim_time,
            scenario=cfg.ns3_mode,
        )

    # ── Patcher les params SITL AVANT de créer les drones (= avant le launch SITL) ──
    _patch_sitl_defaults()

    multirotors, controllers = create_physical_drones(agents, cfg)
    diag_logger = DiagnosticLogger(cfg.output_dir, cfg)
    coordinator = SwarmCoordinator(agents, cfg, diag_logger=diag_logger)
    logger = DataLogger(cfg.output_dir)

    # ── QR Code System : génération + panneau 3D + caméras ──
    from qr_code_system import setup_qr_system, initialize_cameras
    qr_data = os.getenv("QR_CODE_DATA", "DRONE_WAREHOUSE_INSPECTION_001")
    # Position confirmed optimal by simulation:
    # - X=0   : centre du couloir principal (aucune étagère devant)
    # - Y=-5  : face à la zone de spawn des drones (Y≈-2), en espace ouvert
    # - Z=2   : exactement à la hauteur de vol (fly_altitude=2.0m)
    # Rotation (-90,0,0) : normale du panneau pointe +Y → face aux drones
    # qui arrivent depuis Y=-2 en direction Y=-5.
    qr_panel_pos = (0.0, -5.0, 2.0)
    drone_prim_paths = [f"/World/Drone_{a.id:02d}" for a in agents]
    drone_ids = [a.id for a in agents]
    qr_sys = setup_qr_system(
        qr_data=qr_data,
        output_dir=cfg.output_dir,
        drone_prim_paths=drone_prim_paths,
        drone_ids=drone_ids,
        panel_position=qr_panel_pos,
        panel_size=2.5,             # large panel for reliable detection
        cache_ttl=3.0,
        capture_interval=5,
        camera_resolution=(640, 480),
    )
    print(f"[INFO] QR code system ready — data='{qr_data}'")

    world.reset()

    # ── Initialize LiDAR sensors (must happen after world.reset) ──
    for agent in agents:
        if agent.lidar is not None:
            agent.lidar.initialize()

    # ── Initialize cameras (must happen after world.reset) ──
    initialize_cameras(qr_sys["cameras"])

    print(f"[INFO] {cfg.num_drones} drones | grid {cfg.grid_width}×{cfg.grid_height}")
    print("[INFO] Using factory template colliders from USD scene")
    print(f"[INFO] {cfg.sim_steps_per_aif} physics ticks per AIF decision")
    print(f"[INFO] Dashboard → {cfg.output_dir}/aif_state.json")

    # ══════════════════════════════════════════════════════════════
    # Séquence de décollage ArduPilot SITL
    #
    # Le JSON backend Isaac Sim tourne à ~250 Hz. Les params
    # gazebo-iris.parm (patchés ci-dessus) fixent SCHED_LOOP_RATE=50
    # pour que 50*1.8=90 < 250 → pas de PreArm "Main loop slow"
    # ni "Gyro rate", et ARMING_CHECK=0 pour que les PreArm passent.
    #
    # Phase 1 : laisser SITL initialiser (baro, EKF, GPS)
    # Phase 2 : envoyer params backup + GUIDED + force arm
    # Phase 3 : vérifier armement avec retry
    # Phase 4 : takeoff + attendre altitude
    # ══════════════════════════════════════════════════════════════

    # ── Phase 1 : laisser SITL s'initialiser (baro, EKF, GPS) ──
    #    2500 ticks ≈ 25 s en simulation. Nécessaire pour que l'EKF ait
    #    un GPS fix et puisse estimer la position (sinon « Need Position
    #    Estimate » empêche d'armer même avec force_arm en mode GUIDED).
    print("[INFO] Phase 1/3 : initialisation ArduPilot SITL + EKF (2500 ticks) …")
    for _ in range(2500):
        if not running or not sim_app.is_running():
            break
        world.step(render=not cfg.headless)

    # ── Phase 2 : GUIDED + ARM + TAKEOFF (séquentiel par drone) ──
    #    On envoie arm ET takeoff dans la même « respiration » pour
    #    minimiser le temps armé au sol (crash-detect).
    print("[INFO] Phase 2/3 : GUIDED + ARM + TAKEOFF …")
    for ctrl in controllers:
        ctrl.send_backup_params()
    for _ in range(100):
        world.step(render=not cfg.headless)

    for ctrl in controllers:
        ctrl.set_guided()
        for _ in range(30):
            world.step(render=not cfg.headless)
        ctrl.force_arm()
        for _ in range(10):
            world.step(render=not cfg.headless)
        ctrl.takeoff(cfg.fly_altitude)
        print(f"  [SITL] D{ctrl.drone_id} → GUIDED + ARM + TAKEOFF({cfg.fly_altitude:.1f}m)")
        # petit gap entre les drones pour que SITL traite
        for _ in range(200):
            world.step(render=not cfg.headless)

    # ── Phase 3 : attente altitude + retry ciblé ──
    #    On surveille l'altitude via le state tracker (pas de heartbeat =
    #    pas de race condition avec le backend Pegasus).
    #    Retry arm+takeoff uniquement pour les drones qui ne montent pas.
    print(f"[INFO] Phase 3/3 : attente altitude ({cfg.fly_altitude:.1f} m) …")
    drone_flying = [False] * len(controllers)
    RETRY_TICKS = (500, 1000, 1500, 2000, 2500)

    for tick in range(3000):
        if not running or not sim_app.is_running():
            break
        world.step(render=not cfg.headless)
        for agent in agents:
            agent.accumulate_lidar()

        # vérifier altitude
        for i, ctrl in enumerate(controllers):
            if not drone_flying[i] and ctrl.is_flying(cfg.fly_altitude * 0.5):
                drone_flying[i] = True
                print(f"  [TAKEOFF] D{ctrl.drone_id} en vol ✓")

        # retry ciblé pour les drones pas encore en vol
        if tick in RETRY_TICKS:
            for i, ctrl in enumerate(controllers):
                if not drone_flying[i]:
                    ctrl.set_guided()
                    ctrl.force_arm()
                    ctrl.takeoff(cfg.fly_altitude)
                    print(f"  [RETRY] D{ctrl.drone_id} re-arm+takeoff (tick={tick})")

        # monitoring périodique
        if tick % 300 == 299:
            positions = [a.backend.get_position() for a in agents if a.backend]
            alts = [p[2] for p in positions]
            alt_str = ", ".join(f"D{i}={a:.2f}m" for i, a in enumerate(alts))
            n_fly = sum(drone_flying)
            print(f"  [TAKEOFF] tick={tick} {n_fly}/{len(controllers)} en vol | {alt_str}")

        if all(drone_flying):
            print("[INFO] Tous les drones en vol !")
            break

    if not all(drone_flying):
        not_fly = [i for i, f in enumerate(drone_flying) if not f]
        print(f"[WARN] Drones pas en vol après 3000 ticks : {not_fly}")
    print("[INFO] Décollage terminé.\n")

    # ── Resilience demo events ──
    kill_drone_at = getattr(cfg, '_kill_drone_at', -1)

    if kill_drone_at >= 0:
        print(f"[INFO] Resilience demo: drone 0 will be killed at step {kill_drone_at}")

    # ── Start QR decoder thread ──
    qr_decoder = qr_sys["decoder_thread"]
    qr_decoder.start()
    qr_capture = qr_sys["capture_helper"]
    print("[INFO] QR decoder thread started")

    aif_step = 0
    plateau_counter = 0
    plateau_threshold = 0.1   # %/step minimum pour considérer une progression
    plateau_patience = 15     # nb steps consécutifs sans progression
    prev_coverage = 0.0
    while running and aif_step < cfg.max_steps and sim_app.is_running():

        # ── Resilience event: kill drone ──
        if kill_drone_at >= 0 and aif_step == kill_drone_at:
            coordinator.kill_drone(0)
            print(f"\n{'='*60}")
            print(f"  ⚠ RESILIENCE TEST: Drone 0 disabled at step {aif_step}")
            print(f"  Remaining active drones: {len(coordinator.active_agents)}")
            print(f"{'='*60}\n")

        # AIF cycle: perceive (real LiDAR) → fuse → plan (sets waypoints)
        coordinator.step()

        # fly: run physics until drones move toward their waypoints
        for _ in range(cfg.sim_steps_per_aif):
            if not running or not sim_app.is_running():
                break
            world.step(render=not cfg.headless)
            # Accumulate partial LiDAR scans each physics tick
            for agent in agents:
                agent.accumulate_lidar()
            # Capture camera frames (every N ticks, handled internally)
            qr_capture.tick()

        # ── Tâche 3 : exposer les positions courantes à NS-3 ──
        if cfg.ns3_mode != "none":
            _write_drone_positions(agents, cfg)

        # log for dashboard
        state = coordinator.get_full_state(obstacles)
        logger.log(state, coordinator.history)

        m = coordinator.history[-1]
        positions_str = " | ".join(
            f"D{a.id}({'X' if not a.active else f'{a.x:.1f},{a.y:.1f}'})" for a in agents
        )
        phase_tag = f" [{m.get('resilience_phase','normal').upper()}]" if m.get('resilience_phase','normal') != 'normal' else ""
        print(
            f"  Step {aif_step:4d} | "
            f"H={m['mean_entropy']:.3f} | "
            f"Expl={m['exploration_pct']:5.1f}% | "
            f"ΔIG={m['step_info_gain']:.4f} | "
            f"innov={m.get('innovation_mean',0):.3f} | "
            f"active={m.get('active_drones', len(agents))}{phase_tag} | "
            f"{positions_str}"
        )

        # ── Arrêt si target_coverage atteint ──
        cur_coverage = m["exploration_pct"]
        if cur_coverage >= cfg.target_coverage:
            print(f"\n[INFO] Target coverage {cfg.target_coverage}% reached "
                  f"at step {aif_step} (coverage={cur_coverage:.1f}%).")
            break

        # ── Arrêt sur plateau (drones bloqués, plus de progression) ──
        if cur_coverage - prev_coverage < plateau_threshold:
            plateau_counter += 1
        else:
            plateau_counter = 0
        prev_coverage = cur_coverage

        if plateau_counter >= plateau_patience:
            print(f"\n[INFO] Coverage plateau at {cur_coverage:.1f}% "
                  f"(<{plateau_threshold}%/step over {plateau_patience} steps). "
                  f"Exploration considered complete.")
            break

        aif_step += 1

    state = coordinator.get_full_state(obstacles)
    logger.log(state, coordinator.history)
    m = coordinator.history[-1] if coordinator.history else {}
    print(f"\n[INFO] Done — step {aif_step} | entropy {m.get('mean_entropy','?')} | coverage {m.get('exploration_pct','?')}%")

    # ── Stop QR decoder thread and log stats ──
    qr_decoder.stop()
    qr_stats = qr_decoder.stats()
    print(f"[QR] Final stats: decoded={qr_stats['decoded']} "
          f"failed={qr_stats['failed']} total={qr_stats['total']}")
    print(f"[QR] Cache: {qr_stats['cache']}")

    # ── Tâche 3 : stopper NS-3 si on l'a lancé ──
    if ns3_bridge is not None:
        try:
            ns3_bridge.stop_ns3()
        except Exception as e:
            print(f"[WARN] stop_ns3 a échoué : {e}")

    # ── Tâche 5 : générer les artefacts du run (PNG + JSON) ──
    try:
        from run_artifacts import generate_run_artifacts
        run_dir = generate_run_artifacts(
            cfg=cfg,
            history=coordinator.history,
            final_state=state,
            fused_belief=coordinator.fused_belief,
            agents=agents,
            qr_stats=qr_stats,
            ns3_pairs=coordinator.ns3.last_pairs,
        )
        print(f"[RUN] Artefacts → {run_dir}")
    except Exception as e:
        print(f"[WARN] run_artifacts generation failed: {e}")

    sim_app.close()


if __name__ == "__main__":
    main()