"""
SimConfig — Tous les paramètres de la simulation regroupés en un seul dataclass.

Organisation par section :
    1. Environnement & grille
    2. Drones
    3. LiDAR
    4. Active Inference (poids, fusion)
    5. Résilience (phases, seuils, EMA)
    6. Architecture (centralisée vs distribuée, NS-3)
    7. Stresseurs programmés (kill drone, cut cloud, cut links, obstacle)
    8. Run capture (logs/artefacts)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SimConfig:
    # ── 1. Environnement & grille ────────────────────────────────────
    env_width: float = 30.0
    env_height: float = 20.0
    grid_resolution: float = 0.5      # mètres par cellule
    fly_altitude: float = 2.0

    # ── 2. Drones ────────────────────────────────────────────────────
    num_drones: int = 3
    drone_spacing: float = 5.0        # espace de spawn entre drones
    step_size: float = 1.0            # déplacement par action AIF (m)

    # ── 3. LiDAR ─────────────────────────────────────────────────────
    num_rays: int = 360
    lidar_fov_h: float = 360.0
    lidar_fov_v: float = 40.0
    lidar_vert_res: float = 5.0
    lidar_max_range: float = 8.0
    lidar_min_range: float = 0.15
    lidar_hz: float = 10.0
    floor_filter_z: float = 0.25

    # ── 4. Active Inference ──────────────────────────────────────────
    prior_occupancy: float = 0.5
    lo_free: float = -0.55
    lo_occ: float = 0.85
    lo_max: float = 30.0
    occ_threshold: float = 0.65

    w_epistemic: float = 2.5          # privilégie zones incertaines (gain d'info)
    w_pragmatic: float = 0.8          # attire vers les frontières
    w_movement: float = 0.1           # léger coût au mouvement
    w_collision: float = 5.0          # repousse les autres drones
    softmax_temp: float = 0.3         # diversité du tirage d'action
    fusion_mix: float = 0.3           # mix local/fused dans select_action

    waypoint_tol: float = 0.3

    # ── 5. Résilience ────────────────────────────────────────────────
    alpha: int = 30                   # durée phase recovery (steps)
    beta: int = 60                    # durée phase durable (steps)
    H_target: float = 0.44            # entropie cible pour recovery OK
    innov_target: float = 0.16        # innovation cible pour recovery OK
    k_sigma: float = 2.0              # seuil de spike (EMA + k_σ·σ)
    ema_alpha: float = 0.05

    w_entropy_recover: float = 3.0
    w_innov_recover: float = 1.2
    w_deadline: float = 12.0
    w_entropy_durable: float = 1.2
    w_innov_durable: float = 0.8
    w_churn_durable: float = 2.2
    w_maintain: float = 10.0

    # ── 6. Architecture & réseau ─────────────────────────────────────
    planner: str = "aif"              # "aif" | "heuristic"
    arch: str = "centralized"         # "centralized" | "distributed"
    neighbor_radius_m: float = 5.0    # rayon de fusion entre voisins (distribué)
    ns3_mode: str = "wifi"            # "none" | "wifi" | "5g"
    # Latence cloud round-trip (drone→cloud→drone). Inclut l'upload de la belief,
    # le calcul cloud, et le download de la commande d'action.  500 ms = défaut
    # défendable (cloud chargé). Augmenter à 1500 ms pour démontrer l'impact.
    cloud_round_trip_ms: float = 500.0
    # Latence de reconfiguration quand le cloud est coupé : pendant ce délai,
    # les drones gèlent (pas de nouvelle décision, dec/min → 0, covKnown → own
    # belief seule).  Après ce délai, ils basculent en mode distribué (fusion
    # avec voisins).
    switch_latency_ms: float = 2000.0
    ns3_sim_time: int = 600
    physics_dt_s: float = 1.0 / 60.0  # Isaac Sim @60 Hz

    # ── 7. Stresseurs programmés ─────────────────────────────────────
    kill_drone_at_step: int = -1      # -1 = jamais
    kill_drone_id: int = 0
    cut_cloud_at_step: int = -1
    cut_drone_link: str = ""          # "" | "0-1" | "all"
    cut_drone_link_at_step: int = -1
    drop_obstacle_at_step: int = -1   # obstacle dynamique (à venir)
    drop_obstacle_xy: str = ""        # "x,y" en coords monde

    # ── 8. Run capture ───────────────────────────────────────────────
    headless: bool = False
    max_steps: int = 500
    sim_steps_per_aif: int = 60       # ticks physiques entre 2 décisions AIF
    target_coverage: float = 93.0
    output_dir: str = "/tmp"
    run_tag: str = "default"
    runs_dir: str = ""

    # ── 9. Origine du repère (rempli à l'init Isaac Sim) ─────────────
    world_origin_x: float = float("nan")
    world_origin_y: float = float("nan")
    factory_bounds_world: Optional[Tuple[float, float, float, float]] = None
    interior_inset_m: float = 2.5

    # ── Propriétés dérivées ──────────────────────────────────────────

    @property
    def origin_x(self) -> float:
        return -self.env_width / 2 if math.isnan(self.world_origin_x) else self.world_origin_x

    @property
    def origin_y(self) -> float:
        return -self.env_height / 2 if math.isnan(self.world_origin_y) else self.world_origin_y

    @property
    def grid_width(self) -> int:
        return int(self.env_width / self.grid_resolution)

    @property
    def grid_height(self) -> int:
        return int(self.env_height / self.grid_resolution)

    @property
    def max_range_cells(self) -> int:
        return int(self.lidar_max_range / self.grid_resolution)

    @property
    def step_dt_ms(self) -> float:
        """Durée d'un step AIF en millisecondes."""
        return self.sim_steps_per_aif * self.physics_dt_s * 1000.0

    @property
    def ray_angles(self):
        import numpy as np
        return np.linspace(0, 2 * math.pi, self.num_rays, endpoint=False)

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
