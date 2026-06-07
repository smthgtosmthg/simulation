"""
DroneAgent — Représentation logique d'un drone.

Responsabilités :
  - Possède sa propre belief locale (`BeliefGrid`)
  - Lit le LiDAR via un *backend* duck-typé (`accumulate_lidar`/`get_accumulated`)
  - Lit sa pose via un *backend* duck-typé (`get_position_xy`/`get_yaw`)
  - Met à jour sa belief avec inverse sensor model
  - Calcule son `last_innovation` (surprise par rapport à la belief)
  - Fusionne avec les beliefs reçues de voisins (en mode distribué)
  - Exécute une action (set_target sur le backend)

Le DroneAgent NE DÉCIDE PAS ce qu'il faut faire — c'est le rôle de
l'`ArchitecturePlanner` (centralisé ou distribué).  L'agent expose
juste l'interface `execute(action)` pour qu'on lui dise où aller.

Backends attendus (interface duck-typed, fournie par Isaac Sim) :
  - `controller` : `.set_target(x, y, z)`, `.get_position_xy()`, `.get_yaw()`
  - `lidar`      : `.accumulate(yaw)`, `.get_accumulated() -> (angles, ranges, hits)`
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .belief import BeliefGrid, fuse_beliefs_logodds
from .math_utils import clamp


# Anti-stagnation : nb de steps consécutifs sans bouger avant de forcer un mvt
_STAG_THRESHOLD: int = 5


class DroneAgent:
    """Drone logique + belief + interface backend."""

    def __init__(self, drone_id: int, sx: float, sy: float, cfg):
        self.id = drone_id
        self.cfg = cfg
        self.belief = BeliefGrid(cfg)
        self.rng = np.random.default_rng(42 + drone_id * 1000)

        # Trajectoire (en coordonnées grille)
        self.trail: List[Tuple[float, float]] = [(sx, sy)]

        # ── Backends physiques (set via setup_physical) ─────────────
        self.controller: Optional[Any] = None      # SitlController
        self.lidar: Optional[Any] = None           # LidarReader

        # ── État courant ────────────────────────────────────────────
        self.active: bool = True
        self.total_dist: float = 0.0
        self._prev_xy: Tuple[float, float] = (sx, sy)
        self._stag_pos: Tuple[float, float] = (sx, sy)
        self._stag_steps: int = 0

        # ── Mémoire de la dernière décision (exposée au dashboard) ──
        self.last_action: str = "stay"
        self.last_G: float = 0.0
        self.last_ig: float = 0.0
        self.last_innovation: float = 0.0
        self.candidates_diag: List[Dict] = []
        self.selected_idx: int = 0

        # ── Champs alimentés par l'ArchitecturePlanner ──────────────
        self.pending_action: Optional[Tuple[str, float, float]] = None
        self.last_action_fresh: bool = False
        self.last_decision_source: str = "none"    # "cloud" | "local" | "local_fallback" | "local_warmup"
        self.last_plan_belief: Optional[BeliefGrid] = None

        # ── Diagnostic LiDAR (pour DiagnosticLogger) ────────────────
        self.lidar_diag: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

        # ── Boost one-shot d'innovation (utilisé par DynamicObstacleStressor
        #    pour matérialiser un pic visible même si le LiDAR ne détecte
        #    qu'une petite portion du nouvel obstacle).  Consommé par perceive(). ─
        self._innovation_boost: float = 0.0

        # ── Mode distribué : cache des beliefs reçues via MessageQueue ─
        # Clé = src_drone_id, valeur = BeliefGrid (la plus récente reçue)
        self.last_received_belief: Dict[int, BeliefGrid] = {}
        self.local_fused: Optional[BeliefGrid] = None

    # ── Setup physique ──────────────────────────────────────────────

    def setup_physical(self, controller, lidar) -> None:
        self.controller = controller
        self.lidar = lidar

    # ── Pose (en coordonnées grille interne, déduite du backend) ────

    @property
    def x(self) -> float:
        if self.controller:
            return self.controller.get_position_xy()[0] - self.cfg.origin_x
        return self.trail[-1][0]

    @property
    def y(self) -> float:
        if self.controller:
            return self.controller.get_position_xy()[1] - self.cfg.origin_y
        return self.trail[-1][1]

    # ── Perception ──────────────────────────────────────────────────

    def accumulate_lidar(self) -> None:
        """À appeler à chaque tick physique pour empiler les rayons partiels."""
        if not self.active or self.lidar is None or self.controller is None:
            return
        self.lidar.accumulate(self.controller.get_yaw())

    def perceive(self) -> None:
        """Lit le LiDAR accumulé + met à jour la belief + calcule l'innovation."""
        if not self.active:
            self.lidar_diag = None
            self.last_innovation = 0.0
            return

        if self.lidar is not None:
            angles, ranges, hits = self.lidar.get_accumulated()
            self.lidar_diag = (angles.copy(), ranges.copy(), hits.copy())

            # Innovation = surprise sur les hits LiDAR
            innov_sum = 0.0
            n_compare = 0
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
                    innov_sum += abs(1.0 - predicted_occ)
                    n_compare += 1
            self.last_innovation = innov_sum / max(n_compare, 1)

            # Mise à jour bayésienne de la belief depuis le LiDAR
            self.belief.update_from_lidar(
                self.x, self.y, angles, ranges, hits,
                self.cfg.lidar_max_range, self.cfg.lo_free, self.cfg.lo_occ,
            )
        else:
            self.lidar_diag = None
            self.last_innovation = 0.0

        # Boost one-shot : appliqué INDÉPENDAMMENT de la présence du LiDAR.
        # Permet de garantir un pic visible même si :
        #   - le LiDAR ne détecte qu'une poignée de hits sur le nouvel obstacle
        #   - ou même si on tourne sans Isaac Sim (tests / smoke)
        if self._innovation_boost > 0.0:
            self.last_innovation = max(self.last_innovation, self._innovation_boost)
            self._innovation_boost = 0.0

    # ── Fusion locale avec voisins (mode distribué) ─────────────────

    def fuse_with_neighbors(self, neighbors: List["DroneAgent"],
                            prior_lo: float) -> BeliefGrid:
        """Fusionne `self.belief` avec les beliefs cached des voisins.

        Si la latence NS-3 a retardé une belief voisine, on utilise la
        dernière belief reçue (cached) plutôt que le snapshot live — c'est
        ce qui rend la fusion **dépendante du réseau** en distribué.
        """
        beliefs = [self.belief]
        for n in neighbors:
            cached = self.last_received_belief.get(n.id)
            if cached is not None:
                beliefs.append(cached)
            # Sinon : on n'a JAMAIS reçu de belief de ce voisin (lien permanent
            # coupé ou voisin trop récent) → on n'utilise pas son belief.
        self.local_fused = fuse_beliefs_logodds(beliefs, prior_lo)
        return self.local_fused

    # ── Exécution d'une action ──────────────────────────────────────

    def execute(self, action: Tuple[str, float, float]) -> None:
        """Applique l'action : met à jour le waypoint backend + l'état interne.

        L'anti-stagnation reste actif côté agent : si le drone n'a pas bougé
        depuis _STAG_THRESHOLD steps, on remplace l'action par un mvt aléatoire.
        """
        name, dx, dy = action
        cx, cy = self.x, self.y

        # Anti-stagnation
        moved = math.hypot(cx - self._stag_pos[0], cy - self._stag_pos[1])
        if moved < 0.5:
            self._stag_steps += 1
        else:
            self._stag_steps = 0
            self._stag_pos = (cx, cy)

        if self._stag_steps >= _STAG_THRESHOLD:
            # Drone bloqué (mur physique, etc.) → on force un mvt aléatoire pour
            # tenter de débloquer, indépendamment de l'action choisie par le planner.
            from .planner import ACTIONS
            idx = int(self.rng.integers(1, len(ACTIONS)))  # skip "stay"
            name, dx, dy = ACTIONS[idx]
            self._stag_steps = 0
            print(f"  [STAG] D{self.id} stuck → forced {name}")

        tx = clamp(cx + dx * self.cfg.step_size, 0.5, self.cfg.env_width - 0.5)
        ty = clamp(cy + dy * self.cfg.step_size, 0.5, self.cfg.env_height - 0.5)

        if self.controller is not None:
            self.controller.set_target(
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
        # Pour l'export dashboard
        if self.candidates_diag and self.selected_idx < len(self.candidates_diag):
            sel = self.candidates_diag[self.selected_idx]
            self.last_G = sel.get("G", 0.0)
            self.last_ig = sel.get("ig", 0.0)

    # ── Arrêt opérationnel (kill_drone) ─────────────────────────────

    def land(self) -> None:
        """Désactive le drone (panne simulée)."""
        self.active = False
        self.last_action = "LANDED"
        if self.controller:
            wx = self.x + self.cfg.origin_x
            wy = self.y + self.cfg.origin_y
            self.controller.set_target(wx, wy, 0.1)
        print(f"  [RESILIENCE] 🛬 Drone {self.id} LANDED (out of service)")

    # ── Sérialisation pour le dashboard ─────────────────────────────

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
            "decision_source": self.last_decision_source,
        }
