"""
BeliefGrid — Carte d'occupation probabiliste (log-odds) + fusion entre drones.

Représentation : pour chaque cellule, une probabilité d'occupation P(occupé)
stockée sous forme de log-odds L = log(P / (1-P)).  Mise à jour bayésienne
par addition de log-odds (filtrage de Bayes naïf indépendant cellule par
cellule), avec saturation à ±lo_max.

Fusion entre drones : Independent Opinion Pool en log-odds :
    L_fused = L_prior + mean_i(L_i - L_prior)
C'est commutatif (l'ordre des drones n'a pas d'impact) et symétrique
(somme = somme dans n'importe quel ordre).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .math_utils import (
    bernoulli_entropy_v,
    inv_logit,
    inv_logit_v,
    logit,
)


class BeliefGrid:
    """Grille d'occupation 2D en log-odds."""

    def __init__(self, cfg):
        self.width = cfg.grid_width
        self.height = cfg.grid_height
        self.resolution = cfg.grid_resolution
        self.lo_max = cfg.lo_max
        l0 = logit(cfg.prior_occupancy)
        self.logodds = np.full((self.height, self.width), l0, dtype=np.float64)
        self.probability = np.full((self.height, self.width), cfg.prior_occupancy)

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        gx = max(0, min(int(wx / self.resolution), self.width - 1))
        gy = max(0, min(int(wy / self.resolution), self.height - 1))
        return gx, gy

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.width and 0 <= gy < self.height

    def update_cell(self, gx: int, gy: int, dl: float):
        if not self.in_bounds(gx, gy):
            return
        self.logodds[gy, gx] = np.clip(self.logodds[gy, gx] + dl,
                                       -self.lo_max, self.lo_max)
        self.probability[gy, gx] = inv_logit(float(self.logodds[gy, gx]))

    def update_from_lidar(self, ox: float, oy: float,
                          angles: np.ndarray, ranges: np.ndarray,
                          hits: np.ndarray, max_range: float,
                          lo_free: float, lo_occ: float):
        """Inverse sensor model : pour chaque rayon LiDAR, parcourt les
        cellules traversées, marque libres celles avant le hit et occupée
        celle qui correspond au hit final."""
        ogx, ogy = self.world_to_grid(ox, oy)
        for i in range(len(angles)):
            cos_a = math.cos(angles[i])
            sin_a = math.sin(angles[i])
            d = min(float(ranges[i]), max_range) if hits[i] else max_range
            n_steps = int(d / self.resolution)
            for s in range(1, n_steps + 1):
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

    # ── Métriques dérivées ──────────────────────────────────────────

    def mean_entropy(self) -> float:
        return float(bernoulli_entropy_v(self.probability).mean())

    def exploration_ratio(self) -> float:
        """% de cellules avec une croyance franche (p<0.3 ou p>0.7)."""
        known = (self.probability < 0.3) | (self.probability > 0.7)
        return float(known.sum() / known.size)

    def effective_bounds(self, occ_threshold: float = 0.65) -> Tuple[int, int, int, int]:
        """Bounding box des cellules occupées (utile pour cadrer la zone murs)."""
        occ = self.probability >= occ_threshold
        if not occ.any():
            return 0, 0, self.width, self.height
        rows = np.any(occ, axis=1)
        cols = np.any(occ, axis=0)
        rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
        cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
        return cmin, rmin, cmax + 1, rmax + 1

    def interior_exploration_ratio(self, drone_positions=None,
                                   occ_threshold: float = 0.65,
                                   bounds_grid: Optional[Tuple[int, int, int, int]] = None,
                                   interior_area_cells: int = 0) -> float:
        """% de cellules connues dans la bbox intérieure (murs - inset)."""
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

    def observed_inside_walls_ratio(self, occ_threshold: float = 0.65,
                                    bounds_grid: Optional[Tuple[int, int, int, int]] = None) -> float:
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

    # ── Utilitaires ─────────────────────────────────────────────────

    def copy(self) -> "BeliefGrid":
        new = BeliefGrid.__new__(BeliefGrid)
        new.width, new.height = self.width, self.height
        new.resolution, new.lo_max = self.resolution, self.lo_max
        new.logodds = self.logodds.copy()
        new.probability = self.probability.copy()
        return new

    def to_list(self) -> List[List[float]]:
        return np.round(self.probability, 2).tolist()


# ════════════════════════════════════════════════════════════════════
# Fusion entre beliefs
# ════════════════════════════════════════════════════════════════════


def fuse_beliefs_logodds(beliefs: List[BeliefGrid], prior_lo: float) -> BeliefGrid:
    """Independent Opinion Pool : L_fused = L0 + mean(L_i - L0).

    Reflète : chaque drone fournit une "déviation" par rapport au prior,
    on en fait la moyenne et on l'ajoute au prior.  Commutatif.
    """
    fused = beliefs[0].copy()
    n = len(beliefs)
    fused.logodds[:] = prior_lo
    for b in beliefs:
        fused.logodds += (b.logodds - prior_lo) / n
    np.clip(fused.logodds, -fused.lo_max, fused.lo_max, out=fused.logodds)
    fused.probability = inv_logit_v(fused.logodds)
    return fused


def mix_beliefs(local: BeliefGrid, fused: BeliefGrid, lam: float) -> BeliefGrid:
    """Pondère local et fused : L = (1-λ)·L_local + λ·L_fused.

    Utilisé dans select_action() pour mélanger la croyance locale du drone
    avec la croyance fusionnée (cloud ou voisins).
    """
    mixed = local.copy()
    mixed.logodds = (1 - lam) * local.logodds + lam * fused.logodds
    np.clip(mixed.logodds, -mixed.lo_max, mixed.lo_max, out=mixed.logodds)
    mixed.probability = inv_logit_v(mixed.logodds)
    return mixed
