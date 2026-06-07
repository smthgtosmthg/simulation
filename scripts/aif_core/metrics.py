"""
Métriques dérivées — calculées à partir du `history` et des agents.

Trois métriques nouvelles, complémentaires aux métriques de base
(coverage, entropy, innovation) :

1. **discovery_rate** : Δcoverage sur une fenêtre glissante (5 steps par
   défaut).  Mesure la *vitesse d'exploration* (% découvert / step).
   Chute quand un drone meurt, quand les drones se chevauchent (cuts), ou
   quand la latence cloud ralentit les décisions.

2. **coverage_known_to_planner** : moyenne sur drones actifs de
   `plan_belief.interior_exploration_ratio()`.  C'est ce que chaque drone
   *sait* au moment où il décide, à comparer avec la `coverage_pct`
   globale calculée sur la fusion omnisciente.  Le DELTA entre les deux
   = "le manque de partage réseau".  Chute brutale sur cut_links.

3. **decisions_per_min** : nombre de décisions fresh prises par
   l'ensemble du swarm par minute (compteur agrégé sur les drones actifs).
   En centralisé baseline : ~drones × 60.  Au cut cloud, le warmup local
   donne quelques steps stale → léger dip puis remontée.
"""

from __future__ import annotations

from typing import Dict, List, Optional


# ════════════════════════════════════════════════════════════════════
# 1) discovery_rate (calculé sur history)
# ════════════════════════════════════════════════════════════════════


def compute_discovery_rate(history: List[Dict], window: int = 5) -> float:
    """Δexploration_pct sur les `window` derniers steps.

    Renvoie 0 pour les `window` premiers steps (warm-up).
    """
    if len(history) < window + 1:
        return 0.0
    last = history[-1].get("exploration_pct", 0.0) or 0.0
    prev = history[-1 - window].get("exploration_pct", 0.0) or 0.0
    return float((last - prev) / window)


# ════════════════════════════════════════════════════════════════════
# 2) coverage_known_to_planner (calculé live sur les agents)
# ════════════════════════════════════════════════════════════════════


def compute_coverage_known_to_planner(active_agents, cfg) -> float:
    """Moyenne sur drones actifs de leur `last_plan_belief.interior_exploration_ratio()`.

    Si l'agent n'a pas encore eu de plan_belief assignée (warmup), on
    utilise sa belief locale comme proxy.
    """
    if not active_agents:
        return 0.0
    bounds = cfg.factory_bounds_grid()
    area = cfg.interior_area_cells()
    total = 0.0
    for a in active_agents:
        b = a.last_plan_belief if a.last_plan_belief is not None else a.belief
        total += b.interior_exploration_ratio(
            bounds_grid=bounds, interior_area_cells=area,
        )
    return float(total / len(active_agents) * 100.0)


# ════════════════════════════════════════════════════════════════════
# 3) decisions_per_min (compteur fresh sur fenêtre glissante)
# ════════════════════════════════════════════════════════════════════


class DecisionRateTracker:
    """Compte les décisions fresh par minute sur fenêtre glissante.

    Une décision est "fresh" quand `agent.last_action_fresh == True` au
    moment où on l'observe.  Cela vaut True pour :
      - Mode centralisé : drone exécute une action reçue du cloud
      - Mode distribué : drone planifie localement
      - Mode centralisé après cut : fallback local
    Mais vaut False pendant le warmup centralisé (pas encore d'action cloud).
    """

    def __init__(self, cfg, window: int = 10):
        self.cfg = cfg
        self.window = window
        # nb de décisions fresh par step pour les `window` derniers steps
        self._buffer: List[int] = []

    def record(self, active_agents) -> int:
        fresh_count = sum(1 for a in active_agents if a.last_action_fresh)
        self._buffer.append(fresh_count)
        if len(self._buffer) > self.window:
            self._buffer.pop(0)
        return fresh_count

    def rate_per_min(self) -> float:
        if not self._buffer:
            return 0.0
        total_fresh = sum(self._buffer)
        window_seconds = len(self._buffer) * (self.cfg.step_dt_ms / 1000.0)
        if window_seconds <= 0:
            return 0.0
        return float(total_fresh * 60.0 / window_seconds)
