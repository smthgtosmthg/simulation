from __future__ import annotations

from typing import Dict, List, Optional


def compute_discovery_rate(history: List[Dict], window: int = 5) -> float:
    if len(history) < window + 1:
        return 0.0
    last = history[-1].get("exploration_pct", 0.0) or 0.0
    prev = history[-1 - window].get("exploration_pct", 0.0) or 0.0
    return float((last - prev) / window)


def compute_coverage_known_to_planner(active_agents, cfg) -> float:
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


class DecisionRateTracker:

    def __init__(self, cfg, window: int = 10):
        self.cfg = cfg
        self.window = window
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
