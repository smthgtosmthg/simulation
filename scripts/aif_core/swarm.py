from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

from .agent import DroneAgent
from .architecture import CloudPlanner, DistributedPlanner
from .belief import BeliefGrid, fuse_beliefs_logodds
from .math_utils import logit
from .metrics import (
    DecisionRateTracker,
    compute_coverage_known_to_planner,
    compute_discovery_rate,
)
from .network import (
    CLOUD_DST_ALL,
    CLOUD_SRC,
    LinkState,
    MessageQueue,
    NS3LatencyReader,
)
from .resilience import ResilienceManager
from .stressors import StressContext, StressorScheduler


class SwarmCoordinator:

    def __init__(self, agents: List[DroneAgent], cfg, diag_logger=None):
        self.agents = agents
        self.cfg = cfg
        self.diag = diag_logger

        self.prior_lo = logit(cfg.prior_occupancy)
        self.fused_belief: BeliefGrid = agents[0].belief.copy()
        self.step_count: int = 0
        self.history: List[Dict] = []

        self.msg_queue = MessageQueue()
        self.ns3 = NS3LatencyReader(cfg.ns3_mode)
        self.link_state = LinkState()
        self.resilience = ResilienceManager(cfg)
        self.stressors = StressorScheduler(cfg)
        self.dec_rate = DecisionRateTracker(cfg, window=10)

        if cfg.arch == "centralized":
            self.planner = CloudPlanner(cfg, self.msg_queue, self.ns3, self.link_state)
        elif cfg.arch == "distributed":
            self.planner = DistributedPlanner(cfg, self.msg_queue, self.ns3, self.link_state)
        else:
            raise ValueError(f"Unknown arch: {cfg.arch!r}")

        self.inject_obstacle_fn: Optional[Any] = None

        self.interior_area_cells: int = cfg.interior_area_cells()

        sched_summary = self.stressors.summary()
        if sched_summary:
            print(f"[SWARM] Stresseurs programmés : {sched_summary}")
        print(f"[SWARM] Architecture configurée : {cfg.arch} "
              f"(neighbor_radius_m={cfg.neighbor_radius_m}, "
              f"cloud_round_trip_ms={cfg.cloud_round_trip_ms})")

    @property
    def active_agents(self) -> List[DroneAgent]:
        return [a for a in self.agents if a.active]

    def kill_drone(self, drone_id: int) -> None:
        for a in self.agents:
            if a.id == drone_id and a.active:
                a.land()
                self.resilience.trigger(self.step_count, f"drone_{drone_id}_lost")
                return

    def step(self) -> None:
        # 1) stresseurs : peuvent désactiver un drone ou couper un lien
        ctx = StressContext(
            agents=self.agents,
            link_state=self.link_state,
            resilience=self.resilience,
            cfg=self.cfg,
            inject_obstacle_fn=self.inject_obstacle_fn,
        )
        self.stressors.tick(self.step_count, ctx)

        h_before = self.fused_belief.mean_entropy()

        if self.diag:
            self.diag.log_step_header(self.step_count + 1)

        for a in self.agents:
            a.perceive()
            if self.diag and a.active and a.lidar_diag is not None:
                angles, ranges, hits = a.lidar_diag
                self.diag.log_lidar(a.id, a.x, a.y, angles, ranges, hits)

        active = self.active_agents
        innov_mean = (sum(a.last_innovation for a in active) / len(active)) if active else 0.0
        spike = self.resilience.update_innovation_stats(innov_mean)
        if spike and self.step_count > 5:
            self.resilience.trigger(self.step_count, "innovation_spike")

        if self.ns3.active:
            self.ns3.read_latencies()

        if active:
            self.fused_belief = fuse_beliefs_logodds(
                [a.belief for a in active], self.prior_lo,
            )
        h_after = self.fused_belief.mean_entropy()

        self.resilience.update_phase(self.step_count, h_after, innov_mean)
        phase = self.resilience.current_phase(self.step_count)

        for a in active:
            a.last_action_fresh = False
            a.last_decision_source = "none"

        self.planner.broadcast_beliefs(active, self.step_count)

        self._deliver_pending_messages()

        self.planner.plan_step(active, self.fused_belief, phase, self.step_count)

        self.step_count += 1
        metrics_entry = self._compute_step_metrics(active, h_before, h_after, innov_mean, phase)
        self.history.append(metrics_entry)
        # discovery_rate dépend de history → mis à jour après append
        metrics_entry["discovery_rate"] = round(compute_discovery_rate(self.history), 4)

        if self.diag:
            self.diag.log_step_summary(self.step_count, self.agents, self.fused_belief)
            for a in active:
                if a.candidates_diag:
                    others = [(o.x, o.y) for o in active if o.id != a.id]
                    self.diag.log_candidates(a.id, a.x, a.y, others,
                                             a.candidates_diag, a.selected_idx)
                if a.last_plan_belief is not None:
                    self.diag.log_belief(a.id, a.belief, a.last_plan_belief)
            if phase != "normal":
                self.diag._w(f"    [RESILIENCE] phase={phase} "
                             f"cause={self.resilience.state.cause} "
                             f"innov_mean={innov_mean:.4f} "
                             f"innov_ema={self.resilience.innov_ema:.4f}\n")

    def _deliver_pending_messages(self) -> None:
        delivered = self.msg_queue.deliver(self.step_count)
        agents_by_id = {a.id: a for a in self.agents}
        for src, dst, typ, payload in delivered:
            if typ == "belief":
                if dst in agents_by_id:
                    agents_by_id[dst].last_received_belief[src] = payload
            elif typ == "action":
                if src == CLOUD_SRC and dst in agents_by_id:
                    agents_by_id[dst].pending_action = payload

    def _compute_step_metrics(self, active: List[DroneAgent],
                              h_before: float, h_after: float,
                              innov_mean: float, phase: str) -> Dict:
        bounds_grid = self.cfg.factory_bounds_grid()
        drone_positions = [(a.x, a.y) for a in active]

        coverage_pct = (self.fused_belief.interior_exploration_ratio(
            drone_positions, self.cfg.occ_threshold,
            bounds_grid=bounds_grid,
            interior_area_cells=self.interior_area_cells,
        ) * 100.0) if drone_positions else 0.0

        interior_pct = (self.fused_belief.observed_inside_walls_ratio(
            self.cfg.occ_threshold, bounds_grid=bounds_grid,
        ) * 100.0) if drone_positions else 0.0

        coverage_known = compute_coverage_known_to_planner(active, self.cfg)
        fresh_count = self.dec_rate.record(active)
        decisions_pm = self.dec_rate.rate_per_min()

        q = self.msg_queue.stats()

        eff_arch = ("centralized"
                    if (self.cfg.arch == "centralized" and self.link_state.cloud_link_active)
                    else "distributed")

        return {
            "step": self.step_count,
            "mean_entropy": round(h_after, 4),
            "exploration_pct": round(coverage_pct, 2),
            "exploration_pct_interior": round(interior_pct, 2),
            "exploration_pct_raw": round(self.fused_belief.exploration_ratio() * 100.0, 2),
            "step_info_gain": round(max(0.0, h_before - h_after), 4),
            "free_energies": [round(a.last_G, 4) for a in self.agents],
            "info_gains": [round(a.last_ig, 4) for a in self.agents],
            "innovation_mean": round(innov_mean, 4),
            "innovation_ema": round(self.resilience.innov_ema, 4),
            "resilience_phase": phase,
            "active_drones": len(active),
            "arch_effective": eff_arch,
            "arch_configured": self.cfg.arch,
            "cloud_link_active": self.link_state.cloud_link_active,
            "queue_size": q["queue_size"],
            "msg_sent": q["sent"],
            "msg_delivered": q["delivered"],
            "msg_dropped": q["dropped"],
            "msg_dropped_belief": q["dropped_by_type"].get("belief", 0),
            "msg_dropped_action": q["dropped_by_type"].get("action", 0),
            "coverage_known_to_planner": round(coverage_known, 2),
            "decisions_per_min": round(decisions_pm, 2),
            "fresh_decisions_step": int(fresh_count),
        }

    def get_full_state(self, obstacles: List[Dict]) -> Dict:
        eb = self.fused_belief.effective_bounds(self.cfg.occ_threshold)
        q = self.msg_queue.stats()
        ns3_pairs = []
        for (i, j), info in sorted(self.ns3.last_pairs.items()):
            ns3_pairs.append({
                "a": i, "b": j,
                "latency_ms": info.get("latency_ms", 0.0),
                "jitter_ms": info.get("jitter_ms", 0.0),
                "rx_packets": info.get("rx_packets", 0),
            })
        eff_arch = ("centralized"
                    if (self.cfg.arch == "centralized" and self.link_state.cloud_link_active)
                    else "distributed")
        return {
            "step": self.step_count,
            "timestamp": time.time(),
            "environment": {
                "width": self.cfg.env_width, "height": self.cfg.env_height,
                "grid_resolution": self.cfg.grid_resolution,
                "grid_width": self.cfg.grid_width, "grid_height": self.cfg.grid_height,
                "obstacles": obstacles,
                "effective_bounds": {
                    "x1": eb[0], "y1": eb[1], "x2": eb[2], "y2": eb[3],
                },
            },
            "drones": [a.get_state() for a in self.agents],
            "fused_belief": self.fused_belief.to_list(),
            "metrics": self.history[-1] if self.history else {},
            "resilience": self.resilience.state.to_dict(),
            "planner": self.cfg.planner,
            "arch_configured": self.cfg.arch,
            "arch_effective": eff_arch,
            "ns3_mode": self.cfg.ns3_mode,
            "neighbor_radius_m": self.cfg.neighbor_radius_m,
            "network": {
                "ns3_mode": self.cfg.ns3_mode,
                "cloud_link_active": self.link_state.cloud_link_active,
                "all_drone_links_cut": self.link_state.all_drone_links_cut,
                "cut_pairs": self.link_state.cut_pairs_list(),
                "queue_size": q["queue_size"],
                "msg_sent": q["sent"],
                "msg_delivered": q["delivered"],
                "msg_dropped": q["dropped"],
                "ns3_pairs": ns3_pairs,
            },
        }
