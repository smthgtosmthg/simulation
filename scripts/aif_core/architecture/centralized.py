from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from ..belief import BeliefGrid
from ..math_utils import logit
from ..network import CLOUD_SRC, LinkState, MessageQueue, NS3LatencyReader
from ..planner import get_planner
from .distributed import DistributedPlanner


class CloudPlanner:

    def __init__(self, cfg, msg_queue: MessageQueue,
                 ns3: NS3LatencyReader, link_state: LinkState):
        self.cfg = cfg
        self.msg_queue = msg_queue
        self.ns3 = ns3
        self.link_state = link_state
        self.prior_lo = logit(cfg.prior_occupancy)
        self._plan_fn = get_planner(cfg.planner)

        # Fallback distribué après transition
        self._dist_fallback = DistributedPlanner(cfg, msg_queue, ns3, link_state)

        self._switch_active_at_step: Optional[int] = None

    def _phase(self, current_step: int) -> str:
        if self.link_state.cloud_link_active:
            return "cloud"
        if self._switch_active_at_step is None:
            return "switching"
        if current_step < self._switch_active_at_step:
            return "switching"
        return "distributed_fallback"

    def broadcast_beliefs(self, active_agents: List, current_step: int) -> None:
        # Détection initiale du cut → on programme la transition
        if not self.link_state.cloud_link_active and self._switch_active_at_step is None:
            switch_steps = max(1, math.ceil(
                self.cfg.switch_latency_ms / self.cfg.step_dt_ms
            ))
            self._switch_active_at_step = current_step + switch_steps
            self._drop_in_flight_actions()
            print(f"  [CLOUD] Cloud cut détecté à step {current_step} "
                  f"→ basculement vers distribué à step {self._switch_active_at_step} "
                  f"(latence switch = {self.cfg.switch_latency_ms} ms = "
                  f"{switch_steps} step(s))")

        if self._phase(current_step) == "distributed_fallback":
            self._dist_fallback.broadcast_beliefs(active_agents, current_step)

    def _drop_in_flight_actions(self) -> None:
        kept = []
        dropped = 0
        for entry in self.msg_queue._pending:  # noqa: SLF001
            _, src, _, typ, _ = entry
            if typ == "action" and src == CLOUD_SRC:
                dropped += 1
                continue
            kept.append(entry)
        self.msg_queue._pending = kept  # noqa: SLF001
        self.msg_queue.dropped += dropped
        self.msg_queue.dropped_by_type["action"] = (
            self.msg_queue.dropped_by_type.get("action", 0) + dropped
        )
        if dropped > 0:
            print(f"  [CLOUD] ✂ {dropped} action(s) en vol droppée(s) (cloud cut)")

    def plan_step(self, active_agents: List, global_fused_belief: BeliefGrid,
                  phase: str, current_step: int) -> None:
        ph = self._phase(current_step)

        if ph == "cloud":
            self._plan_cloud_mode(active_agents, global_fused_belief, phase, current_step)
            return

        if ph == "switching":
            self._plan_switching_mode(active_agents)
            return

        self._dist_fallback.plan_step(
            active_agents, global_fused_belief, phase, current_step,
        )

    def _plan_cloud_mode(self, active_agents: List,
                         global_fused_belief: BeliefGrid,
                         phase: str, current_step: int) -> None:
        actions = self._cloud_compute_actions(active_agents, global_fused_belief, phase)
        self._cloud_send_actions(actions, current_step)

        for a in active_agents:
            if a.pending_action is not None:
                action = a.pending_action
                a.pending_action = None
                a.last_action_fresh = True
                a.last_decision_source = "cloud"
                a.execute(action)
            else:
                # Warmup : pas encore reçu d'action du cloud
                a.last_action_fresh = False
                a.last_decision_source = "cloud_warmup"
                a.last_plan_belief = a.belief
                a.execute(("stay", 0.0, 0.0))

    def _cloud_compute_actions(self, active_agents: List,
                               global_fused_belief: BeliefGrid,
                               phase: str
                               ) -> Dict[int, Tuple[str, float, float]]:
        actions: Dict[int, Tuple[str, float, float]] = {}
        intended_targets: List[Tuple[float, float]] = []

        for a in active_agents:
            current_others = [(o.x, o.y) for o in active_agents if o.id != a.id]
            others = current_others + intended_targets

            a.last_plan_belief = global_fused_belief
            action, diag, sel = self._plan_fn(
                a.x, a.y, others, a.belief, global_fused_belief,
                self.cfg, a.rng, resilience_phase=phase,
            )
            actions[a.id] = action

            _, dx, dy = action
            intended_targets.append((
                a.x + dx * self.cfg.step_size,
                a.y + dy * self.cfg.step_size,
            ))

            a.candidates_diag = diag
            a.selected_idx = sel

        return actions

    def _cloud_send_actions(self, actions: Dict[int, Tuple[str, float, float]],
                            current_step: int) -> None:
        rtt = self.cfg.cloud_round_trip_ms
        step_dt_ms = self.cfg.step_dt_ms
        for did, action in actions.items():
            self.msg_queue.send(CLOUD_SRC, did, "action", action,
                                current_step, rtt, step_dt_ms)

    def _plan_switching_mode(self, active_agents: List) -> None:
        for a in active_agents:
            a.last_action_fresh = False
            a.last_decision_source = "switching"
            a.last_plan_belief = a.belief
