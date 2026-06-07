"""
Architecture distribuée — Chaque drone décide localement, fusion avec voisins.

Pipeline par step :
  1. broadcast_beliefs : chaque drone diffuse sa belief à tous ses voisins
     dans `cfg.neighbor_radius_m`, à travers la MessageQueue (avec latence
     NS-3 pair-à-pair).  Liens coupés → message droppé.
  2. (entre les deux étapes, le SwarmCoordinator livre les messages prêts —
     les beliefs sont stockées dans `agent.last_received_belief[src_id]`).
  3. plan_step : chaque drone fait une fusion locale (self.belief +
     last_received_belief de ses voisins actuels et non-coupés), puis
     appelle select_action avec cette belief fusionnée.

C'est l'architecture **autonome** : pas de point central.

Cas particulier : **cut all drone↔drone links**.
  Quand tous les liens sont coupés (`link_state.all_drone_links_cut`), on
  matérialise une phase de **transition** de `switch_latency_ms` :
    - Pendant la transition : drones gelés (pas de décision fresh, pas
      d'exécution) → dec/min ↓, discovery_rate ↓, covKnown ↓.
    - Après la transition : drones reprennent en mode "solo" (fusion avec
      voisins = soi-même seul) → metrics remontent mais plafonnent en
      dessous du niveau baseline.
"""

from __future__ import annotations

import math
from typing import List, Optional

from ..belief import BeliefGrid
from ..math_utils import logit
from ..network import LinkState, MessageQueue, NS3LatencyReader
from ..planner import get_planner


class DistributedPlanner:
    """Stratégie distribuée : fusion par voisinage + décision locale."""

    def __init__(self, cfg, msg_queue: MessageQueue,
                 ns3: NS3LatencyReader, link_state: LinkState):
        self.cfg = cfg
        self.msg_queue = msg_queue
        self.ns3 = ns3
        self.link_state = link_state
        self.prior_lo = logit(cfg.prior_occupancy)
        self._plan_fn = get_planner(cfg.planner)

        # État de transition en cas de cut total (analogue à CloudPlanner)
        self._switch_active_at_step: Optional[int] = None

    # ── Phase courante ──────────────────────────────────────────────

    def _phase(self, current_step: int) -> str:
        """'active' | 'switching' | 'solo' (après reconfig)."""
        if not self.link_state.all_drone_links_cut:
            return "active"
        if self._switch_active_at_step is None:
            return "switching"  # cut juste détecté, transition non encore programmée
        if current_step < self._switch_active_at_step:
            return "switching"
        return "solo"

    # ── Étape 1 : broadcasts entre voisins ──────────────────────────

    def broadcast_beliefs(self, active_agents: List, current_step: int) -> None:
        # Détection du cut total → programmer la transition
        if (self.link_state.all_drone_links_cut
                and self._switch_active_at_step is None):
            switch_steps = max(1, math.ceil(
                self.cfg.switch_latency_ms / self.cfg.step_dt_ms
            ))
            self._switch_active_at_step = current_step + switch_steps
            print(f"  [DIST] All drone↔drone links cut at step {current_step} "
                  f"→ reconfig solo à step {self._switch_active_at_step} "
                  f"(latence switch = {self.cfg.switch_latency_ms} ms = "
                  f"{switch_steps} step(s))")

        # Pendant la transition, on ne broadcast pas
        # (les drones sont en train de "réaliser" qu'ils sont seuls)
        if self._phase(current_step) == "switching":
            return

        # Sinon broadcast normal (les cuts par paires sont gérés par send→None→drop)
        step_dt_ms = self.cfg.step_dt_ms
        R = self.cfg.neighbor_radius_m
        for src in active_agents:
            for dst in active_agents:
                if dst.id == src.id:
                    continue
                if math.hypot(src.x - dst.x, src.y - dst.y) > R:
                    continue
                if self.link_state.is_link_cut(src.id, dst.id):
                    self.msg_queue.send(src.id, dst.id, "belief", src.belief,
                                        current_step, None, step_dt_ms)
                    continue
                lat = self.ns3.pair_latency(src.id, dst.id, fallback=0.0) \
                    if self.ns3.active else 0.0
                self.msg_queue.send(src.id, dst.id, "belief", src.belief.copy(),
                                    current_step, lat, step_dt_ms)

    # ── Étape 3 : planification locale par drone ────────────────────

    def plan_step(self, active_agents: List, global_fused_belief: BeliefGrid,
                  phase: str, current_step: int) -> None:
        del global_fused_belief  # non utilisé en distribué
        ph = self._phase(current_step)

        if ph == "switching":
            # Transition : drones gelés, pas de nouvelle décision
            for a in active_agents:
                a.last_action_fresh = False
                a.last_decision_source = "dist_switching"
                a.last_plan_belief = a.belief  # ne sait plus rien d'autre
                # Pas d'execute → drone garde son dernier waypoint
            return

        # Mode actif (avec voisins) ou solo (sans voisins après cut)
        for a in active_agents:
            others = [(o.x, o.y) for o in active_agents if o.id != a.id]

            neighbors = [
                o for o in active_agents
                if o.id != a.id
                and not self.link_state.is_link_cut(a.id, o.id)
                and math.hypot(a.x - o.x, a.y - o.y) <= self.cfg.neighbor_radius_m
            ]

            plan_belief = a.fuse_with_neighbors(neighbors, self.prior_lo)
            a.last_plan_belief = plan_belief

            action, diag, sel = self._plan_fn(
                a.x, a.y, others, a.belief, plan_belief, self.cfg, a.rng,
                resilience_phase=phase,
            )
            a.candidates_diag = diag
            a.selected_idx = sel
            a.last_action_fresh = True
            a.last_decision_source = "local" if ph == "active" else "local_solo"
            a.execute(action)
