from __future__ import annotations

from typing import List, Protocol


class ArchitecturePlanner(Protocol):

    def broadcast_beliefs(self, active_agents: List, current_step: int) -> None:
        ...

    def plan_step(self, active_agents: List, global_fused_belief,
                  phase: str, current_step: int) -> None:
        ...
