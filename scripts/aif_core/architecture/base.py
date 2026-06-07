"""
Interface abstraite des planners d'architecture.

Un planner d'architecture est responsable de :
  1. Décider l'action de chaque drone actif au step courant.
  2. Émettre les messages réseau associés (broadcasts de beliefs, envoi d'actions).
  3. Marquer chaque agent comme ayant pris une décision "fresh" ou "stale".

Deux implémentations :
  - centralized.CloudPlanner  — le cloud décide pour tous
  - distributed.DistributedPlanner — chaque drone décide pour lui-même
"""

from __future__ import annotations

from typing import List, Protocol


class ArchitecturePlanner(Protocol):
    """Contrat minimal d'un planner d'architecture."""

    def broadcast_beliefs(self, active_agents: List, current_step: int) -> None:
        """Programme les envois de beliefs dans la MessageQueue.

        En centralisé : drone → cloud (sera utilisé par cloud à plan_step).
        En distribué  : drone → voisins (pour fusion locale).
        """
        ...

    def plan_step(self, active_agents: List, global_fused_belief,
                  phase: str, current_step: int) -> None:
        """Détermine et applique l'action de chaque drone actif.

        Modifie en place :
          - agent.last_action, agent.candidates_diag, agent.selected_idx
          - agent.last_action_fresh : True si décision fraîche, False si stale
          - agent.last_plan_belief : la belief utilisée pour décider (pour métriques)
        Et déclenche `agent.execute(action)` pour envoyer le waypoint au backend.
        """
        ...
