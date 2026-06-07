"""
Stratégies d'architecture pour la planification multi-drones.

- centralized.CloudPlanner : un "cloud" reçoit les beliefs, les fusionne, décide
  l'action de chaque drone (de manière séquentielle, en tenant compte des cibles
  déjà attribuées), puis renvoie l'action à chaque drone via la MessageQueue.
  → Coordination forte, latence cloud aller-retour, single point of failure.

- distributed.DistributedPlanner : chaque drone reçoit les beliefs de ses voisins
  dans un rayon donné (via la MessageQueue, avec latence NS-3), les fusionne avec
  la sienne, puis décide son action **localement**.
  → Pas de point central, autonomie individuelle, moins de coordination.
"""

from .base import ArchitecturePlanner
from .centralized import CloudPlanner
from .distributed import DistributedPlanner

__all__ = ["ArchitecturePlanner", "CloudPlanner", "DistributedPlanner"]
