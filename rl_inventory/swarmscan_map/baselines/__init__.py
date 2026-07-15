"""Baselines de comparaison (conception §7.2) : planner type Pore et Active Inference.

Les deux pilotent les MÊMES drones dans le MÊME environnement (SwarmScanMapEnv),
avec le MÊME gate de lecture calibré et les MÊMES métriques que la politique RL.
Différence d'information, fidèle à chaque paradigme :
 - Pore : carte + positions des tags CONNUES (waypoints pré-planifiés, secteurs statiques) ;
 - AIF  : AUCUNE position — croyance construite en ligne, comme la politique RL.
"""
