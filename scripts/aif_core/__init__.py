"""
aif_core — Logique de simulation Active Inference pour exploration multi-drones.

Module organisé par responsabilité :
    config.py         — Paramètres de simulation (SimConfig)
    math_utils.py     — Helpers numériques (logit, entropy, softmax)
    belief.py         — Carte de croyance + fusion log-odds
    planner.py        — Sélection d'action (AIF + heuristique)
    network.py        — File de messages, latences NS-3, état des liens
    resilience.py     — Phases recovery / durable, détection de spike
    stressors.py      — Stresseurs programmés (kill drone, cut cloud, cut links)
    architecture/     — Stratégies centralisée vs distribuée
        centralized.py    — Le cloud fusionne, décide, et envoie l'action
        distributed.py    — Chaque drone fusionne avec ses voisins et décide localement
    metrics.py        — Métriques dérivées (discovery_rate, coverage_known, decisions/min)
    agent.py          — DroneAgent (perception, mémoire, exécution)
    swarm.py          — SwarmCoordinator (orchestrateur de tout le pipeline)
    logging.py        — Loggers JSON / diagnostique
"""
