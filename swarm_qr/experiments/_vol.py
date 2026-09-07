"""Chemins de vol pour les expériences, tant que la carte de l'étape 4 n'existe pas.

Les racks sont des boîtes alignées sur x : entre deux racks, la bande d'allée ne contient rien,
donc deux points de la même allée se rejoignent en ligne droite. Pour changer d'allée, on passe
par le couloir libre au bout des racks — la ligne droite les traverserait. Un seul module pour
tous les scripts : deux copies finiraient par diverger.
"""

from __future__ import annotations

import numpy as np

from swarm_qr.env.config import INTERIOR

ALT_TRANSIT_MIN = 1.6     # le couloir nord porte des obstacles bas (1,17 m)
MARGE_COULOIR = 1.0


def allee(layout, x: float) -> int:
    """Numéro de la bande libre qui contient x : le nombre de racks entièrement à sa gauche."""
    return sum(1 for r in layout.racks if r.x_bounds[1] < x)


def couloirs(layout) -> tuple[float, float]:
    y_nord = max(r.y_bounds[1] for r in layout.racks) + MARGE_COULOIR
    y_sud = min(r.y_bounds[0] for r in layout.racks) - MARGE_COULOIR
    lim = (INTERIOR.y_min + 0.8, INTERIOR.y_max - 0.8)
    return float(np.clip(y_nord, *lim)), float(np.clip(y_sud, *lim))


def chemin(layout, depart, cible) -> list[np.ndarray]:
    """Points de passage de `depart` à `cible` : aucun dans la même allée, sinon deux par le
    couloir qui raccourcit le plus le trajet. L'altitude des points reste au-dessus des
    obstacles bas du couloir."""
    p = np.asarray(depart, float)
    q = np.asarray(cible, float)
    if allee(layout, p[0]) == allee(layout, q[0]):
        return []
    y_nord, y_sud = couloirs(layout)
    y = min((y_nord, y_sud), key=lambda c: abs(p[1] - c) + abs(q[1] - c))
    z = max(q[2], ALT_TRANSIT_MIN)
    return [np.array([p[0], y, z]), np.array([q[0], y, z])]


def transit(pilot, layout, x_allee: float, alt: float, psi: float, get_pos, get_yaw) -> None:
    """Amène le drone dans l'allée `x_allee` par le couloir, puis laisse l'appelant descendre
    vers sa cible à l'intérieur de l'allée."""
    p = np.asarray(get_pos(), float)
    for wp in chemin(layout, p, np.array([x_allee, p[1], alt])) or []:
        pilot.goto(wp, psi, get_pos, tol=0.4, get_yaw=get_yaw, timeout_sim_s=120.0)
