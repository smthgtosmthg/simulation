"""Aide de vol commune aux expériences.

Le transit vers une cible passe par le couloir ouvert au bout des racks : la ligne droite les
traverse, et aucun évitement d'obstacles n'existe avant la carte de l'étape 4. Un seul module
pour tous les scripts — deux copies finiraient par diverger.
"""

from __future__ import annotations

import numpy as np

from swarm_qr.env.config import INTERIOR


def transit(pilot, layout, x_allee: float, alt: float, psi: float, get_pos, get_yaw) -> None:
    """Amène le drone dans l'allée `x_allee` sans croiser un rack : d'abord rejoindre le
    couloir libre (au nord ou au sud des racks, le plus proche), le longer jusqu'à l'allée,
    puis laisser l'appelant descendre vers sa cible à l'intérieur de l'allée."""
    y_nord = max(r.y_bounds[1] for r in layout.racks) + 1.0
    y_sud = min(r.y_bounds[0] for r in layout.racks) - 1.0
    p = np.array(get_pos(), float)
    couloir = y_nord if abs(p[1] - y_nord) <= abs(p[1] - y_sud) else y_sud
    couloir = float(np.clip(couloir, INTERIOR.y_min + 0.8, INTERIOR.y_max - 0.8))
    for wp in ((p[0], couloir, alt), (x_allee, couloir, alt)):
        pilot.goto(np.array(wp, float), psi, get_pos, tol=0.4,
                   get_yaw=get_yaw, timeout_sim_s=120.0)
