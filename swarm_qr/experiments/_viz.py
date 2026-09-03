"""Caméra de survol et masquage du toit, sur les caméras natives d'Isaac Sim.
À importer après le démarrage du simulateur. Les fonctions d'image pures sont dans `_img`."""

from __future__ import annotations

import numpy as np
from isaacsim.sensors.camera import Camera

from ._img import board, label, save, side_by_side, to_bgr, video  # noqa: F401

OVERVIEW_PRIM = "/World/Overview"
ROOF_KEYWORDS = ("ceiling", "beam", "lamp", "pillar", "roof")


def hide_roof(stage) -> int:
    """Cache le toit pour que la vue de dessus montre l'intérieur. Purement visuel : la
    géométrie reste en place pour les capteurs et les collisions."""
    from pxr import UsdGeom

    n = 0
    for prim in stage.Traverse():
        if any(k in prim.GetName().lower() for k in ROOF_KEYWORDS):
            UsdGeom.Imageable(prim).MakeInvisible()
            n += 1
    return n


def overview_camera(height: float = 26.0, size: int = 720, focal: float = 18.0) -> Camera:
    """Caméra fixe qui regarde l'entrepôt de haut. Orientation en convention monde
    (avant = +X) : un tangage de +90° fait regarder droit vers le sol."""
    return Camera(
        prim_path=OVERVIEW_PRIM,
        position=np.array([-0.5, 3.0, height]),
        orientation=np.array([0.70710678, 0.0, 0.70710678, 0.0]),
        resolution=(size, size),
    )


def overview_init(cam: Camera, focal: float = 18.0) -> None:
    """Après world.reset() : branche la caméra et règle son objectif."""
    cam.initialize()
    cam.set_focal_length(focal)
    cam.set_horizontal_aperture(20.955)
    cam.set_clipping_range(0.5, 80.0)
