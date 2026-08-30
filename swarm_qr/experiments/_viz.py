"""Caméra de survol et masquage du toit. À importer après le démarrage du simulateur.
Les fonctions d'image pures sont dans `_img`."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCamera, TiledCameraCfg

from ._img import board, label, save, side_by_side, to_bgr, video  # noqa: F401

OVERVIEW_PRIM = "/World/Overview"
ROOF_KEYWORDS = ("ceiling", "beam", "lamp", "pillar", "roof")
LOOK_DOWN = (0.70710678, 0.0, 0.70710678, 0.0)


def hide_roof(stage) -> int:
    """Cache le toit pour que la vue de dessus montre l'intérieur. Purement visuel : la
    géométrie reste en place pour le lidar et les collisions."""
    from pxr import UsdGeom

    n = 0
    for prim in stage.Traverse():
        if any(k in prim.GetName().lower() for k in ROOF_KEYWORDS):
            UsdGeom.Imageable(prim).MakeInvisible()
            n += 1
    return n


def overview_camera(height: float = 26.0, size: int = 720, focal: float = 18.0) -> TiledCamera:
    """Caméra fixe qui regarde l'entrepôt de haut, cadrée sur le bâtiment."""
    return TiledCamera(
        TiledCameraCfg(
            prim_path=OVERVIEW_PRIM,
            offset=TiledCameraCfg.OffsetCfg(pos=(-0.5, 3.0, height), rot=LOOK_DOWN, convention="world"),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=focal, horizontal_aperture=20.955, clipping_range=(0.5, 80.0)
            ),
            width=size,
            height=size,
        )
    )
