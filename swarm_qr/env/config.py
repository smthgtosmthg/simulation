"""Constantes de l'environnement, toutes mesurées dans le fichier USD de l'entrepôt."""

from __future__ import annotations

import math
from dataclasses import dataclass

WAREHOUSE_USD = (
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/4.2/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
)
WAREHOUSE_PRIM = "/World/Warehouse"


@dataclass(frozen=True)
class Interior:
    x_min: float = -10.13
    x_max: float = 9.21
    y_min: float = -12.00
    y_max: float = 17.80
    z_ceiling: float = 9.00
    z_fly_max: float = 5.80

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def length(self) -> float:
        return self.y_max - self.y_min


@dataclass(frozen=True)
class Racks:
    prims: tuple[str, ...] = ("Shelf_0", "Shelf_1", "Shelf_2")
    default_x: tuple[float, ...] = (-9.30, -0.30, 8.68)
    depth: float = 1.40
    length: float = 17.62
    height: float = 6.00
    default_y_min: float = -4.489
    shelf_levels: tuple[float, ...] = (1.325, 2.825, 4.125)


@dataclass(frozen=True)
class MapGrid:
    x_min: float = -11.0
    x_max: float = 10.0
    y_min: float = -13.0
    y_max: float = 19.0
    z_min: float = 0.0
    z_max: float = 6.0
    cell: float = 0.25

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            round((self.x_max - self.x_min) / self.cell),
            round((self.y_max - self.y_min) / self.cell),
            round((self.z_max - self.z_min) / self.cell),
        )


@dataclass(frozen=True)
class Drones:
    count: int = 3
    body_size: float = 0.18
    spawn_z_range: tuple[float, float] = (1.2, 2.2)
    clearance: float = 1.2
    max_speed: float = 1.0


@dataclass(frozen=True)
class Cameras:
    side_width: int = 1024
    side_height: int = 768
    front_width: int = 160
    front_height: int = 120
    fov_deg: float = 60.0
    horizontal_aperture: float = 20.955
    near: float = 0.05
    far: float = 40.0
    update_hz: float = 5.0
    side_offset: float = 0.10      # les latérales sont à 10 cm du centre du corps...
    below: float = 0.11            # ...et 11 cm plus bas ; vérifié en vol (06_position_vraie)

    @property
    def focal_length(self) -> float:
        return self.horizontal_aperture / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))


@dataclass(frozen=True)
class Lidar:
    """Un tour complet à chaque rendu, sans rotation mécanique : le lidar sert à
    cartographier, pas à imiter un capteur tournant. Dix anneaux sur 36 degrés couvrent, à
    2 mètres, une bande verticale de 1,3 m : trois passages à hauteur des trois étagères
    se recouvrent."""

    horizontal_fov_deg: float = 360.0
    vertical_fov_deg: float = 36.0
    horizontal_res_deg: float = 2.0
    vertical_res_deg: float = 4.0
    min_range: float = 0.4          # au-delà des hélices : sinon le drone se mesure lui-même
    max_range: float = 25.0


@dataclass(frozen=True)
class QR:
    box_name_filter: str = "SM_CardBox"
    payload_fmt: str = "BOX_{:03d}"
    panel_ratio: float = 0.4
    offset: float = 0.01
    faces: tuple[tuple[int, int], ...] = ((0, 1), (0, -1))


SIM_DT = 1.0 / 60.0


class RenderClock:
    """Décide quand le simulateur rend une image. Rendre à chaque pas de physique coûte sept
    fois le temps réel ; à 5 images par seconde le coût s'effondre et la lecture des QR reste
    largement assez fréquente pour un drone lent."""

    def __init__(self, hz: float):
        self.every = render_interval(hz)
        self.i = -1

    def step(self, sim) -> bool:
        self.i += 1
        rendered = self.i % self.every == 0
        sim.step(render=rendered)
        return rendered


def render_interval(hz: float) -> int:
    """Nombre de pas de physique entre deux rendus. C'est ce réglage qui décide du coût :
    limiter la relecture du capteur ne sert à rien si le simulateur rend quand même."""
    return max(1, round(1.0 / (hz * SIM_DT)))


INTERIOR = Interior()
RACKS = Racks()
MAP = MapGrid()
DRONES = Drones()
CAMERAS = Cameras()
LIDAR = Lidar()
QR_CFG = QR()

OBSTACLES = (
    (-7.06, -5.19, 12.90, 14.36, 1.17),
    (3.52, 5.38, 12.90, 14.36, 1.17),
)

TRAIN_SEEDS = range(0, 500)
SEALED_SEEDS = range(9000, 9040)
