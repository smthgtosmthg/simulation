"""Assemblage de la scène sur la base Pegasus + ArduPilot SITL.

Entrepôt chargé par URL directe (cache local), racks déplacés selon la graine, cartons
filtrés, QR collés, drones Iris de Pegasus, caméras natives d'Isaac Sim.

S'importe uniquement après le démarrage de SimulationApp. Aucune dépendance à Isaac Lab :
la World de Pegasus et le SimulationContext d'Isaac Lab sont incompatibles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from isaacsim.core.api.world import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from scipy.spatial.transform import Rotation

import omni.usd

from pegasus.simulator.logic.backends.ardupilot_mavlink_backend import (
    ArduPilotMavlinkBackend,
    ArduPilotMavlinkBackendConfig,
)
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.params import ROBOTS, WORLD_SETTINGS

from . import qr_tags
from .config import CAMERAS, DRONES, QR_CFG, RACKS, WAREHOUSE_PRIM, WAREHOUSE_USD
from .layout import Layout, select_boxes

DRONE_PRIM = "/World/Drone_{:02d}"
GROUND_Z = 0.07

def _cam_orientation(yaw_deg: float) -> np.ndarray:
    """Orientation locale d'une caméra visant à `yaw_deg` du cap du drone. La classe Camera
    interprète l'orientation en convention monde (avant = +X, haut = +Z) et fait elle-même la
    conversion vers le repère USD : un simple lacet suffit."""
    h = math.radians(yaw_deg) / 2.0
    return np.array([math.cos(h), 0.0, 0.0, math.sin(h)])


@dataclass
class Scene:
    world: World
    layout: Layout
    drones: list[Multirotor]
    cameras: list[dict[str, Camera]]
    tags: list = field(default_factory=list)
    kept_boxes: list = field(default_factory=list)

    def finalize(self) -> None:
        """À appeler une fois, après world.reset() : branche les caméras au rendu."""
        for per_drone in self.cameras:
            for cam in per_drone.values():
                cam.initialize()
                cam.set_focal_length(CAMERAS.focal_length)
                cam.set_horizontal_aperture(CAMERAS.horizontal_aperture)
                cam.set_clipping_range(CAMERAS.near, CAMERAS.far)

    def positions(self) -> np.ndarray:
        return np.array([d.state.position for d in self.drones])

    def rgb(self, name: str, drone: int = 0) -> np.ndarray:
        return self.cameras[drone][name].get_rgb()


def _add_light(stage) -> None:
    from pxr import UsdLux

    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(2500.0)


def _place_racks(stage, layout: Layout) -> None:
    from pxr import Gf, UsdGeom

    for placement in layout.racks:
        prim = stage.GetPrimAtPath(f"{WAREHOUSE_PRIM}/{placement.prim}")
        if not prim or not prim.IsValid():
            continue
        default_x = RACKS.default_x[RACKS.prims.index(placement.prim)]
        dx = placement.x - default_x
        dy = placement.y_min - RACKS.default_y_min
        xf = UsdGeom.Xformable(prim)
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                cur = op.Get() or Gf.Vec3d(0, 0, 0)
                op.Set(Gf.Vec3d(cur[0] + dx, cur[1] + dy, cur[2]))
                break


def _find_boxes(stage) -> list[str]:
    """Le prim le plus haut dont le nom correspond : ses enfants portent souvent le même nom,
    les compter aussi doublerait cartons et QR."""
    from pxr import UsdGeom

    needle = QR_CFG.box_name_filter.lower()
    out: list[str] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Xformable):
            continue
        if needle not in prim.GetName().lower():
            continue
        path = str(prim.GetPath())
        if any(path.startswith(kept + "/") for kept in out):
            continue
        out.append(path)
    return sorted(out)


def _hide(stage, paths) -> None:
    from pxr import UsdGeom

    for p in paths:
        prim = stage.GetPrimAtPath(p)
        if prim and prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()


def _drone_cameras(drone_prim: str) -> dict[str, Camera]:
    """Deux latérales haute résolution pour lire, une frontale basse résolution pour voir.
    Montées sous le ventre (z -0,11) : au-dessus de ce plan, la coque de l'Iris (z -0,067 à
    +0,047) et les disques d'hélices (z +0,02, rayon 0,13 aux quatre coins) restent à plus de
    25 degrés au-dessus de l'axe optique, hors du champ vertical de ±23,6 degrés."""
    out_dist = 0.10
    specs = {
        "left": (90.0, CAMERAS.side_width, CAMERAS.side_height),
        "right": (-90.0, CAMERAS.side_width, CAMERAS.side_height),
        "front": (0.0, CAMERAS.front_width, CAMERAS.front_height),
    }
    cams = {}
    for name, (yaw, w, h) in specs.items():
        a = math.radians(yaw)
        cams[name] = Camera(
            prim_path=f"{drone_prim}/body/Cam_{name}",
            translation=np.array([out_dist * math.cos(a), out_dist * math.sin(a), -0.11]),
            orientation=_cam_orientation(yaw),
            resolution=(w, h),
        )
    return cams


def _make_drone(index: int, spawn_xy: tuple, with_sitl: bool, pg: PegasusInterface) -> Multirotor:
    config = MultirotorConfig()
    if with_sitl:
        config.backends = [
            ArduPilotMavlinkBackend(
                config=ArduPilotMavlinkBackendConfig(
                    {
                        "vehicle_id": index,
                        "ardupilot_autolaunch": True,
                        "ardupilot_dir": pg.ardupilot_path,
                        "ardupilot_vehicle_model": "gazebo-iris",
                    }
                )
            )
        ]
    else:
        config.backends = []

    return Multirotor(
        DRONE_PRIM.format(index),
        ROBOTS["Iris"],
        index,
        [spawn_xy[0], spawn_xy[1], GROUND_Z],
        Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
        config=config,
    )


def build(layout: Layout, with_sitl: bool = False, n_drones: int | None = None) -> Scene:
    """Construit la scène complète. `with_sitl=False` : drones posés, inertes — suffisant pour
    les tests qui ne volent pas, et aucun terminal ne s'ouvre."""
    n = DRONES.count if n_drones is None else n_drones

    pg = PegasusInterface()
    pg._world = World(**WORLD_SETTINGS["ardupilot"])
    world = pg.world

    add_reference_to_stage(usd_path=WAREHOUSE_USD, prim_path=WAREHOUSE_PRIM)
    world.scene.add_default_ground_plane()

    stage = omni.usd.get_context().get_stage()
    _add_light(stage)
    _place_racks(stage, layout)

    all_boxes = _find_boxes(stage)
    kept = select_boxes(all_boxes, layout)
    _hide(stage, [p for p in all_boxes if p not in set(kept)])

    images = qr_tags.generate_images(len(kept))
    tags = qr_tags.attach(stage, kept, images)

    drones, cameras = [], []
    for i in range(n):
        x, y, _ = layout.spawns[i]
        drones.append(_make_drone(i, (x, y), with_sitl, pg))
        cameras.append(_drone_cameras(DRONE_PRIM.format(i)))

    return Scene(
        world=world,
        layout=layout,
        drones=drones,
        cameras=cameras,
        tags=tags,
        kept_boxes=kept,
    )
