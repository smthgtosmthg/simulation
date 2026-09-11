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
from .config import CAMERAS, DRONES, LIDAR, QR_CFG, RACKS, WAREHOUSE_PRIM, WAREHOUSE_USD
from .layout import Layout, select_boxes

DRONE_PRIM = "/World/Drone_{:02d}"
GROUND_Z = 0.07

# Le rendu a jusqu'à 4 images de retard sur la position réelle : 10 rendus garantissent une
# image à jour. Chaque rendu est précédé d'un peu de physique, sans quoi le pilote ArduPilot,
# qui tourne dans un processus séparé, cesse d'être alimenté et perd le contrôle du drone.
RENDER_LAG = 10
PHYSICS_BETWEEN_RENDERS = 40

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
    lidars: list[str] = field(default_factory=list)
    tags: list = field(default_factory=list)
    kept_boxes: list = field(default_factory=list)
    _lidar_api: object = None

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

    def position(self, drone: int = 0) -> np.ndarray:
        return np.array(self.drones[drone].state.position, float)

    def yaw(self, drone: int = 0) -> float:
        return float(Rotation.from_quat(self.drones[drone].state.attitude).as_euler("ZYX")[0])

    def velocity(self, drone: int = 0) -> np.ndarray:
        return np.array(self.drones[drone].state.linear_velocity, float)

    def rgb(self, name: str, drone: int = 0) -> np.ndarray:
        return self.cameras[drone][name].get_rgb()

    def lidar(self, drone: int = 0):
        """Un tour de lidar en repère monde : directions unitaires et distance par rayon, la
        distance valant l'infini quand rien n'a été touché.

        Deux pièges du capteur, tous deux silencieux :
        - il ne se met à jour qu'au **rendu** : lire après des pas de physique seuls renvoie un
          tour entier de zéros — on lève une erreur plutôt que de remplir la carte de vide ;
        - son angle vertical se compte **vers le bas** : un « zénith » de −15 degrés pointe
          15 degrés vers le haut. Sans le signe, la carte se remplit tête-bêche.
        Les angles sont donnés dans le repère du capteur ; on les tourne avec l'attitude du
        drone, sans quoi la carte se remplirait de travers dès qu'il tourne.
        """
        from omni.isaac.range_sensor import _range_sensor

        if self._lidar_api is None:
            self._lidar_api = _range_sensor.acquire_lidar_sensor_interface()
        chemin = self.lidars[drone]
        d = self._lidar_api.get_linear_depth_data(chemin)
        if d is None:
            raise RuntimeError("lidar sans donnees : le capteur n'est pas initialise")
        portees = np.asarray(d, dtype=float).reshape(-1)
        if not np.count_nonzero(portees > LIDAR.min_range):
            raise RuntimeError("lidar muet : aucun rendu depuis le dernier deplacement")
        az = np.asarray(self._lidar_api.get_azimuth_data(chemin), dtype=float)
        zen = np.asarray(self._lidar_api.get_zenith_data(chemin), dtype=float)
        A, Z = np.meshgrid(az, zen, indexing="ij")
        local = np.stack([np.cos(Z) * np.cos(A), np.cos(Z) * np.sin(A), -np.sin(Z)], axis=-1)
        R = Rotation.from_quat(self.drones[drone].state.attitude).as_matrix()
        dirs = local.reshape(-1, 3) @ R.T
        portees[portees >= LIDAR.max_range - 1e-3] = np.inf
        return dirs, portees

    def capture(self, name: str, drone: int = 0, settle: int = RENDER_LAG) -> np.ndarray:
        """Image à jour d'une caméra du drone. Voir `capture_camera` pour les pièges traités."""
        return capture_camera(self.world, self.cameras[drone][name], settle)


def capture_camera(world: World, cam: Camera, settle: int = RENDER_LAG,
                   physique_entre_rendus: int = PHYSICS_BETWEEN_RENDERS) -> np.ndarray:
    """Image à jour d'une caméra. `physique_entre_rendus=0` est réservé aux scènes sans drone
    SITL : il n'y a alors personne à alimenter et la capture est quatre fois plus rapide."""
    for _ in range(settle):
        for _ in range(physique_entre_rendus):
            world.step(render=False)
        world.step(render=True)
    for _ in range(120):
        img = cam.get_rgb()
        if img is not None and getattr(img, "ndim", 0) == 3 and img.size:
            return img
        world.step(render=True)
    raise RuntimeError("camera vide apres 120 rendus")


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
    """Désactive les cartons non retenus. Les rendre seulement invisibles laissait leur
    collider en place : le lidar et les rayons de contrôle butaient sur des cartons que la
    caméra ne voyait pas (mesuré à l'étape 7 : neuf cartons cachés sur neuf arrêtaient les
    rayons). Un prim désactivé sort de la composition : ni rendu, ni physique."""
    for p in paths:
        prim = stage.GetPrimAtPath(p)
        if prim and prim.IsValid():
            prim.SetActive(False)


def _orientation_monde(yaw_deg: float, plongee_deg: float) -> np.ndarray:
    """Quaternion (w, x, y, z) en convention monde de la classe Camera (avant = +X, haut = +Z) :
    un lacet autour de Z, puis une plongée vers le bas autour de Y."""
    a, b = math.radians(yaw_deg) / 2.0, math.radians(plongee_deg) / 2.0
    ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
    return np.array([ca * cb, -sa * sb, ca * sb, sa * cb])


# Cinq caméras fixes de vidéosurveillance, à 3,5 m sur les murs, qui regardent le long des
# couloirs : à cette hauteur on voit toute la longueur d'un couloir, ses deux faces de rack, et
# à quel étage vole chaque drone. La cinquième, en hauteur dans un coin, voit tout l'entrepôt.
# Positions pour l'entrepôt 9033 (murs à x = -10,5 et 9,25, y = -12,25 et 17,75).
CAMERAS_VIDEO = {
    "sud_ouest":    dict(position=(-9.40, -11.6, 3.5), yaw=90.0,  plongee=8.0,  fov=60.0),
    "sud_central":  dict(position=(-4.96, -11.6, 3.5), yaw=90.0,  plongee=8.0,  fov=60.0),
    "nord_central": dict(position=(-4.96, 17.3, 3.5),  yaw=-90.0, plongee=8.0,  fov=60.0),
    "sud_grande":   dict(position=(2.70, -11.6, 3.5),  yaw=90.0,  plongee=8.0,  fov=80.0),
    "ensemble":     dict(position=(-10.1, -11.6, 5.8), yaw=56.0,  plongee=22.0, fov=95.0),
}
# Le jeu retenu par l'utilisatrice : les deux caméras qui regardent dans les deux couloirs
# principaux, le couloir central et la grande zone. Les trois autres restent disponibles.
CAMERAS_VIDEO_2 = {n: CAMERAS_VIDEO[n] for n in ("sud_central", "sud_grande")}
CAMERAS_VIDEO_3 = {n: CAMERAS_VIDEO[n] for n in ("ensemble", "sud_central", "sud_grande")}
RESOLUTION_VIDEO = (960, 540)


def cameras_fixes(specs: dict = CAMERAS_VIDEO, resolution=RESOLUTION_VIDEO) -> dict[str, Camera]:
    cams = {}
    for nom, c in specs.items():
        cam = Camera(prim_path=f"/World/Video/Cam_{nom}", position=np.array(c["position"], dtype=float),
                     orientation=_orientation_monde(c["yaw"], c["plongee"]), resolution=resolution)
        cam.initialize()                                # branche la caméra au rendu (après world.reset)
        # ouverture ET focale, dans la même unité que les caméras des drones : avec l'ouverture par
        # défaut, une focale de 18 donnait un téléobjectif dix fois trop serré
        cam.set_horizontal_aperture(CAMERAS.horizontal_aperture)
        cam.set_focal_length(CAMERAS.horizontal_aperture / (2.0 * math.tan(math.radians(c["fov"]) / 2.0)))
        cam.set_clipping_range(0.2, 80.0)
        cams[nom] = cam
    return cams


def ajoute_obstacle(nom: str, centre_xy, dims=(1.0, 1.0, 2.0)):
    """Un bloc plein posé au sol en cours de mission, avec son collider : le lidar doit le
    découvrir et la carte doit faire recalculer les chemins. Retourne son emprise."""
    from isaacsim.core.api.objects import FixedCuboid

    lx, ly, lz = dims
    FixedCuboid(prim_path=f"/World/Obstacles/{nom}", position=np.array([centre_xy[0], centre_xy[1], lz / 2.0]),
                scale=np.array([lx, ly, lz]), size=1.0, color=np.array([0.85, 0.35, 0.1]))
    return {"nom": nom, "x": [centre_xy[0] - lx / 2, centre_xy[0] + lx / 2],
            "y": [centre_xy[1] - ly / 2, centre_xy[1] + ly / 2], "z": [0.0, lz]}


def _drone_cameras(drone_prim: str) -> dict[str, Camera]:
    """Deux latérales haute résolution pour lire, une frontale basse résolution pour voir.
    Montées sous le ventre (z -0,11) : au-dessus de ce plan, la coque de l'Iris (z -0,067 à
    +0,047) et les disques d'hélices (z +0,02, rayon 0,13 aux quatre coins) restent à plus de
    25 degrés au-dessus de l'axe optique, hors du champ vertical de ±23,6 degrés."""
    out_dist = CAMERAS.side_offset
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
            translation=np.array([out_dist * math.cos(a), out_dist * math.sin(a), -CAMERAS.below]),
            orientation=_cam_orientation(yaw),
            resolution=(w, h),
        )
    return cams


def _add_lidar(drone_prim: str) -> str:
    """Lidar au centre du corps. Sa portée minimale passe au-delà des hélices, sinon le drone
    se mesurerait lui-même à chaque rayon."""
    import omni.kit.commands

    omni.kit.commands.execute(
        "RangeSensorCreateLidar",
        path="/body/Lidar",
        parent=drone_prim,
        min_range=LIDAR.min_range,
        max_range=LIDAR.max_range,
        draw_points=False,
        draw_lines=False,
        horizontal_fov=LIDAR.horizontal_fov_deg,
        vertical_fov=LIDAR.vertical_fov_deg,
        horizontal_resolution=LIDAR.horizontal_res_deg,
        vertical_resolution=LIDAR.vertical_res_deg,
        rotation_rate=0.0,
        high_lod=True,
        yaw_offset=0.0,
        enable_semantics=False,
    )
    return f"{drone_prim}/body/Lidar"


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

    drones, cameras, lidars = [], [], []
    for i in range(n):
        x, y, _ = layout.spawns[i]
        drones.append(_make_drone(i, (x, y), with_sitl, pg))
        cameras.append(_drone_cameras(DRONE_PRIM.format(i)))
        lidars.append(_add_lidar(DRONE_PRIM.format(i)))

    return Scene(
        world=world,
        layout=layout,
        drones=drones,
        cameras=cameras,
        lidars=lidars,
        tags=tags,
        kept_boxes=kept,
    )
