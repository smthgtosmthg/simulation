"""Construction de la scène : entrepôt, racks déplacés, cartons, QR, drones et capteurs.

Ce module s'importe uniquement après le démarrage du simulateur.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCaster, MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster.patterns import LidarPatternCfg
from isaacsim.core.utils.stage import add_reference_to_stage
import isaacsim.core.utils.prims as prim_utils
import omni.usd

from . import qr_tags
from .config import (
    CAMERAS,
    DRONES,
    LIDAR,
    QR_CFG,
    RACKS,
    SIM_DT,
    WAREHOUSE_PRIM,
    WAREHOUSE_USD,
)
from .layout import Layout, select_boxes

DRONE_ROOT = "/World/Drone"
BODY = "Body"


def _yaw_quat(deg: float) -> tuple[float, float, float, float]:
    h = math.radians(deg) / 2.0
    return (math.cos(h), 0.0, 0.0, math.sin(h))


@dataclass
class Scene:
    layout: Layout
    drones: RigidObject
    cam_left: TiledCamera
    cam_right: TiledCamera
    cam_front: TiledCamera
    lidar: MultiMeshRayCaster
    tags: list = field(default_factory=list)
    kept_boxes: list = field(default_factory=list)

    @property
    def cameras(self) -> dict[str, TiledCamera]:
        return {"left": self.cam_left, "right": self.cam_right, "front": self.cam_front}

    def update(self, dt: float = SIM_DT) -> None:
        self.drones.update(dt)
        for cam in self.cameras.values():
            cam.update(dt)
        self.lidar.update(dt)

    def positions(self) -> torch.Tensor:
        return self.drones.data.root_pos_w

    def orientations(self) -> torch.Tensor:
        return self.drones.data.root_quat_w

    def command_velocity(self, vel: torch.Tensor) -> None:
        """vel : (n_drones, 6) — trois vitesses de déplacement puis trois de rotation, en monde."""
        self.drones.write_root_velocity_to_sim(vel)

    def rgb(self, name: str) -> torch.Tensor:
        return self.cameras[name].data.output["rgb"]

    def ranges(self) -> torch.Tensor:
        hits = self.lidar.data.ray_hits_w
        d = torch.norm(hits - self.lidar.data.pos_w.unsqueeze(1), dim=-1)
        return torch.nan_to_num(d, nan=LIDAR.max_range, posinf=LIDAR.max_range)


def _spawn_warehouse() -> None:
    add_reference_to_stage(WAREHOUSE_USD, WAREHOUSE_PRIM)
    sim_utils.GroundPlaneCfg().func("/World/Ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95))
    )


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
    """Un carton est le prim le plus HAUT dont le nom correspond. Ses enfants portent souvent le
    même nom : les compter aussi doublerait les cartons et les QR."""
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


def _camera_cfg(name: str, width: int, height: int, yaw_deg: float) -> TiledCameraCfg:
    """La caméra est posée sur la coque, pas au centre : au centre elle filmerait l'intérieur
    du corps du drone."""
    out = DRONES.body_size / 2.0 + 0.03
    a = math.radians(yaw_deg)
    pos = (out * math.cos(a), out * math.sin(a), 0.0)
    return TiledCameraCfg(
        prim_path=f"{DRONE_ROOT}_.*/{BODY}/{name}",
        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=_yaw_quat(yaw_deg), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAMERAS.focal_length,
            horizontal_aperture=CAMERAS.horizontal_aperture,
            clipping_range=(CAMERAS.near, CAMERAS.far),
        ),
        width=width,
        height=height,
        update_period=1.0 / CAMERAS.update_hz,
        update_latest_camera_pose=True,
    )


def build(layout: Layout, device: str = "cuda:0") -> Scene:
    _spawn_warehouse()
    stage = omni.usd.get_context().get_stage()
    _place_racks(stage, layout)

    all_boxes = _find_boxes(stage)
    kept = select_boxes(all_boxes, layout)
    _hide(stage, [p for p in all_boxes if p not in set(kept)])

    images = qr_tags.generate_images(len(kept))
    tags = qr_tags.attach(stage, kept, images)

    for k in range(DRONES.count):
        prim_utils.create_prim(f"{DRONE_ROOT}_{k}", "Xform")

    drones = RigidObject(
        RigidObjectCfg(
            prim_path=f"{DRONE_ROOT}_.*/{BODY}",
            spawn=sim_utils.CuboidCfg(
                size=(DRONES.body_size,) * 3,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 0.9)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=layout.spawns[0]),
        )
    )

    cam_left = TiledCamera(_camera_cfg("CamL", CAMERAS.side_width, CAMERAS.side_height, 90.0))
    cam_right = TiledCamera(_camera_cfg("CamR", CAMERAS.side_width, CAMERAS.side_height, -90.0))
    cam_front = TiledCamera(_camera_cfg("CamF", CAMERAS.front_width, CAMERAS.front_height, 0.0))

    lidar = MultiMeshRayCaster(
        MultiMeshRayCasterCfg(
            prim_path=f"{DRONE_ROOT}_.*/{BODY}",
            mesh_prim_paths=[
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr=WAREHOUSE_PRIM, merge_prim_meshes=True, track_mesh_transforms=False
                )
            ],
            pattern_cfg=LidarPatternCfg(
                channels=LIDAR.channels,
                vertical_fov_range=(0.0, 0.0),
                horizontal_fov_range=(-180.0, 180.0),
                horizontal_res=LIDAR.horizontal_res_deg,
            ),
            ray_alignment="base",
            max_distance=LIDAR.max_range,
            update_period=1.0 / LIDAR.update_hz,
            attach_yaw_only=False,
            debug_vis=False,
        )
    )

    return Scene(
        layout=layout,
        drones=drones,
        cam_left=cam_left,
        cam_right=cam_right,
        cam_front=cam_front,
        lidar=lidar,
        tags=tags,
        kept_boxes=kept,
    )


def place_drones(scene: Scene, device: str = "cuda:0") -> None:
    """À appeler après sim.reset() : positionne chaque drone à son point de départ."""
    n = DRONES.count
    state = scene.drones.data.default_root_state.clone()
    for k, (x, y, z) in enumerate(scene.layout.spawns[:n]):
        state[k, 0:3] = torch.tensor([x, y, z], device=state.device)
        state[k, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=state.device)
        state[k, 7:13] = 0.0
    scene.drones.write_root_state_to_sim(state)
