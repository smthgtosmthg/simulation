#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aif_core.agent import DroneAgent
from aif_core.config import SimConfig
from aif_core.loggers import DataLogger, DiagnosticLogger
from aif_core.swarm import SwarmCoordinator


_Backend = None
_Rotation = None


def _ensure_pegasus_imports():
    global _Backend, _Rotation
    if _Backend is None:
        from pegasus.simulator.logic.backends.backend import Backend
        from scipy.spatial.transform import Rotation
        _Backend, _Rotation = Backend, Rotation


AifStateTracker = None


def _create_state_tracker_class():
    global AifStateTracker
    _ensure_pegasus_imports()

    class _AifStateTracker(_Backend):

        def __init__(self, drone_id: int):
            self.drone_id = drone_id
            self.p = np.zeros(3)
            self.v = np.zeros(3)
            self.R = np.eye(3)
            self.w = np.zeros(3)
            self.received_first_state = False
            self._vehicle = None

        def get_position(self) -> np.ndarray:
            return self.p.copy()

        def get_position_xy(self) -> Tuple[float, float]:
            return float(self.p[0]), float(self.p[1])

        def get_yaw(self) -> float:
            if not self.received_first_state:
                return 0.0
            return float(math.atan2(self.R[1, 0], self.R[0, 0]))

        @property
        def vehicle(self):
            return self._vehicle

        def initialize(self, vehicle):
            self._vehicle = vehicle

        def update_state(self, state):
            self.p = np.array(state.position)
            self.v = np.array(state.linear_velocity)
            self.R = _Rotation.from_quat(state.attitude).as_matrix()
            self.w = np.array(state.angular_velocity)
            self.received_first_state = True

        def update_sensor(self, sensor_type, data): pass
        def update_graphical_sensor(self, sensor_type, data): pass
        def input_reference(self): return [0.0, 0.0, 0.0, 0.0]
        def update(self, dt): pass
        def start(self): pass
        def stop(self): pass
        def reset(self): self.received_first_state = False

    AifStateTracker = _AifStateTracker


def _patch_sitl_defaults():
    parm_file = os.path.expanduser(
        "~/ardupilot/Tools/autotest/default_params/gazebo-iris.parm"
    )
    if not os.path.isfile(parm_file):
        print(f"[WARN] gazebo-iris.parm introuvable : {parm_file}")
        return

    required = {
        "ARMING_CHECK":    "0",
        "SCHED_LOOP_RATE": "50",
        "FS_THR_ENABLE":   "0",
        "FS_GCS_ENABLE":   "0",
        "FS_CRASH_CHECK":  "0",
    }

    with open(parm_file) as f:
        lines = f.readlines()

    idx_map: Dict[str, int] = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            parts = stripped.split()
            if len(parts) >= 2:
                idx_map[parts[0]] = i

    modified = False
    for param, value in required.items():
        if param in idx_map:
            i = idx_map[param]
            old = lines[i].strip().split()[1] if len(lines[i].strip().split()) >= 2 else ""
            if old != value:
                lines[i] = f"{param} {value}\n"
                modified = True
                print(f"  [SITL-PARM] {param}: {old} → {value}")
        else:
            lines.append(f"{param} {value}\n")
            modified = True
            print(f"  [SITL-PARM] ajouté {param} = {value}")

    if modified:
        with open(parm_file, "w") as f:
            f.writelines(lines)
        print(f"[INFO] Params SITL patchés dans {parm_file}")
    else:
        print(f"[INFO] {parm_file} — params SITL déjà corrects")


class SitlController:

    GUIDED_MODE = 4
    POS_YAW_MASK = 0b0000_1011_1111_1000

    def __init__(self, drone_id: int, mavlink_backend, state_tracker,
                 spawn_pos: np.ndarray, cfg: SimConfig):
        self.drone_id = drone_id
        self._mav = mavlink_backend
        self._tracker = state_tracker
        self._spawn = spawn_pos.copy()
        self.cfg = cfg
        self.target = spawn_pos.copy()
        self.target_yaw = 0.0
        self.arrived = False
        self._armed = False
        self._cmd_count = 0

    def _conn(self):
        return self._mav._connection

    def _drain(self):
        conn = self._conn()
        if conn is None:
            return
        while conn.recv_match(blocking=False) is not None:
            pass

    def set_guided(self):
        from pymavlink import mavutil as _mav
        conn = self._conn()
        if conn is None:
            return
        conn.target_system = self.drone_id + 1
        conn.target_component = 1
        conn.mav.set_mode_send(
            conn.target_system,
            _mav.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            self.GUIDED_MODE,
        )

    def force_arm(self):
        from pymavlink import mavutil as _mav
        conn = self._conn()
        if conn is None:
            return
        conn.target_system = self.drone_id + 1
        conn.target_component = 1
        self._drain()
        conn.mav.command_long_send(
            conn.target_system, conn.target_component,
            _mav.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 21196, 0, 0, 0, 0, 0,
        )
        self._armed = True

    def is_flying(self, min_alt: float = 0.5) -> bool:
        if not self._tracker.received_first_state:
            return False
        return float(self._tracker.get_position()[2]) > min_alt

    def takeoff(self, altitude: float):
        from pymavlink import mavutil as _mav
        conn = self._conn()
        if conn is None:
            return
        conn.mav.command_long_send(
            conn.target_system, conn.target_component,
            _mav.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude,
        )

    def send_backup_params(self):
        from pymavlink import mavutil as _mav
        conn = self._conn()
        if conn is None:
            return
        conn.target_system = self.drone_id + 1
        conn.target_component = 1
        for name, value in [(b"ARMING_CHECK", 0), (b"SCHED_LOOP_RATE", 50),
                            (b"FS_THR_ENABLE", 0)]:
            conn.mav.param_set_send(
                conn.target_system, conn.target_component,
                name, float(value),
                _mav.mavlink.MAV_PARAM_TYPE_REAL32,
            )

    def set_target(self, x: float, y: float, z: float, yaw: float = 0.0):
        from pymavlink import mavutil as _mav
        self.target = np.array([x, y, z])
        self.target_yaw = yaw
        self.arrived = False

        conn = self._conn()
        if conn is None:
            return

        ned_n = y - self._spawn[1]
        ned_e = x - self._spawn[0]
        ned_d = -(z - self._spawn[2])
        ned_yaw = math.pi / 2 - yaw

        conn.mav.set_position_target_local_ned_send(
            0,
            conn.target_system, conn.target_component,
            _mav.mavlink.MAV_FRAME_LOCAL_NED,
            self.POS_YAW_MASK,
            ned_n, ned_e, ned_d,
            0, 0, 0, 0, 0, 0,
            ned_yaw, 0,
        )

        self._cmd_count += 1
        if self._cmd_count <= 3 or self._cmd_count % 100 == 0:
            print(f"  [SITL] D{self.drone_id} waypoint #{self._cmd_count} "
                  f"isaac=({x:.1f},{y:.1f},{z:.1f}) "
                  f"ned=({ned_n:.1f},{ned_e:.1f},{ned_d:.1f})")

    def get_position(self) -> np.ndarray:
        return self._tracker.get_position()

    def get_position_xy(self) -> Tuple[float, float]:
        return self._tracker.get_position_xy()

    def get_yaw(self) -> float:
        return self._tracker.get_yaw()

    def is_at_target(self) -> bool:
        pos = self.get_position()
        self.arrived = np.linalg.norm(pos[:2] - self.target[:2]) < self.cfg.waypoint_tol
        return self.arrived


class LidarReader:

    def __init__(self, drone_id: int, drone_prim_path: str, cfg: SimConfig):
        from isaacsim.sensors.physx import RotatingLidarPhysX

        self.prim_path = f"{drone_prim_path}/body/Lidar_{drone_id}"
        self.cfg = cfg
        self._read_count = 0

        self.sensor = RotatingLidarPhysX(
            prim_path=self.prim_path,
            rotation_frequency=cfg.lidar_hz,
            fov=(cfg.lidar_fov_h, cfg.lidar_fov_v),
            resolution=(cfg.lidar_fov_h / cfg.num_rays, cfg.lidar_vert_res),
            valid_range=(cfg.lidar_min_range, cfg.lidar_max_range),
        )
        self.sensor.add_linear_depth_data_to_frame()
        self.sensor.add_azimuth_data_to_frame()
        try:
            self.sensor.add_zenith_data_to_frame()
            self._has_zenith = True
        except Exception:
            self._has_zenith = False
            print(f"[WARN] Zenith data unavailable for {self.prim_path}, floor filtering disabled")
        self.sensor.enable_visualization(high_lod=False, draw_points=True, draw_lines=False)
        print(f"[INFO] PhysX LiDAR attached: {self.prim_path}")

        self._acc_angles: List[np.ndarray] = []
        self._acc_depths: List[np.ndarray] = []
        self._acc_hits:   List[np.ndarray] = []

    def initialize(self):
        self.sensor.initialize()
        self.sensor.post_reset()
        print(f"[INFO] LiDAR initialized: {self.prim_path}")

    def _parse_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        frame = self.sensor.get_current_frame()
        depth = frame.get("linear_depth")
        azimuth = frame.get("azimuth")
        if depth is None or azimuth is None or len(depth) == 0:
            return None
        depth = np.asarray(depth, dtype=np.float64).ravel()
        azimuth = np.asarray(azimuth, dtype=np.float64).ravel()
        if azimuth.size > 0 and azimuth.max() > 2 * math.pi + 0.1:
            azimuth = np.deg2rad(azimuth)

        zenith = frame.get("zenith") if self._has_zenith else None
        if zenith is not None and len(zenith) > 0:
            zenith = np.asarray(zenith, dtype=np.float64).ravel()
            if zenith.size > 0 and zenith.max() > 2 * math.pi + 0.1:
                zenith = np.deg2rad(zenith)
            if zenith.size > 0 and np.median(zenith) < math.pi / 4:
                cos_elev = np.cos(zenith)
                sin_elev = np.sin(zenith)
                horizontal_range = depth * np.abs(cos_elev)
                hit_z = self.cfg.fly_altitude + depth * sin_elev
            else:
                sin_zen = np.sin(zenith)
                cos_zen = np.cos(zenith)
                horizontal_range = depth * np.abs(sin_zen)
                hit_z = self.cfg.fly_altitude + depth * cos_zen
            valid_height = hit_z > self.cfg.floor_filter_z
            hits = (horizontal_range >= self.cfg.lidar_min_range) & \
                   (horizontal_range < (self.cfg.lidar_max_range - 0.05)) & valid_height
            depth = horizontal_range
        else:
            hits = (depth >= self.cfg.lidar_min_range) & (depth < (self.cfg.lidar_max_range - 0.05))

        return azimuth, depth, hits

    def accumulate(self, yaw: float = 0.0):
        parsed = self._parse_frame()
        if parsed is None:
            return
        angles, depths, hits = parsed
        world_angles = (angles + yaw) % (2 * math.pi)
        self._acc_angles.append(world_angles)
        self._acc_depths.append(depths)
        self._acc_hits.append(hits)

    def get_accumulated(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._acc_angles:
            n = self.cfg.num_rays
            return (
                np.linspace(0, 2 * math.pi, n, endpoint=False),
                np.full(n, self.cfg.lidar_max_range),
                np.zeros(n, dtype=bool),
            )
        angles = np.concatenate(self._acc_angles)
        depths = np.concatenate(self._acc_depths)
        hits = np.concatenate(self._acc_hits)
        self._acc_angles.clear()
        self._acc_depths.clear()
        self._acc_hits.clear()

        n = self.cfg.num_rays
        bin_width = 2 * math.pi / n
        bins = (angles / bin_width).astype(int) % n

        out_angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
        out_depths = np.full(n, self.cfg.lidar_max_range)
        out_hits = np.zeros(n, dtype=bool)

        for i in range(n):
            mask = bins == i
            if not mask.any():
                continue
            b_d = depths[mask]
            b_h = hits[mask]
            if b_h.any():
                hit_d = b_d[b_h]
                out_depths[i] = hit_d.min()
                out_hits[i] = True
            else:
                out_depths[i] = b_d.max()

        self._read_count += 1
        if self._read_count <= 5 or self._read_count % 50 == 0:
            n_hits = int(out_hits.sum())
            pct = n_hits / max(n, 1) * 100
            print(f"  [LiDAR] {self.prim_path} scan#{self._read_count}: "
                  f"raw_rays={len(angles)}  binned={n}  hits={n_hits} ({pct:.0f}%)")
        return out_angles, out_depths, out_hits


def _import_ns3_bridge():
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    bridge_path = os.path.join(here, "12_ns3_bridge.py")
    if not os.path.isfile(bridge_path):
        return None
    spec = importlib.util.spec_from_file_location("ns3_bridge", bridge_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_drone_positions(agents: List[DroneAgent],
                           path: str = "/tmp/drone_positions.csv"):
    try:
        with open(path, "w") as f:
            f.write("drone_id,x,y,z\n")
            for a in agents:
                if not a.active:
                    continue
                if a.controller is not None:
                    p = a.controller.get_position()
                    x, y, z = float(p[0]), float(p[1]), float(p[2])
                else:
                    x, y, z = float(a.x), float(a.y), 2.0
                f.write(f"{a.id},{x:.4f},{y:.4f},{z:.4f}\n")
    except OSError:
        pass


def parse_args() -> SimConfig:
    p = argparse.ArgumentParser(description="Active Inference drones — Isaac Sim")
    p.add_argument("--num-drones", type=int, default=3)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--env-width", type=float, default=30.0)
    p.add_argument("--env-height", type=float, default=20.0)

    p.add_argument("--planner", choices=["aif", "heuristic"], default="aif")
    p.add_argument("--arch", choices=["centralized", "distributed"], default="centralized")
    p.add_argument("--neighbor-radius-m", type=float, default=5.0)

    p.add_argument("--ns3", choices=["none", "wifi", "5g"], default="wifi")
    p.add_argument("--cloud-round-trip-ms", type=float, default=500.0,
                   help="Latence cloud aller-retour (upload belief + calcul + download action)")
    p.add_argument("--ns3-sim-time", type=int, default=600)

    p.add_argument("--kill-drone-at-step", type=int, default=-1)
    p.add_argument("--kill-drone-id", type=int, default=0)
    p.add_argument("--cut-cloud-at-step", type=int, default=-1)
    p.add_argument("--cut-drone-link", default="",
                   help="i-j (ex: '0-1') ou 'all'")
    p.add_argument("--cut-drone-link-at-step", type=int, default=-1)
    p.add_argument("--drop-obstacle-at-step", type=int, default=-1)
    p.add_argument("--drop-obstacle-xy", default="",
                   help="'x,y' coords monde de l'obstacle dynamique")

    p.add_argument("--run-tag", default="default")
    p.add_argument("--runs-dir", default="")

    a = p.parse_args()
    return SimConfig(
        num_drones=a.num_drones, headless=a.headless,
        max_steps=a.max_steps, env_width=a.env_width, env_height=a.env_height,
        planner=a.planner, arch=a.arch, neighbor_radius_m=a.neighbor_radius_m,
        ns3_mode=a.ns3, cloud_round_trip_ms=a.cloud_round_trip_ms,
        ns3_sim_time=a.ns3_sim_time,
        kill_drone_at_step=a.kill_drone_at_step,
        kill_drone_id=a.kill_drone_id,
        cut_cloud_at_step=a.cut_cloud_at_step,
        cut_drone_link=a.cut_drone_link,
        cut_drone_link_at_step=a.cut_drone_link_at_step,
        drop_obstacle_at_step=a.drop_obstacle_at_step,
        drop_obstacle_xy=a.drop_obstacle_xy,
        run_tag=a.run_tag, runs_dir=a.runs_dir,
    )


def create_sim_app(cfg: SimConfig):
    from isaacsim import SimulationApp
    app_cfg = {
        "headless": cfg.headless,
        "extra_args": ["--/rtx/verifyDriverVersion/enabled=false"],
    }
    if not cfg.headless:
        app_cfg["width"] = 1280
        app_cfg["height"] = 720
    return SimulationApp(app_cfg)


def setup_world():
    from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
    pif = PegasusInterface()
    pif.initialize_world()
    world = pif.world
    world.scene.add_default_ground_plane()
    factory_bounds = _load_factory_environment()
    _add_scene_lighting()
    return world, factory_bounds


def _load_factory_environment():
    try:
        from isaacsim.core.utils.stage import add_reference_to_stage
    except ImportError:
        from omni.isaac.core.utils.stage import add_reference_to_stage

    import omni.usd
    from pxr import UsdGeom

    forced_usd = os.getenv(
        "AIF_FACTORY_USD",
        "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
    ).strip()

    stage = omni.usd.get_context().get_stage()
    print(f"[INFO] Factory USD forced path: {forced_usd}")

    try:
        add_reference_to_stage(usd_path=forced_usd, prim_path="/World/Factory")
    except Exception as e:
        raise RuntimeError(f"Failed to load forced factory USD: {forced_usd} ({e})") from e

    prim = stage.GetPrimAtPath("/World/Factory")
    if not (prim.IsValid() and prim.GetChildren()):
        stage.RemovePrim("/World/Factory")
        raise RuntimeError(f"Forced factory USD is unreachable or empty: {forced_usd}")

    bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(prim)
    rng = bbox.GetRange()
    lo = rng.GetMin()
    hi = rng.GetMax()
    print(f"[INFO] Factory bounding box: X=[{lo[0]:.2f}, {hi[0]:.2f}] "
          f"Y=[{lo[1]:.2f}, {hi[1]:.2f}] Z=[{lo[2]:.2f}, {hi[2]:.2f}]")
    print(f"[INFO] Factory environment loaded OK: {forced_usd}")
    return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))


def _add_scene_lighting():
    from pxr import Gf, UsdGeom, UsdLux
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    stage.DefinePrim("/World/Lights", "Xform")

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(1000.0)
    dome.CreateColorAttr(Gf.Vec3f(0.85, 0.9, 1.0))

    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(3000.0)
    sun.CreateAngleAttr(1.0)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))
    xf = UsdGeom.Xformable(sun.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 30.0, 0.0))

    print("[INFO] Scene lighting added (dome + sun)")


def setup_viewport_camera(cfg: SimConfig):
    if cfg.headless:
        return

    cx = cfg.env_width / 2
    cy = cfg.env_height / 2
    tz = cfg.fly_altitude
    cam_x, cam_y, cam_z = cx, cy - 18.0, 25.0

    try:
        import omni.usd
        from pxr import Gf, UsdGeom
        import omni.kit.viewport.utility as vp_utils

        viewport = vp_utils.get_active_viewport()
        if viewport is None:
            print("[WARN] setup_viewport_camera: no active viewport, skipping.")
            return

        stage = omni.usd.get_context().get_stage()
        persp_prim = stage.GetPrimAtPath("/OmniverseKit_Persp")
        if persp_prim and persp_prim.IsValid():
            UsdGeom.Camera(persp_prim).CreateClippingRangeAttr(Gf.Vec2f(0.01, 500.0))

        try:
            from omni.kit.viewport.utility.camera_state import ViewportCameraState
            cs = ViewportCameraState(viewport)
            cs.set_position_world(Gf.Vec3d(cam_x, cam_y, cam_z), True)
            cs.set_target_world(Gf.Vec3d(cx, cy, tz), True)
            print(f"[INFO] Overview camera → eye=({cam_x:.1f}, {cam_y:.1f}, {cam_z:.1f})")
            return
        except Exception as e1:
            print(f"[INFO] ViewportCameraState unavailable ({e1}), USD camera fallback…")

        cam_path = "/World/OverviewCamera"
        camera = UsdGeom.Camera.Define(stage, cam_path)
        camera.CreateFocalLengthAttr(18.0)
        camera.CreateHorizontalApertureAttr(36.0)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 500.0))
        dY = cy - cam_y
        dZ = tz - cam_z
        pitch_deg = math.degrees(math.atan2(-dZ, dY))
        xf = UsdGeom.Xformable(camera.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(cam_x, cam_y, cam_z))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(pitch_deg, 0.0, 0.0))
        viewport.set_active_camera(cam_path)
    except Exception as e:
        print(f"[WARN] Could not configure viewport camera: {e}")


def create_physical_drones(agents: List[DroneAgent], cfg: SimConfig):
    from pegasus.simulator.params import ROBOTS
    from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
    from pegasus.simulator.logic.sensors.barometer import Barometer
    from pegasus.simulator.logic.sensors.imu import IMU
    from pegasus.simulator.logic.sensors.gps import GPS
    from pegasus.simulator.logic.backends.ardupilot_mavlink_backend import (
        ArduPilotMavlinkBackend, ArduPilotMavlinkBackendConfig,
    )
    from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

    _create_state_tracker_class()

    ardupilot_dir = PegasusInterface().ardupilot_path

    multirotors = []
    controllers: List[SitlController] = []
    for agent in agents:
        gx = agent.trail[0][0]
        gy = agent.trail[0][1]
        sx = gx + cfg.origin_x
        sy = gy + cfg.origin_y
        spawn_pos = np.array([sx, sy, 0.1])

        mav_cfg = ArduPilotMavlinkBackendConfig({
            "vehicle_id": agent.id,
            "ardupilot_autolaunch": True,
            "ardupilot_dir": ardupilot_dir,
            "ardupilot_vehicle_model": "gazebo-iris",
        })
        mav_backend = ArduPilotMavlinkBackend(config=mav_cfg)

        tracker = AifStateTracker(agent.id)

        mc = MultirotorConfig()
        mc.backends = [mav_backend, tracker]
        mc.sensors = [Barometer(), IMU(), GPS()]

        prim_path = f"/World/Drone_{agent.id:02d}"
        drone = Multirotor(
            prim_path, ROBOTS["Iris"], agent.id,
            [sx, sy, 0.1], [0.0, 0.0, 0.0, 1.0], mc,
        )
        multirotors.append(drone)

        controller = SitlController(agent.id, mav_backend, tracker, spawn_pos, cfg)
        controllers.append(controller)

        lidar = LidarReader(agent.id, prim_path, cfg)
        agent.setup_physical(controller, lidar)

        print(
            f"[INFO] Drone {agent.id} — grid({gx:.1f},{gy:.1f}) "
            f"→ world({sx:.1f},{sy:.1f}) — ArduPilot SITL + PhysX LiDAR"
        )

    return multirotors, controllers


def inject_dynamic_obstacle(wx: float, wy: float):
    try:
        from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        idx = int(time.time() * 1000) % 100000
        prim_path = f"/World/DynamicObstacles/Cube_{idx}"

        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.GetSizeAttr().Set(1.0)

        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(wx, wy, 2.0))
        xf.AddScaleOp().Set(Gf.Vec3f(3.0, 3.0, 4.0))

        cube.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.15, 0.15)])

        # double API requise pour que le LiDAR PhysX raycaste le cube
        prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        PhysxSchema.PhysxCollisionAPI.Apply(prim)

        print(f"[OBSTACLE] Cube 3×3×4m rouge injecté à world=({wx:.2f}, {wy:.2f}, z=2.0)")
        print(f"           prim_path={prim_path}  (CollisionAPI + PhysxCollisionAPI)")
    except Exception as e:
        import traceback
        print(f"[OBSTACLE] échec injection : {e}")
        traceback.print_exc()


def main():
    cfg = parse_args()

    running = True

    def _sig(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    print(f"[INFO] AIF config :")
    print(f"  planner={cfg.planner}  arch={cfg.arch}  neighbor_radius_m={cfg.neighbor_radius_m}")
    print(f"  ns3={cfg.ns3_mode}  cloud_round_trip_ms={cfg.cloud_round_trip_ms}")
    print(f"  kill_drone_at_step={cfg.kill_drone_at_step}  "
          f"cut_cloud_at_step={cfg.cut_cloud_at_step}  "
          f"cut_drone_link={cfg.cut_drone_link!r} @ step={cfg.cut_drone_link_at_step}")
    print(f"  drop_obstacle_at_step={cfg.drop_obstacle_at_step}  "
          f"drop_obstacle_xy={cfg.drop_obstacle_xy!r}")
    print(f"  run_tag={cfg.run_tag!r}")

    ns3_bridge = None
    if cfg.ns3_mode != "none":
        ns3_bridge = _import_ns3_bridge()

    print(f"[INFO] Isaac Sim — {'headless' if cfg.headless else 'GUI'}")
    sim_app = create_sim_app(cfg)

    world, factory_bounds = setup_world()

    if factory_bounds:
        margin = 1.0
        bx0, by0, bx1, by1 = factory_bounds
        cfg.world_origin_x = bx0 - margin
        cfg.world_origin_y = by0 - margin
        cfg.env_width = (bx1 - bx0) + 2 * margin
        cfg.env_height = (by1 - by0) + 2 * margin
        cfg.env_width = math.ceil(cfg.env_width / cfg.grid_resolution) * cfg.grid_resolution
        cfg.env_height = math.ceil(cfg.env_height / cfg.grid_resolution) * cfg.grid_resolution
        cfg.factory_bounds_world = (float(bx0), float(by0), float(bx1), float(by1))
        print(f"[INFO] Env adapted to factory: {cfg.env_width:.1f} × {cfg.env_height:.1f} m  "
              f"origin=({cfg.origin_x:.2f}, {cfg.origin_y:.2f})  "
              f"grid={cfg.grid_width}×{cfg.grid_height}")

    setup_viewport_camera(cfg)
    obstacles: List[Dict] = []

    agents: List[DroneAgent] = []
    for i in range(cfg.num_drones):
        gx = cfg.env_width / 2 + (i - (cfg.num_drones - 1) / 2) * cfg.drone_spacing
        gy = max(8.0, cfg.env_height * 0.20)
        agents.append(DroneAgent(i, gx, gy, cfg))

    if ns3_bridge is not None:
        _write_drone_positions(agents)
        ns3_bridge.launch_ns3(
            n_drones=cfg.num_drones,
            sim_time=cfg.ns3_sim_time,
            scenario=cfg.ns3_mode,
        )

    _patch_sitl_defaults()

    multirotors, controllers = create_physical_drones(agents, cfg)
    diag_logger = DiagnosticLogger(cfg.output_dir, cfg)
    coordinator = SwarmCoordinator(agents, cfg, diag_logger=diag_logger)
    coordinator.inject_obstacle_fn = inject_dynamic_obstacle
    logger = DataLogger(cfg.output_dir)

    from qr_code_system import setup_qr_system, initialize_cameras
    qr_data = os.getenv("QR_CODE_DATA", "DRONE_WAREHOUSE_INSPECTION_001")
    qr_panel_pos = (0.0, -5.0, 2.0)
    drone_prim_paths = [f"/World/Drone_{a.id:02d}" for a in agents]
    drone_ids = [a.id for a in agents]
    qr_sys = setup_qr_system(
        qr_data=qr_data,
        output_dir=cfg.output_dir,
        drone_prim_paths=drone_prim_paths,
        drone_ids=drone_ids,
        panel_position=qr_panel_pos,
        panel_size=2.5,
        cache_ttl=3.0,
        capture_interval=5,
        camera_resolution=(640, 480),
    )
    print(f"[INFO] QR code system ready — data='{qr_data}'")

    world.reset()

    for agent in agents:
        if agent.lidar is not None:
            agent.lidar.initialize()

    initialize_cameras(qr_sys["cameras"])

    print(f"[INFO] {cfg.num_drones} drones | grid {cfg.grid_width}×{cfg.grid_height}")
    print(f"[INFO] {cfg.sim_steps_per_aif} physics ticks per AIF decision")
    print(f"[INFO] Dashboard → {cfg.output_dir}/aif_state.json")

    print("[INFO] Phase 1/3 : initialisation ArduPilot SITL + EKF (2500 ticks)…")
    for _ in range(2500):
        if not running or not sim_app.is_running():
            break
        world.step(render=not cfg.headless)

    print("[INFO] Phase 2/3 : GUIDED + ARM + TAKEOFF …")
    for ctrl in controllers:
        ctrl.send_backup_params()
    for _ in range(100):
        world.step(render=not cfg.headless)

    for ctrl in controllers:
        ctrl.set_guided()
        for _ in range(30):
            world.step(render=not cfg.headless)
        ctrl.force_arm()
        for _ in range(10):
            world.step(render=not cfg.headless)
        ctrl.takeoff(cfg.fly_altitude)
        print(f"  [SITL] D{ctrl.drone_id} → GUIDED + ARM + TAKEOFF({cfg.fly_altitude:.1f}m)")
        for _ in range(200):
            world.step(render=not cfg.headless)

    print(f"[INFO] Phase 3/3 : attente altitude ({cfg.fly_altitude:.1f} m)…")
    drone_flying = [False] * len(controllers)
    RETRY_TICKS = (500, 1000, 1500, 2000, 2500)

    for tick in range(3000):
        if not running or not sim_app.is_running():
            break
        world.step(render=not cfg.headless)
        for agent in agents:
            agent.accumulate_lidar()

        for i, ctrl in enumerate(controllers):
            if not drone_flying[i] and ctrl.is_flying(cfg.fly_altitude * 0.5):
                drone_flying[i] = True
                print(f"  [TAKEOFF] D{ctrl.drone_id} en vol ✓")

        if tick in RETRY_TICKS:
            for i, ctrl in enumerate(controllers):
                if not drone_flying[i]:
                    ctrl.set_guided()
                    ctrl.force_arm()
                    ctrl.takeoff(cfg.fly_altitude)
                    print(f"  [RETRY] D{ctrl.drone_id} re-arm+takeoff (tick={tick})")

        if tick % 300 == 299:
            positions = [a.controller.get_position() for a in agents if a.controller]
            alts = [p[2] for p in positions]
            alt_str = ", ".join(f"D{i}={a:.2f}m" for i, a in enumerate(alts))
            n_fly = sum(drone_flying)
            print(f"  [TAKEOFF] tick={tick} {n_fly}/{len(controllers)} en vol | {alt_str}")

        if all(drone_flying):
            print("[INFO] Tous les drones en vol !")
            break

    if not all(drone_flying):
        not_fly = [i for i, f in enumerate(drone_flying) if not f]
        print(f"[WARN] Drones pas en vol après 3000 ticks : {not_fly}")
    print("[INFO] Décollage terminé.\n")

    qr_decoder = qr_sys["decoder_thread"]
    qr_decoder.start()
    qr_capture = qr_sys["capture_helper"]
    print("[INFO] QR decoder thread started")

    aif_step = 0
    plateau_counter = 0
    plateau_threshold = 0.1
    plateau_patience = 15
    prev_coverage = 0.0

    while running and aif_step < cfg.max_steps and sim_app.is_running():
        # perception, fusion, planning, exécution
        coordinator.step()

        # vol physique : N ticks pendant que les drones se déplacent
        for _ in range(cfg.sim_steps_per_aif):
            if not running or not sim_app.is_running():
                break
            world.step(render=not cfg.headless)
            for agent in agents:
                agent.accumulate_lidar()
            qr_capture.tick()

        if cfg.ns3_mode != "none":
            _write_drone_positions(agents)

        state = coordinator.get_full_state(obstacles)
        logger.log(state, coordinator.history)

        m = coordinator.history[-1]
        positions_str = " | ".join(
            f"D{a.id}({'X' if not a.active else f'{a.x:.1f},{a.y:.1f}'})" for a in agents
        )
        phase_tag = (f" [{m.get('resilience_phase','normal').upper()}]"
                     if m.get('resilience_phase','normal') != 'normal' else "")
        print(
            f"  Step {aif_step:4d} | "
            f"H={m['mean_entropy']:.3f} | "
            f"Expl={m['exploration_pct']:5.1f}% | "
            f"ΔIG={m['step_info_gain']:.4f} | "
            f"innov={m.get('innovation_mean',0):.3f} | "
            f"discRate={m.get('discovery_rate',0):.3f} | "
            f"covPlan={m.get('coverage_known_to_planner',0):.1f}% | "
            f"dec/min={m.get('decisions_per_min',0):.0f} | "
            f"active={m.get('active_drones', len(agents))}{phase_tag} | "
            f"{positions_str}"
        )

        cur_coverage = m["exploration_pct"]
        if cur_coverage >= cfg.target_coverage:
            print(f"\n[INFO] Target coverage {cfg.target_coverage}% reached "
                  f"at step {aif_step} (coverage={cur_coverage:.1f}%).")
            break

        if cur_coverage - prev_coverage < plateau_threshold:
            plateau_counter += 1
        else:
            plateau_counter = 0
        prev_coverage = cur_coverage

        if plateau_counter >= plateau_patience:
            print(f"\n[INFO] Coverage plateau at {cur_coverage:.1f}% "
                  f"(<{plateau_threshold}%/step over {plateau_patience} steps). "
                  f"Exploration considered complete.")
            break

        aif_step += 1

    state = coordinator.get_full_state(obstacles)
    logger.log(state, coordinator.history)
    m = coordinator.history[-1] if coordinator.history else {}
    print(f"\n[INFO] Done — step {aif_step} | entropy {m.get('mean_entropy','?')} "
          f"| coverage {m.get('exploration_pct','?')}%")

    qr_decoder.stop()
    qr_stats = qr_decoder.stats()
    print(f"[QR] Final stats: decoded={qr_stats['decoded']} "
          f"failed={qr_stats['failed']} total={qr_stats['total']}")

    if ns3_bridge is not None:
        try:
            ns3_bridge.stop_ns3()
        except Exception as e:
            print(f"[WARN] stop_ns3 a échoué : {e}")

    try:
        from run_artifacts import generate_run_artifacts
        run_dir = generate_run_artifacts(
            cfg=cfg,
            history=coordinator.history,
            final_state=state,
            fused_belief=coordinator.fused_belief,
            agents=agents,
            qr_stats=qr_stats,
            ns3_pairs=coordinator.ns3.last_pairs,
        )
        print(f"[RUN] Artefacts → {run_dir}")
    except Exception as e:
        print(f"[WARN] run_artifacts generation failed: {e}")

    sim_app.close()


if __name__ == "__main__":
    main()
