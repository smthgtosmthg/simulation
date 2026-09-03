from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SimConfig:
    env_width: float = 30.0
    env_height: float = 20.0
    grid_resolution: float = 0.5
    fly_altitude: float = 2.0

    num_drones: int = 3
    drone_spacing: float = 5.0
    step_size: float = 1.0

    num_rays: int = 360
    lidar_fov_h: float = 360.0
    lidar_fov_v: float = 40.0
    lidar_vert_res: float = 5.0
    lidar_max_range: float = 8.0
    lidar_min_range: float = 0.15
    lidar_hz: float = 10.0
    floor_filter_z: float = 0.25

    prior_occupancy: float = 0.5
    lo_free: float = -0.55
    lo_occ: float = 0.85
    lo_max: float = 30.0
    occ_threshold: float = 0.65

    w_epistemic: float = 2.5
    w_pragmatic: float = 0.8
    w_movement: float = 0.1
    w_collision: float = 5.0
    w_clearance: float = 25.0
    clearance_cells: int = 6
    softmax_temp: float = 0.3
    fusion_mix: float = 0.3

    waypoint_tol: float = 0.3

    alpha: int = 30
    beta: int = 60
    H_target: float = 0.44
    innov_target: float = 0.16
    k_sigma: float = 2.0
    ema_alpha: float = 0.05

    w_entropy_recover: float = 3.0
    w_innov_recover: float = 1.2
    w_deadline: float = 12.0
    w_entropy_durable: float = 1.2
    w_innov_durable: float = 0.8
    w_churn_durable: float = 2.2
    w_maintain: float = 10.0

    planner: str = "aif"
    arch: str = "centralized"
    neighbor_radius_m: float = 5.0
    ns3_mode: str = "wifi"
    cloud_round_trip_ms: float = 500.0
    switch_latency_ms: float = 2000.0
    ns3_sim_time: int = 600
    physics_dt_s: float = 1.0 / 60.0

    kill_drone_at_step: int = -1
    kill_drone_id: int = 0
    cut_cloud_at_step: int = -1
    cut_drone_link: str = ""
    cut_drone_link_at_step: int = -1
    drop_obstacle_at_step: int = -1
    drop_obstacle_xy: str = ""

    headless: bool = False
    max_steps: int = 500
    sim_steps_per_aif: int = 60
    target_coverage: float = 93.0
    output_dir: str = "/tmp"
    run_tag: str = "default"
    runs_dir: str = ""

    world_origin_x: float = float("nan")
    world_origin_y: float = float("nan")
    factory_bounds_world: Optional[Tuple[float, float, float, float]] = None
    interior_inset_m: float = 2.5

    @property
    def origin_x(self) -> float:
        return -self.env_width / 2 if math.isnan(self.world_origin_x) else self.world_origin_x

    @property
    def origin_y(self) -> float:
        return -self.env_height / 2 if math.isnan(self.world_origin_y) else self.world_origin_y

    @property
    def grid_width(self) -> int:
        return int(self.env_width / self.grid_resolution)

    @property
    def grid_height(self) -> int:
        return int(self.env_height / self.grid_resolution)

    @property
    def max_range_cells(self) -> int:
        return int(self.lidar_max_range / self.grid_resolution)

    @property
    def step_dt_ms(self) -> float:
        return self.sim_steps_per_aif * self.physics_dt_s * 1000.0

    @property
    def ray_angles(self):
        import numpy as np
        return np.linspace(0, 2 * math.pi, self.num_rays, endpoint=False)

    def factory_bounds_grid(self) -> Optional[Tuple[int, int, int, int]]:
        if self.factory_bounds_world is None:
            return None
        fx0w, fy0w, fx1w, fy1w = self.factory_bounds_world
        gx0 = max(0, int((fx0w - self.origin_x) / self.grid_resolution))
        gy0 = max(0, int((fy0w - self.origin_y) / self.grid_resolution))
        gx1 = min(self.grid_width,
                  int(math.ceil((fx1w - self.origin_x) / self.grid_resolution)))
        gy1 = min(self.grid_height,
                  int(math.ceil((fy1w - self.origin_y) / self.grid_resolution)))
        return gx0, gy0, gx1, gy1

    def interior_bounds_local(self) -> Optional[Tuple[float, float, float, float]]:
        """Bornes de l'intérieur usine (inset compris) en repère local drone.

        Renvoie (x0, y0, x1, y1) en mètres locaux, ou None si l'environnement
        usine n'est pas connu (ex. run headless sans Isaac Sim)."""
        if self.factory_bounds_world is None:
            return None
        fx0w, fy0w, fx1w, fy1w = self.factory_bounds_world
        inset = float(self.interior_inset_m)
        x0 = (fx0w - self.origin_x) + inset
        y0 = (fy0w - self.origin_y) + inset
        x1 = (fx1w - self.origin_x) - inset
        y1 = (fy1w - self.origin_y) - inset
        return x0, y0, x1, y1

    def interior_area_cells(self) -> int:
        if self.factory_bounds_world is None:
            return self.grid_width * self.grid_height
        fx0w, fy0w, fx1w, fy1w = self.factory_bounds_world
        inset = float(self.interior_inset_m)
        interior_w = max(0.0, (fx1w - fx0w) - 2 * inset)
        interior_h = max(0.0, (fy1w - fy0w) - 2 * inset)
        cells = (interior_w * interior_h) / (self.grid_resolution ** 2)
        return max(1, int(round(cells)))
