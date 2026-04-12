#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.stdout.reconfigure(line_buffering=True)


#tus les paramaitres de la simulation sont dans cette classe
@dataclass
class SimConfig:

    # environment (metres)
    env_width: float = 30.0
    env_height: float = 20.0
    grid_resolution: float = 0.5 #taille d'une ceullule de la grille d'occupation
    fly_altitude: float = 2.0 #hauteur de vol des drones

    # drones
    num_drones: int = 3
    drone_spacing: float = 5.0 #espace de séparation entre les drones
    step_size: float = 1.0   # déplacement en mètres par action 

    # lidar 
    num_rays: int = 360 #nombre de rayon par scan 
    lidar_fov_h: float = 360.0      # champs de vision horizontal en degrés
    lidar_fov_v: float = 10.0       # champs de vision vertical en degrés
    lidar_max_range: float = 8.0 # portée maximale du lidar en mètres
    lidar_min_range: float = 0.15 # portée minimale du lidar en mètres
    lidar_hz: float = 10.0 # fréquence de scan du lidar en Hz

    # active inference
    prior_occupancy: float = 0.5 # probabilité a priori d'occupation d'une cellule
    lo_free: float = -0.55 #increment de log-odds pour cellule libre 
    lo_occ: float = 0.85
    lo_max: float = 30.0
    occ_threshold: float = 0.65# pour marquer un ceullule occupé 

    # AIF action selection weights
    w_epistemic: float = 2.5 #pour favoriser les zones de grande incertitude 
    w_pragmatic: float = 0.8 #pour favoriser les zones de frontiere incertaines
    w_movement: float = 0.1 #si on l'augumente le drone bouge moin 
    w_collision: float = 5.0 #pour éviter les collisions avec les autres drones
    softmax_temp: float = 0.3 #si on l'augumente on va aller vers des actions plus variées, sinon on va toujours choisir la même action
    fusion_mix: float = 0.3 #pour mélanger les croyances locales et fusionnées dans la planification

    # le backend de contrôle de vol (PD controller)
    kp_xy: float = 6.0 # le drone corrige plus fort les erreurs de position horizontale
    kd_xy: float = 4.5 # le drone corrige plus fort les erreurs de vitesse horizontale
    kp_z: float = 10.0
    kd_z: float = 6.0
    kp_yaw: float = 2.0 # le drone corrige plus fort les erreurs de lorientation
    waypoint_tol: float = 0.3      # tolérance pour considérer qu'on est arrivé à un waypoint (en mètres)

    # simulation
    headless: bool = False
    max_steps: int = 500 #combien de fois on décide
    sim_steps_per_aif: int = 60   #combien de temps on laisse voler entre deux décisions.
    target_coverage: float = 95.0
    output_dir: str = "/tmp"


    world_origin_x: float = float("nan")  
    world_origin_y: float = float("nan")   

    @property
    def origin_x(self) -> float:
        import math
        return -self.env_width  / 2 if math.isnan(self.world_origin_x) else self.world_origin_x

    @property
    def origin_y(self) -> float:
        import math
        return -self.env_height / 2 if math.isnan(self.world_origin_y) else self.world_origin_y

    @property #nombre de collone 
    def grid_width(self) -> int:
        return int(self.env_width / self.grid_resolution)

    @property #nombre de lignes
    def grid_height(self) -> int:
        return int(self.env_height / self.grid_resolution)

    @property
    def ray_angles(self) -> np.ndarray:
        return np.linspace(0, 2 * math.pi, self.num_rays, endpoint=False)

    @property #portée en nombre de ceullule du lidar
    def max_range_cells(self) -> int:
        return int(self.lidar_max_range / self.grid_resolution)


# ════════════════════════════════════════════════════════════════
# 2. Math Utilities
# ════════════════════════════════════════════════════════════════
#eviter les proba non valides 
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

#convertir une proba en log-odds
def logit(p: float) -> float:
    p = clamp(p, 1e-7, 1 - 1e-7)
    return math.log(p / (1 - p))

#log en proba 
def inv_logit(l: float) -> float:
    return 1.0 / (1.0 + math.exp(-l)) if l >= 0 else math.exp(l) / (1.0 + math.exp(l))

#log en proba version array
def inv_logit_v(arr: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -30, 30)))

#meusure l'incertitude d'une ceullule occupé ou libre
def bernoulli_entropy(p: float) -> float:
    p = clamp(p, 1e-7, 1 - 1e-7)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def bernoulli_entropy_v(arr: np.ndarray) -> np.ndarray:
    p = np.clip(arr, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

#choisir un action parmi les 9 candidates en fonction de leur score G, avec une temperature pour favoriser les actions les mieux notées ou pour favoriser la diversité des actions choisies
def softmax_sample(values: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    temperature = max(temperature, 1e-6)
    shifted = -(values - np.min(values)) / temperature 
    weights = np.exp(shifted - np.max(shifted))
    probs = weights / weights.sum()
    return int(rng.choice(len(values), p=probs))


# ════════════════════════════════════════════════════════════════
# 3. Belief Grid 
# ════════════════════════════════════════════════════════════════

class BeliefGrid:

    def __init__(self, cfg: SimConfig):
        self.width = cfg.grid_width
        self.height = cfg.grid_height
        self.resolution = cfg.grid_resolution
        self.lo_max = cfg.lo_max
        l0 = logit(cfg.prior_occupancy)
        self.logodds = np.full((self.height, self.width), l0, dtype=np.float64)#retourne une grille de log-odds initialisée à la valeur a priori d'occupation
        self.probability = np.full((self.height, self.width), cfg.prior_occupancy)

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        gx = max(0, min(int(wx / self.resolution), self.width - 1))
        gy = max(0, min(int(wy / self.resolution), self.height - 1))
        return gx, gy

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.width and 0 <= gy < self.height
    
#mettre ajour une ceullule 
    def update_cell(self, gx: int, gy: int, dl: float):
        if not self.in_bounds(gx, gy):
            return
        self.logodds[gy, gx] = np.clip(self.logodds[gy, gx] + dl, -self.lo_max, self.lo_max)
        self.probability[gy, gx] = inv_logit(float(self.logodds[gy, gx]))

    def update_from_lidar(
        self,
        ox: float, oy: float,
        angles: np.ndarray,#angle de chaque rayon du lidar
        ranges: np.ndarray,#distance mesurée par chaque rayon du lidar
        hits: np.ndarray,#boolean indiquant si chaque rayon a touché un obstacle ou a atteint la portée maximale
        max_range: float,#portée maximale du lidar en mètres
        lo_free: float,#increment de log-odds pour cellule libre
        lo_occ: float,#increment de log-odds pour cellule occupée
    ):
        ogx, ogy = self.world_to_grid(ox, oy)
        for i in range(len(angles)):
            cos_a = math.cos(angles[i])
            sin_a = math.sin(angles[i])
            d = min(float(ranges[i]), max_range) if hits[i] else max_range #distance à parcourir le long du rayon pour atteindre l'obstacle ou la portée maximale
            n_steps = int(d / self.resolution)
            for s in range(1, n_steps + 1): #pour chaque ceullule traversée par le rayon, on met à jour la croyance de la ceullule en libre ou occupé en fonction de si le rayon a touché un obstacle ou pas
                wx = ox + s * self.resolution * cos_a
                wy = oy + s * self.resolution * sin_a
                gx, gy = self.world_to_grid(wx, wy)
                if not self.in_bounds(gx, gy):
                    break
                if (gx, gy) != (ogx, ogy):
                    self.update_cell(gx, gy, lo_free)
            if hits[i] and ranges[i] <= max_range:
                hx = ox + float(ranges[i]) * cos_a
                hy = oy + float(ranges[i]) * sin_a
                hgx, hgy = self.world_to_grid(hx, hy)
                if self.in_bounds(hgx, hgy) and (hgx, hgy) != (ogx, ogy):
                    self.update_cell(hgx, hgy, lo_occ)

    def mean_entropy(self) -> float:
        return float(bernoulli_entropy_v(self.probability).mean())

    def exploration_ratio(self) -> float:
        known = (self.probability < 0.3) | (self.probability > 0.7)
        return float(known.sum() / known.size)

    def copy(self) -> "BeliefGrid":
        new = BeliefGrid.__new__(BeliefGrid)
        new.width, new.height = self.width, self.height
        new.resolution, new.lo_max = self.resolution, self.lo_max
        new.logodds = self.logodds.copy()
        new.probability = self.probability.copy()
        return new

    def to_list(self) -> List[List[float]]:
        return np.round(self.probability, 2).tolist()


# ════════════════════════════════════════════════════════════════
# 4. Belief Fusion
# ════════════════════════════════════════════════════════════════

def fuse_beliefs_logodds(beliefs: List[BeliefGrid], prior_lo: float) -> BeliefGrid:
    """Independent Opinion Pool in log-odds: L_fused = L0 + mean(L_i - L0)."""
    ref = beliefs[0]
    fused = ref.copy()
    n = len(beliefs)
    fused.logodds[:] = prior_lo
    for b in beliefs:
        fused.logodds += (b.logodds - prior_lo) / n
    np.clip(fused.logodds, -fused.lo_max, fused.lo_max, out=fused.logodds) #Fusion faite par moyenne des écarts au prior
    fused.probability = inv_logit_v(fused.logodds)
    return fused


def mix_beliefs(local: BeliefGrid, fused: BeliefGrid, lam: float) -> BeliefGrid:
    """Blend local and fused: L = (1-λ)·L_local + λ·L_fused."""
    mixed = local.copy()
    mixed.logodds = (1 - lam) * local.logodds + lam * fused.logodds
    np.clip(mixed.logodds, -mixed.lo_max, mixed.lo_max, out=mixed.logodds)
    mixed.probability = inv_logit_v(mixed.logodds)
    return mixed


# ════════════════════════════════════════════════════════════════
# 5. AIF: Expected Free Energy Minimization
# ════════════════════════════════════════════════════════════════

ACTIONS: List[Tuple[str, float, float]] = []
_RAW = [
    ("stay", 0.0, 0.0),
    ("N", 0.0, -1.0), ("NE", 1.0, -1.0), ("E", 1.0, 0.0),
    ("SE", 1.0, 1.0), ("S", 0.0, 1.0), ("SW", -1.0, 1.0),
    ("W", -1.0, 0.0), ("NW", -1.0, -1.0),
]
for _n, _dx, _dy in _RAW:
    _norm = math.hypot(_dx, _dy) or 1.0
    ACTIONS.append((_n, _dx / _norm, _dy / _norm))


def expected_info_gain(wx: float, wy: float, belief: BeliefGrid, cfg: SimConfig) -> float: #score epistémique
    total = 0.0
    for angle in cfg.ray_angles:
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        p_reach = 1.0 #
        for step in range(1, cfg.max_range_cells + 1):
            cx = wx + step * cfg.grid_resolution * cos_a
            cy = wy + step * cfg.grid_resolution * sin_a
            gx, gy = belief.world_to_grid(cx, cy)
            if not belief.in_bounds(gx, gy):
                break
            p_occ = belief.probability[gy, gx]
            total += p_reach * bernoulli_entropy(p_occ)
            p_reach *= (1.0 - p_occ)
            if p_reach < 1e-4:
                break
    return total #plus le score est élevé, plus la position est prometteuse pour réduire l'incertitude de la carte (en observant des zones très incertaines)


def frontier_attraction(wx: float, wy: float, belief: BeliefGrid) -> float: #Score élevé = zone avec beaucoup de cellules incertaines proches.
    gx, gy = belief.world_to_grid(wx, wy)
    window = 5
    total, count = 0.0, 0
    for dy in range(-window, window + 1):
        for dx in range(-window, window + 1):
            nx, ny = gx + dx, gy + dy
            if belief.in_bounds(nx, ny):
                p = belief.probability[ny, nx]
                total += (1.0 - abs(2.0 * p - 1.0)) / (math.hypot(dx, dy) + 1.0)
                count += 1
    return total / max(count, 1)
#Les deux favorisent l’incertain.
#Mais expected_info_gain = incertain visible par rayons.
#frontier_attraction = incertain local dans le voisinage.

def select_action(
    pos_x: float, pos_y: float,
    others: List[Tuple[float, float]],
    belief: BeliefGrid,
    fused: Optional[BeliefGrid],
    cfg: SimConfig,
    rng: np.random.Generator,
) -> Tuple[Tuple[str, float, float], List[Dict], int]:
    """Returns (action_tuple, candidates_diagnostics, selected_index)."""
    plan_belief = mix_beliefs(belief, fused, cfg.fusion_mix) if fused else belief
    n = len(ACTIONS)
    G = np.full(n, 1e6)
    valid = np.zeros(n, dtype=bool)
    cand_diag: List[Dict] = []

    for i, (name, dx, dy) in enumerate(ACTIONS):
        nx = pos_x + dx * cfg.step_size
        ny = pos_y + dy * cfg.step_size
        entry: Dict[str, Any] = {
            "idx": i, "name": name,
            "nx": round(nx, 3), "ny": round(ny, 3),
            "valid": False, "reason": "",
            "ig": 0.0, "frontier": 0.0,
            "move": 0.0, "coll": 0.0, "G": 1e6,
        }
        if not (0.5 <= nx < cfg.env_width - 0.5 and 0.5 <= ny < cfg.env_height - 0.5):
            entry["reason"] = "out-of-bounds"
            cand_diag.append(entry)
            continue
        gx, gy = plan_belief.world_to_grid(nx, ny)
        if plan_belief.probability[gy, gx] >= cfg.occ_threshold:
            entry["reason"] = f"occupied p={plan_belief.probability[gy, gx]:.2f}"
            cand_diag.append(entry)
            continue
        valid[i] = True
        ig = expected_info_gain(nx, ny, plan_belief, cfg)
        fr = frontier_attraction(nx, ny, plan_belief)
        move = 0.0 if name == "stay" else 1.0
        coll = sum(
            1.0 / (math.hypot(nx - ox, ny - oy) + 0.1)
            for ox, oy in others if math.hypot(nx - ox, ny - oy) < 8.0
        )
        G[i] = -cfg.w_epistemic * ig - cfg.w_pragmatic * fr + cfg.w_movement * move + cfg.w_collision * coll
        entry.update({
            "valid": True, "reason": "ok",
            "ig": round(ig, 4), "frontier": round(fr, 4),
            "move": round(move, 2), "coll": round(coll, 4),
            "G": round(G[i], 4),
        })
        cand_diag.append(entry)

    if not valid.any():
        valid[0] = True
        G[0] = 0.0
        if cand_diag:
            cand_diag[0]["valid"] = True
            cand_diag[0]["reason"] = "forced-stay"
            cand_diag[0]["G"] = 0.0
    idx = softmax_sample(G, cfg.softmax_temp, rng)
    return ACTIONS[idx], cand_diag, idx


# ════════════════════════════════════════════════════════════════
# 6. Pegasus Flight Backend (PD Waypoint Controller)
# ════════════════════════════════════════════════════════════════

def _lazy_import_backend():
    from pegasus.simulator.logic.backends.backend import Backend, BackendConfig
    from pegasus.simulator.logic.state import State
    from scipy.spatial.transform import Rotation
    return Backend, BackendConfig, State, Rotation

_Backend = None
_BackendConfig = None
_State = None
_Rotation = None

def _ensure_backend_imports():
    global _Backend, _BackendConfig, _State, _Rotation
    if _Backend is None:
        _Backend, _BackendConfig, _State, _Rotation = _lazy_import_backend()


AifFlightBackend = None  


def _create_backend_class():
    global AifFlightBackend
    _ensure_backend_imports()

    class _AifFlightBackend(_Backend):
      
        def __init__(self, drone_id: int, initial_target: np.ndarray, cfg: SimConfig):
            #param decontrole de vol du drone 
            self.drone_id = drone_id
            self.cfg = cfg
            self.target = initial_target.copy()
            self.target_yaw = 0.0
            self.arrived = False #pour marquer si le drone est arrivé à sa cible, utilisé pour éviter de continuer à appliquer des commandes de mouvement une fois arrivé à la cible

            # State from Pegasus
            self.p = np.zeros(3)
            self.v = np.zeros(3)
            self.R = np.eye(3)
            self.w = np.zeros(3)
            self.received_first_state = False #

            self.input_ref = [0.0, 0.0, 0.0, 0.0] #
            self._vehicle = None 
            self._update_count = 0 

            # PD gains (diagonal matrices)
            self.Kp = np.diag([cfg.kp_xy, cfg.kp_xy, cfg.kp_z])
            self.Kd = np.diag([cfg.kd_xy, cfg.kd_xy, cfg.kd_z])
            # Attitude PD gains
            self.Kr = np.diag([3.0, 3.0, cfg.kp_yaw])
            self.Kw = np.diag([0.5, 0.5, 0.3])
            self.mass = 1.5 
            self.g = 9.81


        def set_target(self, x: float, y: float, z: float, yaw: float = 0.0):
            self.target = np.array([x, y, z])
            self.target_yaw = yaw
            self.arrived = False

        def get_position(self) -> np.ndarray:
            if self.received_first_state:
                return self.p.copy()
            return self.target.copy()

        def get_position_xy(self) -> Tuple[float, float]:
            p = self.get_position()
            return float(p[0]), float(p[1])

        def get_yaw(self) -> float:
            """Extract current yaw from rotation matrix."""
            if not self.received_first_state:
                return self.target_yaw
            return float(math.atan2(self.R[1, 0], self.R[0, 0]))

        def is_at_target(self) -> bool:
            return self.arrived

        # ── Pegasus Backend interface ──

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

        def update_sensor(self, sensor_type: str, data):
            pass

        def update_graphical_sensor(self, sensor_type: str, data):
            pass

        def input_reference(self):
            return self.input_ref

        def update(self, dt: float):
            if not self.received_first_state:
                return

            # ── Position PD ──
            ep = self.p - self.target
            ev = self.v 

            self.arrived = np.linalg.norm(ep[:2]) < self.cfg.waypoint_tol #Si erreur XY < tolérance, drone considéré “arrivé”.

            F_des = -(self.Kp @ ep) - (self.Kd @ ev) + np.array([0.0, 0.0, self.mass * self.g])

            # Current body Z axis
            Z_B = self.R[:, 2]

            # Thrust = projection of desired force onto body Z axis
            u_1 = float(F_des @ Z_B)

            # ── Desired attitude ──
            F_norm = np.linalg.norm(F_des)
            if F_norm > 1e-3:
                Z_b_des = F_des / F_norm
            else:
                Z_b_des = np.array([0.0, 0.0, 1.0])

            X_c_des = np.array([math.cos(self.target_yaw), math.sin(self.target_yaw), 0.0])
            Z_cross_X = np.cross(Z_b_des, X_c_des)
            Z_cross_X_norm = np.linalg.norm(Z_cross_X)
            if Z_cross_X_norm > 1e-6:
                Y_b_des = Z_cross_X / Z_cross_X_norm
            else:
                Y_b_des = np.array([0.0, 1.0, 0.0])
            X_b_des = np.cross(Y_b_des, Z_b_des)

            R_des = np.column_stack([X_b_des, Y_b_des, Z_b_des])

            # ── Attitude PD ──
            # Rotation error (vee map of skew-symmetric error)
            e_R_matrix = R_des.T @ self.R - self.R.T @ R_des
            e_R = 0.5 * np.array([e_R_matrix[2, 1], e_R_matrix[0, 2], e_R_matrix[1, 0]])

            # Angular velocity error (desired angular velocity ≈ 0 for waypoint tracking)
            e_w = self.w

            # Torque = attitude PD
            tau = -(self.Kr @ e_R) - (self.Kw @ e_w)

            # Convert to rotor angular velocities
            if self.vehicle:
                self.input_ref = self.vehicle.force_and_torques_to_velocities(u_1, tau)

            # Debug: print first few updates
            self._update_count += 1
            if self._update_count <= 3 or self._update_count % 500 == 0:
                print(f"  [PD] D{self.drone_id} tick={self._update_count} "
                      f"pos={self.p.round(2)} target={self.target.round(2)} "
                      f"u1={u_1:.2f} tau={tau.round(3)} "
                      f"rotors={[round(r, 1) for r in self.input_ref]}")

        def start(self):
            pass

        def stop(self):
            pass

        def reset(self):
            self.input_ref = [0.0, 0.0, 0.0, 0.0]
            self.received_first_state = False
            self.arrived = False

    AifFlightBackend = _AifFlightBackend


# ════════════════════════════════════════════════════════════════
# 9. LiDAR Reader (PhysX RotatingLidarPhysX — real raycasting)
# ════════════════════════════════════════════════════════════════

class LidarReader:
    """PhysX rotating LiDAR reader with scan accumulation.

    The PhysX RotatingLidarPhysX only returns a partial angular slice per
    physics tick (~60 rays out of 360).  We accumulate partial scans across
    multiple ticks so that `get_accumulated()` returns a full 360° scan.
    Each partial scan is rotated from body-frame to world-frame using the
    drone yaw supplied at accumulation time.
    """

    def __init__(self, drone_id: int, drone_prim_path: str, cfg: SimConfig):
        from isaacsim.sensors.physx import RotatingLidarPhysX

        self.prim_path = f"{drone_prim_path}/body/Lidar_{drone_id}"
        self.cfg = cfg
        self._read_count = 0

        self.sensor = RotatingLidarPhysX(
            prim_path=self.prim_path,
            rotation_frequency=cfg.lidar_hz,
            fov=(cfg.lidar_fov_h, cfg.lidar_fov_v),
            resolution=(cfg.lidar_fov_h / cfg.num_rays, cfg.lidar_fov_v),
            valid_range=(cfg.lidar_min_range, cfg.lidar_max_range),
        )
        self.sensor.add_linear_depth_data_to_frame()
        self.sensor.add_azimuth_data_to_frame()
        self.sensor.enable_visualization(high_lod=False, draw_points=True, draw_lines=False)
        print(f"[INFO] PhysX LiDAR attached: {self.prim_path}")

        # Accumulation buffers (world-frame)
        self._acc_angles: List[np.ndarray] = []
        self._acc_depths: List[np.ndarray] = []
        self._acc_hits:   List[np.ndarray] = []

    def initialize(self):
        self.sensor.initialize()
        self.sensor.post_reset()
        print(f"[INFO] LiDAR initialized: {self.prim_path}")

    # ── internal: parse one partial frame ────────────────────────
    def _parse_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        frame = self.sensor.get_current_frame()
        depth   = frame.get("linear_depth")
        azimuth = frame.get("azimuth")
        if depth is None or azimuth is None or len(depth) == 0:
            return None
        depth   = np.asarray(depth,   dtype=np.float64).ravel()
        azimuth = np.asarray(azimuth, dtype=np.float64).ravel()
        # Isaac Sim may return degrees
        if azimuth.size > 0 and azimuth.max() > 2 * math.pi + 0.1:
            azimuth = np.deg2rad(azimuth)
        hits = (depth >= self.cfg.lidar_min_range) & (depth < (self.cfg.lidar_max_range - 0.05))
        return azimuth, depth, hits

    # ── called every physics tick ────────────────────────────────
    def accumulate(self, yaw: float = 0.0):
        """Read one partial LiDAR frame, rotate to world-frame, append to buffer."""
        parsed = self._parse_frame()
        if parsed is None:
            return
        angles, depths, hits = parsed
        # Body-frame → world-frame
        world_angles = (angles + yaw) % (2 * math.pi)
        self._acc_angles.append(world_angles)
        self._acc_depths.append(depths)
        self._acc_hits.append(hits)

    # ── called once per AIF step in perceive() ───────────────────
    def get_accumulated(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return all accumulated scans merged, then clear the buffer."""
        if not self._acc_angles:
            n = self.cfg.num_rays
            return (
                np.linspace(0, 2 * math.pi, n, endpoint=False),
                np.full(n, self.cfg.lidar_max_range),
                np.zeros(n, dtype=bool),
            )
        angles = np.concatenate(self._acc_angles)
        depths = np.concatenate(self._acc_depths)
        hits   = np.concatenate(self._acc_hits)
        self._acc_angles.clear()
        self._acc_depths.clear()
        self._acc_hits.clear()

        self._read_count += 1
        if self._read_count <= 5 or self._read_count % 50 == 0:
            n_hits = int(hits.sum())
            pct = n_hits / max(len(angles), 1) * 100
            print(f"  [LiDAR] {self.prim_path} scan#{self._read_count}: "
                  f"rays={len(angles)}  hits={n_hits} ({pct:.0f}%)")
        return angles, depths, hits


# ════════════════════════════════════════════════════════════════
# 10. Drone Agent (ties AIF logic to physical drone)
# ════════════════════════════════════════════════════════════════

class DroneAgent: 

    def __init__(self, drone_id: int, sx: float, sy: float, cfg: SimConfig):
        self.id = drone_id
        self.cfg = cfg
        self.belief = BeliefGrid(cfg)
        self.rng = np.random.default_rng(42 + drone_id * 1000)
        self.trail: List[Tuple[float, float]] = [(sx, sy)]
        self.last_action = "stay"
        self.last_G = 0.0
        self.last_ig = 0.0
        self.total_dist = 0.0
        self._prev_xy = (sx, sy)
        self.backend: Optional[AifFlightBackend] = None
        self.lidar: Optional[LidarReader] = None
        # diagnostic data (populated each step for logging)
        self.lidar_diag: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self.candidates_diag: List[Dict] = []
        self.selected_idx: int = 0
        # anti-stagnation
        self._stag_pos = (sx, sy)
        self._stag_steps = 0
        self._STAG_THRESHOLD = 5  # steps stuck → force random move

    def setup_physical(self, backend: AifFlightBackend, lidar: LidarReader):
        self.backend = backend
        self.lidar = lidar

    @property
    def x(self) -> float:
        if self.backend:
            return self.backend.get_position_xy()[0] - self.cfg.origin_x
        return self.trail[-1][0]

    @property
    def y(self) -> float:
        if self.backend:
            return self.backend.get_position_xy()[1] - self.cfg.origin_y
        return self.trail[-1][1]

    def perceive(self):
        """Use accumulated LiDAR scans (full 360°) to update local belief grid."""
        if self.lidar is None:
            self.lidar_diag = None
            return
        angles, ranges, hits = self.lidar.get_accumulated()
        self.lidar_diag = (angles.copy(), ranges.copy(), hits.copy())
        self.belief.update_from_lidar(
            self.x, self.y, angles, ranges, hits,
            self.cfg.lidar_max_range, self.cfg.lo_free, self.cfg.lo_occ,
        )

    def accumulate_lidar(self):
        """Called every physics tick to gather a partial LiDAR scan."""
        if self.lidar is None or self.backend is None:
            return
        self.lidar.accumulate(self.backend.get_yaw())

    def plan(self, others: List[Tuple[float, float]], fused: Optional[BeliefGrid]):
        cx, cy = self.x, self.y

        # ── Anti-stagnation: detect if drone hasn’t moved ──
        moved = math.hypot(cx - self._stag_pos[0], cy - self._stag_pos[1])
        if moved < 0.5:
            self._stag_steps += 1
        else:
            self._stag_steps = 0
            self._stag_pos = (cx, cy)

        if self._stag_steps >= self._STAG_THRESHOLD:
            # Force a random cardinal action to escape
            idx = int(self.rng.integers(1, len(ACTIONS)))  # skip “stay”
            name, dx, dy = ACTIONS[idx]
            # Build a minimal diagnostic entry so log_candidates doesn't crash
            self.candidates_diag = [{
                "idx": idx, "name": name,
                "nx": round(cx + dx * self.cfg.step_size, 3),
                "ny": round(cy + dy * self.cfg.step_size, 3),
                "valid": True, "reason": "forced-stag",
                "ig": 0.0, "frontier": 0.0, "move": 1.0, "coll": 0.0, "G": 0.0,
            }]
            self.selected_idx = 0  # index into the 1-element list
            self._stag_steps = 0
            print(f"  [STAG] D{self.id} stuck {self._STAG_THRESHOLD} steps → forced {name}")
        else:
            (name, dx, dy), self.candidates_diag, self.selected_idx = select_action(
                cx, cy, others, self.belief, fused, self.cfg, self.rng,
            )
            # Update dashboard-exported diagnostics
            if self.candidates_diag and self.selected_idx < len(self.candidates_diag):
                sel = self.candidates_diag[self.selected_idx]
                self.last_G  = sel.get("G", 0.0)
                self.last_ig = sel.get("ig", 0.0)

        tx = clamp(cx + dx * self.cfg.step_size, 0.5, self.cfg.env_width - 0.5)
        ty = clamp(cy + dy * self.cfg.step_size, 0.5, self.cfg.env_height - 0.5)

        if self.backend:
            self.backend.set_target(
                tx + self.cfg.origin_x,
                ty + self.cfg.origin_y,
                self.cfg.fly_altitude,
            )

        self.total_dist += math.hypot(cx - self._prev_xy[0], cy - self._prev_xy[1])
        self._prev_xy = (cx, cy)
        self.trail.append((round(cx, 2), round(cy, 2)))
        if len(self.trail) > 200:
            self.trail = self.trail[-200:]
        self.last_action = name

    def get_state(self) -> Dict:
        heading = 0.0
        if len(self.trail) >= 2:
            dx = self.trail[-1][0] - self.trail[-2][0]
            dy = self.trail[-1][1] - self.trail[-2][1]
            if dx != 0 or dy != 0:
                heading = math.atan2(dy, dx)
        return {
            "id": self.id,
            "x": round(self.x, 3), "y": round(self.y, 3),
            "heading": round(heading, 3),
            "action": self.last_action,
            "free_energy": round(self.last_G, 4),
            "info_gain": round(self.last_ig, 4),
            "total_distance": round(self.total_dist, 2),
            "local_entropy": round(self.belief.mean_entropy(), 4),
            "trail": self.trail[-60:],
        }


# ════════════════════════════════════════════════════════════════
# 11. Swarm Coordinator
# ════════════════════════════════════════════════════════════════

class SwarmCoordinator:
    def __init__(self, agents: List[DroneAgent], cfg: SimConfig,
                 diag_logger: Optional["DiagnosticLogger"] = None):
        self.agents = agents
        self.cfg = cfg
        self.prior_lo = logit(cfg.prior_occupancy)
        self.fused_belief = agents[0].belief.copy()
        self.step_count = 0
        self.history: List[Dict] = []
        self.diag = diag_logger

    def step(self):
        h_before = self.fused_belief.mean_entropy()

        if self.diag:
            self.diag.log_step_header(self.step_count + 1)

        # perception
        for a in self.agents:
            a.perceive()
            if self.diag and a.lidar_diag is not None:
                angles, ranges, hits = a.lidar_diag
                self.diag.log_lidar(a.id, a.x, a.y, angles, ranges, hits)

        # fusion
        self.fused_belief = fuse_beliefs_logodds(
            [a.belief for a in self.agents], self.prior_lo,
        )

        # planning
        for a in self.agents:
            others = [(o.x, o.y) for o in self.agents if o.id != a.id]
            a.plan(others, self.fused_belief)
            if self.diag:
                self.diag.log_candidates(a.id, a.x, a.y, others,
                                         a.candidates_diag, a.selected_idx)
                self.diag.log_belief(a.id, a.belief, self.fused_belief)

        h_after = self.fused_belief.mean_entropy()
        self.step_count += 1
        self.history.append({
            "step": self.step_count,
            "mean_entropy": round(h_after, 4),
            "exploration_pct": round(self.fused_belief.exploration_ratio() * 100, 2),
            "step_info_gain": round(max(0.0, h_before - h_after), 4),
            "free_energies": [round(a.last_G, 4) for a in self.agents],
            "info_gains": [round(a.last_ig, 4) for a in self.agents],
        })

        if self.diag:
            self.diag.log_step_summary(self.step_count, self.agents,
                                       self.fused_belief)

    def get_full_state(self, obstacles: List[Dict]) -> Dict:
        return {
            "step": self.step_count,
            "timestamp": time.time(),
            "environment": {
                "width": self.cfg.env_width, "height": self.cfg.env_height,
                "grid_resolution": self.cfg.grid_resolution,
                "grid_width": self.cfg.grid_width, "grid_height": self.cfg.grid_height,
                "obstacles": obstacles,
            },
            "drones": [a.get_state() for a in self.agents],
            "fused_belief": self.fused_belief.to_list(),
            "metrics": self.history[-1] if self.history else {},
        }


# ════════════════════════════════════════════════════════════════
# 12. Data Logger
# ════════════════════════════════════════════════════════════════

class DataLogger:
    def __init__(self, output_dir: str = "/tmp"):
        self.state_path = os.path.join(output_dir, "aif_state.json")
        self.history_path = os.path.join(output_dir, "aif_history.json")
        self._write(self.history_path, [])

    def log(self, state: Dict, history: List[Dict]):
        self._write(self.state_path, state)
        self._write(self.history_path, history)

    @staticmethod
    def _write(path: str, data: Any):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        os.replace(tmp, path)


# ════════════════════════════════════════════════════════════════
# 12b. Diagnostic Logger  — detailed per-drone log for debugging
# ════════════════════════════════════════════════════════════════

class DiagnosticLogger:
    """Writes a human-readable diagnostic log to help identify
    exploration, obstacle-detection, collision-avoidance and
    planning issues."""

    def __init__(self, output_dir: str, cfg: SimConfig):
        self.cfg = cfg
        self.log_path = os.path.join(output_dir, "aif_diagnostic.log")
        with open(self.log_path, "w") as f:
            f.write("=" * 90 + "\n")
            f.write("  AIF DIAGNOSTIC LOG\n")
            f.write(f"  Started : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Drones  : {cfg.num_drones}\n")
            f.write(f"  Env     : {cfg.env_width} x {cfg.env_height} m  "
                    f"(grid {cfg.grid_width} x {cfg.grid_height}, res {cfg.grid_resolution} m)\n")
            f.write(f"  Weights : epistemic={cfg.w_epistemic}  pragmatic={cfg.w_pragmatic}  "
                    f"movement={cfg.w_movement}  collision={cfg.w_collision}\n")
            f.write(f"  Softmax : temp={cfg.softmax_temp}  fusion_mix={cfg.fusion_mix}\n")
            f.write(f"  Occ thr : {cfg.occ_threshold}   step_size={cfg.step_size} m\n")
            f.write(f"  LiDAR   : rays={cfg.num_rays}  range=[{cfg.lidar_min_range}, "
                    f"{cfg.lidar_max_range}] m  hz={cfg.lidar_hz}\n")
            f.write("=" * 90 + "\n\n")
        print(f"[INFO] Diagnostic log → {self.log_path}")

    # ── helpers ──────────────────────────────────────────────────
    def _w(self, text: str):
        with open(self.log_path, "a") as f:
            f.write(text)

    # ── step header ──────────────────────────────────────────────
    def log_step_header(self, step: int):
        self._w(f"\n{'━' * 90}\n")
        self._w(f"  STEP {step}   ({time.strftime('%H:%M:%S')})\n")
        self._w(f"{'━' * 90}\n")

    # ── LiDAR diagnostics per drone ──────────────────────────────
    def log_lidar(self, drone_id: int, pos_x: float, pos_y: float,
                  angles: np.ndarray, ranges: np.ndarray, hits: np.ndarray):
        n_total = len(hits)
        n_hits = int(hits.sum())
        pct = n_hits / max(n_total, 1) * 100

        self._w(f"\n  ┌─ DRONE {drone_id}  LiDAR  pos=({pos_x:.2f}, {pos_y:.2f})\n")
        self._w(f"  │  Rays total={n_total}  hits={n_hits} ({pct:.1f}%)\n")

        if n_hits > 0:
            hr = ranges[hits]
            ha = angles[hits]
            self._w(f"  │  Hit ranges : min={hr.min():.2f} m  max={hr.max():.2f} m  "
                    f"mean={hr.mean():.2f} m\n")
            # 5 closest obstacles
            order = np.argsort(hr)[:5]
            self._w(f"  │  Closest obstacles:\n")
            for k in order:
                adeg = math.degrees(ha[k]) % 360
                self._w(f"  │    angle={adeg:6.1f}°  dist={hr[k]:.2f} m\n")
            # directional summary (4 quadrants)
            for label, lo, hi in [("FRONT 315-45°", 315, 45),
                                   ("RIGHT 45-135°", 45, 135),
                                   ("BACK 135-225°", 135, 225),
                                   ("LEFT 225-315°", 225, 315)]:
                adeg = np.degrees(ha) % 360
                if lo > hi:  # wraps around 0
                    mask = (adeg >= lo) | (adeg < hi)
                else:
                    mask = (adeg >= lo) & (adeg < hi)
                cnt = int(mask.sum())
                if cnt > 0:
                    mn = hr[mask].min()
                    self._w(f"  │    {label:16s}  hits={cnt:3d}  closest={mn:.2f} m\n")
                else:
                    self._w(f"  │    {label:16s}  hits=  0\n")
        else:
            self._w(f"  │  ⚠  NO obstacles detected by LiDAR\n")
        self._w(f"  └{'─' * 60}\n")

    # ── candidate actions table per drone ────────────────────────
    def log_candidates(self, drone_id: int, pos_x: float, pos_y: float,
                       others: List[Tuple[float, float]],
                       cand: List[Dict], selected_idx: int):
        self._w(f"\n  ┌─ DRONE {drone_id}  Action Selection  pos=({pos_x:.2f}, {pos_y:.2f})\n")

        # inter-drone distances
        if others:
            self._w(f"  │  Other drones:\n")
            for ox, oy in others:
                d = math.hypot(pos_x - ox, pos_y - oy)
                tag = " ⚠ DANGER" if d < 1.5 else " ⚠ CLOSE" if d < 3.0 else ""
                self._w(f"  │    ({ox:.2f}, {oy:.2f})  dist={d:.2f} m{tag}\n")

        # header
        self._w(f"  │\n")
        self._w(f"  │  {'#':>2} {'Act':<5} {'Valid':<6} {'Reason':<14} "
                f"{'nx':>6} {'ny':>6}  "
                f"{'IG':>8} {'Front':>7} {'Move':>5} {'Coll':>7} "
                f"{'G_total':>9}\n")
        self._w(f"  │  {'-' * 88}\n")

        for c in cand:
            sel = " ◄" if c["idx"] == selected_idx else ""
            self._w(f"  │  {c['idx']:2d} {c['name']:<5} "
                    f"{str(c['valid']):<6} {c['reason']:<14} "
                    f"{c['nx']:6.2f} {c['ny']:6.2f}  "
                    f"{c.get('ig', 0):8.3f} {c.get('frontier', 0):7.3f} "
                    f"{c.get('move', 0):5.2f} {c.get('coll', 0):7.3f} "
                    f"{c.get('G', 9999):9.3f}{sel}\n")

        sel_c = cand[selected_idx]
        self._w(f"  │\n")
        self._w(f"  │  ➜ SELECTED: {sel_c['name']}  target=({sel_c['nx']:.2f}, {sel_c['ny']:.2f})  "
                f"G={sel_c.get('G', 0):.4f}\n")
        self._w(f"  └{'─' * 60}\n")

    # ── belief grid stats per drone ──────────────────────────────
    def log_belief(self, drone_id: int, belief: BeliefGrid,
                   fused: Optional[BeliefGrid] = None):
        total = belief.width * belief.height
        occ = int((belief.probability >= self.cfg.occ_threshold).sum())
        free = int((belief.probability < 0.3).sum())
        unc = total - occ - free

        self._w(f"\n  ┌─ DRONE {drone_id}  Belief Grid\n")
        self._w(f"  │  Entropy  : {belief.mean_entropy():.4f}\n")
        self._w(f"  │  Coverage : {belief.exploration_ratio() * 100:.1f}%\n")
        self._w(f"  │  Cells    : occupied={occ} ({occ/total*100:.1f}%)  "
                f"free={free} ({free/total*100:.1f}%)  "
                f"uncertain={unc} ({unc/total*100:.1f}%)\n")
        if fused is not None:
            f_occ = int((fused.probability >= self.cfg.occ_threshold).sum())
            self._w(f"  │  Fused    : entropy={fused.mean_entropy():.4f}  "
                    f"coverage={fused.exploration_ratio() * 100:.1f}%  "
                    f"occ_cells={f_occ}\n")
        self._w(f"  └{'─' * 60}\n")

    # ── step summary (inter-drone + wall proximity) ──────────────
    def log_step_summary(self, step: int, agents, fused: BeliefGrid):
        self._w(f"\n  ── Step {step} Summary ──\n")
        # positions
        for a in agents:
            arrived = ""
            if a.backend is not None:
                arrived = "  arrived=True" if a.backend.is_at_target() else "  arrived=False"
            self._w(f"    D{a.id} pos=({a.x:.2f}, {a.y:.2f})  "
                    f"action={a.last_action}  dist_total={a.total_dist:.2f} m{arrived}\n")

        # pairwise distances
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                d = math.hypot(agents[i].x - agents[j].x, agents[i].y - agents[j].y)
                tag = " ⚠ COLLISION-RISK" if d < 1.5 else " ⚠ CLOSE" if d < 3.0 else ""
                self._w(f"    D{agents[i].id}↔D{agents[j].id} = {d:.2f} m{tag}\n")

        # wall proximity warnings
        for a in agents:
            warns = []
            if a.x < 1.0:
                warns.append(f"LEFT wall x={a.x:.2f}")
            if a.x > self.cfg.env_width - 1.0:
                warns.append(f"RIGHT wall x={a.x:.2f}")
            if a.y < 1.0:
                warns.append(f"BOTTOM wall y={a.y:.2f}")
            if a.y > self.cfg.env_height - 1.0:
                warns.append(f"TOP wall y={a.y:.2f}")
            if warns:
                self._w(f"    ⚠ D{a.id} WALL: {', '.join(warns)}\n")

        # overlap detection (same grid cell)
        cells = {}
        for a in agents:
            gx, gy = fused.world_to_grid(a.x, a.y)
            key = (gx, gy)
            cells.setdefault(key, []).append(a.id)
        for key, ids in cells.items():
            if len(ids) > 1:
                self._w(f"    ⚠ OVERLAP: drones {ids} in same grid cell {key}\n")

        self._w(f"    Fused coverage={fused.exploration_ratio() * 100:.1f}%  "
                f"entropy={fused.mean_entropy():.4f}\n")


# ════════════════════════════════════════════════════════════════
# 13. Main — Isaac Sim Setup & AIF Loop
# ════════════════════════════════════════════════════════════════

def parse_args() -> SimConfig:
    p = argparse.ArgumentParser(description="Active Inference drones — Isaac Sim (real)")
    p.add_argument("--num-drones", type=int, default=3)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--env-width", type=float, default=30.0)
    p.add_argument("--env-height", type=float, default=20.0)
    a = p.parse_args()
    return SimConfig(
        num_drones=a.num_drones, headless=a.headless,
        max_steps=a.max_steps, env_width=a.env_width, env_height=a.env_height,
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
    """Load one forced factory USD path (no path search/fallback).
    Returns the XY bounding box (min_x, min_y, max_x, max_y) in metres."""
    try:
        from isaacsim.core.utils.stage import add_reference_to_stage
    except ImportError:
        from omni.isaac.core.utils.stage import add_reference_to_stage

    import omni.usd
    from pxr import UsdGeom, Gf

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

    # Verify that the prim actually resolved to scene content.
    prim = stage.GetPrimAtPath("/World/Factory")
    if not (prim.IsValid() and prim.GetChildren()):
        stage.RemovePrim("/World/Factory")
        raise RuntimeError(f"Forced factory USD is unreachable or empty: {forced_usd}")

    # ── Compute XY bounding box of the factory ──
    bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_])
    bbox = bbox_cache.ComputeWorldBound(prim)
    rng = bbox.GetRange()
    lo = rng.GetMin()
    hi = rng.GetMax()
    print(f"[INFO] Factory bounding box: "
          f"X=[{lo[0]:.2f}, {hi[0]:.2f}]  Y=[{lo[1]:.2f}, {hi[1]:.2f}]  "
          f"Z=[{lo[2]:.2f}, {hi[2]:.2f}]")
    print(f"[INFO] Factory environment loaded OK: {forced_usd}")
    return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))


def _add_scene_lighting():
    """Add dome + distant lights so the scene is well-lit."""
    from pxr import Gf, UsdGeom, UsdLux
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    stage.DefinePrim("/World/Lights", "Xform")

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(1000.0)
    dome.CreateColorAttr(Gf.Vec3f(0.85, 0.9, 1.0))  # slightly blue sky

    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(3000.0)
    sun.CreateAngleAttr(1.0)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))  # warm sunlight
    xf = UsdGeom.Xformable(sun.GetPrim())
    xf.ClearXformOpOrder()
    # Aim the sun downward at an angle
    xf.AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 30.0, 0.0))

    print("[INFO] Scene lighting added (dome + sun)")


def setup_viewport_camera(cfg: SimConfig):

    if cfg.headless:
        return

    cx = cfg.env_width  / 2  
    cy = cfg.env_height / 2  
    tz = cfg.fly_altitude   

    cam_x = cx
    cam_y = cy - 18.0
    cam_z = 25.0

    try:
        import omni.usd
        from pxr import Gf, UsdGeom
        import omni.kit.viewport.utility as vp_utils

        viewport = vp_utils.get_active_viewport()
        if viewport is None:
            print("[WARN] setup_viewport_camera: no active viewport found, skipping.")
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
            print(
                f"[INFO] Overview camera (ViewportCameraState) → "
                f"eye=({cam_x:.1f}, {cam_y:.1f}, {cam_z:.1f})  "
                f"target=({cx:.1f}, {cy:.1f}, {tz:.1f})"
            )
            return
        except Exception as e1:
            print(f"[INFO] ViewportCameraState unavailable ({e1}), trying USD camera…")

        import math
        cam_path = "/World/OverviewCamera"
        camera = UsdGeom.Camera.Define(stage, cam_path)

        camera.CreateFocalLengthAttr(18.0)
        camera.CreateHorizontalApertureAttr(36.0)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 500.0))

    
        dY = cy    - cam_y   
        dZ = tz    - cam_z  
        pitch_deg = math.degrees(math.atan2(-dZ, dY)) 

        xf = UsdGeom.Xformable(camera.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(cam_x, cam_y, cam_z))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(pitch_deg, 0.0, 0.0))

        viewport.set_active_camera(cam_path)
        print(
            f"[INFO] Overview camera (USD prim) → "
            f"eye=({cam_x:.1f}, {cam_y:.1f}, {cam_z:.1f})  "
            f"pitch={pitch_deg:.1f}°  path={cam_path}"
        )

    except Exception as e:
        print(f"[WARN] Could not configure viewport camera: {e}")


def create_physical_drones(agents: List[DroneAgent], cfg: SimConfig):
    """Create Pegasus Multirotors with AIF flight backends + PhysX LiDAR sensors."""
    from pegasus.simulator.params import ROBOTS
    from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
    from pegasus.simulator.logic.sensors.barometer import Barometer
    from pegasus.simulator.logic.sensors.imu import IMU
    from pegasus.simulator.logic.sensors.gps import GPS

    # create AifFlightBackend as a proper Backend subclass
    _create_backend_class()

    multirotors = []
    for agent in agents:
        gx = agent.trail[0][0]
        gy = agent.trail[0][1]

        sx = gx + cfg.origin_x
        sy = gy + cfg.origin_y

        backend = AifFlightBackend(agent.id, np.array([sx, sy, cfg.fly_altitude]), cfg)

        mc = MultirotorConfig()
        mc.backends = [backend]
        mc.sensors = [Barometer(), IMU(), GPS()]

        prim_path = f"/World/Drone_{agent.id:02d}"
        drone = Multirotor(
            prim_path, ROBOTS["Iris"], agent.id,
            [sx, sy, 0.1], [0.0, 0.0, 0.0, 1.0], mc,
        )
        multirotors.append(drone)

        # attach real PhysX LiDAR
        lidar = LidarReader(agent.id, prim_path, cfg)
        agent.setup_physical(backend, lidar)

        print(
            f"[INFO] Drone {agent.id} — grid({gx:.1f},{gy:.1f}) "
            f"→ world({sx:.1f},{sy:.1f}) — PD backend + PhysX LiDAR"
        )

    return multirotors


def main():
    cfg = parse_args()

    running = True
    def _sig(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    print(f"[INFO] Isaac Sim — {'headless' if cfg.headless else 'GUI'}")
    sim_app = create_sim_app(cfg)

    world, factory_bounds = setup_world()

    # ── Auto-adjust env dimensions from factory USD bounding box ──
    if factory_bounds:
        margin = 1.0  # 1 m padding around the factory
        bx0, by0, bx1, by1 = factory_bounds
        cfg.world_origin_x = bx0 - margin
        cfg.world_origin_y = by0 - margin
        cfg.env_width  = (bx1 - bx0) + 2 * margin
        cfg.env_height = (by1 - by0) + 2 * margin
        # round to grid resolution
        cfg.env_width  = math.ceil(cfg.env_width  / cfg.grid_resolution) * cfg.grid_resolution
        cfg.env_height = math.ceil(cfg.env_height / cfg.grid_resolution) * cfg.grid_resolution
        print(f"[INFO] Env adapted to factory: "
              f"{cfg.env_width:.1f} × {cfg.env_height:.1f} m  "
              f"origin=({cfg.origin_x:.2f}, {cfg.origin_y:.2f})  "
              f"grid={cfg.grid_width}×{cfg.grid_height}")

    setup_viewport_camera(cfg)
    obstacles: List[Dict] = []

    agents: List[DroneAgent] = []
    for i in range(cfg.num_drones):
        gx = cfg.env_width  / 2 + (i - (cfg.num_drones - 1) / 2) * cfg.drone_spacing
        # Spawn well inside factory walls (USD bbox extends beyond walls due to floor)
        gy = max(8.0, cfg.env_height * 0.20)
        agents.append(DroneAgent(i, gx, gy, cfg))

    create_physical_drones(agents, cfg)
    diag_logger = DiagnosticLogger(cfg.output_dir, cfg)
    coordinator = SwarmCoordinator(agents, cfg, diag_logger=diag_logger)
    logger = DataLogger(cfg.output_dir)

    world.reset()

    # ── Initialize LiDAR sensors (must happen after world.reset) ──
    for agent in agents:
        if agent.lidar is not None:
            agent.lidar.initialize()

    print(f"[INFO] {cfg.num_drones} drones | grid {cfg.grid_width}×{cfg.grid_height}")
    print("[INFO] Using factory template colliders from USD scene")
    print(f"[INFO] {cfg.sim_steps_per_aif} physics ticks per AIF decision")
    print(f"[INFO] Dashboard → {cfg.output_dir}/aif_state.json")

    # ── take-off: let PD controllers lift drones to altitude ──
    print("[INFO] Taking off…")
    for _ in range(300):
        if not running or not sim_app.is_running():
            break
        world.step(render=not cfg.headless)
        # Start accumulating LiDAR scans during takeoff
        for agent in agents:
            agent.accumulate_lidar()
    print("[INFO] Take-off complete.\n")

    # ── AIF exploration loop ──
    aif_step = 0
    while running and aif_step < cfg.max_steps and sim_app.is_running():
        # AIF cycle: perceive (real LiDAR) → fuse → plan (sets waypoints)
        coordinator.step()

        # fly: run physics until drones move toward their waypoints
        for _ in range(cfg.sim_steps_per_aif):
            if not running or not sim_app.is_running():
                break
            world.step(render=not cfg.headless)
            # Accumulate partial LiDAR scans each physics tick
            for agent in agents:
                agent.accumulate_lidar()

        # log for dashboard
        state = coordinator.get_full_state(obstacles)
        logger.log(state, coordinator.history)

        m = coordinator.history[-1]
        positions_str = " | ".join(
            f"D{a.id}({a.x:.1f},{a.y:.1f})" for a in agents
        )
        print(
            f"  Step {aif_step:4d} | "
            f"H={m['mean_entropy']:.3f} | "
            f"Expl={m['exploration_pct']:5.1f}% | "
            f"ΔIG={m['step_info_gain']:.4f} | "
            f"{positions_str}"
        )

        if m["exploration_pct"] >= cfg.target_coverage:
            print(f"\n[INFO] Target coverage {cfg.target_coverage}% reached!")
            break

        aif_step += 1

    state = coordinator.get_full_state(obstacles)
    logger.log(state, coordinator.history)
    m = coordinator.history[-1] if coordinator.history else {}
    print(f"\n[INFO] Done — step {aif_step} | entropy {m.get('mean_entropy','?')} | coverage {m.get('exploration_pct','?')}%")
    sim_app.close()


if __name__ == "__main__":
    main()