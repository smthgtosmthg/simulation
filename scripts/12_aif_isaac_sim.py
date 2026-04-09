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
    pos_x: float, pos_y: float,#position du drone 
    others: List[Tuple[float, float]], #position des autres drones
    belief: BeliefGrid,
    fused: Optional[BeliefGrid],
    cfg: SimConfig,
    rng: np.random.Generator,
) -> Tuple[str, float, float]:
    plan_belief = mix_beliefs(belief, fused, cfg.fusion_mix) if fused else belief #on mélange la croyance locale et la croyance fusionnée pour la planification, en fonction du paramètre fusion_mix
    n = len(ACTIONS) #8 +rester 
    G = np.full(n, 1e6) #score initialisé grand pour chauqe action 
    valid = np.zeros(n, dtype=bool) #pou marquer les actions autorisées

    for i, (name, dx, dy) in enumerate(ACTIONS):
        nx = pos_x + dx * cfg.step_size
        ny = pos_y + dy * cfg.step_size
        if not (0.5 <= nx < cfg.env_width - 0.5 and 0.5 <= ny < cfg.env_height - 0.5):#test pour rester dans les limites de l'environnement (en laissant une marge de 0.5m pour éviter les collisions avec les murs)
            continue
        gx, gy = plan_belief.world_to_grid(nx, ny)
        if plan_belief.probability[gy, gx] >= cfg.occ_threshold: #si occupé 
            continue
        valid[i] = True
        ig = expected_info_gain(nx, ny, plan_belief, cfg) #gain d’info attendu (exploration utile).
        fr = frontier_attraction(nx, ny, plan_belief) #attraction de frontière (proximité de zones incertaines).
        move = 0.0 if name == "stay" else 1.0 #favorise les actions de mouvement par rapport à rester sur place (pour éviter de rester bloqué dans une zone sans faire de progrès)
        coll = sum( #pénalité pour la collision avec les autres drones (plus la position est proche des autres drones, plus la pénalité est grande)
            1.0 / (math.hypot(nx - ox, ny - oy) + 0.1)
            for ox, oy in others if math.hypot(nx - ox, ny - oy) < 3.0
        )
        G[i] = -cfg.w_epistemic * ig - cfg.w_pragmatic * fr + cfg.w_movement * move + cfg.w_collision * coll

    if not valid.any():
        valid[0] = True
        G[0] = 0.0 #force stay si aucune action n'est valide 
    idx = softmax_sample(G, cfg.softmax_temp, rng)
    return ACTIONS[idx]


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

    def initialize(self): #active le LiDAR après reset du monde.
        self.sensor.initialize()
        self.sensor.post_reset()
        print(f"[INFO] LiDAR initialized: {self.prim_path}")

    def read(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            angles: azimuth in radians
            ranges: distance per ray (metres)
            hits:   True if ray struck a surface
        """
        frame = self.sensor.get_current_frame()
        depth = frame.get("linear_depth")
        azimuth = frame.get("azimuth")

        self._read_count += 1

        if self._read_count <= 5 or self._read_count % 100 == 0:
            d_info = "None" if depth is None else f"len={len(depth)}"
            a_info = "None" if azimuth is None else f"len={len(azimuth)}"
            print(f"  [LiDAR] {self.prim_path} read#{self._read_count}: depth={d_info} azimuth={a_info}")
            if depth is not None and len(depth) > 0:
                d_arr = np.asarray(depth).ravel()
                n_hits = int((d_arr < self.cfg.lidar_max_range - 0.05).sum())
                print(f"           range=[{d_arr.min():.2f}, {d_arr.max():.2f}] hits={n_hits}/{len(d_arr)}")

        if depth is None or azimuth is None or len(depth) == 0:
            n = self.cfg.num_rays
            return (
                np.linspace(0, 2 * math.pi, n, endpoint=False),
                np.full(n, self.cfg.lidar_max_range),
                np.zeros(n, dtype=bool),
            )

        depth = np.asarray(depth, dtype=np.float64).ravel()
        azimuth = np.asarray(azimuth, dtype=np.float64).ravel()

        # Isaac Sim may return degrees
        if azimuth.size > 0 and azimuth.max() > 2 * math.pi + 0.1:
            azimuth = np.deg2rad(azimuth)

        hits = depth < (self.cfg.lidar_max_range - 0.05)
        return azimuth, depth, hits


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

    def setup_physical(self, backend: AifFlightBackend, lidar: LidarReader):
        self.backend = backend
        self.lidar = lidar

    @property
    def x(self) -> float:
        if self.backend:
            return self.backend.get_position_xy()[0]
        return self.trail[-1][0]

    @property
    def y(self) -> float:
        if self.backend:
            return self.backend.get_position_xy()[1]
        return self.trail[-1][1]

    def perceive(self):
        """Read real LiDAR data and update local belief grid."""
        if self.lidar is None:
            return
        angles, ranges, hits = self.lidar.read()
        self.belief.update_from_lidar(
            self.x, self.y, angles, ranges, hits,
            self.cfg.lidar_max_range, self.cfg.lo_free, self.cfg.lo_occ,
        )

    def plan(self, others: List[Tuple[float, float]], fused: Optional[BeliefGrid]):
       #appelle select action pour choisir une action ==> calculer la cible et envoyer la cible au backend 
        name, dx, dy = select_action(
            self.x, self.y, others, self.belief, fused, self.cfg, self.rng,
        )
        tx = clamp(self.x + dx * self.cfg.step_size, 0.5, self.cfg.env_width - 0.5)
        ty = clamp(self.y + dy * self.cfg.step_size, 0.5, self.cfg.env_height - 0.5)

        if self.backend:
            self.backend.set_target(tx, ty, self.cfg.fly_altitude)

        cx, cy = self.x, self.y
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
    def __init__(self, agents: List[DroneAgent], cfg: SimConfig):
        self.agents = agents
        self.cfg = cfg
        self.prior_lo = logit(cfg.prior_occupancy)
        self.fused_belief = agents[0].belief.copy()
        self.step_count = 0
        self.history: List[Dict] = []

    def step(self): #cycle complet multidrone : perception, fusion, planification, logging
        h_before = self.fused_belief.mean_entropy()
#chaque drone met à jour sa carte locale avec son LiDAR.
        for a in self.agents:
            a.perceive()
#combine toutes les cartes locales en une carte globale:
        self.fused_belief = fuse_beliefs_logodds(
            [a.belief for a in self.agents], self.prior_lo,
        )
#planification : chaque drone choisit une action en fonction de sa carte locale et de la carte globale fusionnée, ainsi que de la position des autres drones.
        for a in self.agents:
            others = [(o.x, o.y) for o in self.agents if o.id != a.id]
            a.plan(others, self.fused_belief)

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
    _load_factory_environment()
    _add_scene_lighting()
    return world


def _load_factory_environment():
    """Load one forced factory USD path (no path search/fallback)."""
    try:
        from isaacsim.core.utils.stage import add_reference_to_stage
    except ImportError:
        from omni.isaac.core.utils.stage import add_reference_to_stage

    import omni.usd
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

    print(f"[INFO] Factory environment loaded OK: {forced_usd}")


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
        sx = cfg.env_width / 2 + (agent.id - (cfg.num_drones - 1) / 2) * cfg.drone_spacing
        sy = cfg.env_height / 2

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

        print(f"[INFO] Drone {agent.id} @ ({sx:.1f}, {sy:.1f}) — PD backend + PhysX LiDAR")

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

    world = setup_world()
    # Factory scene is authoritative; avoid injecting legacy handcrafted obstacles.
    obstacles: List[Dict] = []

    agents: List[DroneAgent] = []
    for i in range(cfg.num_drones):
        sx = cfg.env_width / 2 + (i - (cfg.num_drones - 1) / 2) * cfg.drone_spacing
        sy = cfg.env_height / 2
        agents.append(DroneAgent(i, sx, sy, cfg))

    create_physical_drones(agents, cfg)
    coordinator = SwarmCoordinator(agents, cfg)
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
