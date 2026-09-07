"""Pilotage d'un drone ArduPilot par MAVLink, pendant que la simulation avance.

La règle absolue : jamais d'attente bloquante — chaque attente fait avancer le monde. Les
ports suivent la convention SITL : commande sur tcp 5762 + 10 × identifiant.

Séquence validée par la sonde de vol du 30-08 : s'annoncer comme station sol, demander les
flux, attendre le fixe GPS 3D plus 20 s de marge, confirmer le mode guidé dans le battement de
cœur, armer avec réessais, décoller en lisant l'acquittement.

Ce module ne contient aucune loi de commande : elle vit dans `swarm_qr.control`. Ici, on parle
au pilote automatique, on convertit les repères, et on tient l'horloge simulée.
"""

from __future__ import annotations

import math
import time

import numpy as np
from pymavlink import mavutil

from pegasus.simulator.params import WORLD_SETTINGS

PHYS_DT = WORLD_SETTINGS["ardupilot"]["physics_dt"]
GUIDED = 4


class Clock:
    """L'horloge simulée, partagée par tous les pilotes d'un même monde. Faire avancer le monde
    fait avancer tous les drones à la fois : le temps est unique, il doit l'être ici aussi."""

    def __init__(self, world):
        self.world = world
        self.t = 0.0

    def pump(self, sim_seconds: float) -> None:
        n = int(round(sim_seconds / PHYS_DT))
        for _ in range(n):
            self.world.step(render=False)
        self.t += n * PHYS_DT


class Pilot:
    def __init__(self, world, vehicle_id: int = 0, clock: Clock | None = None):
        self.world = world
        self.vehicle_id = vehicle_id
        self.port = f"tcp:127.0.0.1:{5762 + vehicle_id * 10}"
        self.mav = None
        self.clock = clock or Clock(world)
        self._ned_in_world = None
        self._origin_world = None

    @property
    def sim_clock(self) -> float:
        return self.clock.t

    # --- boucle ---

    def pump(self, sim_seconds: float) -> None:
        self.clock.pump(sim_seconds)

    def _beat(self) -> None:
        self.mav.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
        )

    # --- connexion ---

    def connect(self, timeout_s: float = 90.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            self.pump(1.0)
            if self.mav is None:
                try:
                    self.mav = mavutil.mavlink_connection(self.port, retries=0)
                except Exception:
                    self.mav = None
                    continue
            if self.mav.recv_match(type="HEARTBEAT", blocking=False):
                self._beat()
                self.mav.mav.request_data_stream_send(
                    self.mav.target_system, self.mav.target_component,
                    mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1,
                )
                print(f"[PILOT {self.port}] lien etabli")
                return True
        return False

    def wait_ekf(self, timeout_sim_s: float = 60.0, timeout_wall_s: float = 900.0) -> bool:
        """L'estimateur a besoin de son origine GPS avant tout décollage — un ordre envoyé
        avant est refusé en silence. L'attente se compte en temps simulé, parce que c'est ce
        temps-là que voit l'autopilote : en temps réel, la même attente durerait deux fois
        plus longtemps sur une machine chargée."""
        t0 = self.sim_clock
        mur0 = time.monotonic()
        fix_since = None
        last_beat = 0.0
        while self.sim_clock - t0 < timeout_sim_s and time.monotonic() - mur0 < timeout_wall_s:
            self.pump(0.25)
            now = self.sim_clock
            if now - last_beat > 1.0:
                self._beat()
                last_beat = now
            while True:
                msg = self.mav.recv_match(blocking=False)
                if msg is None:
                    break
                kind = msg.get_type()
                if kind == "STATUSTEXT":
                    text = msg.text if isinstance(msg.text, str) else msg.text.decode()
                    if "is using GPS" in text:
                        self.pump(3.0)
                        print(f"[PILOT {self.port}] EKF pret ({now - t0:.0f} s simulees)")
                        return True
                elif kind == "GPS_RAW_INT" and msg.fix_type >= 3 and fix_since is None:
                    fix_since = now
            if fix_since is not None and now - fix_since > 8.0:
                print(f"[PILOT {self.port}] EKF pret via GPS ({now - t0:.0f} s simulees)")
                return True
        return False

    # --- commandes ---

    def _drain(self) -> None:
        while self.mav.recv_match(blocking=False) is not None:
            pass

    def set_guided(self) -> bool:
        for _ in range(10):
            self.mav.mav.set_mode_send(
                self.mav.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, GUIDED
            )
            self.pump(1.0)
            hb = self.mav.recv_match(type="HEARTBEAT", blocking=False)
            if hb and hb.custom_mode == GUIDED:
                return True
        return False

    def arm(self, attempts: int = 20) -> bool:
        for i in range(attempts):
            self.mav.mav.command_long_send(
                self.mav.target_system, self.mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0,
            )
            self.pump(2.0)
            self._drain()
            if self.mav.motors_armed():
                print(f"[PILOT {self.port}] arme (essai {i + 1})")
                return True
        return False

    def ack(self, command: int, timeout_sim_s: float = 1.5):
        t0 = self.sim_clock
        while self.sim_clock - t0 < timeout_sim_s:
            self.pump(0.1)
            msg = self.mav.recv_match(type="COMMAND_ACK", blocking=False)
            if msg and msg.command == command:
                return msg.result
        return None

    def takeoff(self, alt: float, get_z, timeout_sim_s: float = 40.0) -> bool:
        """`get_z` : fonction qui renvoie l'altitude vraie du drone (lue dans la simulation)."""
        z0 = get_z()
        t0 = self.sim_clock
        while self.sim_clock - t0 < timeout_sim_s:
            if get_z() - z0 < 0.3:
                self.mav.mav.command_long_send(
                    self.mav.target_system, self.mav.target_component,
                    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, alt,
                )
                self.ack(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF)
            self.pump(3.0)
            if get_z() > alt - 0.5:
                print(f"[PILOT {self.port}] decollage confirme : {get_z():.2f} m")
                return True
        return False

    def velocity(self, vx: float, vy: float, vz: float, yaw_rad: float | None = None,
                 yaw_rate: float | None = None) -> None:
        """Vitesses en repère NED (vz positif vers le bas), avec au choix un cap absolu — tenu
        par l'estimateur du drone — ou une vitesse de rotation en rad/s, positive du nord vers
        l'est."""
        if yaw_rate is not None:
            mask, yaw, rate = 0b0000011111000111, 0.0, float(yaw_rate)
        elif yaw_rad is not None:
            mask, yaw, rate = 0b0000100111000111, float(yaw_rad), 0.0
        else:
            mask, yaw, rate = 0b0000111111000111, 0.0, 0.0
        self.mav.mav.set_position_target_local_ned_send(
            0, self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            mask,
            0, 0, 0, vx, vy, vz, 0, 0, 0, yaw, rate,
        )

    def position(self, x: float, y: float, z: float, yaw_rad: float) -> None:
        """Consigne de position en repère NED, relative à l'origine de l'estimateur : c'est le
        pilote automatique qui fait alors le profil de vitesse et le freinage."""
        self.mav.mav.set_position_target_local_ned_send(
            0, self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000101111111000,
            x, y, z, 0, 0, 0, 0, 0, 0, yaw_rad, 0,
        )

    def get_param(self, name: str, timeout_sim_s: float = 5.0) -> float:
        self.mav.mav.param_request_read_send(
            self.mav.target_system, self.mav.target_component, name.encode(), -1)
        return self._attendre_param(name, timeout_sim_s)

    def set_param(self, name: str, value: float, timeout_sim_s: float = 5.0) -> float:
        self.mav.mav.param_set_send(
            self.mav.target_system, self.mav.target_component, name.encode(), float(value),
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        lu = self._attendre_param(name, timeout_sim_s)
        print(f"[PILOT {self.port}] {name} = {lu:g}")
        return lu

    def _attendre_param(self, name: str, timeout_sim_s: float) -> float:
        t0 = self.sim_clock
        while self.sim_clock - t0 < timeout_sim_s:
            self.pump(0.1)
            msg = self.mav.recv_match(type="PARAM_VALUE", blocking=False)
            if msg is not None:
                pid = msg.param_id if isinstance(msg.param_id, str) else msg.param_id.decode()
                if pid.rstrip("\x00") == name:
                    return float(msg.param_value)
        raise RuntimeError(f"parametre {name} sans reponse")

    # --- repères ---

    def calibrate_frame(self, get_pos, get_yaw=None) -> None:
        """Oriente le repère NED d'ArduPilot dans notre monde, sans hypothèse de convention.
        Avec `get_yaw` (cap vrai du drone en monde) : mesure exacte, en comparant au cap NED
        que le drone rapporte dans son message ATTITUDE. Sans : impulsion de vitesse vers le
        nord NED et lecture du déplacement — polluée d'environ 10 degrés par les transitoires
        du contrôleur, à éviter dès qu'un cap vrai est disponible.

        Situe aussi l'origine de l'estimateur dans le monde, pour les consignes de position."""
        if get_yaw is not None:
            att = self._attendre("ATTITUDE")
            north = get_yaw() + att.yaw
            x_w = np.array([np.cos(north), np.sin(north), 0.0])
        else:
            p0 = np.array(get_pos())
            for _ in range(12):
                self.velocity(0.8, 0.0, 0.0)
                self.pump(0.15)
            p1 = np.array(get_pos())
            for _ in range(15):
                self.velocity(0.0, 0.0, 0.0)
                self.pump(0.2)
            x_w = p1 - p0
            x_w[2] = 0.0
            x_w /= max(np.linalg.norm(x_w), 1e-6)
        z_w = np.array([0.0, 0.0, -1.0])          # NED : z vers le bas
        y_w = np.cross(z_w, x_w)
        self._ned_in_world = np.stack([x_w, y_w, z_w], axis=1)
        loc = self._attendre("LOCAL_POSITION_NED")
        self._origin_world = (np.array(get_pos(), float)
                              - self._ned_in_world @ np.array([loc.x, loc.y, loc.z]))
        print(f"[PILOT {self.port}] nord NED mesure en monde : {np.round(x_w, 2)}")

    def _attendre(self, kind: str, timeout_sim_s: float = 10.0):
        self._drain()
        t0 = self.sim_clock
        while self.sim_clock - t0 < timeout_sim_s:
            self.pump(0.1)
            msg = self.mav.recv_match(type=kind, blocking=False)
            if msg is not None:
                return msg
        raise RuntimeError(f"pas de message {kind} pour calibrer le repere")

    def ensure_frame(self, get_pos, get_yaw=None) -> None:
        if self._ned_in_world is None:
            self.calibrate_frame(get_pos, get_yaw)

    def yaw_ned(self, yaw_world_rad: float) -> float:
        R = self._ned_in_world
        cw = np.array([math.cos(yaw_world_rad), math.sin(yaw_world_rad), 0.0])
        return float(math.atan2(np.dot(cw, R[:, 1]), np.dot(cw, R[:, 0])))

    def velocity_world(self, v_world, yaw_world_rad: float | None = None,
                       yaw_rate_world: float | None = None) -> None:
        """`yaw_rate_world` : rotation en rad/s dans le sens du monde (z vers le haut) ; le
        repère NED a z vers le bas, la même rotation y change de signe."""
        v = self._ned_in_world.T @ np.asarray(v_world, float)
        if yaw_rate_world is not None:
            self.velocity(float(v[0]), float(v[1]), float(v[2]), yaw_rate=-float(yaw_rate_world))
            return
        yaw = None if yaw_world_rad is None else self.yaw_ned(yaw_world_rad)
        self.velocity(float(v[0]), float(v[1]), float(v[2]), yaw)

    def position_world(self, p_world, yaw_world_rad: float) -> None:
        p = self._ned_in_world.T @ (np.asarray(p_world, float) - self._origin_world)
        self.position(float(p[0]), float(p[1]), float(p[2]), self.yaw_ned(yaw_world_rad))

    def hold(self, yaw_world_rad: float | None = None) -> None:
        self.velocity_world(np.zeros(3), yaw_world_rad)

    # --- entrées de haut niveau ---

    def goto(self, target_world, yaw_world_rad: float, get_pos,
             speed: float = 0.8, tol: float = 0.15, timeout_sim_s: float = 45.0,
             get_yaw=None, yaw_tol_rad: float = 0.09) -> bool:
        """Rejoint un point du monde et s'y tient. Enveloppe bloquante du contrôleur de
        `swarm_qr.control`, pour les expériences à un seul drone."""
        from swarm_qr import control

        ctrl = control.Controleur(self, get_pos, get_yaw, v_approche=speed, tol=tol,
                                  tol_cap=yaw_tol_rad)
        ctrl.assigne(control.Consigne(np.asarray(target_world, float), yaw_world_rad),
                     budget_s=timeout_sim_s)
        return control.rejoindre(ctrl).phase is control.Phase.ATTEINT

    def ready(self, altitude: float, get_z) -> bool:
        """La séquence complète : lien, estimateur, mode, armement, décollage."""
        return (
            self.connect()
            and self.wait_ekf()
            and self.set_guided()
            and self.arm()
            and self.takeoff(altitude, get_z)
        )
