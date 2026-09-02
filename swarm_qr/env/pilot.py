"""Pilotage d'un drone ArduPilot par MAVLink, pendant que la simulation avance.

La règle absolue : jamais d'attente bloquante — chaque attente fait avancer le monde. Les
ports suivent la convention SITL : commande sur tcp 5762 + 10 × identifiant.

Séquence validée par la sonde de vol du 30-08 : s'annoncer comme station sol, demander les
flux, attendre le fixe GPS 3D plus 20 s de marge, confirmer le mode guidé dans le battement de
cœur, armer avec réessais, décoller en lisant l'acquittement.
"""

from __future__ import annotations

import time

from pymavlink import mavutil

from pegasus.simulator.params import WORLD_SETTINGS

PHYS_DT = WORLD_SETTINGS["ardupilot"]["physics_dt"]
GUIDED = 4


class Pilot:
    def __init__(self, world, vehicle_id: int = 0):
        self.world = world
        self.port = f"tcp:127.0.0.1:{5762 + vehicle_id * 10}"
        self.mav = None
        self.sim_clock = 0.0

    # --- boucle ---

    def pump(self, sim_seconds: float) -> None:
        n = int(sim_seconds / PHYS_DT)
        for _ in range(n):
            self.world.step(render=False)
        self.sim_clock += n * PHYS_DT

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

    def velocity(self, vx: float, vy: float, vz: float, yaw_rad: float | None = None) -> None:
        """Vitesses en repère NED (vz positif vers le bas), cap optionnel."""
        if yaw_rad is None:
            mask = 0b0000111111000111
            yaw = 0.0
        else:
            mask = 0b0000100111000111
            yaw = yaw_rad
        self.mav.mav.set_position_target_local_ned_send(
            0, self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            mask,
            0, 0, 0, vx, vy, vz, 0, 0, 0, yaw, 0,
        )

    def calibrate_frame(self, get_pos, get_yaw=None) -> None:
        """Oriente le repère NED d'ArduPilot dans notre monde, sans hypothèse de convention.
        Avec `get_yaw` (cap vrai du drone en monde) : mesure exacte, en comparant au cap NED
        que le drone rapporte dans son message ATTITUDE. Sans : impulsion de vitesse vers le
        nord NED et lecture du déplacement — polluée d'environ 10 degrés par les transitoires
        du contrôleur, à éviter dès qu'un cap vrai est disponible."""
        import numpy as np

        if get_yaw is not None:
            self._drain()
            t0 = self.sim_clock
            msg = None
            while msg is None and self.sim_clock - t0 < 10.0:
                self.pump(0.1)
                msg = self.mav.recv_match(type="ATTITUDE", blocking=False)
            if msg is None:
                raise RuntimeError("pas de message ATTITUDE pour calibrer le repere")
            north = get_yaw() + msg.yaw
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
        print(f"[PILOT {self.port}] nord NED mesure en monde : {np.round(x_w, 2)}")

    def goto(self, target_world, yaw_world_rad: float, get_pos,
             speed: float = 0.8, tol: float = 0.15, timeout_sim_s: float = 45.0,
             get_yaw=None, yaw_tol_rad: float = 0.09) -> bool:
        """Rejoint un point du monde par asservissement proportionnel en vitesse NED.

        Si `get_yaw` est fourni (cap réel du drone, en monde), l'arrivée exige aussi le cap :
        le drone tourne bien plus lentement qu'il ne se déplace, valider la seule position
        laisse la caméra de travers.

        Le délai se compte en temps simulé. Mesuré en temps réel, il dépendrait de la charge
        de la machine : le même vol réussirait à vide et échouerait pendant un calcul lourd."""
        import math

        import numpy as np

        if not hasattr(self, "_ned_in_world"):
            self.calibrate_frame(get_pos, get_yaw)
        R = self._ned_in_world
        # le cap NED se deduit du meme reperage : angle du cap monde dans la base NED
        cw = np.array([np.cos(yaw_world_rad), np.sin(yaw_world_rad), 0.0])
        yaw_ned = float(np.arctan2(np.dot(cw, R[:, 1]), np.dot(cw, R[:, 0])))
        t0 = self.sim_clock
        while self.sim_clock - t0 < timeout_sim_s:
            err_w = np.array(target_world) - np.array(get_pos())
            yaw_ok = True
            if get_yaw is not None:
                d = yaw_world_rad - get_yaw()
                yaw_ok = abs(math.atan2(math.sin(d), math.cos(d))) < yaw_tol_rad
            if np.linalg.norm(err_w) < tol and yaw_ok:
                self.velocity(0.0, 0.0, 0.0, yaw_ned)
                self.pump(0.3)
                return True
            v_w = err_w * 0.9
            n = np.linalg.norm(v_w)
            if n > speed:
                v_w *= speed / n
            v_ned = R.T @ v_w
            self.velocity(float(v_ned[0]), float(v_ned[1]), float(v_ned[2]), yaw_ned)
            self.pump(0.15)
        return False

    def ready(self, altitude: float, get_z) -> bool:
        """La séquence complète : lien, estimateur, mode, armement, décollage."""
        return (
            self.connect()
            and self.wait_ekf()
            and self.set_guided()
            and self.arm()
            and self.takeoff(altitude, get_z)
        )
