"""Sonde de vol automatique, sans fenêtre — les trois mesures qui décident.

Même recette que sonde_gui.py (chaîne validée : entrepôt en cache, World au pas ArduPilot,
backend en lancement automatique). Le script fait tout lui-même par MAVLink, sur le port de
secours du SITL (tcp:5762), pendant que Pegasus garde son propre lien sur 14550 :

  1. mode guidé, armement, décollage à 3 m ;
  2. STATIONNAIRE : 10 s immobile → écart-type et rayon maximal, en centimètres.
     C'est la mesure qui conditionne toute la lecture de QR ;
  3. ARRÊT : vitesse commandée à 1 m/s puis zéro → temps et distance pour s'immobiliser.
     C'est l'inertie réelle du drone ;
  4. DÉBIT : pas de simulation par seconde (physique à 1/800 s → temps réel = 800 pas/s).

  DISPLAY=:1 ~/isaac5_env/bin/python swarm_qr/experiments/05_sitl/vol_auto.py
  (DISPLAY reste nécessaire : le lancement automatique d'ArduPilot ouvre un terminal.)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {"headless": True, "extra_args": ["--/rtx/verifyDriverVersion/enabled=false"]}
)

import traceback  # noqa: E402

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.api.world import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from pymavlink import mavutil  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from pegasus.simulator.logic.backends.ardupilot_mavlink_backend import (  # noqa: E402
    ArduPilotMavlinkBackend,
    ArduPilotMavlinkBackendConfig,
)
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface  # noqa: E402
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig  # noqa: E402
from pegasus.simulator.params import ROBOTS, WORLD_SETTINGS  # noqa: E402

from swarm_qr.env.config import WAREHOUSE_PRIM, WAREHOUSE_USD  # noqa: E402

SPAWN = (-5.0, 0.0, 0.10)
CONTROL_PORT = "tcp:127.0.0.1:5762"
PHYS_DT = WORLD_SETTINGS["ardupilot"]["physics_dt"]
GUIDED = 4

REPORT: dict = {}


def build() -> tuple:
    pg = PegasusInterface()
    pg._world = World(**WORLD_SETTINGS["ardupilot"])
    world = pg.world

    add_reference_to_stage(usd_path=WAREHOUSE_USD, prim_path=WAREHOUSE_PRIM)
    world.scene.add_default_ground_plane()

    backend = ArduPilotMavlinkBackend(
        config=ArduPilotMavlinkBackendConfig(
            {
                "vehicle_id": 0,
                "ardupilot_autolaunch": True,
                "ardupilot_dir": pg.ardupilot_path,
                "ardupilot_vehicle_model": "gazebo-iris",
            }
        )
    )
    config = MultirotorConfig()
    config.backends = [backend]
    drone = Multirotor(
        "/World/Drone_00", ROBOTS["Iris"], 0, list(SPAWN),
        Rotation.from_euler("XYZ", [0, 0, 0], degrees=True).as_quat(), config=config,
    )
    world.reset()
    return world, drone


class Pilot:
    """Commandes MAVLink pendant que la simulation avance : jamais d'attente bloquante."""

    def __init__(self, world):
        self.world = world
        self.mav = None

    def pump(self, seconds: float) -> None:
        for _ in range(int(seconds / PHYS_DT)):
            self.world.step(render=False)

    def connect(self, timeout_s: float = 60.0) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            self.pump(1.0)
            if self.mav is None:
                try:
                    self.mav = mavutil.mavlink_connection(CONTROL_PORT, retries=0)
                except Exception:
                    self.mav = None
                    continue
            hb = self.mav.recv_match(type="HEARTBEAT", blocking=False)
            if hb is not None:
                print(f"[VOL] lien de commande etabli sur {CONTROL_PORT}")
                # S'annoncer comme station sol et demander les flux : sans ca, ArduPilot
                # n'envoie presque rien sur un port secondaire (MAVProxy le fait tout seul).
                self.mav.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
                )
                self.mav.mav.request_data_stream_send(
                    self.mav.target_system, self.mav.target_component,
                    mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1,
                )
                return True
        return False

    def drain(self):
        while self.mav.recv_match(blocking=False) is not None:
            pass

    def wait_ekf(self, timeout_s: float = 150.0) -> bool:
        """Attendre que l'estimateur ait une position. Deux signaux acceptes : le message
        « EKF3 IMU0 is using GPS » (la consigne d'INSTRUCTIONS.md), ou un GPS fixe 3D suivi
        de 20 s de marge. Temps MUR : le SITL vit en temps reel."""
        t0 = time.monotonic()
        fix_since = None
        last_beat = 0.0
        while time.monotonic() - t0 < timeout_s:
            self.pump(0.25)
            now = time.monotonic()
            if now - last_beat > 1.0:
                self.mav.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
                )
                last_beat = now
            while True:
                msg = self.mav.recv_match(blocking=False)
                if msg is None:
                    break
                kind = msg.get_type()
                if kind == "STATUSTEXT":
                    text = msg.text if isinstance(msg.text, str) else msg.text.decode()
                    print(f"[VOL] ardupilot : {text}")
                    if "is using GPS" in text:
                        print(f"[VOL] EKF pret via statustext ({now - t0:.0f} s)")
                        self.pump(3.0)
                        return True
                elif kind == "GPS_RAW_INT" and msg.fix_type >= 3 and fix_since is None:
                    fix_since = now
                    print(f"[VOL] GPS fixe 3D ({now - t0:.0f} s), marge de 20 s...")
            if fix_since is not None and now - fix_since > 20.0:
                print(f"[VOL] EKF pret via GPS ({now - t0:.0f} s)")
                return True
        print("[VOL] EKF jamais pret")
        return False

    def set_guided(self) -> bool:
        for _ in range(10):
            self.mav.mav.set_mode_send(
                self.mav.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, GUIDED
            )
            self.pump(1.0)
            hb = self.mav.recv_match(type="HEARTBEAT", blocking=False)
            if hb and hb.custom_mode == GUIDED:
                print("[VOL] mode GUIDED confirme")
                return True
        print("[VOL] mode GUIDED jamais confirme")
        return False

    def ack(self, command: int, timeout_s: float = 3.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            self.pump(0.1)
            msg = self.mav.recv_match(type="COMMAND_ACK", blocking=False)
            if msg and msg.command == command:
                return msg.result
        return None

    def arm(self, attempts: int = 15) -> bool:
        for i in range(attempts):
            self.mav.mav.command_long_send(
                self.mav.target_system, self.mav.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0,
            )
            self.pump(2.0)
            self.drain()
            if self.mav.motors_armed():
                print(f"[VOL] arme (essai {i + 1})")
                return True
        return False

    def takeoff(self, alt: float) -> None:
        self.mav.mav.command_long_send(
            self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, alt,
        )

    def velocity(self, vx: float, vy: float, vz: float) -> None:
        self.mav.mav.set_position_target_local_ned_send(
            0, self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            0b0000111111000111,  # vitesses seules
            0, 0, 0, vx, vy, vz, 0, 0, 0, 0, 0,
        )


def main() -> None:
    print("[VOL] construction de la scene...")
    world, drone = build()
    omni.timeline.get_timeline_interface().play()

    pilot = Pilot(world)
    if not pilot.connect():
        REPORT["lien_commande"] = False
        print("[VOL] ECHEC : pas de lien de commande")
        return
    REPORT["lien_commande"] = True

    z0 = float(drone.state.position[2])
    REPORT["ekf"] = pilot.wait_ekf()
    if not REPORT["ekf"]:
        return
    REPORT["mode_guided"] = pilot.set_guided()
    if not REPORT["mode_guided"]:
        return
    if not pilot.arm():
        REPORT["arme"] = False
        print("[VOL] ECHEC : armement refuse")
        return
    REPORT["arme"] = True

    # --- decollage, avec acquittement et re-essais ---
    z = z0
    t0 = time.monotonic()
    acks = []
    while time.monotonic() - t0 < 90.0:
        if float(drone.state.position[2]) - z0 < 0.3:
            pilot.takeoff(3.0)
            result = pilot.ack(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF)
            acks.append(result)
            print(f"[VOL] takeoff envoye, acquittement = {result}"
                  f" (0 = accepte)")
        pilot.pump(3.0)
        z = float(drone.state.position[2])
        print(f"[VOL] altitude {z:.2f} m")
        if z > 2.5:
            break
    REPORT["takeoff_acks"] = [int(a) if a is not None else None for a in acks]
    REPORT["decollage"] = {"altitude_m": round(z, 2), "ok": z > 2.5}
    print(f"[VOL] altitude apres decollage : {z:.2f} m -> "
          f"{'DECOLLAGE CONFIRME' if z > 2.5 else 'ECHEC'}")
    if z <= 2.5:
        return
    pilot.pump(4.0)

    # --- mesure 1 : stationnaire ---
    n = int(10.0 / PHYS_DT)
    poses = np.empty((n, 3))
    for i in range(n):
        world.step(render=False)
        poses[i] = drone.state.position
    std_cm = poses.std(axis=0) * 100.0
    radius_cm = float(np.linalg.norm(poses[:, :2] - poses[:, :2].mean(axis=0), axis=1).max() * 100)
    REPORT["stationnaire"] = {
        "duree_s": 10.0,
        "ecart_type_cm": [round(float(s), 2) for s in std_cm],
        "rayon_max_cm": round(radius_cm, 2),
    }
    print(f"[VOL] stationnaire 10 s : ecart-type xyz = "
          f"{np.round(std_cm, 2).tolist()} cm, rayon max = {radius_cm:.1f} cm")

    # --- mesure 2 : arret depuis 1 m/s ---
    for _ in range(int(3.0 / 0.2)):
        pilot.velocity(1.0, 0.0, 0.0)
        pilot.pump(0.2)
    v = float(np.linalg.norm(drone.state.linear_velocity[:2]))
    print(f"[VOL] vitesse atteinte : {v:.2f} m/s")
    p_stop = np.array(drone.state.position[:2])
    t0 = time.monotonic()
    stop_time, stop_dist = None, None
    while time.monotonic() - t0 < 15.0:
        pilot.velocity(0.0, 0.0, 0.0)
        pilot.pump(0.1)
        if float(np.linalg.norm(drone.state.linear_velocity[:2])) < 0.05:
            stop_time = time.monotonic() - t0
            stop_dist = float(np.linalg.norm(np.array(drone.state.position[:2]) - p_stop))
            break
    REPORT["arret"] = {
        "vitesse_initiale_ms": round(v, 2),
        "temps_s": round(stop_time, 2) if stop_time else None,
        "distance_m": round(stop_dist, 2) if stop_dist else None,
    }
    print(f"[VOL] arret depuis {v:.2f} m/s : "
          f"{stop_time:.2f} s, {stop_dist:.2f} m" if stop_time else "[VOL] arret non atteint en 15 s")

    # --- mesure 3 : debit ---
    steps = 2400
    t0 = time.perf_counter()
    for _ in range(steps):
        world.step(render=False)
    dt = time.perf_counter() - t0
    sps = steps / dt
    REPORT["debit"] = {
        "pas_par_s": round(sps, 1),
        "dt_physique": PHYS_DT,
        "x_temps_reel": round(sps * PHYS_DT, 2),
    }
    print(f"[VOL] debit : {sps:.0f} pas/s ({sps * PHYS_DT:.2f}x le temps reel)")

    print("[VOL] SONDE TERMINEE")


try:
    main()
except Exception:
    traceback.print_exc()
finally:
    (HERE / "resultat_vol.json").write_text(json.dumps(REPORT, indent=2, ensure_ascii=False))
    simulation_app.close()
