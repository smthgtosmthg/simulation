"""Sonde SITL — un drone ArduPilot dans notre entrepôt, fenêtre graphique.

La recette vient de ce qui a déjà marché, rien n'est improvisé :
- démarrage, boucle et arrêt propre : scripts/11_isaac_sim_drones.py et 12_aif_isaac_sim.py,
  le pipeline validé du projet (8 runs archivés dans logs/runs/) ;
- backend ArduPilot : l'exemple officiel de Pegasus (examples/11_ardupilot_multi_vehicle.py) ;
- pas de temps : WORLD_SETTINGS["ardupilot"] (physique à 1/800 s), le réglage que l'interface
  officielle de Pegasus applique pour ArduPilot — l'exemple, lui, l'oublie ;
- décor : NOTRE entrepôt par son URL directe, déjà en cache local. Jamais les décors cloud de
  Pegasus : leur téléchargement bloque la fenêtre, et c'est lui qui déclenchait le dialogue
  « l'application ne répond pas » de GNOME.

Ce qui se passe au lancement :
1. la fenêtre Isaac s'ouvre (~10 s, jusqu'à ~45 s au tout premier lancement, shaders) ;
2. l'entrepôt apparaît, le drone Iris est posé dans l'allée ;
3. un terminal MAVProxy s'ouvre tout seul (lancement automatique d'ArduPilot SITL) ;
4. dans ce terminal, taper :   mode guided     puis     arm throttle     puis     takeoff 3
5. le drone décolle ; sa position s'affiche aussi ici, dans ce terminal-ci.

Arrêt : Ctrl+C ici, ou fermer la fenêtre Isaac. Pegasus tue alors le SITL et MAVProxy.
Si GNOME affiche « ne répond pas » pendant un chargement : cliquer « Attendre ».

  DISPLAY=:1 ~/isaac5_env/bin/python swarm_qr/experiments/05_sitl/sonde_gui.py
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": 1280,
        "height": 720,
        "extra_args": ["--/rtx/verifyDriverVersion/enabled=false"],
    }
)

# Tout ce qui suit exige que l'application soit démarrée.
import omni.timeline  # noqa: E402
from isaacsim.core.api.world import World  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from pegasus.simulator.logic.backends.ardupilot_mavlink_backend import (  # noqa: E402
    ArduPilotMavlinkBackend,
    ArduPilotMavlinkBackendConfig,
)
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface  # noqa: E402
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig  # noqa: E402
from pegasus.simulator.params import ROBOTS, WORLD_SETTINGS  # noqa: E402

from swarm_qr.env.config import WAREHOUSE_PRIM, WAREHOUSE_USD  # noqa: E402

SPAWN = (-5.0, 0.0, 0.10)  # dans l'allée ouest, dégagée du sol au plafond


def build_world(pg: PegasusInterface) -> World:
    """La World au pas de temps officiel ArduPilot (1/800 s), pas celui de l'exemple."""
    pg._world = World(**WORLD_SETTINGS["ardupilot"])
    return pg.world


def load_scene(world: World) -> None:
    """Notre entrepôt (URL directe, cache local) + sol + lumière."""
    add_reference_to_stage(usd_path=WAREHOUSE_USD, prim_path=WAREHOUSE_PRIM)
    world.scene.add_default_ground_plane()

    from pxr import UsdLux

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(2500.0)


def create_drone(pg: PegasusInterface) -> Multirotor:
    """Un Iris avec le backend ArduPilot en lancement automatique."""
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

    return Multirotor(
        "/World/Drone_00",
        ROBOTS["Iris"],
        0,
        list(SPAWN),
        Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
        config=config,
    )


def main() -> None:
    running = True

    def on_signal(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    timeline = omni.timeline.get_timeline_interface()
    pg = PegasusInterface()
    world = build_world(pg)

    print("[SONDE] chargement de l'entrepot...")
    load_scene(world)
    drone = create_drone(pg)
    world.reset()

    print("[SONDE] scene prete, demarrage de la simulation")
    print("[SONDE] un terminal MAVProxy va s'ouvrir : y taper")
    print("[SONDE]   mode guided   puis   arm throttle   puis   takeoff 3")
    timeline.play()

    last_print = 0.0
    while running and simulation_app.is_running():
        world.step(render=True)
        now = time.monotonic()
        if now - last_print >= 2.0:
            p = drone.state.position
            print(f"[SONDE] drone x={p[0]:+.2f} y={p[1]:+.2f} z={p[2]:+.2f} m")
            last_print = now

    print("[SONDE] fermeture...")
    timeline.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
