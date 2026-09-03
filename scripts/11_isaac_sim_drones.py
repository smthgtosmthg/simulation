#!/usr/bin/env python3

import argparse
import signal
import sys

sys.stdout.reconfigure(line_buffering=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Isaac Sim multi-drone simulation")
    parser.add_argument("--num-drones", type=int, default=3)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--spacing", type=float, default=3.0)
    return parser.parse_args()


def create_sim_app(headless):
    from isaacsim import SimulationApp

    config = {
        "headless": headless,
        "extra_args": ["--/rtx/verifyDriverVersion/enabled=false"],
    }
    if not headless:
        config["width"] = 1280
        config["height"] = 720
    return SimulationApp(config)


def setup_scene():
    from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

    try:
        from isaacsim.core.utils.stage import add_reference_to_stage
    except ImportError:
        from omni.isaac.core.utils.stage import add_reference_to_stage

    pegasus_if = PegasusInterface()
    pegasus_if.initialize_world()
    world = pegasus_if.world
    world.scene.add_default_ground_plane()

    warehouse_assets = [
        "omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Environments/Simple_Warehouse/warehouse.usd",
        "omniverse://localhost/NVIDIA/Assets/Isaac/Environments/Simple_Warehouse/warehouse.usd",
    ]
    for usd in warehouse_assets:
        try:
            add_reference_to_stage(usd_path=usd, prim_path="/World/Warehouse")
            break
        except Exception:
            continue

    return world


def create_drones(num_drones, spacing):
    from pegasus.simulator.params import ROBOTS
    from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig

    drones = []
    for i in range(num_drones):
        x = -((num_drones - 1) * spacing / 2.0) + i * spacing
        config = MultirotorConfig()
        config.backends = []

        drone = Multirotor(
            f"/World/Drone_{i:02d}",
            ROBOTS["Iris"],
            i,
            [x, 0.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
            config,
        )
        drones.append(drone)
        print(f"[INFO] Drone {i} @ ({x:.1f}, 0.0, 0.1)")
    return drones


def main():
    args = parse_args()

    running = True
    def on_signal(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    print(f"[INFO] Chargement Isaac Sim (mode {'headless' if args.headless else 'GUI'})...")
    sim_app = create_sim_app(args.headless)

    world = setup_scene()
    create_drones(args.num_drones, args.spacing)

    world.reset()
    print(f"[INFO] Simulation active : {args.num_drones} drone(s). Ctrl+C pour arrêter.")

    while running and sim_app.is_running():
        world.step(render=not args.headless)

    print("[INFO] Fermeture...")
    sim_app.close()


if __name__ == "__main__":
    main()
