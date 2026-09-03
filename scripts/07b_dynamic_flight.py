#!/usr/bin/env python3

import argparse
import math
import os
import signal
import sys
import time


def _bootstrap_python_with_pymavlink():
    """Re-exec the script with a known project Python if available."""
    if os.environ.get("SIM_MC02_PYMAVLINK_BOOTSTRAPPED") == "1":
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "installation", "venv", "bin", "python3"),
        os.path.expanduser(
            "~/ns-allinone-3.40/ns-3.40/contrib/sionna/model/ns3sionna/"
            "sionna-venv/bin/python3"
        ),
    ]

    current_python = os.path.abspath(sys.executable) if sys.executable else ""
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if not os.path.isfile(candidate) or not os.access(candidate, os.X_OK):
            continue
        if candidate == current_python:
            continue

        env = os.environ.copy()
        env["SIM_MC02_PYMAVLINK_BOOTSTRAPPED"] = "1"
        os.execve(candidate, [candidate, os.path.abspath(__file__), *sys.argv[1:]], env)

try:
    from pymavlink import mavutil
except ImportError:
    _bootstrap_python_with_pymavlink()
    print("ERREUR: pymavlink non installé.")
    print("Active le venv du projet ou lance scripts/installation/01_install_dependencies.sh.")
    sys.exit(1)

MODE_GUIDED = 4
MODE_LAND   = 9


# On fait passer les drones à travers des zones variées :
#   - Derrière les étagères
#   - Au centre 
#   - Aux extrémités 
#   - À différentes altitudes 

WAYPOINTS = {
    0: [
        (-10.0,  5.5,  1.5),  
        ( 0.5,   0.5,  5.5),  
        (-12.0, -6.0,  1.5),  
        ( 10.0,  5.5,  1.5),  
        ( 0.0,   0.0,  5.8),  
        (-8.0,  -5.5,  1.5),  
    ],
    1: [
        ( 0.0,   8.0,  1.5),  
        ( 0.0,  -8.0,  5.5),  
        ( 7.5,   5.5,  1.5),  
        (-7.5,  -5.5,  5.5),  
        ( 1.0,   0.5,  5.5),  
        ( 7.5,   5.5,  1.0),  
    ],
    2: [
        ( 12.0,  0.0,  2.0),  
        (-12.0,  0.0,  5.5),  
        ( 8.0,  -5.5,  1.5),  
        (-0.5,   0.5,  5.5),  
        ( 10.0,  5.5,  1.0),  
        (-8.0,   5.5,  5.5),  
    ],
}

POS_CSV = "/tmp/drone_positions.csv"
running = True


def signal_handler(sig, frame):
    global running
    running = False
    print("\n  ⛔ Interruption reçue...")

def connect_drone(instance, timeout=30):
    port = 5760 + instance * 10
    addr = f"tcp:127.0.0.1:{port}"
    print(f"  Drone {instance} : connexion à {addr}...", end=" ", flush=True)
    try:
        conn = mavutil.mavlink_connection(addr, source_system=255)
        conn.wait_heartbeat(timeout=timeout)
        print(f"OK (sysid={conn.target_system})")
        return conn
    except Exception as e:
        print(f"ERREUR : {e}")
        return None


def set_mode(conn, mode_id, timeout=10):
    conn.mav.set_mode_send(
        conn.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )
    t0 = time.time()
    while time.time() - t0 < timeout:
        hb = conn.recv_match(type='HEARTBEAT', blocking=True, timeout=2)
        if hb and hb.custom_mode == mode_id:
            return True
    return False


def arm_drone(conn, timeout=30):
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0
    )
    t0 = time.time()
    while time.time() - t0 < timeout:
        hb = conn.recv_match(type='HEARTBEAT', blocking=True, timeout=2)
        if hb and (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
            return True
    return False


def takeoff(conn, altitude):
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude
    )


def get_altitude(conn, timeout=2):
    msg = conn.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=timeout)
    if msg:
        return -msg.z
    return None


def get_position(conn, timeout=1):
    """Retourne (x, y, z) ou None."""
    msg = conn.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=timeout)
    if msg:
        return (msg.x, msg.y, -msg.z)
    return None


def request_data_streams(conn, rate_hz=4):
    conn.mav.request_data_stream_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_POSITION,
        rate_hz, 1
    )


def wait_altitude(conn, target_alt, tolerance=1.0, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        alt = get_altitude(conn, timeout=2)
        if alt is not None and abs(alt - target_alt) < tolerance:
            return True
        time.sleep(0.5)
    return False


def goto_local(conn, x, y, z):
    """Envoyer le drone vers une position locale (NED frame)."""
    conn.mav.set_position_target_local_ned_send(
        0,                                         
        conn.target_system,
        conn.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        0b0000111111111000,                        
        x, y, -z,                                  
        0, 0, 0,                                   
        0, 0, 0,                                   
        0, 0                                       
    )


def distance_3d(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def write_positions_csv(connections):
    """Écrit les positions actuelles dans le CSV + retourne un résumé."""
    parts = []
    with open(POS_CSV, "w") as pf:
        for i, conn in enumerate(connections):
            msg = conn.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=1)
            if msg:
                x, y, z = msg.x, msg.y, -msg.z
                pf.write(f"{i},{x:.4f},{y:.4f},{z:.4f}\n")
                parts.append(f"D{i}({x:6.1f},{y:6.1f},{z:4.1f}m)")
            else:
                parts.append(f"D{i}(  N/A  )")
    return "  ".join(parts)



def navigate_to_waypoint(connections, targets, tolerance=3.0, timeout=6):
    """Envoie tous les drones vers leurs waypoints.
    Timeout COURT (6s) — on enchaîne immédiatement, jamais figé.
    Re-envoie la commande goto toutes les 1s pour garder le mouvement."""
    t0 = time.time()
    while time.time() - t0 < timeout and running:
        for i, conn in enumerate(connections):
            if i in targets:
                x, y, z = targets[i]
                goto_local(conn, x, y, z)

        time.sleep(0.5)
        pos_line = write_positions_csv(connections)

        all_reached = True
        for i, conn in enumerate(connections):
            if i not in targets:
                continue
            pos = get_position(conn)
            if pos is None or distance_3d(pos, targets[i]) > tolerance:
                all_reached = False

        elapsed = time.time() - t0
        print(f"    {'✓' if all_reached else '→'} {elapsed:4.0f}s | {pos_line}")

        if all_reached:
            return True

    return False



def main():
    parser = argparse.ArgumentParser(
        description='Vol dynamique multi-drones — Patrouille warehouse')
    parser.add_argument('--drones', type=int, default=3)
    parser.add_argument('--altitude', type=float, default=4.0,
                        help='Altitude initiale (défaut: 4)')
    parser.add_argument('--alt-step', type=float, default=1.0)
    parser.add_argument('--ekf-wait', type=int, default=20)
    parser.add_argument('--loops', type=int, default=2,
                        help='Nombre de boucles de patrouille (défaut: 2)')
    args = parser.parse_args()

    n_drones = min(args.drones, 3)
    signal.signal(signal.SIGINT, signal_handler)

    W = 70
    print()
    print("┌" + "─" * (W - 2) + "┐")
    print("│" + "  08b — VOL DYNAMIQUE : PATROUILLE WAREHOUSE".ljust(W - 2) + "│")
    print("│" + f"  {n_drones} drones · {args.loops} boucles · {len(WAYPOINTS[0])} waypoints/drone".ljust(W - 2) + "│")
    print("│" + f"  Max 6s/WP · Positions → {POS_CSV}".ljust(W - 2) + "│")
    print("└" + "─" * (W - 2) + "┘")
    print()

    print(f"[1/6] Connexion aux {n_drones} drones...")
    connections = []
    for i in range(n_drones):
        conn = connect_drone(i)
        if conn is None:
            print(f"\nERREUR: Drone {i} non accessible.")
            print("Vérifie que 06_launch_multi_drones.sh tourne.")
            sys.exit(1)
        connections.append(conn)
    print(f"  => Tous connectés !\n")

    for conn in connections:
        request_data_streams(conn)

    print(f"[2/6] Attente calibration EKF ({args.ekf_wait}s)...")
    for s in range(args.ekf_wait, 0, -1):
        sys.stdout.write(f"\r  {s}s...")
        sys.stdout.flush()
        time.sleep(1)
    print("\r  => EKF prêt !\n")

    print("[3/6] Mode GUIDED...")
    for i, conn in enumerate(connections):
        ok = set_mode(conn, MODE_GUIDED)
        print(f"  Drone {i} : {'GUIDED' if ok else 'TIMEOUT'}")
    print()

    print("[4/6] Armement...")
    for i, conn in enumerate(connections):
        ok = arm_drone(conn, timeout=15)
        print(f"  Drone {i} : {'ARMED' if ok else 'ECHEC'}")
        if not ok:
            time.sleep(5)
            ok = arm_drone(conn, timeout=15)
            print(f"  Drone {i} (retry) : {'ARMED' if ok else 'ECHEC'}")
    print()

    print("[5/6] Décollage...")
    target_alts = []
    for i, conn in enumerate(connections):
        alt = args.altitude + i * args.alt_step
        target_alts.append(alt)
        takeoff(conn, alt)
        print(f"  Drone {i} : takeoff → {alt}m")
        time.sleep(2)

    print("\n  Attente des altitudes cibles...")
    for s in range(30):
        time.sleep(1)
        pos_line = write_positions_csv(connections)
        all_ok = all(
            (lambda a: a is not None and abs(a - target_alts[i]) < 1.0)(get_altitude(c))
            for i, c in enumerate(connections)
        )
        if s % 3 == 0:
            print(f"  ↗ {s}s | {pos_line}")
        if all_ok:
            break
    print()

    print("[6/6] ═══ DÉBUT DE LA PATROUILLE ═══")
    print(f"  {args.loops} boucles × {len(WAYPOINTS[0])} waypoints")
    print(f"  Max 6s/WP — mouvement continu, jamais figé !")
    print(f"  Les étagères (3m haut) vont bloquer le signal !")
    print()

    total_wps = len(WAYPOINTS[0])

    for loop in range(args.loops):
        if not running:
            break
        print(f"  ┌── Boucle {loop + 1}/{args.loops} ──────────────────────────")

        for wp_idx in range(total_wps):
            if not running:
                break

            targets = {}
            descriptions = []
            for i in range(n_drones):
                wps = WAYPOINTS.get(i, WAYPOINTS[0])
                wp = wps[wp_idx % len(wps)]
                targets[i] = wp
                descriptions.append(f"D{i}→({wp[0]:+.0f},{wp[1]:+.0f},{wp[2]:.0f}m)")

            print(f"  │")
            print(f"  ├─ WP {wp_idx + 1}/{total_wps}: {' · '.join(descriptions)}")

            navigate_to_waypoint(connections, targets, tolerance=3.0, timeout=6)

        print(f"  └── Boucle {loop + 1} terminée")
        print()

    print("ATTERRISSAGE...")
    for i, conn in enumerate(connections):
        set_mode(conn, MODE_LAND)
        print(f"  Drone {i} : LAND")

    for s in range(30):
        time.sleep(1)
        pos_line = write_positions_csv(connections)
        if s % 5 == 4:
            print(f"  ↘ {s+1}s | {pos_line}")
        all_landed = all(
            (lambda a: a is not None and a < 0.3)(get_altitude(c))
            for c in connections
        )
        if all_landed:
            print("  ✓ Tous posés !")
            break

    print()
    print("=" * W)
    print("  PATROUILLE TERMINÉE !")
    print("=" * W)

    for conn in connections:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
