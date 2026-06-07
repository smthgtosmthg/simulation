#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Optional


NS3_DIR = os.path.expanduser("~/ns-allinone-3.40/ns-3.40")
NS3_BIN = os.path.join(NS3_DIR, "ns3")

POS_CSV = "/tmp/drone_positions.csv"
WIFI_OUT_CSV = "/tmp/ns3_output.csv"
LTE5G_OUT_CSV = "/tmp/drone_latency_ns3.csv"

SCENARIOS = {
    "wifi": "drone-wifi-scenario",
    "5g":   "drone-5g-nr-scenario",
}

_ns3_process: Optional[subprocess.Popen] = None


def _ns3_available() -> bool:
    return os.path.isfile(NS3_BIN)


def launch_ns3(n_drones: int = 3, sim_time: int = 600,
               scenario: str = "wifi",
               pos_csv: str = POS_CSV,
               channel_model: str = "log-distance") -> bool:
    global _ns3_process

    if scenario not in SCENARIOS:
        print(f"  [NS3-BRIDGE] ⚠ Scénario inconnu : {scenario!r} "
              f"(valides : {list(SCENARIOS)})")
        return False

    if not _ns3_available():
        print(f"  [NS3-BRIDGE] ⚠ NS-3 non trouvé : {NS3_BIN}")
        print(f"  [NS3-BRIDGE]   → Latences = 0, simulation continue sans NS-3.")
        return False

    # Nettoyer le CSV de sortie pour ne pas relire un ancien run
    out_csv = WIFI_OUT_CSV if scenario == "wifi" else LTE5G_OUT_CSV
    if os.path.exists(out_csv):
        try:
            os.remove(out_csv)
        except OSError:
            pass

    scenario_name = SCENARIOS[scenario]
    args_str = (
        f"{scenario_name} "
        f"--nDrones={n_drones} "
        f"--posFile={pos_csv} "
        f"--outFile={out_csv} "
        f"--simTime={sim_time}"
    )
    if scenario == "wifi":
        args_str += f" --channelModel={channel_model}"

    cmd = [NS3_BIN, "run", args_str, "--no-build"]
    print(f"  [NS3-BRIDGE] ⏳ Lancement {scenario_name} (n={n_drones}, t={sim_time}s)…")
    try:
        _ns3_process = subprocess.Popen(
            cmd, cwd=NS3_DIR,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        # Laisser 2s pour détecter un échec immédiat
        time.sleep(2.0)
        if _ns3_process.poll() is not None:
            err = b""
            if _ns3_process.stderr is not None:
                try:
                    err = _ns3_process.stderr.read()
                except Exception:
                    err = b""
            err_txt = err.decode(errors="replace")[-400:]
            print(f"  [NS3-BRIDGE] ⚠ NS-3 a crashé : {err_txt}")
            _ns3_process = None
            return False
        print(f"  [NS3-BRIDGE] ✓ NS-3 lancé (PID {_ns3_process.pid})")
        return True
    except Exception as e:
        print(f"  [NS3-BRIDGE] ⚠ Échec lancement NS-3 : {e}")
        _ns3_process = None
        return False


def is_alive() -> bool:
    return _ns3_process is not None and _ns3_process.poll() is None


def stop_ns3() -> None:
    global _ns3_process
    if _ns3_process is None:
        return
    if _ns3_process.poll() is None:
        try:
            _ns3_process.terminate()
            try:
                _ns3_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _ns3_process.kill()
            print("  [NS3-BRIDGE] NS-3 arrêté.")
        except Exception as e:
            print(f"  [NS3-BRIDGE] erreur stop : {e}")
    _ns3_process = None


def write_drone_positions(positions: dict, path: str = POS_CSV) -> None:
    # Format CSV : drone_id,x,y,z (réécriture complète à chaque step)
    try:
        with open(path, "w") as f:
            for did in sorted(positions):
                x, y, z = positions[did]
                f.write(f"{did},{x:.4f},{y:.4f},{z:.4f}\n")
    except OSError as e:
        print(f"  [NS3-BRIDGE] ⚠ écriture positions impossible ({path}) : {e}")


def _signal_handler(sig, frame):
    stop_ns3()
    sys.exit(0)


def main():
    p = argparse.ArgumentParser(description="NS-3 bridge for Isaac Sim AIF")
    p.add_argument("--scenario", choices=list(SCENARIOS), default="wifi")
    p.add_argument("--n-drones", type=int, default=3)
    p.add_argument("--sim-time", type=int, default=600)
    p.add_argument("--pos-csv", default=POS_CSV)
    args = p.parse_args()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    ok = launch_ns3(n_drones=args.n_drones, sim_time=args.sim_time,
                    scenario=args.scenario, pos_csv=args.pos_csv)
    if not ok:
        print("  [NS3-BRIDGE] échec du lancement, sortie.")
        return 1
    print("  [NS3-BRIDGE] tourne en avant-plan — Ctrl+C pour stopper.")
    try:
        while is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    stop_ns3()
    return 0


if __name__ == "__main__":
    sys.exit(main())
