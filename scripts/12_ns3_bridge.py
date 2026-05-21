#!/usr/bin/env python3
"""
NS-3 bridge pour 12_aif_isaac_sim.py (Tâche 3 du PROMPT_CLAUDE_CODE.md).

Rôle :
 - Lance ns3 (drone-wifi-scenario ou drone-5g-nr-scenario) en arrière-plan.
 - NS-3 lit les positions courantes depuis /tmp/drone_positions.csv
   (qu'Isaac Sim écrit après chaque step AIF).
 - NS-3 écrit les latences dans /tmp/ns3_output.csv (WiFi) ou
   /tmp/drone_latency_ns3.csv (5G).
 - NS3LatencyReader (dans 12_aif_isaac_sim.py) relit ce CSV à chaque step
   et injecte les latences dans la MessageQueue.

Inspiré de scripts/08_wifi_bridge.py (NS-3 WiFi) et 09_5g_lena_bridge.py.

Usage programmatique (depuis 12_aif_isaac_sim.py) :
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("ns3_bridge", ".../12_ns3_bridge.py")
    mod  = module_from_spec(spec); spec.loader.exec_module(mod)
    ok = mod.launch_ns3(n_drones=3, sim_time=600, scenario="wifi")
    ...
    mod.stop_ns3()

Usage CLI direct (pour debug standalone) :
    python scripts/12_ns3_bridge.py --scenario wifi --n-drones 3 --sim-time 600
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Optional


# ── Chemins NS-3 (mêmes valeurs que dans 08_wifi_bridge.py / 09_5g_lena_bridge.py) ──
NS3_DIR = os.path.expanduser("~/ns-allinone-3.40/ns-3.40")
NS3_BIN = os.path.join(NS3_DIR, "ns3")

# Fichiers d'échange Isaac Sim ↔ NS-3
POS_CSV = "/tmp/drone_positions.csv"          # INPUT: positions courantes
WIFI_OUT_CSV = "/tmp/ns3_output.csv"          # OUTPUT WiFi
LTE5G_OUT_CSV = "/tmp/drone_latency_ns3.csv"  # OUTPUT 5G NR

# Scénarios disponibles (compilés via ./waf build dans NS-3)
SCENARIOS = {
    "wifi": "drone-wifi-scenario",
    "5g":   "drone-5g-nr-scenario",
}

# Process NS-3 actif (module global pour permettre stop_ns3())
_ns3_process: Optional[subprocess.Popen] = None


def _ns3_available() -> bool:
    """True si NS-3 est installé et compilé."""
    return os.path.isfile(NS3_BIN)


def launch_ns3(n_drones: int = 3, sim_time: int = 600,
               scenario: str = "wifi",
               pos_csv: str = POS_CSV,
               channel_model: str = "log-distance") -> bool:
    """Lance NS-3 en subprocess (mode temps réel).

    Args:
        n_drones: nombre de drones simulés.
        sim_time: durée de simulation NS-3 (secondes).
        scenario: "wifi" | "5g".
        pos_csv: chemin du CSV de positions (Isaac Sim écrit, NS-3 lit).
        channel_model: log-distance | nakagami | etc. (passé au scénario).

    Returns:
        True si NS-3 a démarré, False sinon (degradation gracieuse).
    """
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
    # On passe les arguments comme une seule chaîne (comportement de `ns3 run`)
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
        # Laisser 2 secondes pour détecter un échec immédiat
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
    """Arrête proprement NS-3 (terminate + kill au timeout)."""
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


# ── Helpers pour les positions (utilisés depuis Isaac Sim via main) ──

def write_drone_positions(positions: dict, path: str = POS_CSV) -> None:
    """Écrit le CSV des positions au format attendu par NS-3 / le script 08.

    Format : `drone_id,x,y,z` (une ligne par drone). Réécriture complète
    pour que NS-3 lise toujours l'état courant."""
    try:
        with open(path, "w") as f:
            for did in sorted(positions):
                x, y, z = positions[did]
                f.write(f"{did},{x:.4f},{y:.4f},{z:.4f}\n")
    except OSError as e:
        print(f"  [NS3-BRIDGE] ⚠ écriture positions impossible ({path}) : {e}")


# ── Mode standalone (debug) ──

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
