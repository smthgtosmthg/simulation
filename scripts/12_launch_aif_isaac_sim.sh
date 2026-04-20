#!/usr/bin/env bash

set -euo pipefail

N_DRONES=${1:-3}
shift 2>/dev/null || true   # consume $1 so "$@" holds remaining flags
EXTRA_ARGS="$@"
VENV_DIR="${HOME}/isaac_sim_env"
ACTIVATE_SCRIPT="${VENV_DIR}/activate_isaac.sh"
WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"
ARDUPILOT_DIR="${HOME}/ardupilot"

log()  { echo "[INFO] $*"; }
warn() { echo "[WARN] $*"; }
die()  { echo "[ERROR] $*"; exit 1; }

# --- 0. Vérifier ArduPilot SITL ---
[[ -d "${ARDUPILOT_DIR}" ]] \
    || die "ArduPilot non trouvé dans ${ARDUPILOT_DIR}. Cloner avec : git clone https://github.com/ArduPilot/ardupilot.git ~/ardupilot"
[[ -f "${ARDUPILOT_DIR}/build/sitl/bin/arducopter" ]] \
    || die "ArduPilot SITL pas compilé. Lancer : cd ~/ardupilot && ./waf configure --board sitl && ./waf copter"

# Nettoyer les processus SITL résiduels
log "Nettoyage des processus SITL résiduels…"
pkill -f "arducopter" 2>/dev/null || true
pkill -f "sim_vehicle.py" 2>/dev/null || true
pkill -f "mavproxy" 2>/dev/null || true
sleep 1

# Patcher gazebo-iris.parm pour le JSON backend Isaac Sim
PARM_FILE="${ARDUPILOT_DIR}/Tools/autotest/default_params/gazebo-iris.parm"
if [[ -f "${PARM_FILE}" ]]; then
    for entry in "ARMING_CHECK 0" "SCHED_LOOP_RATE 50" "FS_THR_ENABLE 0" "FS_GCS_ENABLE 0" "FS_CRASH_CHECK 0"; do
        pname="${entry%% *}"
        if grep -q "^${pname} " "${PARM_FILE}"; then
            sed -i "s/^${pname} .*/${entry}/" "${PARM_FILE}"
        else
            echo "${entry}" >> "${PARM_FILE}"
        fi
    done
    log "Params SITL vérifiés dans ${PARM_FILE}"
fi

# --- Auto-detect display ---
if [[ -z "${HEADLESS:-}" ]]; then
    if xdpyinfo -display "${DISPLAY:-}" >/dev/null 2>&1; then
        HEADLESS=0
    else
        FOUND_DISPLAY=""
        for sock in /tmp/.X11-unix/X*; do
            d=":${sock##*/tmp/.X11-unix/X}"
            if xdpyinfo -display "$d" >/dev/null 2>&1; then
                warn "DISPLAY invalide, basculement vers $d"
                export DISPLAY="$d"
                FOUND_DISPLAY="$d"
                break
            fi
        done
        if [[ -n "$FOUND_DISPLAY" ]]; then
            HEADLESS=0
        else
            HEADLESS=1
            warn "Aucun display X11 détecté, mode headless forcé."
        fi
    fi
fi

# --- 1. Activate Isaac Sim environment ---
if [[ -f "${ACTIVATE_SCRIPT}" ]]; then
    # shellcheck disable=SC1090
    source "${ACTIVATE_SCRIPT}"
else
    die "Isaac Sim env not found (${ACTIVATE_SCRIPT}). Run ./install_isaac_sim.sh first."
fi

# --- 2. GPU check ---
command -v nvidia-smi &>/dev/null || die "nvidia-smi not found. Check NVIDIA drivers."
log "GPU : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# --- 3. Verify imports ---
python -c "import isaacsim" 2>/dev/null || die "Isaac Sim not importable."
python -c "import pegasus"  2>/dev/null || die "Pegasus not importable."
python -c "from pymavlink import mavutil" 2>/dev/null \
    || die "pymavlink not installed. Run: pip install pymavlink"

# --- 4. Launch AIF simulation (ArduPilot SITL) ---
HEADLESS_FLAG=""
if [[ "${HEADLESS}" == "1" ]]; then
    HEADLESS_FLAG="--headless"
fi
log "Launching AIF exploration (ArduPilot SITL): ${N_DRONES} drone(s), mode $([ "${HEADLESS}" == "1" ] && echo headless || echo GUI)"

export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
export OMNI_KIT_ACCEPT_EULA=YES
export AIF_FACTORY_USD="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"

log "Factory USD forced: ${AIF_FACTORY_USD}"

# shellcheck disable=SC2086
python "${WORKSPACE}/scripts/12_aif_isaac_sim.py" \
    --num-drones "${N_DRONES}" \
    ${HEADLESS_FLAG} \
    ${EXTRA_ARGS}
