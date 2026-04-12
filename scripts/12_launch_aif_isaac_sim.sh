#!/usr/bin/env bash

set -euo pipefail

N_DRONES=${1:-3}
shift 2>/dev/null || true   # consume $1 so "$@" holds remaining flags
EXTRA_ARGS="$@"
VENV_DIR="${HOME}/isaac_sim_env"
ACTIVATE_SCRIPT="${VENV_DIR}/activate_isaac.sh"
WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"

log()  { echo "[INFO] $*"; }
warn() { echo "[WARN] $*"; }
die()  { echo "[ERROR] $*"; exit 1; }

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

# --- 4. Launch AIF simulation ---
HEADLESS_FLAG=""
if [[ "${HEADLESS}" == "1" ]]; then
    HEADLESS_FLAG="--headless"
fi
log "Launching AIF exploration: ${N_DRONES} drone(s), mode $([ "${HEADLESS}" == "1" ] && echo headless || echo GUI)"

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
