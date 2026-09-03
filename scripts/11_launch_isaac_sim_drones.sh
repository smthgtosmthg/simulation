#!/usr/bin/env bash

set -euo pipefail

N_DRONES=${1:-3}
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

# --- 1. Environnement Isaac Sim ---
if [[ -f "${ACTIVATE_SCRIPT}" ]]; then
    # shellcheck disable=SC1090
    source "${ACTIVATE_SCRIPT}"
else
    die "Environnement Isaac Sim introuvable (${ACTIVATE_SCRIPT}). Lancez d'abord ./install_isaac_sim.sh"
fi

# --- 2. Vérifier GPU NVIDIA ---
command -v nvidia-smi &>/dev/null || die "nvidia-smi introuvable. Vérifiez les drivers NVIDIA."
log "GPU : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# --- 3. Vérifier les imports ---
python -c "import isaacsim" 2>/dev/null || die "Isaac Sim non importable."
python -c "import pegasus"  2>/dev/null || die "Pegasus non importable."

# --- 4. Lancer la simulation ---
HEADLESS_FLAG=""
if [[ "${HEADLESS}" == "1" ]]; then
    HEADLESS_FLAG="--headless"
fi
log "Lancement : ${N_DRONES} drone(s), mode $([ "${HEADLESS}" == "1" ] && echo headless || echo GUI)"

export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
export OMNI_KIT_ACCEPT_EULA=YES

# shellcheck disable=SC2086
python "${WORKSPACE}/scripts/11_isaac_sim_drones.py" \
    --num-drones "${N_DRONES}" \
    ${HEADLESS_FLAG}
