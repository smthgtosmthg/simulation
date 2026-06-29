#!/usr/bin/env bash
# Lanceur Isaac Sim 5.1 (reprend la méthode éprouvée de scripts/12_launch_aif_isaac_sim.sh).
#   GUI       : bash launch.sh <script.py> [args]
#   headless  : bash launch.sh <script.py> --headless [args]
set -euo pipefail

VENV="${HOME}/isaac5_env"
SCRIPT="${1:?usage: bash launch.sh <script.py> [args...]}"
shift || true

export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
export OMNI_KIT_ACCEPT_EULA=YES
export AIF_FACTORY_USD="${AIF_FACTORY_USD:-http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd}"

if [[ "$*" != *--headless* ]]; then
    export XDG_RUNTIME_DIR="/run/user/$(id -u)"
    for xa in "${XDG_RUNTIME_DIR}/gdm/Xauthority" "${HOME}/.Xauthority"; do
        [[ -r "$xa" ]] && { export XAUTHORITY="$xa"; break; }
    done
    export DISPLAY="${ISAAC_DISPLAY:-:1}"
    echo "[launch] GUI DISPLAY=${DISPLAY}  XAUTHORITY=${XAUTHORITY:-?}"
    xdpyinfo -display "${DISPLAY}" >/dev/null 2>&1 && echo "[launch] display ${DISPLAY} OK" || echo "[launch] ATTENTION: ${DISPLAY} ne répond pas"
fi

source "${VENV}/bin/activate"
exec python "${SCRIPT}" --kit_args="--/rtx/verifyDriverVersion/enabled=false" "$@"
