#!/usr/bin/env bash

set -euo pipefail

ISAAC_SIM_VERSION="4.5.0"
PYTHON_VERSION="3.10"
VENV_DIR="${HOME}/isaac_sim_env"
PEGASUS_REPO="https://github.com/PegasusSimulator/PegasusSimulator.git"
PEGASUS_BRANCH="main"
PEGASUS_DIR="${HOME}/PegasusSimulator"
MIN_DRIVER_VERSION="535.129"
DRY_RUN=false

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

die() { log_error "$@"; exit 1; }

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--help]"
            echo "  --dry-run   Vérifie les prérequis sans installer"
            exit 0 ;;
        *) die "Argument inconnu: $arg" ;;
    esac
done

# ─── Étape 1 : Vérification du système ─────────────────────────────────────
check_prerequisites() {
    log_info "═══ Vérification des prérequis ═══"

    # OS
    if [[ -f /etc/os-release ]]; then
        source /etc/os-release
        log_info "OS détecté : ${PRETTY_NAME}"
    else
        log_warn "Impossible de détecter l'OS"
    fi

    # GPU NVIDIA
    if ! command -v nvidia-smi &>/dev/null; then
        die "nvidia-smi introuvable. Installez les drivers NVIDIA >= ${MIN_DRIVER_VERSION}."
    fi

    local driver_version
    driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)
    # Comparer version complète (ex: 535.129 vs 535.32)
    local driver_ok
    driver_ok=$(python3 -c "
from packaging.version import Version
try:
    ok = Version('${driver_version}') >= Version('${MIN_DRIVER_VERSION}')
except: ok = float('${driver_version}'.split('.')[0]) >= float('${MIN_DRIVER_VERSION}'.split('.')[0])
print('yes' if ok else 'no')
" 2>/dev/null || echo "yes")
    if [[ "${driver_ok}" == "no" ]]; then
        die "Driver NVIDIA ${driver_version} trop ancien (minimum: ${MIN_DRIVER_VERSION}). Mettez à jour : sudo apt install nvidia-driver-535"
    fi
    log_ok "Driver NVIDIA : ${driver_version} (>= ${MIN_DRIVER_VERSION})"

    # GPU info
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    local gpu_mem
    gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    log_ok "GPU : ${gpu_name} (${gpu_mem})"

    # Python
    if command -v "python${PYTHON_VERSION}" &>/dev/null; then
        log_ok "Python ${PYTHON_VERSION} trouvé : $(command -v python${PYTHON_VERSION})"
    else
        log_warn "Python ${PYTHON_VERSION} non trouvé. Il sera installé."
    fi

    # Espace disque (Isaac Sim ~15 GB)
    local free_gb
    free_gb=$(df -BG --output=avail "${HOME}" | tail -1 | tr -dc '0-9')
    if (( free_gb < 20 )); then
        log_warn "Espace disque faible : ${free_gb} Go libres (20 Go recommandé)."
    else
        log_ok "Espace disque : ${free_gb} Go libres"
    fi

    log_info "═══ Prérequis OK ═══"
}

# ─── Étape 2 : Installation des dépendances système ────────────────────────
install_system_deps() {
    log_info "═══ Installation des dépendances système ═══"

    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        build-essential \
        cmake \
        git \
        curl \
        wget \
        "python${PYTHON_VERSION}" \
        "python${PYTHON_VERSION}-venv" \
        "python${PYTHON_VERSION}-dev" \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        > /dev/null 2>&1

    log_ok "Dépendances système installées"
}

# ─── Étape 3 : Création de l'environnement virtuel ─────────────────────────
setup_venv() {
    log_info "═══ Création de l'environnement Python (${VENV_DIR}) ═══"

    if [[ -d "${VENV_DIR}" ]]; then
        log_warn "Environnement existant détecté. Réutilisation."
    else
        "python${PYTHON_VERSION}" -m venv "${VENV_DIR}"
        log_ok "Environnement virtuel créé"
    fi

    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    pip install --upgrade pip setuptools wheel -q
    log_ok "pip, setuptools, wheel mis à jour"
}

# ─── Étape 4 : Installation d'Isaac Sim via pip ────────────────────────────
install_isaac_sim() {
    log_info "═══ Installation d'Isaac Sim ${ISAAC_SIM_VERSION} via pip ═══"
    log_info "Cette étape peut prendre 10-20 minutes selon la connexion..."

    # Accepter le EULA NVIDIA Omniverse de manière non-interactive
    export OMNI_KIT_ACCEPT_EULA=YES

    # NVIDIA héberge certains packages sur son propre index PyPI
    local NVIDIA_INDEX="https://pypi.nvidia.com"

    # [all] installe tous les sous-packages (core, app, sensor, robot…)
    pip install \
        "isaacsim[all]==${ISAAC_SIM_VERSION}" \
        --extra-index-url "${NVIDIA_INDEX}" \
        -q

    # Extensions en cache (évite le téléchargement au runtime)
    pip install \
        "isaacsim[extscache]==${ISAAC_SIM_VERSION}" \
        --extra-index-url "${NVIDIA_INDEX}" \
        -q

    log_ok "Isaac Sim ${ISAAC_SIM_VERSION} installé"
}

# ─── Étape 5 : Installation de Pegasus Simulator ───────────────────────────
install_pegasus() {
    log_info "═══ Installation de Pegasus Simulator (drones) ═══"

    # Accepter le EULA avant tout import (sinon prompt interactif bloquant)
    export OMNI_KIT_ACCEPT_EULA=YES

    # Pegasus nécessite ISAACSIM_PATH pour trouver Isaac Sim
    local isaacsim_path
    isaacsim_path=$(python -c "import importlib.util; print(importlib.util.find_spec('isaacsim').submodule_search_locations[0])" 2>/dev/null)
    # Fallback : localiser via pip show
    if [[ -z "${isaacsim_path}" ]]; then
        isaacsim_path=$(pip show isaacsim 2>/dev/null | grep '^Location:' | awk '{print $2}')/isaacsim
    fi
    if [[ -z "${isaacsim_path}" || ! -d "${isaacsim_path}" ]]; then
        die "Impossible de localiser le package isaacsim dans le venv."
    fi
    export ISAACSIM_PATH="${isaacsim_path}"
    export ISAACSIM_PYTHON="$(command -v python)"
    log_ok "ISAACSIM_PATH=${ISAACSIM_PATH}"

    if [[ -d "${PEGASUS_DIR}" ]]; then
        log_info "Pegasus déjà cloné, mise à jour..."
        pushd "${PEGASUS_DIR}" > /dev/null
        git pull --quiet
        popd > /dev/null
    else
        git clone --branch "${PEGASUS_BRANCH}" --depth 1 \
            "${PEGASUS_REPO}" "${PEGASUS_DIR}" --quiet
    fi

    # Installation en mode éditable pour pouvoir modifier facilement
    pip install -e "${PEGASUS_DIR}/extensions/pegasus.simulator" -q

    log_ok "Pegasus Simulator installé"
}

# ─── Étape 6 : Dépendances Python supplémentaires ──────────────────────────
install_python_extras() {
    log_info "═══ Installation des dépendances Python supplémentaires ═══"

    pip install -q \
        numpy \
        scipy \
        pyyaml \
        matplotlib

    log_ok "Dépendances Python supplémentaires installées"
}

# ─── Étape 7 : Vérification post-installation ──────────────────────────────
verify_installation() {
    log_info "═══ Vérification de l'installation ═══"

    local all_ok=true

    # Vérif Isaac Sim
    if python -c "import isaacsim; print('Isaac Sim', isaacsim.__path__[0])" 2>/dev/null; then
        log_ok "Isaac Sim importable"
    else
        log_error "Isaac Sim import échoué"
        all_ok=false
    fi

    # Vérif Pegasus
    if python -c "import pegasus; print('Pegasus Simulator OK')" 2>/dev/null; then
        log_ok "Pegasus Simulator importable"
    else
        log_error "Pegasus Simulator import échoué"
        all_ok=false
    fi

    if $all_ok; then
        log_ok "═══ Installation complète et vérifiée ═══"
    else
        log_warn "═══ Installation terminée avec des avertissements ═══"
    fi
}

# ─── Étape 8 : Génération du script d'activation ───────────────────────────
generate_activate_helper() {
    local helper="${VENV_DIR}/activate_isaac.sh"

    # Récupère le chemin Isaac Sim pour le persister dans le helper
    export OMNI_KIT_ACCEPT_EULA=YES
    local isaacsim_path
    isaacsim_path=$(python -c "import importlib.util; print(importlib.util.find_spec('isaacsim').submodule_search_locations[0])" 2>/dev/null || echo "")
    if [[ -z "${isaacsim_path}" ]]; then
        isaacsim_path=$(pip show isaacsim 2>/dev/null | grep '^Location:' | awk '{print $2}')/isaacsim
    fi

    cat > "${helper}" << ACTIVATE_EOF
#!/usr/bin/env bash
# Source ce fichier pour activer l'environnement Isaac Sim :
#   source ~/isaac_sim_env/activate_isaac.sh
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
source "\${SCRIPT_DIR}/bin/activate"
export OMNI_KIT_ACCEPT_EULA=YES
export ISAACSIM_PATH="${isaacsim_path}"
export ISAACSIM_PYTHON="\$(command -v python)"
echo "✔ Environnement Isaac Sim activé (\$(python --version))"
echo "  ISAACSIM_PATH=\${ISAACSIM_PATH}"
ACTIVATE_EOF
    chmod +x "${helper}"
    log_ok "Helper d'activation : source ${helper}"
}

# ─── Résumé final ───────────────────────────────────────────────────────────
print_summary() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║          INSTALLATION TERMINÉE AVEC SUCCÈS              ║${NC}"
    echo -e "${GREEN}╠══════════════════════════════════════════════════════════╣${NC}"
    echo -e "${GREEN}║${NC}  Isaac Sim     : ${ISAAC_SIM_VERSION}                              ${GREEN}║${NC}"
    echo -e "${GREEN}║${NC}  Pegasus       : ${PEGASUS_DIR}            ${GREEN}║${NC}"
    echo -e "${GREEN}║${NC}  Env. Python   : ${VENV_DIR}                  ${GREEN}║${NC}"
    echo -e "${GREEN}╠══════════════════════════════════════════════════════════╣${NC}"
    echo -e "${GREEN}║${NC}  Activation :                                          ${GREEN}║${NC}"
    echo -e "${GREEN}║${NC}    source ${VENV_DIR}/activate_isaac.sh    ${GREEN}║${NC}"
    echo -e "${GREEN}║${NC}                                                        ${GREEN}║${NC}"
    echo -e "${GREEN}║${NC}  Lancer les drones :                                   ${GREEN}║${NC}"
    echo -e "${GREEN}║${NC}    python launch_drones.py --num-drones 4              ${GREEN}║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# ─── Main ───────────────────────────────────────────────────────────────────
main() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║       Isaac Sim + Pegasus Simulator — Installation      ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""

    check_prerequisites

    if $DRY_RUN; then
        log_info "Mode --dry-run : arrêt après vérification des prérequis."
        exit 0
    fi

    install_system_deps
    setup_venv
    install_isaac_sim
    install_pegasus
    install_python_extras
    verify_installation
    generate_activate_helper
    print_summary
}

main "$@"
