#!/bin/bash
set -e

echo "=============================================="
echo " [1/5] Installation des dépendances système"
echo "=============================================="

sudo apt-get update

# Outils de build essentiels
sudo apt-get install -y \
    git \
    wget \
    curl \
    cmake \
    build-essential \
    ninja-build \
    pkg-config \
    astyle \
    dmidecode \
    libxml2-utils \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev

# Python 3 et bibliothèques nécessaires pour ArduPilot
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-jinja2 \
    python3-jsonschema \
    python3-numpy \
    python3-empy \
    python3-toml \
    python3-packaging \
    python3-setuptools \
    python3-lxml \
    python3-matplotlib \
    python3-serial \
    python3-pexpect \
    python3-future \
    python3-tz \
    python3-opencv \
    python3-wxgtk4.0

# GStreamer (pour le streaming vidéo des drones)
sudo apt-get install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly

# Outils Python supplémentaires via pip (installed in a virtualenv to avoid
# modifying the system-managed Python environment which may be protected
# by the OS / PEP 668).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtualenv at $VENV_DIR"
    sudo apt-get install -y python3-venv || true
    python3 -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install "numpy<2" pymavlink MAVProxy pexpect dronekit dronekit-sitl
deactivate

echo ""
echo "==> Dépendances système installées avec succès !"
echo "==> Passe au script suivant : 02_install_gazebo.sh"
