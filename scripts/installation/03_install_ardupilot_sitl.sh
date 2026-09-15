#!/bin/bash
set -e

echo "=============================================="
echo " [3/5] Installation d'ArduPilot SITL + Gazebo"
echo "=============================================="

# Partie A : Cloner et configurer ArduPilot
ARDUPILOT_DIR="$HOME/ardupilot"

if [ -d "$ARDUPILOT_DIR" ]; then
    echo "ArduPilot existe déjà dans $ARDUPILOT_DIR"
    echo "Pour réinstaller, supprime le dossier : rm -rf $ARDUPILOT_DIR"
    echo "Mise à jour du repo existant..."
    cd "$ARDUPILOT_DIR"
    git pull
    git submodule update --init --recursive
else
    echo "Clonage d'ArduPilot..."
    cd "$HOME"
    git clone --recursive https://github.com/ArduPilot/ardupilot.git
    cd "$ARDUPILOT_DIR"
fi

# --- Lancer le script d'installation officiel d'ArduPilot ---
echo ""
echo "Exécution du script de setup ArduPilot (install-prereqs-ubuntu.sh)..."
cd "$ARDUPILOT_DIR"
Tools/environment_install/install-prereqs-ubuntu.sh -y

# Recharger le profil pour avoir les paths
. ~/.profile 2>/dev/null || true

# --- Compiler ArduPilot pour SITL (ArduCopter) ---
echo ""
echo "Compilation d'ArduPilot SITL — ArduCopter (première compilation, ~5-10min)..."
cd "$ARDUPILOT_DIR"
./waf configure --board sitl
./waf copter

echo ""
echo "==> ArduPilot SITL (ArduCopter) compilé avec succès !"

# Partie B : Installer le plugin ardupilot_gazebo
echo ""
echo "Installation du plugin ardupilot_gazebo pour Gazebo Harmonic..."

PLUGIN_DIR="$HOME/ardupilot_gazebo"

if [ -d "$PLUGIN_DIR" ]; then
    echo "ardupilot_gazebo existe déjà dans $PLUGIN_DIR"
    echo "Mise à jour..."
    cd "$PLUGIN_DIR"
    git pull
else
    cd "$HOME"
    git clone https://github.com/ArduPilot/ardupilot_gazebo.git
    cd "$PLUGIN_DIR"
fi

# Compiler le plugin
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j$(nproc)

echo ""
echo "==> Plugin ardupilot_gazebo compilé avec succès !"

# Partie C : Configurer les variables d'environnement
echo ""
echo "Configuration des variables d'environnement..."

# Ajouter au .bashrc si pas déjà présent
MARKER="# ArduPilot Gazebo Plugin"
if ! grep -q "$MARKER" ~/.bashrc; then
    cat >> ~/.bashrc << 'EOF'

# ArduPilot Gazebo Plugin
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:$GZ_SIM_RESOURCE_PATH
EOF
    echo "Variables ajoutées à ~/.bashrc"
else
    echo "Variables déjà configurées dans ~/.bashrc"
fi

# Charger pour la session courante
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/ardupilot_gazebo/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
export GZ_SIM_RESOURCE_PATH=$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:$GZ_SIM_RESOURCE_PATH

echo ""
echo " ArduPilot SITL + Plugin Gazebo installés !"
echo ""
