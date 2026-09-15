#!/bin/bash
set -e

echo "=============================================="
echo " [2/5] Installation de Gazebo Harmonic"
echo "=============================================="

# --- Ajouter le dépôt OSRF Gazebo ---
sudo apt-get install -y lsb-release gnupg

# Clé GPG
sudo curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
    -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg

# Dépôt
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | \
    sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

sudo apt-get update

# --- Installer Gazebo Harmonic ---
sudo apt-get install -y gz-harmonic

echo ""
echo "==> Vérification de l'installation de Gazebo :"
gz sim --version 2>/dev/null && echo "Gazebo Harmonic installé avec succès !" || echo "ATTENTION : gz sim non trouvé, vérifier l'installation."

