#!/bin/bash
set -e

NS3_VERSION="3.40"
NS3_DIR="$HOME/ns-allinone-${NS3_VERSION}"
NS3_SRC="$NS3_DIR/ns-${NS3_VERSION}"

echo "=============================================="
echo " [4/5] Installation de NS-3.${NS3_VERSION}"
echo "=============================================="

# Dépendances NS-3
echo ""
echo "[1/4] Installation des dépendances NS-3..."

sudo apt-get update
sudo apt-get install -y \
    gcc g++ python3 python3-pip python3-dev \
    cmake ninja-build \
    git mercurial \
    qt5-qmake qtbase5-dev \
    libgsl-dev \
    libgtk-3-dev \
    sqlite3 libsqlite3-dev \
    libxml2 libxml2-dev \
    libboost-all-dev \
    libc6-dev \
    pkg-config \
    dpkg-dev

# Dépendances spécifiques pour NS3-Sionna (ZMQ + ProtoBuf)
echo ""
echo "[2/4] Installation des dépendances NS3-Sionna (ZMQ, ProtoBuf)..."

sudo apt-get install -y \
    libzmq5 libzmq5-dev \
    libprotobuf-dev \
    protobuf-compiler \
    python3-venv

echo ""
echo "==> Dépendances installées !"

# Télécharger NS-3
echo ""
echo "[3/4] Téléchargement de ns-allinone-${NS3_VERSION}..."

cd "$HOME"

if [ -d "$NS3_DIR" ]; then
    echo "NS-3 existe déjà dans $NS3_DIR"
    echo "Pour réinstaller : rm -rf $NS3_DIR"
else
    TARBALL="ns-allinone-${NS3_VERSION}.tar.bz2"
    if [ ! -f "$TARBALL" ]; then
        wget "https://www.nsnam.org/releases/${TARBALL}"
    fi
    echo "Extraction..."
    tar xf "$TARBALL"
    rm -f "$TARBALL"
fi

# Compiler NS-3 (sans les modules Sionna pour l'instant)
echo ""
echo "[4/4] Compilation de NS-3.${NS3_VERSION}..."
echo "      (Première compilation ~5-10 min)"

cd "$NS3_SRC"

# Configurer NS-3 avec les exemples activés
./ns3 configure --enable-examples --enable-tests

# Compiler
./ns3 build

echo ""
echo " NS-3.${NS3_VERSION} installé avec succès !"
echo ""

