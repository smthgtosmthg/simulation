#!/bin/bash
set -e

NS3_VERSION="3.40"
NS3_SRC="$HOME/ns-allinone-${NS3_VERSION}/ns-${NS3_VERSION}"
SIONNA_DIR="$NS3_SRC/contrib/sionna"

echo "=============================================="
echo " [5/5] Installation de NS3-Sionna"
echo "=============================================="

# Vérifications
if [ ! -d "$NS3_SRC" ]; then
    echo "ERREUR : NS-3 non trouvé dans $NS3_SRC"
    echo "Lance d'abord : ./09_install_ns3.sh"
    exit 1
fi

if [ ! -f "$NS3_SRC/ns3" ]; then
    echo "ERREUR : NS-3 semble incomplet (pas de ./ns3)"
    echo "Relance : ./09_install_ns3.sh"
    exit 1
fi

# Étape 1 : Cloner NS3-Sionna dans contrib/
echo ""
echo "[1/4] Clonage de NS3-Sionna dans contrib/sionna..."

mkdir -p "$NS3_SRC/contrib"
cd "$NS3_SRC/contrib"

if [ -d "$SIONNA_DIR" ]; then
    echo "NS3-Sionna existe déjà. Mise à jour..."
    cd "$SIONNA_DIR"
    git pull
else
    git clone https://github.com/tkn-tub/ns3sionna.git ./sionna
fi

# Étape 2 : Recompiler NS-3 avec le module Sionna
echo ""
echo "[2/4] Recompilation de NS-3 avec le module Sionna..."

cd "$NS3_SRC"
./ns3 configure --enable-examples
./ns3 build

echo ""
echo "==> NS-3 + module Sionna compilés avec succès !"

# Étape 3 : Installer le serveur Python Sionna (venv)
echo ""
echo "[3/4] Installation du serveur Python Sionna..."

cd "$SIONNA_DIR/model/ns3sionna"

if [ -d "sionna-venv" ]; then
    echo "Virtual env existe déjà."
else
    echo "Création du virtual env Python..."
    python3 -m venv sionna-venv
fi

# Activer le venv et installer les dépendances
source sionna-venv/bin/activate

echo "Installation des dépendances Python (Sionna, ZMQ, etc.)..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Test des imports..."
python3 tests/test_imports.py && echo "==> Tous les imports OK !" || echo "ATTENTION : certains imports ont échoué"

deactivate

# Étape 4 : Créer les scripts de lancement
echo ""
echo "[4/4] Configuration des variables d'environnement..."

MARKER="# NS3-Sionna Paths"
if ! grep -q "$MARKER" ~/.bashrc; then
    cat >> ~/.bashrc << EOF

$MARKER
export NS3_DIR=$NS3_SRC
export NS3_SIONNA_DIR=$SIONNA_DIR
EOF
    echo "Chemins ajoutés à ~/.bashrc"
else
    echo "Chemins déjà configurés."
fi

echo ""
echo " NS3-Sionna installé avec succès !"
echo ""
echo "  NS-3 : $NS3_SRC"
echo "  Sionna: $SIONNA_DIR"
echo ""
echo "  ┌───────────────────────────────────────────────┐"
echo "  │  Pour lancer le serveur Sionna (Terminal 1) : │"
echo "  │  cd $SIONNA_DIR/model/ns3sionna"
echo "  │  source sionna-venv/bin/activate"
echo "  │  ./run.sh                                     │"
echo "  ├───────────────────────────────────────────────┤"
echo "  │  Pour lancer un exemple NS-3 (Terminal 2) :   │"
echo "  │  cd $NS3_SRC"
echo "  │  ./ns3 run "scratch/drone-wifi-scenario --nDrones=3 --simTime=60"              │"
echo "  └───────────────────────────────────────────────┘"

