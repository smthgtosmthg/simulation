#!/usr/bin/env bash
# Campagne de l'étape 7. Chaque phase se lance seule :
#   bash campagne.sh rendu      images + cadres des 6 entrepôts d'entraînement et des 2 scellés
#   bash campagne.sh controle   planches à regarder + confrontation à la projection de l'étape 2
#   bash campagne.sh entraine   les variantes (n1024, n640 ; s1024 seulement si demandé)
#   bash campagne.sh banc       le jugement, sans simulateur
# Jamais deux phases GPU en même temps : le rendu et l'entraînement se partagent 8 Go.

set -u
PY=/home/djihene_guitoun/isaac5_env/bin/python
ICI="$(cd "$(dirname "$0")" && pwd)"
export DISPLAY=:1 PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false
cd "$ICI" || exit 1

rendu_un() {
  local seed="$1"; shift
  echo "=========== rendu entrepot $seed ==========="
  timeout -s KILL 3600 "$PY" rendu.py --seed "$seed" "$@" 2>&1 | grep -vE "gpu.foundation|PNG|ros2|omni.kit.app._impl\] \[py stderr\]: $" 
  echo "--- entrepot $seed : code de sortie ${PIPESTATUS[0]}"
  sleep 3
}

case "${1:-}" in
  rendu)
    for s in 0 1 2 3 4 5; do rendu_un "$s" --images 500; done
    rendu_un 9033 --images 400 --relabel optique,sans_qr,vol,traversee
    rendu_un 9019 --images 400 --relabel 9019
    ;;
  controle)
    "$PY" controle.py --planche jeu/rendu_0 jeu/rendu_3 jeu/rendu_9033 jeu/rendu_9019 jeu/etape2_optique jeu/etape2_traversee --par-jeu 4
    "$PY" controle.py --verifie
    ;;
  entraine)
    "$PY" entraine.py --variantes "${2:-n1024,n640}"
    ;;
  banc)
    "$PY" banc.py
    ;;
  vol)
    # l'œil appris en mission : même patrouille que l'étape 4, réduite à une étagère et deux
    # allées, jugée par l'analyse de l'étape 4 (pistes sur un vrai panneau, coûts, sécurité)
    pgrep -x arducopter >/dev/null && { echo "ATTENTION : un SITL tourne encore"; exit 1; }
    timeout -s KILL 7200 "$PY" ../09_carte/banc.py --mode vol --etages 1 --allees 2 --detecteur auto --sortie "$ICI/vol" 2>&1 | grep -vE "gpu.foundation|PNG|ros2"
    echo "--- vol : code de sortie ${PIPESTATUS[0]}"
    "$PY" ../09_carte/analyse.py --dossier "$ICI/vol"
    ;;
  *)
    echo "phase inconnue : rendu | controle | entraine | banc"; exit 1;;
esac
echo "PHASE ${1} FINIE"
