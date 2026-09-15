#!/bin/bash
# L'évaluation finale : un vol par cas, avec vidéo, jugement et assemblage vidéo.
#   bash evaluation.sh             les 4 cas du système : géométrie seule, λ = 0
# Rien d'autre ne doit tourner sur la machine pendant les vols.
set -u
ICI="$(cd "$(dirname "$0")" && pwd)"
SQ="$ICI/../.."
PY=~/isaac5_env/bin/python
export DISPLAY=:1 PYTHONUNBUFFERED=1
OPT="--lam 0.0"   # géométrie seule : le modèle n'est pas consulté
TAG="eval"
vole() {  # nom, arguments propres au cas
  local nom="$1"; shift
  local sortie="$ICI/${TAG}_$nom"
  if [ -f "$sortie/mission.json" ]; then echo "=== $nom deja fait"; return; fi
  echo "=== $TAG $nom : depart $(date +%H:%M)"
  ( cd "$SQ" && timeout -s KILL 7200 $PY mission.py --drones 3 --budget 600 --detecteur auto --video $OPT "$@" --sortie "$sortie" \
      2>&1 | grep -vE 'gpu.foundation|PNG|ros2|Duplicate input|enough inputs' )
  ( cd "$SQ" && $PY experiments/11_mission/analyse.py --dossier "$sortie" )
  ( cd "$SQ" && $PY experiments/11_mission/video.py --dossier "$sortie" --sans-images )   # le disque est petit
  echo "=== $TAG $nom : fin $(date +%H:%M)"
}
vole nominal  --seed 9033
vole panne    --seed 9033 --panne 1:200
vole 9019     --seed 9019
vole obstacle --seed 9033 --obstacle=-4.96,4.0,200
echo "=== EVALUATION FINIE $(date +%H:%M)"
