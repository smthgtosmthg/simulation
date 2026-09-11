#!/bin/bash
# La référence de Pore et al. (Symmetry 2026) : un vol par cas, même protocole que le système.
#   bash campagne.sh
# Rien d'autre ne doit tourner sur la machine pendant les vols.
set -u
ICI="$(cd "$(dirname "$0")" && pwd)"
SQ="$ICI/../.."
PY=~/isaac5_env/bin/python
export DISPLAY=:1 PYTHONUNBUFFERED=1
vole() {
  local nom="$1"; shift
  local sortie="$ICI/pore_$nom"
  if [ -f "$sortie/mission.json" ]; then echo "=== $nom deja fait"; return; fi
  echo "=== pore $nom : depart $(date +%H:%M)"
  ( cd "$SQ" && timeout -s KILL 7200 $PY pore_mission.py --drones 3 --budget 600 --video "$@" --sortie "$sortie" \
      2>&1 | grep -vE 'gpu.foundation|PNG|ros2|Duplicate input|enough inputs' )
  ( cd "$SQ" && $PY experiments/11_mission/video.py --dossier "$sortie" --sans-images )
  echo "=== pore $nom : fin $(date +%H:%M)"
}
vole nominal  --seed 9033
vole panne    --seed 9033 --panne 1:200
vole 9019     --seed 9019
vole obstacle --seed 9033 --obstacle=-4.96,4.0,200
echo "=== CAMPAGNE PORE FINIE $(date +%H:%M)"
