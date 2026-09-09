#!/usr/bin/env bash
# Campagne de l'étape 5 : les missions complètes à trois drones, puis leur jugement.
#   bash campagne.sh nominale     3 drones, entrepôt 9033, 600 s simulées
#   bash campagne.sh panne        idem, le drone 1 tombe en panne à 200 s
#   bash campagne.sh autre        3 drones, entrepôt 9019
#   bash campagne.sh guide        le banc hors ligne du guide sur les instantanés des trois missions
#   bash campagne.sh guidee       3 drones, entrepôt 9019, le guide branché (λ = 1)
#   bash campagne.sh juge NOM     le jugement seul, sans simulateur
set -u
PY=/home/djihene_guitoun/isaac5_env/bin/python
ICI="$(cd "$(dirname "$0")" && pwd)"
export DISPLAY=:1 PYTHONUNBUFFERED=1 YOLO_AUTOINSTALL=false
cd "$ICI" || exit 1

vole() {
  local nom="$1"; shift
  pgrep -x arducopter >/dev/null && { echo "ATTENTION : un SITL tourne encore"; exit 1; }
  echo "=========== mission $nom ==========="
  timeout -s KILL 10800 "$PY" ../../mission.py --sortie "$ICI/$nom" "$@" 2>&1 | grep -vE "gpu.foundation|PNG|ros2"
  echo "--- mission $nom : code de sortie ${PIPESTATUS[0]}"
  sleep 5
  "$PY" analyse.py --dossier "$ICI/$nom"
}

case "${1:-}" in
  nominale) vole nominale --seed 9033 --drones 3 --budget 600 --detecteur auto ;;
  panne)    vole panne    --seed 9033 --drones 3 --budget 600 --detecteur auto --panne 1:200 ;;
  autre)    vole autre    --seed 9019 --drones 3 --budget 600 --detecteur auto ;;
  guidee)   vole guidee   --seed 9019 --drones 3 --budget 600 --detecteur auto --guide smolvlm --lam 1.0 ;;
  guide)    "$PY" ../12_guide/banc.py --missions "$ICI/nominale" "$ICI/panne" "$ICI/autre" --modeles smolvlm smolvlm-2b ;;
  juge)     "$PY" analyse.py --dossier "$ICI/${2:?nom}" ;;
  *) echo "phase inconnue : nominale | panne | autre | juge NOM"; exit 1 ;;
esac
echo "PHASE ${1} FINIE"
