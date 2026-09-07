#!/usr/bin/env bash
# Campagne de l'étape 4. Les tests puis les contrôles passent avant la patrouille : si le
# lidar ne dit pas la vérité, rien ne sert de voler.
#
#   bash campagne.sh 2>&1 | tee /tmp/campagne_carte.log

set -u
PY=/home/djihene_guitoun/isaac5_env/bin/python
ICI="$(cd "$(dirname "$0")" && pwd)"
export DISPLAY=:1 PYTHONUNBUFFERED=1

passe() {
  local nom="$1" duree="$2"; shift 2
  echo "=========== $nom ==========="
  pgrep -x arducopter >/dev/null && { echo "ATTENTION : un SITL tourne encore"; return 1; }
  timeout -s KILL "$duree" "$PY" "$@"
  echo "--- $nom : code de sortie $?"
  sleep 5
}

cd "$ICI" || exit 1
"$PY" -m pytest "$ICI/../../tests" -q || exit 1
passe "controles de la chaine lidar"  1800  banc.py --mode verifie
passe "patrouille, trois etageres"   10800  banc.py --mode vol --etages 0,1,2
echo "=========== analyse ==========="
"$PY" analyse.py
echo "CAMPAGNE FINIE"
