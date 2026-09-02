#!/usr/bin/env bash
# Campagne complète de l'étape 2. Chaque passe écrit ses propres fichiers : un échec ne fait
# perdre qu'elle. Entre deux passes, on vérifie qu'aucun pilote automatique ne traîne.
#
#   bash campagne.sh 2>&1 | tee /tmp/campagne.log

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
passe "banc optique (2000 poses)"   5400 banc.py --mode optique   --poses 2000
passe "fausses alertes (300 poses)" 1800 banc.py --mode sans-qr   --poses 300
passe "second entrepot (400 poses)" 2400 banc.py --mode optique   --poses 400 --seed 9019 --nom 9019
passe "vol : poses tenues"          4800 banc.py --mode vol
passe "traversees a 4 vitesses"     4800 banc.py --mode traversee

echo "=========== analyse ==========="
"$PY" analyse.py
echo "CAMPAGNE FINIE"
