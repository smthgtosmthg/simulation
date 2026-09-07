#!/usr/bin/env bash
# Campagne complète de l'étape 3. Chaque passe écrit ses propres fichiers : un échec ne fait
# perdre qu'elle. Entre deux passes, on vérifie qu'aucun pilote automatique ne traîne.
#
#   bash campagne.sh 2>&1 | tee /tmp/campagne_controle.log

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
passe "freinage : trois lois"        3600  banc.py --mode freinage
passe "cent poses"                  14400  banc.py --mode poses --poses 100
passe "essaim : trois drones"        5400  banc.py --mode essaim

echo "=========== analyse ==========="
"$PY" analyse.py
echo "CAMPAGNE FINIE"
