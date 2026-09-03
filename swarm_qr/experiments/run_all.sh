#!/usr/bin/env bash
# Lance les quatre tests de l'étape 1.
#   bash swarm_qr/experiments/run_all.sh
# Un processus Isaac par mesure : le simulateur ne sait pas reconstruire une scène en cours de
# route. `timeout -s KILL` est indispensable, Isaac ignore le signal d'arrêt normal.

set -u
cd "$(dirname "$0")/../.."
PY="$HOME/isaac5_env/bin/python"
E="swarm_qr/experiments"
KIT="--kit_args=--/rtx/verifyDriverVersion/enabled=false"
run() { PYTHONUNBUFFERED=1 timeout -s KILL 1200 "$PY" "$@" "$KIT" 2>&1 \
        | grep --line-buffered -E "^\[RESULTAT\]|^ *[0-9.]+ m :|^QR vise|^graine |^passage |^video |Traceback|Error:" \
        || echo "  echec ou temps depasse"; }

echo "=== Test 1 : reproductibilite ==="
run $E/01_reproductibilite/run.py --seed 7 --pass A
run $E/01_reproductibilite/run.py --seed 7 --pass B
"$PY" $E/01_reproductibilite/run.py --compare

echo; echo "=== Test 2 : variation ==="
for s in 1 2 3 4 5 6; do run $E/02_variation/run.py --seed $s; done
"$PY" $E/02_variation/run.py --board

echo; echo "=== Test 3 : images et video ==="
run $E/03_images_qr/run.py --seed 7

echo; echo "=== Test 4 : debit ==="
rm -f $E/04_debit/mesures.json
for hz in 1 5 15 60; do run $E/04_debit/run.py --hz $hz --steps 240; done
"$PY" $E/04_debit/run.py --plot
