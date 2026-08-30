#!/usr/bin/env bash
# Balayage du débit de rendu.
# Un processus Isaac par configuration : le simulateur ne supporte pas de
# reconstruire une scène en cours de route.
#
#   bash experiments/03_render_bench/run_all.sh
#
# Compter ~40 s de démarrage d'Isaac par configuration.
# `timeout -s KILL` est indispensable : Isaac ignore le signal d'arrêt normal.

set -u
cd "$(dirname "$0")/../.."
PY="$HOME/isaac5_env/bin/python"
SCRIPT="experiments/03_render_bench/render_bench.py"
CSV="experiments/03_render_bench/resultats.csv"
KIT="--kit_args=--/rtx/verifyDriverVersion/enabled=false"

rm -f "$CSV"

# (envs, caméras par env, résolution)
CONFIGS="
1 1 64
1 2 64
3 2 64
8 2 64
16 2 64
32 2 64
8 2 128
"

echo "$CONFIGS" | while read -r envs cams res; do
  [ -z "$envs" ] && continue
  echo "--- ${envs} envs x ${cams} cams @ ${res}px ---"
  PYTHONUNBUFFERED=1 timeout -s KILL 300 "$PY" "$SCRIPT" \
      --envs "$envs" --cams "$cams" --res "$res" --steps 60 "$KIT" 2>&1 \
    | grep --line-buffered -E "^\[VÉRIF\]|^\[RESULTAT\]|Traceback|Error:" \
    || echo "  BLOQUÉ ou échec (tué à 300 s)"
done

echo
echo "=== Récapitulatif ==="
column -s, -t < "$CSV" 2>/dev/null || cat "$CSV"
