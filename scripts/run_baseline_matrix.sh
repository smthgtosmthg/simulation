#!/usr/bin/env bash
# Matrice d'évaluation des baselines (conception §7) — split test, gate nominal, seeds fixes.
# Chaque cellule écrit son JSON dans results/baselines/. Relançable cellule par cellule.
#   bash scripts/run_baseline_matrix.sh            # aif + serpentine, toutes les cellules
#   bash scripts/run_baseline_matrix.sh aif        # une seule politique
set -euo pipefail
cd "$(dirname "$0")/.."

POLICIES=(${1:-aif serpentine})
RUNNER="rl_inventory/swarmscan_map/baselines/run_baseline.py"
COMMON="--headless --num_envs 16 --rounds 2 --seed 1000"

for pol in "${POLICIES[@]}"; do
    bash rl_inventory/launch.sh "$RUNNER" $COMMON --policy "$pol" --scenario nominal
    for t in 0.25 0.5 0.75; do
        bash rl_inventory/launch.sh "$RUNNER" $COMMON --policy "$pol" --scenario failure1 --failure_t "$t" \
            --out "results/baselines/${pol}_failure1_t${t}_team3_seed1000.json"
    done
    bash rl_inventory/launch.sh "$RUNNER" $COMMON --policy "$pol" --scenario failure2 --failure_t 0.5
    bash rl_inventory/launch.sh "$RUNNER" $COMMON --policy "$pol" --scenario noise
    bash rl_inventory/launch.sh "$RUNNER" $COMMON --policy "$pol" --scenario noise_hard
    bash rl_inventory/launch.sh "$RUNNER" $COMMON --policy "$pol" --scenario qr_minus
    bash rl_inventory/launch.sh "$RUNNER" $COMMON --policy "$pol" --scenario qr_plus
    bash rl_inventory/launch.sh "$RUNNER" $COMMON --policy "$pol" --scenario nominal --team 2
    bash rl_inventory/launch.sh "$RUNNER" $COMMON --policy "$pol" --scenario nominal --team 4
done
echo "MATRICE TERMINÉE → results/baselines/"
