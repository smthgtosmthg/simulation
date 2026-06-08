#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Matrice complète des runs AIF — Plan d'expérimentation pour PFE.
#
# Couvre 4 axes d'évaluation :
#   1. AIF vs Heuristique (planificateur)
#   2. Centralisé vs Distribué (architecture)
#   3. Réseau (WiFi vs 5G, latence cloud chargé)
#   4. Résilience aux stresseurs (kill drone, cut cloud, cut links, obstacle)
#
# Chaque run produit `logs/runs/run_YYYYMMDD_HHMMSS_<tag>/` avec :
#   - config.json, history.json, state_final.json
#   - 10 PNG (coverage, entropy, innovation, discovery_rate,
#     coverage_known_vs_global, decisions_per_min, resilience_phases,
#     belief_map_final, trajectories, ns3_latencies)
#
# Override via env :
#   N_DRONES=3 MAX_STEPS=80 ./run_all_experiments.sh
#   ONLY="aif_cent_baseline,aif_dist_baseline" ./run_all_experiments.sh
#   HEADLESS=1 ./run_all_experiments.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCH="$SCRIPT_DIR/12_launch_aif_isaac_sim.sh"
RUNS_DIR="$WORKSPACE/logs/runs"
mkdir -p "$RUNS_DIR"

N_DRONES=${N_DRONES:-3}
MAX_STEPS=${MAX_STEPS:-80}
ONLY="${ONLY:-}"

run_one() {
    local TAG="$1"; shift
    if [[ -n "$ONLY" ]] && ! [[ ",$ONLY," == *",$TAG,"* ]]; then
        echo "  [SKIP] $TAG (not in ONLY=$ONLY)"
        return 0
    fi
    echo
    echo "════════════════════════════════════════════════════════════════"
    echo "  RUN  : $TAG"
    echo "  Time : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Flags: $*"
    echo "════════════════════════════════════════════════════════════════"
    "$LAUNCH" "$N_DRONES" --max-steps "$MAX_STEPS" \
        --run-tag "$TAG" "$@" || {
        echo "  [WARN] run $TAG returned non-zero; continuing."
    }
}

# ════════════════════════════════════════════════════════════════════
# Baselines (4 runs)
# ════════════════════════════════════════════════════════════════════
#run_one aif_cent_baseline       --planner aif       --arch centralized --ns3 wifi
#run_one aif_dist_baseline       --planner aif       --arch distributed --ns3 wifi
#run_one heur_cent_baseline      --planner heuristic --arch centralized --ns3 wifi
#run_one heur_dist_baseline      --planner heuristic --arch distributed --ns3 wifi

# ════════════════════════════════════════════════════════════════════
# Charge réseau (latence cloud × 3 → 2 steps de retard)
# ════════════════════════════════════════════════════════════════════
#run_one aif_cent_cloud_loaded   --planner aif       --arch centralized --ns3 wifi \
#                                --cloud-round-trip-ms 1500

# ════════════════════════════════════════════════════════════════════
# Résilience — stresseurs isolés
# ════════════════════════════════════════════════════════════════════
# Kill drone
#run_one aif_cent_kill_d0_s20    --planner aif       --arch centralized --ns3 wifi \
#                                --kill-drone-at-step 20 --kill-drone-id 0
#run_one heur_cent_kill_d0_s20   --planner heuristic --arch centralized --ns3 wifi \
#                                --kill-drone-at-step 20 --kill-drone-id 0

# Cut cloud (centralisé → fallback local)
#run_one aif_cent_cut_cloud_s20  --planner aif       --arch centralized --ns3 wifi \
#                                --cut-cloud-at-step 20


#run_one aif_dist_cut_links_s20  --planner aif       --arch distributed --ns3 wifi \
#                                --neighbor-radius-m 12 \
#                                --cut-drone-link all --cut-drone-link-at-step 20


run_one aif_cent_obstacle_s20   --planner aif       --arch centralized --ns3 wifi \
                                --drop-obstacle-at-step 20 --drop-obstacle-xy "4.0,3.5"

# ════════════════════════════════════════════════════════════════════
# Multi-stress
# ════════════════════════════════════════════════════════════════════
#run_one aif_cent_full_chain     --planner aif       --arch centralized --ns3 wifi \
#                                --cut-cloud-at-step 15 \
#                                --kill-drone-at-step 30 --kill-drone-id 1

# ════════════════════════════════════════════════════════════════════
# Couche réseau alternative
# ════════════════════════════════════════════════════════════════════
#run_one aif_cent_ns3_5g         --planner aif       --arch centralized --ns3 5g

echo
echo "  Runs terminés → $RUNS_DIR"
echo "  Open the dashboard, tab 'Runs', to browse runs interactively."
