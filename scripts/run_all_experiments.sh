#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Plan d'exécution complet (Tâche 6 du PROMPT_CLAUDE_CODE.md)
#
# Lance séquentiellement 12 configurations couvrant :
#   - Baseline AIF / Heuristique × Centralisé / Distribué
#   - Impact de la latence NS-3 (WiFi puis 5G)
#   - Résilience : perte drone, cut cloud, cut tous les liens drone↔drone
#   - Multi-stress (full_chain)
#
# Chaque run écrit ses artefacts dans :
#   logs/runs/run_YYYYMMDD_HHMMSS_<tag>/
# (cf. scripts/run_artifacts.py)
#
# À la fin, génère le rapport comparatif :
#   logs/runs/comparison_report.md
#   logs/runs/comparison_*.png
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCH="$SCRIPT_DIR/12_launch_aif_isaac_sim.sh"
RUNS_DIR="$WORKSPACE/logs/runs"
mkdir -p "$RUNS_DIR"

# ── Paramètres communs (cap à 80 steps : exploration plateau ~step 52, cf. prompt §6) ──
N_DRONES=${N_DRONES:-3}
MAX_STEPS=${MAX_STEPS:-80}

# ── Permettre d'override la liste via env ──
ONLY="${ONLY:-}"   # ex: ONLY="aif_cent_baseline,heur_cent_baseline" pour ne lancer que ces deux

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
    # GUI par défaut (Isaac Sim affiche sa fenêtre, GPU visible).
    # Pour forcer headless sur cette batterie : `HEADLESS=1 ./run_all_experiments.sh`
    "$LAUNCH" "$N_DRONES" --max-steps "$MAX_STEPS" \
        --run-tag "$TAG" "$@" || {
        echo "  [WARN] run $TAG returned non-zero; continuing."
    }
}

# ───────────────── Matrice de 12 runs ─────────────────

# 1-4 : baselines (NS-3 wifi activé partout, pas de cut)
#run_one aif_cent_baseline      --planner aif       --arch centralized --ns3 wifi
#run_one heur_cent_baseline     --planner heuristic --arch centralized --ns3 wifi
#run_one aif_dist_baseline      --planner aif       --arch distributed --ns3 wifi
#run_one heur_dist_baseline     --planner heuristic --arch distributed --ns3 wifi


# 7-8 : résilience perte drone
#run_one aif_cent_kill_d0_s20   --planner aif       --arch centralized --ns3 wifi --kill-drone-at-step 20
#run_one heur_cent_kill_d0_s20  --planner heuristic --arch centralized --ns3 wifi --kill-drone-at-step 20

# 9-10 : résilience cut liens
run_one aif_cent_cut_cloud_s20 --planner aif --arch centralized --ns3 wifi --cut-cloud-at-step 20
run_one aif_dist_cut_links_s20 --planner aif --arch distributed --ns3 wifi --cut-drone-link all --cut-drone-link-at-step 20

# 11 : compare WiFi vs 5G
#run_one aif_cent_ns3_5g        --planner aif --arch centralized --ns3 5g



# ───────────────── Rapport comparatif ─────────────────

echo
echo "  → Generating comparison report …"
python3 "$SCRIPT_DIR/generate_comparison_report.py" --runs-dir "$RUNS_DIR"
echo
echo "  Comparison report → $RUNS_DIR/comparison_report.md"
echo "  Open the dashboard, tab 'Runs', to browse runs interactively."
