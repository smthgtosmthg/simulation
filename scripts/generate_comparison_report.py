#!/usr/bin/env python3
"""
Génère le rapport comparatif des runs (Tâche 6 du PROMPT_CLAUDE_CODE.md).

Lit tous les sous-dossiers `logs/runs/run_*` produits par
`scripts/run_all_experiments.sh` (chacun contient `config.json` et
`history.json` — cf. `scripts/run_artifacts.py`).  Produit :

    logs/runs/comparison_report.md
    logs/runs/comparison_aif_vs_heur_coverage.png
    logs/runs/comparison_aif_vs_heur_entropy.png
    logs/runs/comparison_cent_vs_dist_coverage.png
    logs/runs/comparison_cent_vs_dist_entropy.png
    logs/runs/comparison_resilience_<stress>_<metric>.png  (un par scénario)

Le rapport markdown contient :
  - Tableau récapitulatif (tag, planner, arch, ns3, coverage_final, entropy_final,
    time_to_recovery, durable_count)
  - Section AIF vs Heuristique (courbes + diagnostic)
  - Section Centralisé vs Distribué
  - Section Résilience par stresseur
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Style cohérent avec run_artifacts.py / dashboard
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor":   "#1e293b",
    "axes.edgecolor":   "#334155",
    "axes.labelcolor":  "#94a3b8",
    "axes.titlecolor":  "#f8fafc",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "grid.color":       "#334155",
    "grid.alpha":       0.35,
    "text.color":       "#f8fafc",
    "savefig.facecolor": "#0f172a",
})


# ──────────────────────────────────────────────────────────────────
# Chargement des runs
# ──────────────────────────────────────────────────────────────────


def load_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    if not runs_dir.is_dir():
        return runs
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("run_"):
            continue
        cfg_p = d / "config.json"
        hist_p = d / "history.json"
        if not cfg_p.exists() or not hist_p.exists():
            continue
        try:
            cfg = json.loads(cfg_p.read_text())
            history = json.loads(hist_p.read_text())
        except (IOError, json.JSONDecodeError):
            continue
        # state_final pour récupérer resilience.events
        state_final = {}
        sp = d / "state_final.json"
        if sp.exists():
            try:
                state_final = json.loads(sp.read_text())
            except (IOError, json.JSONDecodeError):
                pass
        runs.append({
            "dir": d,
            "tag": cfg.get("run_tag") or d.name,
            "config": cfg,
            "history": history,
            "state_final": state_final,
        })
    return runs


def _by_tag(runs, *tags):
    """Retourne les runs dont run_tag est dans `tags`, dans l'ordre passé."""
    out = []
    by = {r["tag"]: r for r in runs}
    # certains tags peuvent ne pas exister → on skippe
    for t in tags:
        if t in by:
            out.append(by[t])
    return out


def _series(history, key, default=0.0):
    xs = [h.get("step", i) for i, h in enumerate(history)]
    ys = [float(h.get(key, default) or default) for h in history]
    return xs, ys


# ──────────────────────────────────────────────────────────────────
# Plots comparatifs
# ──────────────────────────────────────────────────────────────────


PALETTE = ["#22d3ee", "#34d399", "#a78bfa", "#fbbf24", "#f472b6", "#fb923c",
           "#3b82f6", "#10b981", "#ef4444"]


def _overlay_plot(runs, key, title, ylabel, out_path, y_range=None):
    if not runs:
        return False
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, r in enumerate(runs):
        xs, ys = _series(r["history"], key)
        ax.plot(xs, ys, "-", color=PALETTE[i % len(PALETTE)],
                lw=2.0, label=r["tag"])
    ax.set_title(title)
    ax.set_xlabel("AIF step"); ax.set_ylabel(ylabel)
    ax.grid(True, axis="y")
    if y_range:
        ax.set_ylim(*y_range)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def _phase_bands(ax, history):
    bands = []
    if not history:
        return
    cur = history[0].get("resilience_phase", "normal")
    s = history[0].get("step", 0)
    for h in history[1:]:
        ph = h.get("resilience_phase", "normal")
        if ph != cur:
            bands.append((cur, s, h.get("step", s)))
            cur = ph
            s = h.get("step", s)
    bands.append((cur, s, history[-1].get("step", s)))
    cmap = {"recovery": "#ef4444", "durable": "#fbbf24"}
    for ph, st, en in bands:
        if ph in cmap:
            ax.axvspan(st, en, color=cmap[ph], alpha=0.18)


def _resilience_overlay(runs, key, title, ylabel, out_path, y_range=None):
    if not runs:
        return False
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, r in enumerate(runs):
        _phase_bands(ax, r["history"])  # phase bands per run (légère superposition)
        xs, ys = _series(r["history"], key)
        ax.plot(xs, ys, "-", color=PALETTE[i % len(PALETTE)],
                lw=2.0, label=r["tag"])
        # marqueurs verticaux pour les events
        events = (r.get("state_final", {}).get("resilience", {}) or {}).get("events", [])
        for ev in events:
            step = ev.get("step", -1)
            if step >= 0:
                ax.axvline(step, color=PALETTE[i % len(PALETTE)], lw=0.8,
                           alpha=0.35, ls="--")
    ax.set_title(title)
    ax.set_xlabel("AIF step"); ax.set_ylabel(ylabel)
    ax.grid(True, axis="y")
    if y_range:
        ax.set_ylim(*y_range)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


# ──────────────────────────────────────────────────────────────────
# Métriques résumées
# ──────────────────────────────────────────────────────────────────


def _summary_row(r) -> Dict[str, Any]:
    cfg = r["config"]
    hist = r["history"]
    last = hist[-1] if hist else {}
    state_final = r.get("state_final", {})
    res = state_final.get("resilience", {}) if state_final else {}
    events = res.get("events", []) if res else []
    t_recovery = None
    for ev in events:
        if (ev.get("type") == "recovery_reached") or (ev.get("event") == "recovered"):
            t_recovery = ev.get("step")
            break
    return {
        "tag": cfg.get("run_tag") or r["tag"],
        "planner": cfg.get("planner", "?"),
        "arch": cfg.get("arch", "?"),
        "ns3": cfg.get("ns3_mode", "none"),
        "steps": len(hist),
        "coverage_final": last.get("exploration_pct", 0.0),
        "coverage_known_final": last.get("coverage_known_to_planner", 0.0),
        "entropy_final":  last.get("mean_entropy", 0.0),
        "innov_final":    last.get("innovation_mean", 0.0),
        "discovery_rate_final": last.get("discovery_rate", 0.0),
        "decisions_per_min_final": last.get("decisions_per_min", 0.0),
        "active_final":   last.get("active_drones", cfg.get("num_drones", 0)),
        "msg_dropped":    last.get("msg_dropped", 0),
        "queue_size":     last.get("queue_size", 0),
        "time_to_recovery": t_recovery,
        "durable_count": res.get("durable_count", 0),
    }


def _md_table(rows: List[Dict[str, Any]]) -> str:
    cols = ["tag", "planner", "arch", "ns3", "steps",
            "coverage_final", "coverage_known_final",
            "entropy_final", "innov_final",
            "discovery_rate_final", "decisions_per_min_final",
            "active_final", "msg_dropped", "queue_size",
            "time_to_recovery", "durable_count"]
    head = "| " + " | ".join(cols) + " |\n"
    sep  = "| " + " | ".join("---" for _ in cols) + " |\n"
    body = ""
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c)
            if v is None:
                vals.append("—")
            elif isinstance(v, float):
                vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        body += "| " + " | ".join(vals) + " |\n"
    return head + sep + body


# ──────────────────────────────────────────────────────────────────
# Sections du rapport
# ──────────────────────────────────────────────────────────────────


COMPARE_GROUPS = [
    # (title, list_of_run_tags, comment)
    ("AIF vs Heuristique — centralized",
     ["aif_cent_baseline", "heur_cent_baseline"],
     "Compare la stratégie d'action à environnement égal (sans NS-3, sans cut)."),
    ("AIF vs Heuristique — distributed",
     ["aif_dist_baseline", "heur_dist_baseline"],
     "Même comparaison en architecture distribuée."),
    ("Centralized vs Distributed — AIF",
     ["aif_cent_baseline", "aif_dist_baseline"],
     "Impact pur de l'architecture, planner identique."),
    ("Centralized vs Distributed — Heuristique",
     ["heur_cent_baseline", "heur_dist_baseline"],
     "Idem pour le planner heuristique."),
    ("Impact NS-3 WiFi — AIF/centralized",
     ["aif_cent_baseline", "aif_cent_ns3_wifi"],
     "Ajout de la latence WiFi NS-3."),
    ("Impact NS-3 WiFi — AIF/distributed",
     ["aif_dist_baseline", "aif_dist_ns3_wifi"],
     "Idem en distribué."),
    ("WiFi vs 5G — AIF/centralized",
     ["aif_cent_ns3_wifi", "aif_cent_ns3_5g"],
     "Compare deux couches réseau."),
]

RESILIENCE_GROUPS = [
    ("Perte de drone @20 — AIF vs Heuristique",
     ["aif_cent_kill_d0_s20", "heur_cent_kill_d0_s20"]),
    ("Cut cloud @20 — auto-failover (AIF/centralized)",
     ["aif_cent_baseline", "aif_cent_cut_cloud_s20"]),
    ("Cut all drone links @20 — autonomie pure (AIF/distributed)",
     ["aif_dist_baseline", "aif_dist_cut_links_s20"]),
    ("Multi-stress (cut cloud@15 + kill d1@30)",
     ["aif_cent_baseline", "aif_cent_full_chain"]),
]


def build_report(runs_dir: Path) -> Path:
    runs = load_runs(runs_dir)
    if not runs:
        print(f"  [report] no runs found in {runs_dir}")
        return runs_dir / "comparison_report.md"

    # ── 1) Comparaisons baseline ──
    sections_md = []
    for title, tags, comment in COMPARE_GROUPS:
        sel = _by_tag(runs, *tags)
        if not sel:
            continue
        safe = title.lower().replace(" ", "_").replace("/", "_")
        cov_png = runs_dir / f"comparison_{safe}_coverage.png"
        ent_png = runs_dir / f"comparison_{safe}_entropy.png"
        _overlay_plot(sel, "exploration_pct", f"{title} — coverage",
                      "coverage (%)", cov_png, y_range=(0, 100))
        _overlay_plot(sel, "mean_entropy", f"{title} — mean entropy",
                      "entropy", ent_png)
        block = (
            f"### {title}\n\n"
            f"{comment}\n\n"
            f"![coverage]({cov_png.name})\n\n"
            f"![entropy]({ent_png.name})\n\n"
        )
        sections_md.append(block)

    # ── 2) Résilience ──
    res_sections = []
    for title, tags in RESILIENCE_GROUPS:
        sel = _by_tag(runs, *tags)
        if not sel:
            continue
        safe = title.lower().replace(" ", "_").replace("/", "_")
        cov_png = runs_dir / f"resilience_{safe}_coverage.png"
        ent_png = runs_dir / f"resilience_{safe}_entropy.png"
        inn_png = runs_dir / f"resilience_{safe}_innovation.png"
        dr_png = runs_dir / f"resilience_{safe}_discovery_rate.png"
        ck_png = runs_dir / f"resilience_{safe}_coverage_known.png"
        _resilience_overlay(sel, "exploration_pct", f"{title} — coverage (global)",
                            "coverage (%)", cov_png, y_range=(0, 100))
        _resilience_overlay(sel, "mean_entropy", f"{title} — entropy",
                            "entropy", ent_png)
        _resilience_overlay(sel, "innovation_ema", f"{title} — innovation (EMA)",
                            "innovation EMA", inn_png)
        _resilience_overlay(sel, "discovery_rate", f"{title} — discovery rate",
                            "% / step", dr_png)
        _resilience_overlay(sel, "coverage_known_to_planner",
                            f"{title} — coverage known to planner",
                            "coverage planner (%)", ck_png, y_range=(0, 100))
        block = (
            f"### {title}\n\n"
            f"![coverage global]({cov_png.name})\n\n"
            f"![discovery rate]({dr_png.name})\n\n"
            f"![coverage known to planner]({ck_png.name})\n\n"
            f"![entropy]({ent_png.name})\n\n"
            f"![innovation]({inn_png.name})\n\n"
        )
        res_sections.append(block)

    # ── 3) Tableau récapitulatif ──
    summary_rows = [_summary_row(r) for r in runs]
    table_md = _md_table(summary_rows)

    # ── 4) Markdown final ──
    report = []
    report.append(f"# Comparison report — AIF / Heuristique / NS-3 / Résilience\n")
    report.append(f"Generated from `{runs_dir}` "
                  f"({len(runs)} runs)\n\n")
    report.append("## Résumé\n\n")
    report.append(table_md + "\n")
    report.append("## Comparaisons systématiques\n\n")
    report.extend(sections_md)
    report.append("## Courbes de résilience par stresseur\n\n")
    report.extend(res_sections)
    report.append("\n---\n_Generated by `scripts/generate_comparison_report.py`._\n")

    md_path = runs_dir / "comparison_report.md"
    md_path.write_text("".join(report), encoding="utf-8")
    print(f"  [report] markdown : {md_path}")
    return md_path


def main():
    p = argparse.ArgumentParser(description="AIF runs comparison report")
    default_runs = Path(__file__).resolve().parent.parent / "logs" / "runs"
    p.add_argument("--runs-dir", type=Path, default=default_runs)
    args = p.parse_args()
    build_report(args.runs_dir)


if __name__ == "__main__":
    main()
