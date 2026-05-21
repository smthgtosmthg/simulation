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


def _resilience_overlay(runs, key, title, ylabel, out_path, y_range=None,
                        transform=None):
    """Overlay une métrique pour plusieurs runs avec bandes de phase + events.

    transform : optionnel, fonction (history) -> (xs, ys) pour calculer
    une dérivée / rolling au lieu de lire `key` brut.
    """
    if not runs:
        return False
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, r in enumerate(runs):
        _phase_bands(ax, r["history"])  # phase bands per run (légère superposition)
        if transform is not None:
            xs, ys = transform(r["history"])
        else:
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
# Transforms : métriques non-monotones (peuvent chuter au stresseur)
# ──────────────────────────────────────────────────────────────────


def _coverage_rate_series(history, window: int = 5):
    """Δcoverage / Δstep lissé sur fenêtre glissante."""
    xs, cov = _series(history, "exploration_pct")
    n = len(cov)
    rate = [0.0] * n
    for i in range(n):
        a = max(0, i - window)
        dy = cov[i] - cov[a]
        dx = max(1, xs[i] - xs[a])
        rate[i] = dy / dx
    return xs, rate


def _info_gain_smooth(history, window: int = 3):
    xs, ig = _series(history, "step_info_gain")
    n = len(ig)
    out = [0.0] * n
    for i in range(n):
        a = max(0, i - window + 1)
        chunk = ig[a:i + 1]
        out[i] = sum(chunk) / len(chunk)
    return xs, out


def _delivery_ratio_series(history):
    xs = [h.get("step", i) for i, h in enumerate(history)]
    out = []
    for h in history:
        sent = float(h.get("msg_sent", 0) or 0)
        deliv = float(h.get("msg_delivered", 0) or 0)
        out.append(deliv / sent if sent > 0 else 1.0)
    return xs, out


# ──────────────────────────────────────────────────────────────────
# Resilience triangle : R(t) = perf_stressed(t) / perf_baseline(t)
# ──────────────────────────────────────────────────────────────────


def _align_on_steps(hist_a, hist_b, key):
    """Aligne deux séries sur leurs steps communs. Renvoie (steps, ya, yb)."""
    xa, ya = _series(hist_a, key)
    xb, yb = _series(hist_b, key)
    map_b = dict(zip(xb, yb))
    common = [s for s in xa if s in map_b]
    aligned_a = [ya[xa.index(s)] for s in common]
    aligned_b = [map_b[s] for s in common]
    return common, aligned_a, aligned_b


def _resilience_triangle(baseline_run, stressed_run, key, ylabel, out_path,
                         title_suffix=""):
    """Plot R(t) = perf_stressed(t) / perf_baseline(t) au même step.

    R < 1 = chute par rapport à la baseline ; R = 1 = parité ; R > 1 = au-dessus.
    L'aire ∫(1 − R(t))⁺ dt sur la période post-stress quantifie le « coût de
    résilience » (resilience triangle de Bruneau et al., 2003).
    """
    steps, ys_str, ys_base = _align_on_steps(
        stressed_run["history"], baseline_run["history"], key
    )
    if not steps:
        return False
    # R(t) = stressed / baseline (clampe denom à eps pour éviter div/0)
    R = []
    for a, b in zip(ys_str, ys_base):
        denom = b if abs(b) > 1e-9 else 1e-9
        R.append(a / denom)

    # step où le stresseur s'active (premier event de stress_start)
    events = (stressed_run.get("state_final", {}).get("resilience", {})
              or {}).get("events", [])
    t_stress = None
    for ev in events:
        if ev.get("type") == "stress_start":
            t_stress = ev.get("step")
            break

    # cost = aire sous la ligne y=1 pour les steps post-stress où R < 1
    cost = 0.0
    if t_stress is not None and len(steps) > 1:
        for i in range(1, len(steps)):
            if steps[i] < t_stress:
                continue
            dx = steps[i] - steps[i - 1]
            mid = 0.5 * (R[i] + R[i - 1])
            deficit = max(0.0, 1.0 - mid)
            cost += deficit * dx

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True,
                                          gridspec_kw={"height_ratios": [2, 1]})

    # ── haut : valeurs absolues alignées ──
    ax_top.plot(steps, ys_base, "-", color=PALETTE[0], lw=2.0,
                label=f"{baseline_run['tag']} (baseline)")
    ax_top.plot(steps, ys_str, "-", color=PALETTE[4], lw=2.0,
                label=f"{stressed_run['tag']} (stressed)")
    ax_top.fill_between(steps, ys_str, ys_base,
                         where=[a < b for a, b in zip(ys_str, ys_base)],
                         interpolate=True, color="#ef4444", alpha=0.18,
                         label="déficit (chute)")
    if t_stress is not None:
        ax_top.axvline(t_stress, color="#ef4444", lw=1.2, ls="--",
                       label=f"stress @ {t_stress}")
    ax_top.set_ylabel(ylabel)
    ax_top.set_title(f"Resilience triangle — {key}{title_suffix}")
    ax_top.grid(True, axis="y"); ax_top.legend(loc="best", fontsize=8)

    # ── bas : ratio R(t) ──
    ax_bot.axhline(1.0, color="#94a3b8", lw=1.0, ls=":")
    ax_bot.plot(steps, R, "-", color="#fbbf24", lw=2.0)
    ax_bot.fill_between(steps, R, 1.0, where=[r < 1.0 for r in R],
                         interpolate=True, color="#ef4444", alpha=0.25)
    if t_stress is not None:
        ax_bot.axvline(t_stress, color="#ef4444", lw=1.2, ls="--")
    ax_bot.set_ylabel("R(t) = stressed/baseline")
    ax_bot.set_xlabel("AIF step")
    ax_bot.set_title(f"resilience cost ∫(1−R)⁺ dt = {cost:.2f}", fontsize=10)
    ax_bot.grid(True, axis="y")

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
        "entropy_final":  last.get("mean_entropy", 0.0),
        "innov_final":    last.get("innovation_mean", 0.0),
        "active_final":   last.get("active_drones", cfg.get("num_drones", 0)),
        "msg_dropped":    last.get("msg_dropped", 0),
        "queue_size":     last.get("queue_size", 0),
        "time_to_recovery": t_recovery,
        "durable_count": res.get("durable_count", 0),
    }


def _md_table(rows: List[Dict[str, Any]]) -> str:
    cols = ["tag", "planner", "arch", "ns3", "steps",
            "coverage_final", "entropy_final", "innov_final",
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
        _resilience_overlay(sel, "exploration_pct", f"{title} — coverage",
                            "coverage (%)", cov_png, y_range=(0, 100))
        _resilience_overlay(sel, "mean_entropy", f"{title} — entropy",
                            "entropy", ent_png)
        _resilience_overlay(sel, "innovation_ema", f"{title} — innovation (EMA)",
                            "innovation EMA", inn_png)

        # ── métriques non-monotones (peuvent chuter au stresseur) ──
        ig_png = runs_dir / f"resilience_{safe}_info_gain.png"
        rate_png = runs_dir / f"resilience_{safe}_coverage_rate.png"
        innov_raw_png = runs_dir / f"resilience_{safe}_innov_raw.png"
        deliv_png = runs_dir / f"resilience_{safe}_delivery.png"
        _resilience_overlay(sel, None,
                            f"{title} — per-step info gain (rolling 3)",
                            "info gain", ig_png,
                            transform=_info_gain_smooth)
        _resilience_overlay(sel, None,
                            f"{title} — coverage rate (Δ%/step, rolling 5)",
                            "Δcov / step", rate_png,
                            transform=_coverage_rate_series)
        _resilience_overlay(sel, "innovation_mean",
                            f"{title} — innovation (raw)",
                            "innov_mean", innov_raw_png)
        _resilience_overlay(sel, None,
                            f"{title} — delivery ratio",
                            "delivered/sent", deliv_png,
                            transform=_delivery_ratio_series,
                            y_range=(0, 1.05))

        block = (
            f"### {title}\n\n"
            f"**Cumulatif (monotone)** :\n\n"
            f"![coverage]({cov_png.name})\n\n"
            f"![entropy]({ent_png.name})\n\n"
            f"![innovation_ema]({inn_png.name})\n\n"
            f"**Métriques non-monotones — peuvent chuter au stresseur** :\n\n"
            f"![info_gain]({ig_png.name})\n\n"
            f"![coverage_rate]({rate_png.name})\n\n"
            f"![innov_raw]({innov_raw_png.name})\n\n"
            f"![delivery_ratio]({deliv_png.name})\n\n"
        )

        # ── resilience triangle : stressed vs baseline ──
        # convention RESILIENCE_GROUPS : tags[0] = baseline, tags[1] = stressed
        if len(sel) >= 2:
            baseline_run, stressed_run = sel[0], sel[1]
            tri_cov_png = runs_dir / f"resilience_{safe}_triangle_coverage.png"
            tri_ig_png = runs_dir / f"resilience_{safe}_triangle_info_gain.png"
            ok_cov = _resilience_triangle(
                baseline_run, stressed_run, "exploration_pct",
                "coverage (%)", tri_cov_png,
                title_suffix=f" — {title}",
            )
            ok_ig = _resilience_triangle(
                baseline_run, stressed_run, "step_info_gain",
                "info gain / step", tri_ig_png,
                title_suffix=f" — {title}",
            )
            tri_block = "**Resilience triangle (stressed / baseline)** :\n\n"
            if ok_cov:
                tri_block += f"![triangle_coverage]({tri_cov_png.name})\n\n"
            if ok_ig:
                tri_block += f"![triangle_info_gain]({tri_ig_png.name})\n\n"
            block += tri_block

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
