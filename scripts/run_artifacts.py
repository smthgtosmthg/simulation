#!/usr/bin/env python3
"""
Génération des artefacts par run (Tâche 5.4 du PROMPT_CLAUDE_CODE.md).

Appelé en fin de main() dans 12_aif_isaac_sim.py — produit un dossier
logs/runs/run_YYYYMMDD_HHMMSS_<tag>/ contenant :
    - config.json          : dump du SimConfig (params actifs)
    - history.json         : courbes step-par-step
    - state_final.json     : snapshot final
    - belief_map_final.png : grille de croyance fusionnée (matplotlib)
    - trajectories.png     : trajectoires des drones par couleur
    - entropy.png          : courbe d'entropie
    - coverage.png         : courbe de coverage interior
    - innovation.png       : mean + EMA
    - resilience_phases.png: entropie avec bandes de phases
    - ns3_latencies.png    : (si NS-3 actif) heatmap latences inter-drones

Toutes les figures utilisent matplotlib (déjà standard). Aucune dépendance
externe ajoutée.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Palette alignée sur le dashboard ──
DRONE_COLORS = ["#22d3ee", "#34d399", "#a78bfa", "#fbbf24", "#f472b6", "#fb923c"]
PHASE_COLORS = {
    "normal":   "#10b981",   # green
    "recovery": "#ef4444",   # red
    "durable":  "#fbbf24",   # yellow
}


def _safe_cfg_dict(cfg) -> Dict[str, Any]:
    """Sérialise un dataclass SimConfig (ignore les attrs internes)."""
    if is_dataclass(cfg):
        d = asdict(cfg)
    else:
        d = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    # Convertir les types non-JSON (float NaN, paths)
    for k, v in list(d.items()):
        if isinstance(v, float) and (v != v):  # NaN
            d[k] = None
    return d


def _workspace_root() -> Path:
    """Remonte au dossier du repo (parent du dossier scripts/)."""
    return Path(__file__).resolve().parent.parent


def _resolve_runs_dir(cfg) -> Path:
    explicit = getattr(cfg, "runs_dir", "") or ""
    if explicit:
        return Path(explicit).expanduser().resolve()
    return _workspace_root() / "logs" / "runs"


def make_run_dir(cfg) -> Path:
    """Crée le dossier run_YYYYMMDD_HHMMSS_<tag>/ et le retourne."""
    base = _resolve_runs_dir(cfg)
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = (getattr(cfg, "run_tag", "default") or "default").strip().replace(" ", "_")
    run_dir = base / f"run_{ts}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ──────────────────────────────────────────────────────────────────
# Génération des figures
# ──────────────────────────────────────────────────────────────────


def _setup_matplotlib():
    """Backend non-interactif + style sombre cohérent avec le dashboard."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
    return plt


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _plot_belief_map(fused_belief, run_dir: Path, agents: List = None):
    """Carte de croyance fusionnée (Y-flipped pour matcher le dashboard)."""
    plt = _setup_matplotlib()
    arr = np.asarray(fused_belief.probability, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    # Y-flip pour que (0,0) soit en bas (convention Isaac/dashboard)
    im = ax.imshow(arr, origin="lower", cmap="RdYlGn_r", vmin=0.0, vmax=1.0,
                   interpolation="nearest", aspect="equal")
    ax.set_title("Belief Map — Fused Occupancy (final)")
    ax.set_xlabel("grid x"); ax.set_ylabel("grid y")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04)
    cbar.set_label("P(occupied)")
    if agents:
        for a in agents:
            try:
                gx = a.x / fused_belief.resolution
                gy = a.y / fused_belief.resolution
                color = DRONE_COLORS[a.id % len(DRONE_COLORS)]
                ax.plot(gx, gy, "o", color=color, markersize=8,
                        markeredgecolor="white", markeredgewidth=1.2,
                        label=f"D{a.id}")
            except Exception:
                pass
        ax.legend(loc="upper right", fontsize=8)
    _save(fig, run_dir / "belief_map_final.png")


def _plot_trajectories(agents: List, fused_belief, run_dir: Path):
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 5))
    # background = belief map en filigrane
    arr = np.asarray(fused_belief.probability, dtype=float)
    ax.imshow(arr, origin="lower", cmap="Greys", vmin=0.0, vmax=1.0,
              alpha=0.5, aspect="equal")
    for a in agents:
        try:
            color = DRONE_COLORS[a.id % len(DRONE_COLORS)]
            xs = [t[0] / fused_belief.resolution for t in a.trail]
            ys = [t[1] / fused_belief.resolution for t in a.trail]
            ax.plot(xs, ys, "-", color=color, lw=1.6, label=f"D{a.id}")
            if xs:
                ax.plot(xs[-1], ys[-1], "o", color=color, markersize=8,
                        markeredgecolor="white", markeredgewidth=1.2)
        except Exception:
            pass
    ax.set_title("Drone trajectories")
    ax.set_xlabel("grid x"); ax.set_ylabel("grid y")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, run_dir / "trajectories.png")


def _series(history: List[Dict], key: str) -> Tuple[List[int], List[float]]:
    xs = [h.get("step", i) for i, h in enumerate(history)]
    ys = [float(h.get(key, 0.0) or 0.0) for h in history]
    return xs, ys


def _plot_simple(history, key: str, title: str, ylabel: str,
                 path: Path, color: str = "#3b82f6", y_range=None):
    plt = _setup_matplotlib()
    xs, ys = _series(history, key)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(xs, ys, "-", color=color, lw=1.8)
    ax.fill_between(xs, ys, alpha=0.15, color=color)
    ax.set_title(title)
    ax.set_xlabel("AIF step"); ax.set_ylabel(ylabel)
    ax.grid(True, axis="y")
    if y_range:
        ax.set_ylim(*y_range)
    _save(fig, path)


def _plot_innovation(history, run_dir: Path):
    plt = _setup_matplotlib()
    xs, mean = _series(history, "innovation_mean")
    _, ema = _series(history, "innovation_ema")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(xs, mean, "-", color="#f472b6", lw=1.2, label="mean")
    ax.plot(xs, ema, "-", color="#fbbf24", lw=2.0, label="EMA")
    ax.set_title("Innovation (mean + EMA)")
    ax.set_xlabel("AIF step"); ax.set_ylabel("innovation")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, run_dir / "innovation.png")


def _phase_bands(ax, history):
    """Surimpose des bandes verticales colorées selon resilience_phase."""
    if not history:
        return
    # On regroupe les indices consécutifs avec la même phase
    cur_phase = history[0].get("resilience_phase", "normal")
    start = history[0].get("step", 0)
    for h in history[1:]:
        ph = h.get("resilience_phase", "normal")
        if ph != cur_phase:
            end = h.get("step", start + 1)
            if cur_phase != "normal":
                ax.axvspan(start, end, color=PHASE_COLORS.get(cur_phase, "#888"),
                           alpha=0.18)
            cur_phase = ph
            start = end
    # dernière bande
    end = history[-1].get("step", start + 1)
    if cur_phase != "normal":
        ax.axvspan(start, end, color=PHASE_COLORS.get(cur_phase, "#888"),
                   alpha=0.18)


def _plot_resilience(history, run_dir: Path, events: List[Dict] = None):
    plt = _setup_matplotlib()
    xs, ent = _series(history, "mean_entropy")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    _phase_bands(ax, history)
    ax.plot(xs, ent, "-", color="#3b82f6", lw=1.8, label="entropy")
    # marqueurs verticaux pour les events
    if events:
        for ev in events:
            step = ev.get("step", -1)
            if step < 0:
                continue
            t = ev.get("type") or ev.get("event") or "evt"
            ax.axvline(step, color="#ef4444", lw=1.0, alpha=0.6, ls="--")
            ax.text(step, max(ent) if ent else 1.0, f" {t}",
                    rotation=90, fontsize=7, va="top", color="#fbbf24")
    ax.set_title("Resilience curve — entropy + phase bands")
    ax.set_xlabel("AIF step"); ax.set_ylabel("mean entropy")
    ax.grid(True, axis="y"); ax.legend(loc="upper right", fontsize=8)
    _save(fig, run_dir / "resilience_phases.png")


def _rolling_mean(values: List[float], window: int) -> List[float]:
    n = len(values)
    if n == 0 or window <= 1:
        return list(values)
    out = []
    for i in range(n):
        a = max(0, i - window + 1)
        chunk = values[a:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def _coverage_rate(history: List[Dict], window: int = 5) -> Tuple[List[int], List[float]]:
    """Δcoverage/Δstep lissé sur une fenêtre glissante."""
    xs, cov = _series(history, "exploration_pct")
    n = len(cov)
    rate = [0.0] * n
    for i in range(n):
        a = max(0, i - window)
        dy = cov[i] - cov[a]
        dx = max(1, xs[i] - xs[a])
        rate[i] = dy / dx
    return xs, rate


def _delivery_ratio(history: List[Dict]) -> Tuple[List[int], List[float]]:
    """msg_delivered / msg_sent cumulés. NaN → 1.0 (rien à délivrer ⇒ OK)."""
    xs = [h.get("step", i) for i, h in enumerate(history)]
    ratio = []
    for h in history:
        sent = float(h.get("msg_sent", 0) or 0)
        deliv = float(h.get("msg_delivered", 0) or 0)
        ratio.append(deliv / sent if sent > 0 else 1.0)
    return xs, ratio


def _plot_dynamics(history, run_dir: Path, events: List[Dict] = None):
    """Métriques non-monotones — peuvent chuter au moment du stresseur.

    4 sous-courbes : step_info_gain, Δcoverage/Δstep (rolling),
    innovation_mean (brut, pas EMA), delivery ratio cumulé.
    """
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(4, 1, figsize=(8.5, 9.5), sharex=True)

    xs_ig, ig = _series(history, "step_info_gain")
    ig_smooth = _rolling_mean(ig, window=3)
    ax0 = axes[0]
    _phase_bands(ax0, history)
    ax0.plot(xs_ig, ig, "-", color="#94a3b8", lw=0.8, alpha=0.5, label="raw")
    ax0.plot(xs_ig, ig_smooth, "-", color="#22d3ee", lw=2.0, label="rolling(3)")
    ax0.set_ylabel("info gain / step")
    ax0.set_title("Per-step info gain — chute = perception/comms dégradées")
    ax0.grid(True, axis="y"); ax0.legend(loc="upper right", fontsize=8)

    xs_r, rate = _coverage_rate(history, window=5)
    ax1 = axes[1]
    _phase_bands(ax1, history)
    ax1.plot(xs_r, rate, "-", color="#10b981", lw=2.0)
    ax1.fill_between(xs_r, rate, alpha=0.15, color="#10b981")
    ax1.set_ylabel("Δcoverage / step (%)")
    ax1.set_title("Coverage rate (rolling 5) — chute = exploration ralentit")
    ax1.grid(True, axis="y")

    xs_in, innov = _series(history, "innovation_mean")
    ax2 = axes[2]
    _phase_bands(ax2, history)
    ax2.plot(xs_in, innov, "-", color="#f472b6", lw=1.6)
    ax2.set_ylabel("innovation (raw)")
    ax2.set_title("Innovation mean (brut) — pic = surprise sur les obs")
    ax2.grid(True, axis="y")

    xs_d, ratio = _delivery_ratio(history)
    ax3 = axes[3]
    _phase_bands(ax3, history)
    ax3.plot(xs_d, ratio, "-", color="#fbbf24", lw=2.0)
    ax3.set_ylim(0.0, 1.05)
    ax3.set_ylabel("delivered / sent")
    ax3.set_xlabel("AIF step")
    ax3.set_title("Delivery ratio cumulé — chute = liens coupés / drops")
    ax3.grid(True, axis="y")

    # marqueurs d'events de résilience sur les 4 axes
    if events:
        for ev in events:
            step = ev.get("step", -1)
            if step < 0:
                continue
            t = ev.get("type") or ev.get("event") or "evt"
            for ax in axes:
                ax.axvline(step, color="#ef4444", lw=1.0, alpha=0.6, ls="--")
            axes[0].text(step, max(ig) if ig else 1.0, f" {t}",
                         rotation=90, fontsize=7, va="top", color="#fbbf24")

    _save(fig, run_dir / "dynamics.png")


def _plot_ns3(ns3_pairs: Dict[Tuple[int, int], Dict[str, float]],
              n_drones: int, run_dir: Path):
    if not ns3_pairs:
        return
    plt = _setup_matplotlib()
    n = max(n_drones, max(max(p) for p in ns3_pairs.keys()) + 1)
    M = np.full((n, n), np.nan)
    for (i, j), info in ns3_pairs.items():
        lat = info.get("latency_ms", 0.0)
        M[i, j] = lat
        M[j, i] = lat
    fig, ax = plt.subplots(figsize=(5, 4))
    cmap = "viridis"
    im = ax.imshow(M, cmap=cmap, vmin=0)
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                        fontsize=7, color="white")
    ax.set_title("NS-3 latency matrix (ms)")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xlabel("drone j"); ax.set_ylabel("drone i")
    fig.colorbar(im, ax=ax, fraction=0.046, label="latency (ms)")
    _save(fig, run_dir / "ns3_latencies.png")


# ──────────────────────────────────────────────────────────────────
# Point d'entrée appelé depuis main()
# ──────────────────────────────────────────────────────────────────


def generate_run_artifacts(
    cfg,
    history: List[Dict],
    final_state: Dict[str, Any],
    fused_belief,
    agents: List,
    qr_stats: Optional[Dict] = None,
    ns3_pairs: Optional[Dict[Tuple[int, int], Dict[str, float]]] = None,
) -> Path:
    """Produit le dossier run_YYYYMMDD_HHMMSS_<tag>/ et renvoie son chemin.

    En cas d'échec d'un plot (matplotlib indispo, etc.), on continue : les
    JSON sont toujours écrits — c'est le contrat minimal pour la page Runs.
    """
    run_dir = make_run_dir(cfg)

    # 1) JSON : config + history + state final
    try:
        with open(run_dir / "config.json", "w") as f:
            json.dump(_safe_cfg_dict(cfg), f, indent=2)
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, separators=(",", ":"))
        with open(run_dir / "state_final.json", "w") as f:
            json.dump(final_state, f, separators=(",", ":"))
        if qr_stats:
            with open(run_dir / "qr_stats.json", "w") as f:
                json.dump(qr_stats, f, indent=2)
    except Exception as e:
        print(f"  [run_artifacts] JSON dump warning: {e}")

    # 2) PNGs — on isole chaque appel pour ne pas perdre les autres si un crash
    plot_calls = [
        ("belief map", lambda: _plot_belief_map(fused_belief, run_dir, agents)),
        ("trajectories", lambda: _plot_trajectories(agents, fused_belief, run_dir)),
        ("entropy", lambda: _plot_simple(history, "mean_entropy",
                                         "Mean entropy",
                                         "entropy", run_dir / "entropy.png",
                                         color="#3b82f6")),
        ("coverage", lambda: _plot_simple(history, "exploration_pct",
                                          "Coverage (%)",
                                          "%", run_dir / "coverage.png",
                                          color="#10b981", y_range=(0, 100))),
        ("innovation", lambda: _plot_innovation(history, run_dir)),
        ("resilience", lambda: _plot_resilience(
            history, run_dir,
            events=(final_state.get("resilience", {}) or {}).get("events", []),
        )),
        ("dynamics", lambda: _plot_dynamics(
            history, run_dir,
            events=(final_state.get("resilience", {}) or {}).get("events", []),
        )),
    ]
    if ns3_pairs:
        plot_calls.append(
            ("ns3", lambda: _plot_ns3(ns3_pairs,
                                      n_drones=getattr(cfg, "num_drones", 3),
                                      run_dir=run_dir))
        )
    for name, fn in plot_calls:
        try:
            fn()
        except Exception as e:
            print(f"  [run_artifacts] plot {name} skipped: {e}")

    # 3) Petit README facile à lire pour le prof
    try:
        m = history[-1] if history else {}
        readme = (
            f"# Run {run_dir.name}\n\n"
            f"- Generated : {datetime.now().isoformat(timespec='seconds')}\n"
            f"- Tag       : {getattr(cfg, 'run_tag', 'default')}\n"
            f"- Planner   : {getattr(cfg, 'planner', '?')}\n"
            f"- Arch      : {getattr(cfg, 'arch', '?')} (eff: "
            f"{final_state.get('arch_effective','?')})\n"
            f"- NS-3      : {getattr(cfg, 'ns3_mode', 'none')}\n"
            f"- Steps     : {len(history)}\n"
            f"- Final entropy  : {m.get('mean_entropy', '?')}\n"
            f"- Final coverage : {m.get('exploration_pct', '?')} %\n"
        )
        with open(run_dir / "README.md", "w") as f:
            f.write(readme)
    except Exception:
        pass

    return run_dir
