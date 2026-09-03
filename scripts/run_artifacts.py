#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# palette alignée sur le dashboard
DRONE_COLORS = ["#22d3ee", "#34d399", "#a78bfa", "#fbbf24", "#f472b6", "#fb923c"]
PHASE_COLORS = {
    "normal":   "#10b981",
    "recovery": "#ef4444",
    "durable":  "#fbbf24",
}


def _safe_cfg_dict(cfg) -> Dict[str, Any]:
    if is_dataclass(cfg):
        d = asdict(cfg)
    else:
        d = {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    for k, v in list(d.items()):
        if isinstance(v, float) and (v != v):
            d[k] = None
    return d


def _workspace_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_runs_dir(cfg) -> Path:
    explicit = getattr(cfg, "runs_dir", "") or ""
    if explicit:
        return Path(explicit).expanduser().resolve()
    return _workspace_root() / "logs" / "runs"


def make_run_dir(cfg) -> Path:
    base = _resolve_runs_dir(cfg)
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = (getattr(cfg, "run_tag", "default") or "default").strip().replace(" ", "_")
    run_dir = base / f"run_{ts}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _setup_matplotlib():
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
    plt = _setup_matplotlib()
    arr = np.asarray(fused_belief.probability, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
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
    _phase_bands(ax, history)
    ax.plot(xs, mean, "-", color="#f472b6", lw=1.2, label="mean")
    ax.plot(xs, ema, "-", color="#fbbf24", lw=2.0, label="EMA")
    ax.set_title("Innovation (mean + EMA)")
    ax.set_xlabel("AIF step"); ax.set_ylabel("innovation")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, run_dir / "innovation.png")


def _plot_discovery_rate(history, run_dir: Path):
    plt = _setup_matplotlib()
    xs, ys = _series(history, "discovery_rate")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    _phase_bands(ax, history)
    ax.plot(xs, ys, "-", color="#10b981", lw=2.0)
    ax.fill_between(xs, ys, alpha=0.15, color="#10b981")
    ax.axhline(0, color="#94a3b8", lw=0.5, alpha=0.5)
    ax.set_title("Discovery rate — Δcoverage / step (5-step window)")
    ax.set_xlabel("AIF step"); ax.set_ylabel("% / step")
    ax.grid(True, axis="y")
    _save(fig, run_dir / "discovery_rate.png")


def _plot_coverage_known_vs_global(history, run_dir: Path):
    plt = _setup_matplotlib()
    xs, global_cov = _series(history, "exploration_pct")
    _, known_cov = _series(history, "coverage_known_to_planner")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    _phase_bands(ax, history)
    ax.plot(xs, global_cov, "-", color="#10b981", lw=2.0, label="global (omniscient)")
    ax.plot(xs, known_cov, "-", color="#f472b6", lw=2.0, label="known to planner")
    ax.fill_between(xs, known_cov, global_cov,
                    where=[g > k for g, k in zip(global_cov, known_cov)],
                    color="#ef4444", alpha=0.15, label="network knowledge gap")
    ax.set_title("Coverage : globale vs connue par le drone planificateur")
    ax.set_xlabel("AIF step"); ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, run_dir / "coverage_known_vs_global.png")


def _plot_decisions_per_min(history, run_dir: Path):
    plt = _setup_matplotlib()
    xs, ys = _series(history, "decisions_per_min")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    _phase_bands(ax, history)
    ax.plot(xs, ys, "-", color="#a78bfa", lw=2.0)
    ax.fill_between(xs, ys, alpha=0.15, color="#a78bfa")
    ax.set_title("Vitesse de décision — actions fresh / min (fenêtre 10 steps)")
    ax.set_xlabel("AIF step"); ax.set_ylabel("décisions / min")
    ax.grid(True, axis="y")
    _save(fig, run_dir / "decisions_per_min.png")


def _phase_bands(ax, history):
    if not history:
        return
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


def generate_run_artifacts(
    cfg,
    history: List[Dict],
    final_state: Dict[str, Any],
    fused_belief,
    agents: List,
    qr_stats: Optional[Dict] = None,
    ns3_pairs: Optional[Dict[Tuple[int, int], Dict[str, float]]] = None,
) -> Path:
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

    # 2) PNGs — chaque appel isolé pour ne pas perdre les autres
    plot_calls = [
        ("belief map", lambda: _plot_belief_map(fused_belief, run_dir, agents)),
        ("trajectories", lambda: _plot_trajectories(agents, fused_belief, run_dir)),
        ("entropy", lambda: _plot_simple(history, "mean_entropy",
                                         "Mean entropy",
                                         "entropy", run_dir / "entropy.png",
                                         color="#3b82f6")),
        ("coverage", lambda: _plot_simple(history, "exploration_pct",
                                          "Coverage — % bbox USD",
                                          "%", run_dir / "coverage.png",
                                          color="#10b981", y_range=(0, 100))),
        ("coverage_interior", lambda: _plot_simple(
            history, "exploration_pct_interior",
            "Coverage intérieur — % bbox murs",
            "%", run_dir / "coverage_interior.png",
            color="#0ea5e9", y_range=(0, 100))),
        ("innovation", lambda: _plot_innovation(history, run_dir)),
        ("resilience", lambda: _plot_resilience(
            history, run_dir,
            events=(final_state.get("resilience", {}) or {}).get("events", []),
        )),
        ("discovery_rate", lambda: _plot_discovery_rate(history, run_dir)),
        ("coverage_known_vs_global", lambda: _plot_coverage_known_vs_global(history, run_dir)),
        ("decisions_per_min", lambda: _plot_decisions_per_min(history, run_dir)),
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
            f"- Final coverage (bbox USD)     : {m.get('exploration_pct', '?')} %\n"
            f"- Final coverage (bbox murs)    : {m.get('exploration_pct_interior', '?')} %\n"
        )
        with open(run_dir / "README.md", "w") as f:
            f.write(readme)
    except Exception:
        pass

    return run_dir
