"""
Résilience — Détection de stress, phases de récupération.

Trois phases possibles :
  - **normal**   : régime nominal, poids `select_action` par défaut.
  - **recovery** : déclenchée par un trigger (innovation spike, drone perdu,
                   lien coupé). Boost de l'exploration (w_entropy_recover,
                   w_innov_recover) pour re-mapper rapidement.
  - **durable**  : après α steps en recovery (ou recovered tôt), on entre en
                   "maintien" — pénalité forte si l'entropie remonte.

Transition recovery → durable :
  Recovered ssi `H ≤ H_target` ET `innov ≤ innov_target`.
  Au bout de β steps consécutifs en "recovered" → stress résolu, retour normal.

Détection de spike (déclenche stress automatiquement) :
  innov_mean > innov_ema + k_sigma · σ
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ResilienceState:
    """État brut de la résilience (sérialisé pour le dashboard)."""

    stress_active: bool = False
    stress_t0: int = -1
    recovered_at: int = -1
    durable_count: int = 0
    cause: str = ""
    events: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "stress_active": self.stress_active,
            "cause": self.cause,
            "phase": "normal" if not self.stress_active else self.cause,
            "stress_t0": self.stress_t0,
            "recovered_at": self.recovered_at,
            "durable_count": self.durable_count,
            "events": self.events[-20:],
        }


class ResilienceManager:
    """Encapsule la machine à états + les stats EMA d'innovation.

    Usage :
        rm = ResilienceManager(cfg)
        for step in range(...):
            innov_mean = compute_innov(...)
            spike = rm.update_innovation_stats(innov_mean)
            if spike and step > 5:
                rm.trigger(step, "innovation_spike")
            rm.update_phase(step, h_mean, innov_mean)
            phase = rm.current_phase(step)   # "normal" | "recovery" | "durable"
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = ResilienceState()
        # Stats d'innovation
        self.innov_ema: float = 0.0
        self.innov_var: float = 0.0

    # ── Trigger explicite (stresseur connu : kill drone, cut cloud, …) ──

    def trigger(self, step: int, cause: str) -> None:
        if self.state.stress_active:
            # Déjà en stress : on ajoute juste l'event (cause cumulative)
            self.state.events.append({
                "step": step, "type": "stress_addon", "cause": cause,
            })
            return
        self.state.stress_active = True
        self.state.stress_t0 = step
        self.state.recovered_at = -1
        self.state.durable_count = 0
        self.state.cause = cause
        self.state.events.append({
            "step": step, "type": "stress_start", "cause": cause,
        })
        print(f"  [RESILIENCE] ⚠ STRESS ACTIVATED at step {step}: {cause}")

    # ── Update innovation EMA et détection de spike ─────────────────

    def update_innovation_stats(self, innov_mean: float) -> bool:
        """Met à jour EMA + variance de l'innovation, retourne True si spike."""
        err = innov_mean - self.innov_ema
        self.innov_ema += self.cfg.ema_alpha * err
        self.innov_var += self.cfg.ema_alpha * ((err * err) - self.innov_var)
        sigma = math.sqrt(max(self.innov_var, 1e-8))
        return innov_mean > (self.innov_ema + self.cfg.k_sigma * sigma)

    # ── Update de phase (à appeler chaque step) ─────────────────────

    def update_phase(self, step: int, h_mean: float, innov_mean: float) -> None:
        """Vérifie si on est recovered, si on quitte le stress."""
        if not self.state.stress_active:
            return
        cfg = self.cfg
        recovered_now = (h_mean <= cfg.H_target) and (innov_mean <= cfg.innov_target)
        if self.state.recovered_at < 0:
            if recovered_now:
                self.state.recovered_at = step
                self.state.durable_count = 0
                self.state.events.append({
                    "step": step, "type": "recovery_reached",
                    "entropy": round(h_mean, 4),
                    "innovation": round(innov_mean, 4),
                })
                print(f"  [RESILIENCE] ✓ Recovery reached at step {step} (H={h_mean:.4f})")
        else:
            if recovered_now:
                self.state.durable_count += 1
            else:
                self.state.durable_count = 0
            if self.state.durable_count >= cfg.beta:
                self.state.stress_active = False
                self.state.events.append({
                    "step": step, "type": "stress_resolved",
                    "duration": step - self.state.stress_t0,
                })
                print(f"  [RESILIENCE] ✓ STRESS RESOLVED at step {step} "
                      f"(duration={step - self.state.stress_t0} steps)")

    # ── Phase courante (lecture seule) ──────────────────────────────

    def current_phase(self, step: int) -> str:
        """'normal' | 'recovery' | 'durable'."""
        if not self.state.stress_active:
            return "normal"
        elapsed = step - self.state.stress_t0
        if elapsed <= self.cfg.alpha:
            return "recovery"
        return "durable"
