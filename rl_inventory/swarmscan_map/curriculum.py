"""Curriculum ADR bidirectionnel sur le gate de lecture (OpenAI ADR / DORAEMON), piloté par le
taux de lecture mesuré. Pur python/torch, testable sans Isaac."""

from __future__ import annotations

from .config_map import CurriculumConfig, GateConfig


def _interp(loose: float, nominal: float, t: float) -> float:
    return loose + (nominal - loose) * t


class GateCurriculum:
    """Un niveau entier 0..notches : 0 = gate tolérant, notches = gate nominal.

    Monte d'un cran quand l'EMA du taux de lecture dépasse success_hi, redescend sous success_lo.
    Le dwell ne se resserre qu'au dernier cran (condition la plus dure à découvrir).
    """

    def __init__(self, gate: GateConfig, cfg: CurriculumConfig):
        self.gate = gate
        self.cfg = cfg
        self.level = 0
        self.ema = 0.0
        self.episodes_at_level = 0
        self.episodes_after_nominal = 0
        self.confirm = 0

    @property
    def progress(self) -> float:
        # ÉCRÊTÉ à 1.0 : _interp extrapole au-delà du gate nominal si level > notches.
        # verify_env.py imposait --level 7 par défaut alors que notches est passé de 12 à 6 :
        # progress = 1.167 donnait read_distance = 4.0 + (1.25−4.0)·1.167 = 0.79 m, soit un
        # gate 37 % PLUS SERRÉ que le nominal. Toutes les validations d'environnement ont été
        # faites à ce gate-là, et ses zéros ont été lus comme « l'environnement est sain ».
        return min(self.level / self.cfg.notches, 1.0)

    @property
    def nominal(self) -> bool:
        return self.level >= self.cfg.notches

    def thresholds(self) -> dict[str, float]:
        t = self.progress
        g = self.gate
        return {
            "read_distance_m": _interp(*g.adr_read_distance_m, t),
            "view_angle_deg": _interp(*g.adr_view_angle_deg, t),
            "max_speed_mps": _interp(*g.adr_max_speed_mps, t),
            "max_yawrate_rps": _interp(*g.adr_max_yawrate_rps, t),
            "dwell_steps": float(g.adr_dwell_steps[1] if self.nominal else g.adr_dwell_steps[0]),
        }

    def spawn_near_prob(self) -> float:
        return _interp(*self.cfg.spawn_near_prob, self.progress)

    def dropout_prob(self) -> float:
        if self.cfg.dropout_after_nominal and not self.nominal:
            return self.cfg.dropout_prob[0]
        ramp = min(1.0, self.episodes_after_nominal / 500.0)
        return _interp(*self.cfg.dropout_prob, ramp)

    def thresholds_hi_lo(self) -> tuple[float, float]:
        hi = max(self.cfg.success_hi - self.cfg.hi_decay_per_notch * self.level, self.cfg.hi_min)
        lo = max(hi - self.cfg.lo_gap, self.cfg.lo_min)
        return hi, lo

    def on_episodes_end(self, read_fracs: list[float]):
        """À appeler avec le taux de lecture (au gate COURANT) de chaque épisode terminé."""
        hi, lo = self.thresholds_hi_lo()
        for f in read_fracs:
            self.ema = (1 - self.cfg.ema_alpha) * self.ema + self.cfg.ema_alpha * f
            self.episodes_at_level += 1
            if self.nominal:
                self.episodes_after_nominal += 1
            self.confirm = self.confirm + 1 if self.ema > hi + self.cfg.promo_margin else 0
        promotable = (self.confirm >= self.cfg.confirm_episodes
                      and self.episodes_at_level >= self.cfg.min_episodes_per_notch)
        if promotable and self.level < self.cfg.notches:
            self.level += 1
            self.episodes_at_level = 0
            self.confirm = 0
        elif self.ema < lo and self.level > 0 and self.episodes_at_level >= self.cfg.min_episodes_down:
            self.level -= 1
            self.episodes_at_level = 0
            self.confirm = 0

    def state(self) -> dict[str, float]:
        return {"level": float(self.level), "ema": self.ema, "dropout_p": self.dropout_prob()}
