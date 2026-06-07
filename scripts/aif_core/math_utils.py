"""
Helpers numériques pour la simulation AIF.

clamp, logit/inv_logit (probabilité ↔ log-odds), entropie de Bernoulli,
échantillonnage softmax.  Aucune dépendance autre que numpy/math standard.
"""

from __future__ import annotations

import math

import numpy as np


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def logit(p: float) -> float:
    """Probabilité → log-odds (avec garde-fou anti 0/1)."""
    p = clamp(p, 1e-7, 1 - 1e-7)
    return math.log(p / (1 - p))


def inv_logit(l: float) -> float:
    """Log-odds → probabilité (numériquement stable pour |l| grand)."""
    if l >= 0:
        return 1.0 / (1.0 + math.exp(-l))
    return math.exp(l) / (1.0 + math.exp(l))


def inv_logit_v(arr: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -30, 30)))


def bernoulli_entropy(p: float) -> float:
    """H(p) = -p·log(p) - (1-p)·log(1-p). Maximum à p=0.5 (≈ 0.693)."""
    p = clamp(p, 1e-7, 1 - 1e-7)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def bernoulli_entropy_v(arr: np.ndarray) -> np.ndarray:
    p = np.clip(arr, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def softmax_sample(values: np.ndarray, temperature: float,
                   rng: np.random.Generator) -> int:
    """Échantillonne un indice selon P(i) ∝ exp(-(values[i]-min)/T).

    Plus T est petit → plus on tend vers argmin déterministe.
    Plus T est grand → plus le tirage est uniforme.
    """
    temperature = max(temperature, 1e-6)
    shifted = -(values - np.min(values)) / temperature
    weights = np.exp(shifted - np.max(shifted))
    probs = weights / weights.sum()
    return int(rng.choice(len(values), p=probs))
