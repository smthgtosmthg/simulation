from __future__ import annotations

import math

import numpy as np


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def logit(p: float) -> float:
    # Probabilité → log-odds (garde-fou anti 0/1)
    p = clamp(p, 1e-7, 1 - 1e-7)
    return math.log(p / (1 - p))


def inv_logit(l: float) -> float:
    # Log-odds → probabilité (stable pour |l| grand)
    if l >= 0:
        return 1.0 / (1.0 + math.exp(-l))
    return math.exp(l) / (1.0 + math.exp(l))


def inv_logit_v(arr: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -30, 30)))


def bernoulli_entropy(p: float) -> float:
    p = clamp(p, 1e-7, 1 - 1e-7)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


def bernoulli_entropy_v(arr: np.ndarray) -> np.ndarray:
    p = np.clip(arr, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def softmax_sample(values: np.ndarray, temperature: float,
                   rng: np.random.Generator) -> int:
    temperature = max(temperature, 1e-6)
    shifted = -(values - np.min(values)) / temperature
    weights = np.exp(shifted - np.max(shifted))
    probs = weights / weights.sum()
    return int(rng.choice(len(values), p=probs))
