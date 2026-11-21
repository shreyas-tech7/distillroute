"""Accuracy, macro-F1 and their 95% intervals.

Accuracy gets a Wilson score interval. Macro-F1 gets a seeded percentile bootstrap, because it has
no simple closed form. Macro-F1 weighs all 77 intents the same, so a student that ignores the rare
intents cannot hide behind the common ones.
"""

from __future__ import annotations

import math

import numpy as np


def accuracy(gold: list[str], pred: list[str]) -> float:
    return sum(g == p for g, p in zip(gold, pred, strict=True)) / len(gold) if gold else 0.0


def macro_f1(gold: list[str], pred: list[str], labels: list[str] | None = None) -> float:
    """Mean of per-label F1 over the gold label set. A label never predicted and never gold is skipped."""
    labels = sorted(set(gold)) if labels is None else labels
    scores = []
    for c in labels:
        tp = sum(1 for g, p in zip(gold, pred, strict=True) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred, strict=True) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred, strict=True) if g == c and p != c)
        if tp + fp + fn == 0:
            continue
        scores.append(2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores) if scores else 0.0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def bootstrap_f1(gold: list[str], pred: list[str], reps: int = 300, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap of macro-F1 over tickets, vectorized with a confusion count per resample."""
    labels = sorted(set(gold))
    index = {c: i for i, c in enumerate(labels)}
    k = len(labels)
    g = np.array([index[x] for x in gold])
    p = np.array([index.get(x, k) for x in pred])  # k means "a label outside the gold set"
    rng = np.random.default_rng(seed)
    n = len(g)
    out = np.empty(reps)
    for r in range(reps):
        i = rng.integers(0, n, n)
        gi, pi = g[i], p[i]
        tp = np.bincount(gi[gi == pi], minlength=k)[:k]
        gold_n = np.bincount(gi, minlength=k)[:k]
        pred_n = np.bincount(pi, minlength=k + 1)[:k]
        denom = gold_n + pred_n
        present = denom > 0
        out[r] = np.mean(2 * tp[present] / denom[present])
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def score(gold: list[str], pred: list[str], seed: int = 0) -> dict:
    k = sum(g == p for g, p in zip(gold, pred, strict=True))
    lo, hi = wilson(k, len(gold))
    flo, fhi = bootstrap_f1(gold, pred, seed=seed)
    return {
        "n": len(gold),
        "correct": k,
        "accuracy": k / len(gold) if gold else 0.0,
        "accuracy_ci": [lo, hi],
        "macro_f1": macro_f1(gold, pred),
        "macro_f1_ci": [flo, fhi],
    }


def agreement(a: list[str], b: list[str]) -> float:
    return accuracy(a, b)
