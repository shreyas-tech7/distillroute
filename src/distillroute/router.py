"""The hybrid router: the student answers when it is confident, the LLM gets the rest.

The threshold comes from the development split only. Rule: take the lowest confidence threshold
at which the student's accuracy on the tickets it keeps reaches the target. A lower threshold keeps
more tickets, so it saves more money, so the lowest one that meets the target wins.
"""

from __future__ import annotations

import numpy as np

GRID = [round(x, 2) for x in np.linspace(0.0, 0.99, 100)]


def sweep(gold: list[str], pred: list[str], conf: np.ndarray, grid=GRID) -> list[dict]:
    right = np.array([g == p for g, p in zip(gold, pred, strict=True)])
    out = []
    for t in grid:
        keep = conf >= t
        n_keep = int(keep.sum())
        out.append(
            {
                "threshold": t,
                "kept_share": n_keep / len(gold),
                "kept_accuracy": float(right[keep].mean()) if n_keep else 1.0,
            }
        )
    return out


def pick_threshold(gold: list[str], pred: list[str], conf: np.ndarray, target: float = 0.95) -> dict:
    """The lowest threshold whose kept accuracy on this split reaches `target`. Call it on dev only."""
    rows = sweep(gold, pred, conf)
    for r in rows:
        if r["kept_share"] > 0 and r["kept_accuracy"] >= target:
            return {"target": target, **r}
    return {"target": target, **rows[-1]}


def route(pred: list[str], conf: np.ndarray, threshold: float) -> list[bool]:
    """True where the student keeps the ticket, False where it goes to the LLM."""
    return [bool(c >= threshold) for c in conf]


LLM_WRONG = "__llm_wrong__"


def hybrid_predictions(gold: list[str], pred: list[str], kept: list[bool], llm_right: bool) -> list[str]:
    """Hybrid answers with a bounding LLM: always right (upper bound) or always wrong (lower bound).

    Record mode did not run, so the real LLM answers are unknown. The two bounds hold the truth.
    """
    fallback = list(gold) if llm_right else [LLM_WRONG] * len(gold)
    return [p if k else f for p, k, f in zip(pred, kept, fallback, strict=True)]
