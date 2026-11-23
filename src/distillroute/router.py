"""The hybrid router: the student answers when it is confident, the LLM gets the rest.

The threshold comes from the development split only. Rule: take the lowest confidence threshold
at which the Wilson lower bound of the student's accuracy on the tickets it keeps reaches the
target. The lower bound leaves a margin, so the target is more likely to hold on new tickets. A
lower threshold keeps more tickets, so it saves more money, so the lowest one that meets the rule
wins. When no threshold meets it, the highest one comes back with `met` set to False.
"""

from __future__ import annotations

import numpy as np

from .metrics import wilson

GRID = [round(x, 2) for x in np.linspace(0.0, 0.99, 100)]


def sweep(gold: list[str], pred: list[str], conf: np.ndarray, grid=GRID) -> list[dict]:
    right = np.array([g == p for g, p in zip(gold, pred, strict=True)])
    out = []
    for t in grid:
        keep = conf >= t
        n_keep = int(keep.sum())
        k = int(right[keep].sum())
        out.append(
            {
                "threshold": t,
                "kept_share": n_keep / len(gold),
                "kept_accuracy": k / n_keep if n_keep else 1.0,
                "kept_accuracy_lo": wilson(k, n_keep)[0] if n_keep else 0.0,
            }
        )
    return out


def pick_threshold(gold: list[str], pred: list[str], conf: np.ndarray, target: float = 0.95) -> dict:
    """The lowest threshold whose kept accuracy lower bound reaches `target`. Call it on dev only."""
    rows = sweep(gold, pred, conf)
    for r in rows:
        if r["kept_share"] > 0 and r["kept_accuracy_lo"] >= target:
            return {"target": target, "met": True, **r}
    return {"target": target, "met": False, **rows[-1]}


def route(conf: np.ndarray, threshold: float) -> list[bool]:
    """True where the student keeps the ticket, False where it goes to the LLM."""
    return [bool(c >= threshold) for c in conf]


def wrong_label(gold: str, pred: str, classes: list[str]) -> str:
    """A real class that is not the gold one: the student's own guess when that is wrong, else the
    next class in sorted order. A real wrong class adds a false positive to some class, so macro-F1
    drops the way it would with a real wrong answer."""
    if pred != gold:
        return pred
    i = classes.index(gold) if gold in classes else -1
    return classes[(i + 1) % len(classes)]


def hybrid_predictions(
    gold: list[str], pred: list[str], kept: list[bool], llm_right: bool, classes: list[str] | None = None
) -> list[str]:
    """Hybrid answers with a bounding LLM: always right (upper bound) or always wrong (lower bound).

    Record mode did not run, so the real LLM answers are unknown. Accuracy lies between the two.
    For the lower case each sent ticket gets a wrong real class (see `wrong_label`).
    """
    classes = sorted(classes or set(gold) | set(pred))
    out = []
    for g, p, k in zip(gold, pred, kept, strict=True):
        if k:
            out.append(p)
        else:
            out.append(g if llm_right else wrong_label(g, p, classes))
    return out
