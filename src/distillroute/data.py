"""Load the datasets, check their hashes and make the splits.

BANKING77 ships with an official train and test file. The development split comes out of the
training file only, so the test file is never used for training, tuning or threshold choice.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SOURCES = DATA / "SOURCES.json"


@dataclass(frozen=True)
class Split:
    texts: list[str]
    labels: list[str]

    def __len__(self) -> int:
        return len(self.texts)


@dataclass(frozen=True)
class Dataset:
    name: str
    labels: list[str]
    train: Split
    dev: Split
    test: Split


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_hashes(root: Path = DATA) -> dict[str, bool]:
    """Compare each data file with the hash recorded in data/SOURCES.json."""
    meta = json.loads((root / "SOURCES.json").read_text())
    out = {}
    for ds in meta["datasets"].values():
        for rel, want in ds["sha256"].items():
            out[rel] = sha256(root / rel) == want
    return out


def read_csv(path: Path) -> Split:
    texts, labels = [], []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"].strip())
            labels.append(row["category"].strip())
    return Split(texts, labels)


def stratified_split(split: Split, frac: float, seed: int) -> tuple[Split, Split]:
    """Cut `frac` of each label out of `split`. Returns (rest, cut)."""
    by_label: dict[str, list[int]] = {}
    for i, y in enumerate(split.labels):
        by_label.setdefault(y, []).append(i)
    rng = random.Random(seed)
    cut: set[int] = set()
    for y in sorted(by_label):
        idx = by_label[y][:]
        rng.shuffle(idx)
        cut.update(idx[: max(1, round(len(idx) * frac))])
    keep = [i for i in range(len(split)) if i not in cut]
    take = sorted(cut)
    return (
        Split([split.texts[i] for i in keep], [split.labels[i] for i in keep]),
        Split([split.texts[i] for i in take], [split.labels[i] for i in take]),
    )


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def drop_overlap(split: Split, other: Split) -> tuple[Split, int]:
    """Remove rows of `split` whose normalized text also appears in `other`."""
    seen = {norm(t) for t in other.texts}
    keep = [i for i, t in enumerate(split.texts) if norm(t) not in seen]
    return Split([split.texts[i] for i in keep], [split.labels[i] for i in keep]), len(split) - len(keep)


def banking77(seed: int = 0, dev_frac: float = 0.15, root: Path = DATA) -> Dataset:
    """The official test file stays whole. The official train file shares 7 texts with it, and
    those 7 are dropped before the dev split, so no test text reaches training or tuning."""
    base = root / "banking77"
    labels = json.loads((base / "categories.json").read_text())
    test = read_csv(base / "test.csv")
    full, _ = drop_overlap(read_csv(base / "train.csv"), test)
    train, dev = stratified_split(full, dev_frac, seed)
    return Dataset("banking77", labels, train, dev, test)


def synthetic(seed: int = 0, root: Path = DATA) -> Dataset:
    """The 600 synthetic tickets, cut 60/20/20 into train, dev and test by label."""
    full = read_csv(root / "synthetic" / "tickets.csv")
    rest, test = stratified_split(full, 0.2, seed)
    train, dev = stratified_split(rest, 0.25, seed + 1)
    return Dataset("synthetic", sorted(set(full.labels)), train, dev, test)


def load(name: str, seed: int = 0) -> Dataset:
    if name == "banking77":
        return banking77(seed)
    if name == "synthetic":
        return synthetic(seed)
    raise ValueError(f"unknown dataset {name!r}")


def sample(split: Split, n: int, seed: int) -> Split:
    """A seeded random sample of n rows, the texts a labeling budget would pay for."""
    idx = list(range(len(split)))
    random.Random(seed).shuffle(idx)
    idx = sorted(idx[:n])
    return Split([split.texts[i] for i in idx], [split.labels[i] for i in idx])
