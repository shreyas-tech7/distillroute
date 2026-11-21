"""Run the whole pipeline and write the results files.

1. Load and split the data.
2. Get labels for a budgeted training sample: teacher labels from the cache when they exist,
   gold labels when the teacher step did not run.
3. Train the students.
4. Pick the hybrid threshold on the development split.
5. Evaluate on the test split.

Every timing runs on 2 pinned cores. The results files carry the date, CPU, seed and label source.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import costs, data, metrics, router
from .students import HashEmbedder, Student
from .teacher import Cache, Teacher

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "fixtures" / "teacher"
SIZES = [100, 250, 500, 800]
CURVE_SEEDS = [1, 2, 3]
NOISE = [0.0, 0.1, 0.2, 0.3]
BUDGET = 800
BASELINE_SAMPLE = 300
TARGET = 0.95
CORES = (0, 1)


def pin_cores() -> list[int]:
    """Run on cores 0 and 1 only, so every timing means "on 2 cores"."""
    try:
        os.sched_setaffinity(0, set(CORES))
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return []


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def make_embedder():
    """FastEmbed when its model downloads, else the hash embedder. Returns (embedder, note)."""
    if os.environ.get("DISTILLROUTE_FASTEMBED") == "1":
        try:
            from .students import FastEmbedder

            return FastEmbedder(), "fastembed BAAI/bge-small-en-v1.5"
        except Exception as e:  # noqa: BLE001 (any download failure means the fallback)
            return HashEmbedder(), f"hash embedder, FastEmbed failed: {type(e).__name__}"
    return HashEmbedder(), "hash embedder (FastEmbed model download blocked here, see BLOCKERS.md)"


def teacher_labels(ds: data.Dataset, texts: list[str], model: str, folder: Path = FIXTURES) -> list[str] | None:
    """Replay recorded teacher labels for every text, or None when any one is missing."""
    t = Teacher(ds.labels, Cache(folder / ds.name), model)
    out = []
    for x in texts:
        try:
            y = t.label(x)
        except KeyError:
            return None
        out.append(y or "__unparsed__")
    return out


def p50_latency_ms(student: Student, texts: list[str]) -> float:
    times = []
    for x in texts:
        t = time.perf_counter()
        student.predict([x])
        times.append(time.perf_counter() - t)
    return float(np.median(times) * 1000)


def label_noise(labels: list[str], classes: list[str], rate: float, seed: int) -> list[str]:
    """Swap `rate` of the labels for a different random class. Simulates a noisy teacher."""
    rng = random.Random(seed)
    out = list(labels)
    for i in rng.sample(range(len(out)), round(rate * len(out))):
        out[i] = rng.choice([c for c in classes if c != out[i]])
    return out


def eval_student(student: Student, ds: data.Dataset, seed: int, prices: dict, llm: dict) -> dict:
    pred, conf = student.predict(ds.test.texts)
    lat = p50_latency_ms(student, ds.test.texts[:300])
    row = metrics.score(ds.test.labels, pred, seed=seed)
    row.update(
        {
            "p50_ms": lat,
            "llm_share": 0.0,
            "usd_per_10k": costs.student_cost_per_10k(lat / 1000, prices),
        }
    )
    # Hybrid: threshold from dev only.
    dpred, dconf = student.predict(ds.dev.texts)
    pick = router.pick_threshold(ds.dev.labels, dpred, dconf, TARGET)
    kept = router.route(pred, conf, pick["threshold"])
    share = 1 - sum(kept) / len(kept)
    kept_right = sum(1 for k, g, p in zip(kept, ds.test.labels, pred, strict=True) if k and g == p)
    hybrid = {
        "threshold": pick["threshold"],
        "target": TARGET,
        "dev_kept_share": pick["kept_share"],
        "dev_kept_accuracy": pick["kept_accuracy"],
        "llm_share": share,
        "kept": sum(kept),
        "kept_correct": kept_right,
        "kept_accuracy_ci": list(metrics.wilson(kept_right, sum(kept))),
        "upper": metrics.score(
            ds.test.labels, router.hybrid_predictions(ds.test.labels, pred, kept, True), seed=seed
        ),
        "lower": metrics.score(
            ds.test.labels, router.hybrid_predictions(ds.test.labels, pred, kept, False), seed=seed
        ),
        "usd_per_10k": row["usd_per_10k"] + share * llm["usd_per_10k"],
        "p50_ms": lat if share < 0.5 else None,
        "dev_sweep": router.sweep(ds.dev.labels, dpred, dconf)[::5],
    }
    return {"student": row, "hybrid": hybrid}


def run(name: str = "banking77", seed: int = 0, budget: int = BUDGET) -> dict:
    cores = pin_cores()
    prices = costs.load_prices()
    ds = data.load(name, seed)
    train = data.sample(ds.train, min(budget, len(ds.train)), seed)
    teach = teacher_labels(ds, train.texts, prices["llm"]["model"])
    if teach is None:
        labels, source = train.labels, "gold (the teacher step did not run, so this shows training, not distillation)"
    else:
        labels, source = teach, f"teacher {prices['llm']['model']} from the fixture cache"
    llm_sample = data.sample(ds.test, min(BASELINE_SAMPLE, len(ds.test)), seed)
    llm = costs.llm_cost_per_10k(ds.test.texts, ds.labels, prices)
    embedder, embed_note = make_embedder()
    rows = {}
    for kind in ("tfidf", "embed"):
        s = Student(kind, seed=seed, embedder=embedder if kind == "embed" else None)
        s.fit(train.texts, labels)
        out = eval_student(s, ds, seed, prices, llm)
        if teach is not None:
            out["student"]["teacher_agreement"] = metrics.agreement(labels, train.labels)
        rows[s.name] = out
    return {
        "meta": {
            "date": dt.date.today().isoformat(),
            "dataset": name,
            "seed": seed,
            "cpu": cpu_model(),
            "cores": cores,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "budget": len(train),
            "labels": source,
            "embedder": embed_note,
            "train": len(ds.train),
            "dev": len(ds.dev),
            "test": len(ds.test),
            "classes": len(ds.labels),
            "prices": prices,
        },
        "llm_only": {
            "status": "not run: record mode blocked (no provider key, provider hosts refused)",
            "planned_sample": len(llm_sample),
            "llm_share": 1.0,
            **llm,
        },
        "rows": rows,
    }


def curve(name: str = "banking77", seeds=CURVE_SEEDS, sizes=SIZES) -> dict:
    """Test accuracy and macro-F1 against the number of labeled training examples, plus noise."""
    cores = pin_cores()
    ds = data.load(name, 0)
    out = {"meta": {"date": dt.date.today().isoformat(), "dataset": name, "seeds": list(seeds),
                    "cpu": cpu_model(), "cores": cores, "labels": "gold (the teacher step did not run)",
                    "embedder": make_embedder()[1]}, "points": [], "noise": []}
    for kind in ("tfidf", "embed"):
        for n in [*sizes, len(ds.train)]:
            for sd in seeds:
                tr = data.sample(ds.train, n, sd)
                s = Student(kind, seed=sd).fit(tr.texts, tr.labels)
                p, _ = s.predict(ds.test.texts)
                sc = metrics.score(ds.test.labels, p, sd)
                out["points"].append({**sc, "test_n": sc["n"], "student": s.name, "n": n, "seed": sd})
        for rate in NOISE:
            for sd in seeds:
                tr = data.sample(ds.train, BUDGET, sd)
                noisy = label_noise(tr.labels, ds.labels, rate, sd)
                s = Student(kind, seed=sd).fit(tr.texts, noisy)
                p, _ = s.predict(ds.test.texts)
                out["noise"].append({**metrics.score(ds.test.labels, p, sd), "student": s.name, "rate": rate, "seed": sd})
    return out


def timeit(kind: str, name: str = "banking77") -> dict:
    """Train one student on the full training split and report wall time and peak memory."""
    cores = pin_cores()
    ds = data.load(name, 0)
    t = time.perf_counter()
    Student(kind).fit(ds.train.texts, ds.train.labels)
    secs = time.perf_counter() - t
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return {"student": kind, "n": len(ds.train), "seconds": secs, "peak_mb": peak_mb, "cores": cores}


def timings(name: str = "banking77") -> dict:
    out = {"meta": {"date": dt.date.today().isoformat(), "cpu": cpu_model(), "dataset": name,
                    "how": "fresh child process per student, pinned to cores 0 and 1, full training split"},
           "students": []}
    for kind in ("tfidf", "embed"):
        cmd = [sys.executable, "-m", "distillroute.cli", "timeit", "--student", kind, "--dataset", name]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=True)  # noqa: S603
        out["students"].append(json.loads(res.stdout))
    return out


def write(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
