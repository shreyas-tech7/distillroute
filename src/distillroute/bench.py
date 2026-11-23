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
DEV_BUDGET = 300
BASELINE_SAMPLE = 300
TARGET = 0.95
CORES = (0, 1)


def pin_cores() -> list[int]:
    """Run on cores 0 and 1 only, so every timing means "on 2 cores".

    numpy starts its BLAS worker threads at import, before this runs, and a thread keeps the
    affinity it had. So every thread of the process is pinned, not only the main one.
    """
    try:
        cores = set(CORES) & set(range(os.cpu_count() or 1)) or {0}
        for tid in os.listdir("/proc/self/task"):
            try:
                os.sched_setaffinity(int(tid), cores)
            except OSError:
                pass
        os.sched_setaffinity(0, cores)
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return []


def thread_cores() -> list[int]:
    """The union of cores any thread of this process may run on. Used to check the pinning."""
    seen: set[int] = set()
    try:
        for tid in os.listdir("/proc/self/task"):
            seen |= os.sched_getaffinity(int(tid))
    except (AttributeError, OSError):
        return []
    return sorted(seen)


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


_MISSING = object()


def model_dir(model: str) -> str:
    """The folder name for one teacher model's recordings."""
    return model.replace("/", "__").replace(":", "_")


def teacher_labels(
    ds: data.Dataset, texts: list[str], model: str, folder: Path = FIXTURES
) -> list[str | None] | None:
    """Replay recorded labels for every text. None when nothing is recorded for these texts.

    A partial recording is an error, never a quiet fallback to gold. An answer that names no known
    label comes back as None, and the caller drops that row.
    """
    t = Teacher(ds.labels, Cache(folder / ds.name / model_dir(model)), model)
    got: list = []
    for x in texts:
        try:
            got.append(t.label(x))
        except KeyError:
            got.append(_MISSING)
    missing = sum(1 for y in got if y is _MISSING)
    if missing == len(texts):
        return None
    if missing:
        raise RuntimeError(f"{missing} of {len(texts)} teacher labels are not recorded for {model}")
    return got


def recorded_models(ds: data.Dataset, folder: Path = FIXTURES) -> list[str]:
    """Models with a recording folder for this dataset. Each folder holds model.txt with the name."""
    base = folder / ds.name
    return sorted(p.read_text().strip() for p in base.glob("*/model.txt")) if base.exists() else []


def latency_ms(student: Student, texts: list[str]) -> tuple[float, float]:
    """p50 and mean time to answer one ticket at a time."""
    times = []
    for x in texts:
        t = time.perf_counter()
        student.predict([x])
        times.append(time.perf_counter() - t)
    return float(np.median(times) * 1000), float(np.mean(times) * 1000)


def label_split(ds: data.Dataset, split: data.Split, models: list[str], folder: Path = FIXTURES):
    """Labels for a split from the first recorded teacher that covers it, else gold.

    Returns (rows kept, their labels, model or None). Rows the teacher answered with an unknown
    label are dropped, so they never train or tune anything.
    """
    for model in models:
        got = teacher_labels(ds, split.texts, model, folder)
        if got is None:
            continue
        keep = [i for i, y in enumerate(got) if y is not None]
        sub = data.Split([split.texts[i] for i in keep], [split.labels[i] for i in keep])
        return sub, [got[i] for i in keep], model
    return split, list(split.labels), None


def budget_splits(ds: data.Dataset, budget: int, dev_budget: int | None, seed: int):
    """The training sample and the dev sample that the label budget pays for. None means all of dev."""
    train = data.sample(ds.train, min(budget, len(ds.train)), seed)
    dev = ds.dev if dev_budget is None else data.sample(ds.dev, min(dev_budget, len(ds.dev)), seed)
    return train, dev


def label_noise(labels: list[str], classes: list[str], rate: float, seed: int) -> list[str]:
    """Swap `rate` of the labels for a different random class. Simulates a noisy teacher."""
    rng = random.Random(seed)  # noqa: S311 (seeded sampling, not crypto)
    out = list(labels)
    for i in rng.sample(range(len(out)), round(rate * len(out))):
        out[i] = rng.choice([c for c in classes if c != out[i]])
    return out


def eval_student(
    student: Student,
    ds: data.Dataset,
    seed: int,
    prices: dict,
    llm: dict,
    dev: data.Split | None = None,
    dev_labels: list[str] | None = None,
) -> dict:
    """Score on test. The hybrid threshold comes from `dev` scored against `dev_labels`, which are
    the same kind of labels the student trained on (teacher or gold). Test labels never tune it."""
    dev = dev if dev is not None else ds.dev
    dev_labels = dev_labels if dev_labels is not None else dev.labels
    pred, conf = student.predict(ds.test.texts)
    lat, mean_ms = latency_ms(student, ds.test.texts[:300])
    row = metrics.score(ds.test.labels, pred, seed=seed)
    row.update(
        {
            "p50_ms": lat,
            "mean_ms": mean_ms,
            "llm_share": 0.0,
            "usd_per_10k": costs.student_cost_per_10k(mean_ms / 1000, prices),
        }
    )
    # Hybrid: threshold from dev only.
    dpred, dconf = student.predict(dev.texts)
    pick = router.pick_threshold(dev_labels, dpred, dconf, TARGET)
    kept = router.route(conf, pick["threshold"])
    share = 1 - sum(kept) / len(kept)
    kept_right = sum(1 for k, g, p in zip(kept, ds.test.labels, pred, strict=True) if k and g == p)
    hybrid = {
        "threshold": pick["threshold"],
        "target": TARGET,
        "dev_kept_share": pick["kept_share"],
        "dev_kept_accuracy": pick["kept_accuracy"],
        "dev_kept_accuracy_lo": pick["kept_accuracy_lo"],
        "dev_target_met": pick["met"],
        "llm_share": share,
        "kept": sum(kept),
        "kept_correct": kept_right,
        "kept_accuracy": kept_right / sum(kept) if sum(kept) else None,
        "kept_accuracy_ci": list(metrics.wilson(kept_right, sum(kept))),
        "test_target_met": bool(sum(kept)) and kept_right / sum(kept) >= TARGET,
        "upper": metrics.score(
            ds.test.labels, router.hybrid_predictions(ds.test.labels, pred, kept, True, ds.labels), seed=seed
        ),
        "lower": metrics.score(
            ds.test.labels, router.hybrid_predictions(ds.test.labels, pred, kept, False, ds.labels), seed=seed
        ),
        "usd_per_10k": costs.student_cost_per_10k(mean_ms / 1000, prices) + share * llm["usd_per_10k"],
        "p50_ms": lat if share < 0.5 else None,
        "dev_sweep": router.sweep(dev_labels, dpred, dconf)[::5],
    }
    return {"student": row, "hybrid": hybrid}


def run(
    name: str = "banking77",
    seed: int = 0,
    budget: int = BUDGET,
    dev_budget: int | None = DEV_BUDGET,
    folder: Path = FIXTURES,
) -> dict:
    """Train both students on `budget` labels, tune the router on `dev_budget` labels, test once.

    Both label sets come from the same source: a recorded teacher when every text has a recording
    under `folder`, else gold. `record` writes recordings for exactly these two samples.
    """
    cores = pin_cores()
    prices = costs.load_prices()
    ds = data.load(name, seed)
    train0, dev0 = budget_splits(ds, budget, dev_budget, seed)
    train, labels, model = label_split(ds, train0, recorded_models(ds, folder), folder)
    dev, dev_labels, dev_model = label_split(ds, dev0, [model] if model else [], folder)
    if model is None:
        source = "gold (the teacher step did not run, so this shows training, not distillation)"
    elif dev_model is None:
        raise RuntimeError(f"teacher {model} labeled the training sample but not the dev sample")
    else:
        source = f"teacher {model} from the fixture cache"
    llm = costs.llm_cost_per_10k(ds.test.texts, ds.labels, prices)
    embedder, embed_note = make_embedder()
    rows = {}
    for kind in ("tfidf", "embed"):
        s = Student(kind, seed=seed, embedder=embedder if kind == "embed" else None)
        s.fit(train.texts, labels)
        out = eval_student(s, ds, seed, prices, llm, dev, dev_labels)
        if model is not None:
            out["student"]["teacher_agreement"] = metrics.agreement(labels, train.labels)
            out["student"]["teacher_dropped"] = len(train0) - len(train)
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
            "budget": len(train0),
            "dev_labels": len(dev0),
            "label_count": len(train0) + len(dev0),
            "teacher_model": model,
            "train_test_overlap_dropped": ds.dropped,
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
            "planned_sample": min(BASELINE_SAMPLE, len(ds.test)),
            "llm_share": 1.0,
            **llm,
        },
        "rows": rows,
    }


def curve(name: str = "banking77", seeds=CURVE_SEEDS, sizes=SIZES) -> dict:
    """Test accuracy and macro-F1 against the number of labeled training examples, plus noise."""
    cores = pin_cores()
    ds = data.load(name, 0)
    out = {
        "meta": {
            "date": dt.date.today().isoformat(),
            "dataset": name,
            "seeds": list(seeds),
            "cpu": cpu_model(),
            "cores": cores,
            "labels": "gold (the teacher step did not run)",
            "embedder": "hash embedder (the curve always uses it)",
        },
        "points": [],
        "noise": [],
    }
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
                out["noise"].append(
                    {**metrics.score(ds.test.labels, p, sd), "student": s.name, "rate": rate, "seed": sd}
                )
    return out


def timeit(kind: str, name: str = "banking77") -> dict:
    """Train one student on the full training split and report wall time and peak memory."""
    cores = pin_cores()
    ds = data.load(name, 0)
    t = time.perf_counter()
    Student(kind).fit(ds.train.texts, ds.train.labels)
    secs = time.perf_counter() - t
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return {
        "student": kind,
        "n": len(ds.train),
        "seconds": secs,
        "peak_mb": peak_mb,
        "cores": cores,
        "thread_cores": thread_cores(),
    }


def _pin_child() -> None:
    """Runs in the child before exec, so numpy's threads start on cores 0 and 1 already."""
    try:
        os.sched_setaffinity(0, set(CORES) & set(range(os.cpu_count() or 1)) or {0})
    except (AttributeError, OSError):
        pass


def timings(name: str = "banking77") -> dict:
    out = {
        "meta": {
            "date": dt.date.today().isoformat(),
            "cpu": cpu_model(),
            "dataset": name,
            "how": (
                "fresh child process per student, pinned to cores 0 and 1 before numpy starts its threads, "
                "BLAS threads capped at 2, full training split"
            ),
        },
        "students": [],
    }
    for kind in ("tfidf", "embed"):
        cmd = [sys.executable, "-m", "distillroute.cli", "timeit", "--student", kind, "--dataset", name]
        env = {**os.environ, "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
        res = subprocess.run(  # noqa: S603 (fixed argv, no shell)
            cmd, capture_output=True, text=True, timeout=600, check=True, env=env, preexec_fn=_pin_child
        )
        out["students"].append(json.loads(res.stdout))
    return out


def write(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
