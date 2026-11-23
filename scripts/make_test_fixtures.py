"""Write the fake teacher recordings under tests/fixtures/teacher. Run it again to check them.

They stand in for a real recording in the tests: 40 training answers and 20 dev answers for the
synthetic domain, seed 0, model "fixture-model". Every tenth answer is a wrong label, so teacher
agreement is 0.9, and dev answer 5 names no label, so the run must drop that row.
"""

from __future__ import annotations

import sys
from pathlib import Path

from distillroute import bench, data, teacher

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "teacher"
MODEL = "fixture-model"


def answers() -> dict[str, bytes]:
    ds = data.load("synthetic", 0)
    train, dev = bench.budget_splits(ds, 40, 20, 0)
    out = {}
    for part, split in (("train", train), ("dev", dev)):
        for i, (x, y) in enumerate(zip(split.texts, split.labels, strict=True)):
            if part == "dev" and i == 5:
                label = "I am not sure"
            elif i % 10 == 9:
                label = ds.labels[(ds.labels.index(y) + 1) % len(ds.labels)]
            else:
                label = y
            key = teacher.request_hash(teacher.build_request(x, ds.labels, MODEL))
            out[key] = teacher.fake_response(label)
    return out


def main() -> int:
    folder = OUT / "synthetic" / bench.model_dir(MODEL)
    folder.mkdir(parents=True, exist_ok=True)
    for key, raw in answers().items():
        (folder / f"{key}.json").write_bytes(raw)
    (folder / "model.txt").write_text(MODEL + "\n")
    print(f"wrote {len(answers())} fixtures to {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
