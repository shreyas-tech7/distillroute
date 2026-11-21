"""Command line: distillroute <command>."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from functools import partial

from . import bench, data, synth, teacher

ROOT = bench.ROOT


def cmd_synth(a) -> int:
    n = synth.write(ROOT / "data" / "synthetic" / "tickets.csv", a.seed)
    print(f"wrote {n} synthetic tickets")
    return 0


def cmd_hashes(a) -> int:
    res = data.check_hashes()
    for rel, ok in res.items():
        print("ok  " if ok else "BAD ", rel)
    return 0 if all(res.values()) else 1


def cmd_run(a) -> int:
    for name in a.dataset:
        res = bench.run(name, a.seed, a.budget)
        bench.write(res, ROOT / "results" / f"{name}-b{res['meta']['budget']}-seed{a.seed}.json")
        for k, v in res["rows"].items():
            s, h = v["student"], v["hybrid"]
            print(
                f"{name} {k}: acc {s['accuracy']:.3f} f1 {s['macro_f1']:.3f} "
                f"hybrid llm share {h['llm_share']:.2f} lower {h['lower']['accuracy']:.3f} upper {h['upper']['accuracy']:.3f}"
            )
    return 0


def cmd_curve(a) -> int:
    res = bench.curve(a.dataset)
    bench.write(res, ROOT / "results" / f"curve-{a.dataset}.json")
    print(f"wrote {len(res['points'])} curve points and {len(res['noise'])} noise points")
    return 0


def cmd_timings(a) -> int:
    res = bench.timings(a.dataset)
    bench.write(res, ROOT / "results" / "timings.json")
    for s in res["students"]:
        print(f"{s['student']}: {s['seconds']:.1f} s, peak {s['peak_mb']:.0f} MB, cores {s['cores']}")
    return 0


def cmd_timeit(a) -> int:
    print(json.dumps(bench.timeit(a.student, a.dataset)))
    return 0


def cmd_record(a) -> int:
    """Label the budgeted sample through a live provider. Writes the status either way."""
    prices = json.loads((ROOT / "data" / "prices.json").read_text())
    ds = data.load(a.dataset, 0)
    model = teacher.PROVIDERS[a.provider][2] if a.provider != "gemini" else prices["llm"]["model"]
    t = teacher.Teacher(
        ds.labels,
        teacher.Cache(bench.FIXTURES / ds.name),
        model,
        send=partial(teacher.http_send, a.provider),
        record=True,
    )
    texts = data.sample(ds.train, a.budget, 0).texts
    status = {
        "date": dt.date.today().isoformat(),
        "provider": a.provider,
        "model": model,
        "requested": len(texts),
    }
    try:
        for x in texts:
            t.raw(x)
        status.update({"result": "recorded", "calls": t.calls, "cache_hits": t.hits})
        code = 0
    except teacher.RecordBlocked as e:
        status.update({"result": "blocked", "error": str(e), "calls": t.calls})
        code = 2
    path = ROOT / "results" / "record-status.json"
    allst = json.loads(path.read_text()) if path.exists() else {}
    allst[a.provider] = status
    bench.write(allst, path)
    print(json.dumps(status))
    return code


def cmd_readme(a) -> int:
    from . import readme

    return readme.main(check=a.check)


def cmd_demo(a) -> int:
    """A fast run on the synthetic domain that needs no network and no results files."""
    res = bench.run("synthetic", 0, 360)
    for k, v in res["rows"].items():
        s, h = v["student"], v["hybrid"]
        print(
            f"synthetic {k}: accuracy {s['accuracy']:.3f} macro-F1 {s['macro_f1']:.3f}, "
            f"threshold {h['threshold']:.2f} sends {100 * h['llm_share']:.0f}% to the LLM, "
            f"hybrid accuracy between {h['lower']['accuracy']:.3f} and {h['upper']['accuracy']:.3f}"
        )
    print(f"LLM only: {res['llm_only']['status']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="distillroute", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth", help="write the synthetic domain")
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(fn=cmd_synth)
    sub.add_parser("hashes", help="check data file hashes").set_defaults(fn=cmd_hashes)
    s = sub.add_parser("run", help="train, route and evaluate")
    s.add_argument("--dataset", nargs="+", default=["banking77", "synthetic"])
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--budget", type=int, default=bench.BUDGET)
    s.set_defaults(fn=cmd_run)
    s = sub.add_parser("curve", help="learning curve and label noise sweep")
    s.add_argument("--dataset", default="banking77")
    s.set_defaults(fn=cmd_curve)
    s = sub.add_parser("timings", help="train time and peak memory on 2 cores")
    s.add_argument("--dataset", default="banking77")
    s.set_defaults(fn=cmd_timings)
    s = sub.add_parser("timeit", help=argparse.SUPPRESS)
    s.add_argument("--student", required=True)
    s.add_argument("--dataset", default="banking77")
    s.set_defaults(fn=cmd_timeit)
    s = sub.add_parser("record", help="record teacher labels from a live provider")
    s.add_argument("--provider", choices=sorted(teacher.PROVIDERS), default="groq")
    s.add_argument("--dataset", default="banking77")
    s.add_argument("--budget", type=int, default=bench.BUDGET)
    s.set_defaults(fn=cmd_record)
    s = sub.add_parser("readme", help="write the README tables and docs/index.html from results")
    s.add_argument("--check", action="store_true", help="fail if a generated file is out of date")
    s.set_defaults(fn=cmd_readme)
    sub.add_parser("demo", help="fast offline run on the synthetic domain").set_defaults(fn=cmd_demo)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
