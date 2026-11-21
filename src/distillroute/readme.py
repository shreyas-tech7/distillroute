"""Write the number blocks in the README and the docs, and docs/index.html, from the results files.

Nobody types a number into the README. Each block sits between `<!-- gen:NAME -->` and
`<!-- /gen:NAME -->`, and this module rewrites it. `--check` fails when a block is stale.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
MAIN = "results/banking77-b800-seed0.json"
SECOND = "results/banking77-b800-seed1.json"
FULL = "results/banking77-b8498-seed0.json"
SYNTH = "results/synthetic-b360-seed0.json"
CURVE = "results/curve-banking77.json"
TIMING = "results/timings.json"
RECORD = "results/record-status.json"
FILES = ["README.md", "docs/INTERVIEW_QA.md", "docs/WALKTHROUGH.md"]
BLOCK = re.compile(r"(<!-- gen:(\w+) -->\n)(.*?)(<!-- /gen:\2 -->)", re.S)
NAMES = {"tfidf": "TF-IDF + logistic regression", "embed-hash": "Hash embedding + logistic regression"}


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text())


def p(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def ci(x: float, lo_hi: list[float]) -> str:
    return f"{p(x)} ({p(lo_hi[0])} to {p(lo_hi[1])})"


def usd(x: float) -> str:
    return f"${x:.6f}" if x < 0.01 else f"${x:.2f}"


def table(head: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * len(head)) + " |"] + [
        "| " + " | ".join(r) + " |" for r in rows
    ]


def ordered(res: dict) -> list:
    return [(k, res["rows"][k]) for k in NAMES if k in res["rows"]]


def main_rows(res: dict) -> list[list[str]]:
    llm = res["llm_only"]
    rows = [
        [
            "LLM only (" + res["meta"]["prices"]["llm"]["model"] + ")",
            "not run",
            "not run",
            usd(llm["usd_per_10k"]) + " (estimate)",
            "not measured",
            "100%",
        ]
    ]
    for key, v in ordered(res):
        s = v["student"]
        rows.append(
            [
                NAMES[key],
                ci(s["accuracy"], s["accuracy_ci"]),
                ci(s["macro_f1"], s["macro_f1_ci"]),
                usd(s["usd_per_10k"]),
                f"{s['p50_ms']:.2f} ms",
                "0%",
            ]
        )
    for key, v in ordered(res):
        h = v["hybrid"]
        lo, hi = h["lower"], h["upper"]
        lat = f"{h['p50_ms']:.2f} ms" if h["p50_ms"] is not None else "LLM latency (not measured)"
        rows.append(
            [
                f"Hybrid, {NAMES[key]} first",
                f"{p(lo['accuracy'])} to {p(hi['accuracy'])} (bounds)",
                f"{p(lo['macro_f1'])} to {p(hi['macro_f1'])} (bounds)",
                usd(h["usd_per_10k"]) + " (estimate)",
                lat,
                p(h["llm_share"]),
            ]
        )
    return rows


HEAD = [
    "System",
    "Accuracy (95% CI)",
    "Macro-F1 (95% CI)",
    "Cost per 10,000 tickets",
    "p50 latency",
    "Sent to LLM",
]


def hybrid_rows(res: dict) -> list[list[str]]:
    out = []
    for key, v in ordered(res):
        h = v["hybrid"]
        out.append(
            [
                NAMES[key],
                f"{h['threshold']:.2f}",
                p(h["dev_kept_accuracy"]),
                p(1 - h["llm_share"]),
                f"{h['kept_correct']}/{h['kept']} = {ci(h['kept_correct'] / max(h['kept'], 1), h['kept_accuracy_ci'])}",
            ]
        )
    return out


def meta_line(res: dict, rel: str) -> str:
    m = res["meta"]
    return (
        f"Run of {m['date']}, seed {m['seed']}, {m['budget']} labeled training texts, labels: {m['labels']}. "
        f"Students: numpy, {m['embedder']}. Scored on the full test split ({m['test']} tickets, {m['classes']} intents). "
        f"CPU: {m['cpu']}, cores {m['cores']}. File `{rel}`."
    )


def curve_summary(cur: dict) -> dict:
    out: dict = {}
    for pt in cur["points"]:
        out.setdefault(pt["student"], {}).setdefault(pt["n"], []).append(pt)
    return out


def curve_rows(cur: dict) -> list[list[str]]:
    rows = []
    for student, by_n in curve_summary(cur).items():
        for n, pts in sorted(by_n.items()):
            accs = [x["accuracy"] for x in pts]
            f1s = [x["macro_f1"] for x in pts]
            rows.append(
                [
                    NAMES[student],
                    str(n),
                    p(mean(accs)),
                    f"{p(min(accs))} to {p(max(accs))}",
                    p(mean(f1s)),
                    f"{p(min(f1s))} to {p(max(f1s))}",
                ]
            )
    return rows


def noise_rows(cur: dict) -> list[list[str]]:
    by: dict = {}
    for pt in cur["noise"]:
        by.setdefault((pt["student"], pt["rate"]), []).append(pt)
    return [
        [NAMES[s], p(r, 0), p(mean(x["accuracy"] for x in pts)), p(mean(x["macro_f1"] for x in pts))]
        for (s, r), pts in sorted(by.items(), key=lambda kv: (list(NAMES).index(kv[0][0]), kv[0][1]))
    ]


def why(status: dict) -> str:
    err = status.get("error", "")
    if "403" in err and "proxy" in err:
        return "the network proxy refused the host"
    if "HTTP" in err:
        return f"the host answered {err.split('answered ')[-1]} with no key"
    return err or f"{status.get('calls', 0)} calls"


def blocks() -> dict[str, str]:
    res, sec, full, syn = load(MAIN), load(SECOND), load(FULL), load(SYNTH)
    cur, tim, rec = load(CURVE), load(TIMING), load(RECORD)
    pr = res["meta"]["prices"]
    llm = res["llm_only"]
    b = {}
    b["main"] = "\n".join(table(HEAD, main_rows(res)) + ["", meta_line(res, MAIN)])
    b["hybrid"] = "\n".join(
        table(
            [
                "Student first",
                "Threshold (from dev)",
                "Kept accuracy on dev",
                "Kept share on test",
                "Kept accuracy on test (95% CI)",
            ],
            hybrid_rows(res),
        )
    )
    b["second"] = "\n".join(table(HEAD, main_rows(sec)) + ["", meta_line(sec, SECOND)])
    b["full"] = "\n".join(table(HEAD, main_rows(full)) + ["", meta_line(full, FULL)])
    b["synthetic"] = "\n".join(table(HEAD, main_rows(syn)) + ["", meta_line(syn, SYNTH)])
    m = cur["meta"]
    b["curve"] = "\n".join(
        table(
            [
                "Student",
                "Labeled examples",
                "Accuracy, mean",
                "Accuracy, range",
                "Macro-F1, mean",
                "Macro-F1, range",
            ],
            curve_rows(cur),
        )
        + [
            "",
            f"Run of {m['date']}, seeds {m['seeds']}, labels: {m['labels']}. Each point trains "
            f"on a fresh seeded sample and scores on the full test split. The largest size is the whole "
            f"training split. File `{CURVE}`.",
        ]
    )
    b["noise"] = "\n".join(
        table(["Student", "Wrong labels", "Accuracy, mean", "Macro-F1, mean"], noise_rows(cur))
        + [
            "",
            f"Training sets of 800 with a share of labels swapped for a random wrong intent, "
            f"seeds {m['seeds']}. This simulates a noisy teacher. It is not a real teacher. File `{CURVE}`.",
        ]
    )
    b["timing"] = "\n".join(
        table(
            ["Student", "Training texts", "Train time", "Peak memory", "Cores"],
            [
                [
                    NAMES.get(s["student"], NAMES["embed-hash"] if s["student"] == "embed" else s["student"]),
                    str(s["n"]),
                    f"{s['seconds']:.1f} s",
                    f"{s['peak_mb']:.0f} MB",
                    str(s["cores"]),
                ]
                for s in tim["students"]
            ],
        )
        + [
            "",
            f"Run of {tim['meta']['date']} on {tim['meta']['cpu']}, {tim['meta']['how']}. File `{TIMING}`.",
        ]
    )
    b["cost"] = (
        f"LLM price: `{pr['llm']['model']}` at ${pr['llm']['input_usd_per_million']} per million input tokens "
        f"and ${pr['llm']['output_usd_per_million']} per million output tokens ({pr['llm']['tier']}), read on "
        f"{pr['llm']['read_on']} from {pr['llm']['source']}. Tokens are estimated at {pr['chars_per_token']} "
        f"characters per token. One call has about {llm['mean_input_tokens']:.0f} input tokens, and the 77 label "
        f"names are most of them. Student price: the {pr['compute']['name']} at "
        f"${pr['compute']['usd_per_minute']} per minute, read on {pr['compute']['read_on']} from "
        f"{pr['compute']['source']}, times the measured p50 time per ticket. Labeling the {res['meta']['budget']} "
        f"training texts once costs about {usd(llm['usd_per_10k'] * res['meta']['budget'] / 10_000)} at the same "
        f"LLM price. These are comparison prices, not a bill."
    )
    rr = " ".join(f"{k} ({v['model']}): {v['result']}, {why(v)}." for k, v in sorted(rec.items()))
    b["record"] = f"Record mode on {next(iter(rec.values()))['date']}. {rr} File `{RECORD}`."
    src = json.loads((ROOT / "data" / "SOURCES.json").read_text())["datasets"]["banking77"]
    b["dataset"] = (
        f"BANKING77 holds {res['meta']['train'] + res['meta']['dev'] + 7 + res['meta']['test']} customer "
        f"service queries in {res['meta']['classes']} intents. It is licensed under {src['license']}. "
        f"Attribution: {src['attribution']} Source: {src['source']}. Downloaded on {src['downloaded_on']}, "
        f"unchanged, with SHA-256 hashes in `data/SOURCES.json`. The official train file repeats 7 test texts, "
        f"and distillroute drops those 7 before it splits off the development set."
    )
    best_key = max(res["rows"], key=lambda k: res["rows"][k]["student"]["accuracy"])
    best = res["rows"][best_key]["student"]
    fb = full["rows"]["tfidf"]
    b["headline"] = (
        f"With {res['meta']['budget']} labels, the best student gets {p(best['accuracy'])} accuracy on the "
        f"BANKING77 test split, and the hybrid has to send {p(res['rows']['embed-hash']['hybrid']['llm_share'], 0)} "
        f"of tickets to the LLM to keep the student's share at the {p(res['rows']['tfidf']['hybrid']['target'], 0)} "
        f"target. With all {full['meta']['budget']} training labels, TF-IDF gets {p(fb['student']['accuracy'])} "
        f"and the hybrid sends {p(fb['hybrid']['llm_share'], 0)}."
    )
    return b


def render(text: str, b: dict[str, str]) -> str:
    def sub(m):
        name = m.group(2)
        return m.group(1) + b[name] + "\n" + m.group(4) if name in b else m.group(0)

    return BLOCK.sub(sub, text)


def svg_curve(cur: dict, width: int = 640, height: int = 320) -> str:
    colors = {"tfidf": "#2f6fdb", "embed-hash": "#d9822b"}
    summ = curve_summary(cur)
    ns = sorted({n for by in summ.values() for n in by})
    x0, x1, y0, y1 = 60, width - 20, height - 40, 20
    import math

    lx = [math.log(n) for n in ns]

    def X(n):
        return x0 + (math.log(n) - lx[0]) / (lx[-1] - lx[0]) * (x1 - x0)

    def Y(a):
        return y0 - a * (y0 - y1)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Learning curve">',
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="currentColor"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="currentColor"/>',
    ]
    for t in (0.25, 0.5, 0.75, 1.0):
        parts.append(
            f'<text x="{x0 - 8}" y="{Y(t) + 4:.0f}" text-anchor="end" font-size="11">{int(t * 100)}%</text>'
        )
    for n in ns:
        parts.append(f'<text x="{X(n):.0f}" y="{y0 + 16}" text-anchor="middle" font-size="11">{n}</text>')
    for i, (student, by) in enumerate(summ.items()):
        pts = [(X(n), Y(mean(x["accuracy"] for x in by[n]))) for n in sorted(by)]
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        c = colors.get(student, "#555")
        parts.append(f'<polyline points="{path}" fill="none" stroke="{c}" stroke-width="2"/>')
        parts += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{c}"/>' for x, y in pts]
        parts.append(
            f'<text x="{x0 + 10}" y="{y1 + 14 + 14 * i}" font-size="12" fill="{c}">{html.escape(NAMES[student])}</text>'
        )
    parts.append(
        f'<text x="{(x0 + x1) / 2:.0f}" y="{height - 6}" text-anchor="middle" font-size="11">labeled training examples (log scale)</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def svg_sweep(res: dict, key: str, width: int = 640, height: int = 280) -> str:
    sw = res["rows"][key]["hybrid"]["dev_sweep"]
    x0, x1, y0, y1 = 60, width - 20, height - 40, 20
    pts = [(x0 + r["kept_share"] * (x1 - x0), y0 - r["kept_accuracy"] * (y0 - y1)) for r in sw]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return "\n".join(
        [
            f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Threshold sweep">',
            f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="currentColor"/>',
            f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="currentColor"/>',
            f'<polyline points="{path}" fill="none" stroke="#2f6fdb" stroke-width="2"/>',
            f'<text x="{(x0 + x1) / 2:.0f}" y="{height - 6}" text-anchor="middle" font-size="11">share the student keeps (dev split)</text>',
            f'<text x="14" y="{(y0 + y1) / 2:.0f}" font-size="11" transform="rotate(-90 14 {(y0 + y1) / 2:.0f})" text-anchor="middle">accuracy on kept tickets</text>',
            f'<text x="{x0 - 8}" y="{y1 + 4}" text-anchor="end" font-size="11">100%</text>',
            f'<text x="{x0 - 8}" y="{y0 + 4}" text-anchor="end" font-size="11">0%</text>',
            f'<text x="{x0}" y="{y0 + 16}" text-anchor="middle" font-size="11">0%</text>',
            f'<text x="{x1}" y="{y0 + 16}" text-anchor="middle" font-size="11">100%</text>',
            "</svg>",
        ]
    )


def md_table_html(md: str) -> str:
    lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    if not lines:
        return ""
    cells = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines]
    head, body = cells[0], cells[2:]
    out = [
        "<table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in head) + "</tr></thead><tbody>"
    ]
    out += ["<tr>" + "".join(f"<td>{html.escape(c.strip('`'))}</td>" for c in r) + "</tr>" for r in body]
    return "\n".join(out + ["</tbody></table>"])


def note(md: str) -> str:
    rest = [ln for ln in md.splitlines() if ln and not ln.startswith("|")]
    return "".join(f"<p class=note>{html.escape(ln)}</p>" for ln in rest)


def index_html(b: dict[str, str]) -> str:
    res, cur = load(MAIN), load(CURVE)
    sections = [
        ("main", "BANKING77 test split, 800 labels"),
        ("hybrid", "How the hybrid threshold was picked"),
        ("full", "Same pipeline with every training label"),
        ("curve", "Learning curve"),
        ("noise", "Simulated teacher noise"),
        ("timing", "Train time and memory on 2 cores"),
        ("synthetic", "Synthetic small-business domain"),
    ]
    body = []
    for key, title in sections:
        body.append(f'<h2 id="{key}">{html.escape(title)}</h2>')
        if key == "curve":
            body.append(svg_curve(cur))
        if key == "hybrid":
            body.append(svg_sweep(res, "tfidf"))
        body.append(md_table_html(b[key]) + note(b[key]))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>distillroute results</title>
<style>
:root {{ color-scheme: light dark; --fg:#1b1f24; --bg:#ffffff; --muted:#57606a; --line:#d0d7de; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg:#e6edf3; --bg:#0d1117; --muted:#9aa4ae; --line:#30363d; }} }}
body {{ font: 15px/1.5 system-ui, sans-serif; color: var(--fg); background: var(--bg); max-width: 980px; margin: 0 auto; padding: 16px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; display: block; overflow-x: auto; }}
th, td {{ border: 1px solid var(--line); padding: 4px 8px; text-align: left; white-space: nowrap; }}
.note {{ color: var(--muted); font-size: 13px; }}
svg {{ width: 100%; max-width: 640px; color: var(--fg); }} svg text {{ fill: currentColor; }}
</style></head><body>
<h1>distillroute results</h1>
<p>Generated by <code>distillroute readme</code> from the files in <code>results/</code>. Open this file from disk. It loads nothing from the network.</p>
<p>{html.escape(b["headline"])}</p>
<p class=note>{html.escape(b["record"])}</p>
{chr(10).join(body)}
<h2 id="cost">Cost model</h2><p class=note>{html.escape(b["cost"])}</p>
<h2 id="data">Data</h2><p class=note>{html.escape(b["dataset"])}</p>
</body></html>
"""


def main(check: bool = False) -> int:
    b = blocks()
    stale = []
    targets = {rel: render((ROOT / rel).read_text(), b) for rel in FILES if (ROOT / rel).exists()}
    targets["docs/index.html"] = index_html(b)
    for rel, new in targets.items():
        path = ROOT / rel
        old = path.read_text() if path.exists() else ""
        if old != new:
            stale.append(rel)
            if not check:
                path.write_text(new)
    if check and stale:
        print("stale generated files:", ", ".join(stale), "(run `distillroute readme`)")
        return 1
    print("generated files are up to date" if check else f"wrote {len(stale)} files")
    return 0
