"""The results files meet the brief, and the README matches them."""

import json
import re
from pathlib import Path

from distillroute import bench, readme

ROOT = Path(__file__).resolve().parents[1]
GEN = re.compile(r"<!-- gen:(\w+) -->.*?<!-- /gen:\1 -->", re.S)
FENCE = re.compile(r"```.*?```", re.S)


def load(rel):
    return json.loads((ROOT / rel).read_text())


def test_readme_matches_results():
    """Broken case: a number typed into the README by hand, or results changed without `readme`."""
    assert readme.main(check=True) == 0
    res = load(readme.MAIN)
    s = res["rows"]["tfidf"]["student"]
    assert readme.ci(s["accuracy"], s["accuracy_ci"]) in (ROOT / "README.md").read_text()


def test_no_numbers_typed_outside_generated_blocks():
    """Broken case: a result percent written into the prose."""
    for rel in ("README.md", "docs/INTERVIEW_QA.md", "docs/WALKTHROUGH.md"):
        text = GEN.sub("", FENCE.sub("", (ROOT / rel).read_text()))
        assert set(re.findall(r"\d+(?:\.\d+)?%", text)) <= {"95%", "15%"}, rel


def test_main_table_has_every_row_and_metric():
    """Done-means 3: LLM only, each student and the hybrid on the BANKING77 test split."""
    res = load(readme.MAIN)
    assert res["meta"]["dataset"] == "banking77" and res["meta"]["test"] == 3080
    assert res["llm_only"]["usd_per_10k"] > 0 and res["llm_only"]["llm_share"] == 1.0
    for key in ("tfidf", "embed-hash"):
        s, h = res["rows"][key]["student"], res["rows"][key]["hybrid"]
        assert s["n"] == 3080
        assert s["accuracy_ci"][0] <= s["accuracy"] <= s["accuracy_ci"][1]
        assert s["macro_f1_ci"][0] <= s["macro_f1"] <= s["macro_f1_ci"][1]
        assert s["p50_ms"] > 0 and s["usd_per_10k"] > 0
        assert h["lower"]["accuracy"] <= h["upper"]["accuracy"]
        assert 0 <= h["llm_share"] <= 1


def test_curve_comes_from_real_runs():
    """Done-means 4: points at 100, 250, 500 and 800, for both students and three seeds."""
    cur = load(readme.CURVE)
    for student in ("tfidf", "embed-hash"):
        ns = {p["n"] for p in cur["points"] if p["student"] == student}
        assert set(bench.SIZES) <= ns
        assert len([p for p in cur["points"] if p["student"] == student and p["n"] == 800]) == 3
    assert {p["rate"] for p in cur["noise"]} == set(bench.NOISE)


def test_timing_under_budget():
    """Done-means 2: each student trains in under 2 minutes on 2 cores, peak memory under 2 GB."""
    tim = load(readme.TIMING)
    assert {s["student"] for s in tim["students"]} == {"tfidf", "embed"}
    for s in tim["students"]:
        assert s["cores"] == [0, 1]
        assert s["seconds"] < 120 and s["peak_mb"] < 2048


def test_record_status_reported():
    """Done-means 5 fallback: a blocked record run says why, and the README says so."""
    rec = load(readme.RECORD)
    assert rec and all(v["result"] in ("recorded", "blocked") for v in rec.values())
    if all(v["result"] == "blocked" for v in rec.values()):
        assert "gold" in load(readme.MAIN)["meta"]["labels"]
        assert "The teacher did not run" in (ROOT / "README.md").read_text()
