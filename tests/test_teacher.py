from pathlib import Path

import pytest

from distillroute import bench, data, teacher

ROOT_FIX = Path(__file__).parent / "fixtures" / "teacher"
FIX = ROOT_FIX / "synthetic" / "fixture-model"


def test_request_hash_is_stable_and_order_free():
    a = teacher.build_request("hi", ["x", "y"], "m")
    b = dict(reversed(list(a.items())))
    assert teacher.request_hash(a) == teacher.request_hash(b)
    assert teacher.request_hash(a) != teacher.request_hash(teacher.build_request("hi!", ["x", "y"], "m"))


def test_replay_is_byte_for_byte():
    """Broken case: the cache parsed and rewrote JSON, so replayed bytes differed from the recording."""
    ds = data.load("synthetic", 0)
    tr = data.sample(ds.train, 40, 0)
    t = teacher.Teacher(ds.labels, teacher.Cache(FIX), "fixture-model")
    for x in tr.texts:
        key = teacher.request_hash(teacher.build_request(x, ds.labels, "fixture-model"))
        assert t.raw(x) == (FIX / f"{key}.json").read_bytes()
    assert t.hits == 40 and t.calls == 0


def test_nothing_recorded_means_none():
    ds = data.load("synthetic", 0)
    tr = data.sample(ds.train, 40, 0)
    assert bench.teacher_labels(ds, tr.texts, "fixture-model", folder=ROOT_FIX / "x") is None
    assert bench.teacher_labels(ds, tr.texts, "another-model", folder=ROOT_FIX) is None


def test_run_trains_on_recorded_teacher_labels():
    """Broken case: record and run hashed with different model names, so a recording never replayed
    and the run quietly trained on gold."""
    res = bench.run("synthetic", 0, 40, 20, folder=ROOT_FIX)
    assert res["meta"]["teacher_model"] == "fixture-model"
    assert res["meta"]["labels"].startswith("teacher fixture-model")
    assert res["meta"]["label_count"] == 60
    for row in res["rows"].values():
        assert row["student"]["teacher_agreement"] == 0.9
        assert row["student"]["teacher_dropped"] == 0


def test_run_without_recordings_says_gold(tmp_path):
    res = bench.run("synthetic", 0, 40, 20, folder=tmp_path)
    assert res["meta"]["teacher_model"] is None and res["meta"]["labels"].startswith("gold")


def test_unparsed_dev_answer_is_dropped(monkeypatch):
    """Broken case: an answer that named no label became a fake class that trained the student."""
    seen = []
    real = bench.router.pick_threshold

    def spy(gold, pred, conf, target):
        seen.append(len(gold))
        return real(gold, pred, conf, target)

    monkeypatch.setattr(bench.router, "pick_threshold", spy)
    bench.run("synthetic", 0, 40, 20, folder=ROOT_FIX)
    assert seen == [19, 19]


def test_partial_recording_is_an_error(tmp_path):
    """Broken case: a half-finished recording quietly fell back to gold labels."""
    ds = data.load("synthetic", 0)
    tr = data.sample(ds.train, 40, 0)
    folder = tmp_path / "synthetic" / "fixture-model"
    folder.mkdir(parents=True)
    for p in sorted(FIX.glob("*.json"))[:5]:
        (folder / p.name).write_bytes(p.read_bytes())
    with pytest.raises(RuntimeError):
        bench.teacher_labels(ds, tr.texts, "fixture-model", folder=tmp_path)


def test_record_and_run_agree_on_the_requests(tmp_path):
    """Broken case: record used seed 0 and the provider model, run used the run seed and the price
    model, so their request hashes never matched."""
    from distillroute import cli

    sent = []

    def fake_send(provider, req, **_):
        sent.append(req)
        return teacher.fake_response(req["messages"][1]["content"].splitlines()[1])  # the first label

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(teacher, "http_send", fake_send)
    monkeypatch.setattr(bench, "FIXTURES", tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "prices.json").write_text((bench.ROOT / "data" / "prices.json").read_text())
    try:
        args = "record --provider groq --dataset synthetic --seed 1 --budget 30 --dev-budget 10"
        assert cli.main(args.split()) == 0
    finally:
        monkeypatch.undo()
    assert len(sent) == 40
    res = bench.run("synthetic", 1, 30, 10, folder=tmp_path)
    assert res["meta"]["teacher_model"] == teacher.PROVIDERS["groq"][2]


def test_request_hash_is_the_same_in_a_new_process():
    import subprocess
    import sys

    req = teacher.build_request("my card is late", ["card_arrival"], "m")
    code = "from distillroute import teacher as t;print(t.request_hash(t.build_request('my card is late',['card_arrival'],'m')))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)  # noqa: S603
    assert out.stdout.strip() == teacher.request_hash(req)


def test_replay_miss_never_calls_out():
    """Broken case: a cache miss in replay mode fell through to a live provider call."""
    called = []
    t = teacher.Teacher(["a"], teacher.Cache(Path("/nonexistent")), "m", send=lambda r: called.append(r))
    with pytest.raises(KeyError):
        t.raw("new ticket")
    assert called == []


def test_record_writes_once(tmp_path):
    t = teacher.Teacher(
        ["a", "b"], teacher.Cache(tmp_path), "m", send=lambda r: teacher.fake_response("b"), record=True
    )
    assert t.label("x") == "b"
    assert t.label("x") == "b"
    assert t.calls == 1 and t.hits == 1
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize(
    ("content", "want"),
    [
        ("card_arrival", "card_arrival"),
        ("`Card_Arrival`.", "card_arrival"),
        ("Label: card_arrival", "card_arrival"),
        ("no idea", None),
    ],
)
def test_parse_label(content, want):
    import json

    raw = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
    assert teacher.parse_label(raw, ["card_arrival", "card_linking"]) == want
    assert teacher.parse_label(b"not json", ["a"]) is None


def test_http_send_without_key_is_blocked(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(teacher.RecordBlocked):
        teacher.http_send("groq", teacher.build_request("x", ["a"], "m"), timeout=2, tries=1)


class _Opener:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def open(self, req, timeout):
        self.calls += 1
        raise self.errors.pop(0)


def _http_error(code, headers=None):
    import email.message
    import urllib.error

    h = email.message.Message()
    for k, v in (headers or {}).items():
        h[k] = v
    return urllib.error.HTTPError("https://example.invalid", code, "x", h, None)


def test_http_send_refused_host_is_blocked(monkeypatch):
    """Broken case: a 403 from a proxy was retried or treated as an answer."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    op = _Opener([_http_error(403)])
    with pytest.raises(teacher.RecordBlocked):
        teacher.http_send("groq", teacher.build_request("x", ["a"], "m"), tries=3, opener=op)
    assert op.calls == 1


def test_http_send_honors_retry_after(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    waits = []
    monkeypatch.setattr(teacher.time, "sleep", waits.append)
    op = _Opener([_http_error(429, {"Retry-After": "7"}), _http_error(429)])
    with pytest.raises(teacher.RecordBlocked):
        teacher.http_send("groq", teacher.build_request("x", ["a"], "m"), tries=2, opener=op)
    assert waits == [7.0, 4.0]


def test_redirects_are_refused():
    """Broken case: urllib followed a redirect and sent the Authorization header to another host."""
    import urllib.request

    h = teacher._NoRedirect()
    req = urllib.request.Request("https://api.groq.com/x", headers={"Authorization": "Bearer k"})
    assert h.redirect_request(req, None, 302, "Found", {}, "https://evil.invalid/") is None
