from pathlib import Path

import pytest

from distillroute import bench, data, teacher

FIX = Path(__file__).parent / "fixtures" / "teacher"


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


def test_recorded_labels_feed_the_students():
    ds = data.load("synthetic", 0)
    tr = data.sample(ds.train, 40, 0)
    got = bench.teacher_labels(ds, tr.texts, "fixture-model", folder=FIX.parent / "x")
    assert got is None  # wrong folder, so nothing is recorded
    t = teacher.Teacher(ds.labels, teacher.Cache(FIX), "fixture-model")
    labels = [t.label(x) for x in tr.texts]
    agree = sum(a == b for a, b in zip(labels, tr.labels, strict=True)) / 40
    assert agree == 0.9


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


def test_http_send_without_network_is_blocked():
    """The conftest guard stops the call, and record mode reports it as blocked, never as a label."""
    with pytest.raises((teacher.RecordBlocked, RuntimeError)):
        teacher.http_send("groq", teacher.build_request("x", ["a"], "m"), timeout=2, tries=1)
