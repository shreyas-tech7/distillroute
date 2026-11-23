from pathlib import Path

from distillroute import data, synth


def test_hashes_match_sources():
    """Broken case: a data file changed after download, so its hash no longer matches SOURCES.json."""
    assert all(data.check_hashes().values())


def test_banking77_sizes():
    ds = data.banking77()
    assert len(ds.labels) == 77
    assert len(ds.train) + len(ds.dev) == 10003 - 7  # 7 official train texts repeat in test
    assert len(ds.test) == 3080


def test_dev_comes_from_train_and_test_never_leaks():
    """Broken case: the dev split or the training sample holds test texts."""
    ds = data.banking77()
    test = {data.norm(t) for t in ds.test.texts}
    norm = lambda s: {data.norm(t) for t in s.texts}  # noqa: E731
    assert not test & norm(ds.dev) and not test & norm(ds.train)
    assert not set(ds.dev.texts) - set(data.read_csv(data.DATA / "banking77" / "train.csv").texts)
    assert not set(ds.dev.texts) & set(ds.train.texts)
    budget = data.sample(ds.train, 800, 0)
    assert not test & norm(budget)


def test_overlap_drop_catches_the_official_duplicates():
    """Broken case: the raw official files share 7 texts, so training on them leaks the test set."""
    raw = data.read_csv(data.DATA / "banking77" / "train.csv")
    test = data.read_csv(data.DATA / "banking77" / "test.csv")
    _, dropped = data.drop_overlap(raw, test)
    assert dropped == 7


def test_synthetic_splits_are_disjoint():
    ds = data.synthetic()
    tr, dv, te = (list(zip(s.texts, s.labels, strict=True)) for s in (ds.train, ds.dev, ds.test))
    assert len(tr) + len(dv) + len(te) == 600
    assert not set(te) & set(tr) and not set(te) & set(dv) and not set(dv) & set(tr)


def test_sample_is_seeded():
    ds = data.synthetic()
    assert data.sample(ds.train, 50, 3).texts == data.sample(ds.train, 50, 3).texts
    assert data.sample(ds.train, 50, 3).texts != data.sample(ds.train, 50, 4).texts


def test_synthetic_domain_shape(tmp_path: Path):
    """Broken case: the generator drifts, so the committed CSV no longer matches its seed."""
    rows = synth.generate(7)
    assert len(rows) == 600
    assert len({t for t, _ in rows}) == 600
    counts = {c: sum(1 for _, y in rows if y == c) for c in synth.CATEGORIES}
    assert len(counts) == 12 and set(counts.values()) == {50}
    out = tmp_path / "t.csv"
    synth.write(out, 7)
    assert out.read_bytes() == (data.DATA / "synthetic" / "tickets.csv").read_bytes()
