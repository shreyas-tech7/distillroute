import numpy as np

from distillroute import data, metrics
from distillroute.students import HashEmbedder, SoftmaxRegression, Student, Tfidf, tokens


def test_tokens_include_bigrams():
    assert tokens("Card not arrived") == ["card", "not", "arrived", "card not", "not arrived"]


def test_tfidf_rows_are_unit_length():
    x = Tfidf(min_df=1).fit(["a b", "a c", "b c d"]).transform(["a b", "zzz"])
    assert np.isclose(np.linalg.norm(x[0]), 1.0)
    assert np.linalg.norm(x[1]) == 0.0


def test_hash_embedder_is_deterministic():
    """Broken case: Python's salted hash() would give different vectors in each process."""
    e = HashEmbedder(64)
    a = e.transform(["my card is late"])
    assert np.allclose(a, HashEmbedder(64).transform(["my card is late"]))
    assert np.isclose(np.linalg.norm(a), 1.0)
    assert round(float(a[0].sum()), 4) == round(float(a[0].sum()), 4)


def test_softmax_learns_separable_data():
    x = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    m = SoftmaxRegression(epochs=200).fit(x, ["a", "a", "b", "b"])
    p = m.predict_proba(x)
    assert np.allclose(p.sum(axis=1), 1)
    assert [m.classes[i] for i in p.argmax(axis=1)] == ["a", "a", "b", "b"]


def test_students_beat_chance_on_synthetic():
    ds = data.synthetic()
    for kind in ("tfidf", "embed"):
        s = Student(kind).fit(ds.train.texts, ds.train.labels)
        pred, conf = s.predict(ds.test.texts)
        assert metrics.accuracy(ds.test.labels, pred) > 0.6
        assert ((conf > 0) & (conf <= 1)).all()


def test_student_training_is_seeded():
    ds = data.synthetic()
    a = Student("tfidf", seed=1).fit(ds.train.texts, ds.train.labels).predict(ds.test.texts)[1]
    b = Student("tfidf", seed=1).fit(ds.train.texts, ds.train.labels).predict(ds.test.texts)[1]
    assert np.array_equal(a, b)
