import pytest

from distillroute import metrics


def test_accuracy_and_macro_f1_known_values():
    gold = ["a", "a", "a", "b"]
    pred = ["a", "a", "a", "a"]
    assert metrics.accuracy(gold, pred) == 0.75
    # F1(a) = 6/7, F1(b) = 0, so macro-F1 = 3/7. Accuracy hides the dead class, macro-F1 does not.
    assert metrics.macro_f1(gold, pred) == pytest.approx(3 / 7)


def test_wilson_interval():
    lo, hi = metrics.wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)
    assert metrics.wilson(0, 0) == (0.0, 0.0)


def test_bootstrap_interval_holds_point_and_is_seeded():
    gold = ["a", "b", "c"] * 40
    pred = ["a", "b", "b"] * 40
    f1 = metrics.macro_f1(gold, pred)
    lo, hi = metrics.bootstrap_f1(gold, pred, seed=1)
    assert lo <= f1 <= hi
    assert (lo, hi) == metrics.bootstrap_f1(gold, pred, seed=1)


def test_bootstrap_counts_out_of_set_predictions_as_errors():
    """Broken case: a prediction outside the gold labels was counted as a hit for some class."""
    gold = ["a", "b"] * 50
    pred = ["zzz", "b"] * 50
    lo, hi = metrics.bootstrap_f1(gold, pred, seed=0)
    assert hi <= metrics.macro_f1(gold, ["b", "b"] * 50) + 0.2
    assert metrics.macro_f1(gold, pred) == pytest.approx(0.5)
