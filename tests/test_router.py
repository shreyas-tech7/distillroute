import numpy as np

from distillroute import bench, costs, data, router


def test_pick_threshold_lowest_that_meets_target():
    gold = ["a"] * 10
    pred = ["a"] * 8 + ["b"] * 2
    conf = np.array([0.9] * 8 + [0.3, 0.2])
    pick = router.pick_threshold(gold, pred, conf, target=0.6)
    assert 0.3 < pick["threshold"] <= 0.9
    assert pick["kept_accuracy"] == 1.0 and pick["kept_share"] == 0.8 and pick["met"]


def test_threshold_rule_uses_the_lower_bound():
    """Broken case: the rule took point accuracy, so 8 right of 8 counted as meeting 95%."""
    gold = ["a"] * 10
    pred = ["a"] * 8 + ["b"] * 2
    conf = np.array([0.9] * 8 + [0.3, 0.2])
    pick = router.pick_threshold(gold, pred, conf, target=0.95)
    assert not pick["met"]  # Wilson lower bound of 8/8 is about 0.68
    many = router.pick_threshold(["a"] * 200, ["a"] * 200, np.full(200, 0.9), target=0.95)
    assert many["met"] and many["kept_accuracy_lo"] >= 0.95


def test_hybrid_bounds():
    gold = ["a", "b", "c", "d"]
    pred = ["a", "x", "c", "y"]
    kept = [True, True, False, False]
    classes = ["a", "b", "c", "d", "x", "y"]
    assert router.hybrid_predictions(gold, pred, kept, True, classes) == ["a", "x", "c", "d"]
    low = router.hybrid_predictions(gold, pred, kept, False, classes)
    assert low == ["a", "x", "d", "y"]  # wrong real classes, never the gold one
    assert all(lo in classes for lo in low)


def test_macro_f1_lower_bound_is_not_above_the_student():
    """Broken case: a placeholder label added no false positives, so the "lower" macro-F1 came out
    above the student alone."""
    from distillroute import metrics

    ds = data.load("synthetic", 0)
    from distillroute.students import Student

    s = Student("tfidf").fit(ds.train.texts, ds.train.labels)
    pred, conf = s.predict(ds.test.texts)
    kept = router.route(conf, 0.5)
    low = router.hybrid_predictions(ds.test.labels, pred, kept, False, ds.labels)
    assert metrics.macro_f1(ds.test.labels, low) <= metrics.macro_f1(ds.test.labels, pred)


def test_threshold_is_chosen_on_dev_not_test(monkeypatch):
    """Broken case: eval_student passed test labels to pick_threshold."""
    ds = data.load("synthetic", 0)
    seen = []
    real = router.pick_threshold

    def spy(gold, pred, conf, target):
        seen.append(list(gold))
        return real(gold, pred, conf, target)

    monkeypatch.setattr(router, "pick_threshold", spy)
    from distillroute.students import Student

    s = Student("tfidf").fit(ds.train.texts, ds.train.labels)
    prices = costs.load_prices()
    bench.eval_student(s, ds, 0, prices, costs.llm_cost_per_10k(ds.test.texts[:5], ds.labels, prices))
    assert seen == [ds.dev.labels]
    assert ds.test.labels not in seen


def test_label_noise_rate():
    labels = ["a"] * 100
    noisy = bench.label_noise(labels, ["a", "b", "c"], 0.2, 0)
    assert sum(x != "a" for x in noisy) == 20


def test_cost_model_uses_price_table():
    prices = costs.load_prices()
    llm = costs.llm_cost_per_10k(["my card is late"], ["card_arrival", "card_linking"], prices)
    assert llm["usd_per_10k"] > 0 and llm["mean_input_tokens"] > 10
    assert costs.compute_cost(60, prices) == prices["compute"]["usd_per_minute"]
