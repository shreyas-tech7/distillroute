import numpy as np

from distillroute import bench, costs, data, router


def test_pick_threshold_lowest_that_meets_target():
    gold = ["a"] * 10
    pred = ["a"] * 8 + ["b"] * 2
    conf = np.array([0.9] * 8 + [0.3, 0.2])
    pick = router.pick_threshold(gold, pred, conf, target=0.95)
    assert 0.3 < pick["threshold"] <= 0.9
    assert pick["kept_accuracy"] == 1.0 and pick["kept_share"] == 0.8


def test_hybrid_bounds():
    gold = ["a", "b", "c", "d"]
    pred = ["a", "x", "c", "y"]
    kept = [True, True, False, False]
    assert router.hybrid_predictions(gold, pred, kept, True) == ["a", "x", "c", "d"]
    assert router.hybrid_predictions(gold, pred, kept, False) == [
        "a",
        "x",
        router.LLM_WRONG,
        router.LLM_WRONG,
    ]


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


def test_label_noise_rate():
    labels = ["a"] * 100
    noisy = bench.label_noise(labels, ["a", "b", "c"], 0.2, 0)
    assert sum(x != "a" for x in noisy) == 20


def test_cost_model_uses_price_table():
    prices = costs.load_prices()
    llm = costs.llm_cost_per_10k(["my card is late"], ["card_arrival", "card_linking"], prices)
    assert llm["usd_per_10k"] > 0 and llm["mean_input_tokens"] > 10
    assert costs.compute_cost(60, prices) == prices["compute"]["usd_per_minute"]
