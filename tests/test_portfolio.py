"""Portfolio governance has to be right where the framework spends on itself.

A layer is a spending line (limit, spend, measured return, sample count); the
portfolio ranks by return-per-cost and shifts the next window's budget
deterministically: a paying layer grows, a non-paying one shrinks, a quiet one
is held, and none is ever zeroed. The fixtures pin the surfaces that make this
honest: deterministic reallocation, the hold floor on evidence, the cap on
growth, the diminishing flag, finite-input guards, and the archive aggregation
that turns rule records into layer rows.
"""

import pytest

from agentdescent import (
    ConsolidationStore,
    EvolutionResult,
    LayerBudget,
    PortfolioReport,
    layers_from_archive,
    portfolio_report,
    problem_descriptor,
)


def _layer(name, budget, spent, return_, samples, **kw):
    return LayerBudget(name=name, budget=budget, spent=spent, return_=return_,
                       samples=samples, **kw)


def test_reallocation_grows_shrinks_and_holds():
    good = _layer("meta:sampler", budget=100, spent=10, return_=1.0, samples=5)
    bad = _layer("meta:selection", budget=100, spent=100, return_=0.1, samples=5)
    quiet = _layer("meta:acceptance", budget=50, spent=50, return_=0.5, samples=1)
    report = portfolio_report([good, bad, quiet], floor=0.05, min_samples=3,
                              grow=0.5, shrink=0.5)
    assert isinstance(report, PortfolioReport)
    # good: rpc 0.1 >= floor, 5 samples -> grows 50%
    assert report.reallocated["meta:sampler"] == pytest.approx(150)
    # bad: rpc 0.001 < floor -> shrinks 50%
    assert report.reallocated["meta:selection"] == pytest.approx(50)
    # quiet: 1 < 3 samples -> held exactly
    assert report.reallocated["meta:acceptance"] == pytest.approx(50)
    assert "grown" in report.reasons["meta:sampler"]
    assert "shrunk" in report.reasons["meta:selection"]
    assert "held" in report.reasons["meta:acceptance"]


def test_no_layer_is_ever_zeroed():
    bad = _layer("meta:selection", budget=10, spent=10, return_=0.0, samples=9)
    report = portfolio_report([bad], floor=0.05, shrink=0.9)
    assert report.reallocated["meta:selection"] >= 0.05
    assert report.reallocated["meta:selection"] > 0
    # the floor is 5% of the layer's own budget, not an absolute near-zero value
    assert report.reallocated["meta:selection"] == pytest.approx(1.0)


def test_min_budget_is_a_real_absolute_floor():
    bad = _layer("meta:selection", budget=100, spent=100, return_=0.0, samples=9)
    report = portfolio_report([bad], floor=0.05, shrink=0.9, min_budget=30)
    assert report.reallocated["meta:selection"] == pytest.approx(30)


def test_a_layer_with_no_spend_is_held_not_shrunk():
    unspent = _layer("meta:sampler", budget=100, spent=0, return_=0.5, samples=5)
    report = portfolio_report([unspent], floor=0.05, shrink=0.5)
    assert report.reallocated["meta:sampler"] == pytest.approx(100)
    assert "held" in report.reasons["meta:sampler"]


def test_the_cap_limits_growth():
    good = _layer("meta:sampler", budget=100, spent=10, return_=1.0, samples=5)
    report = portfolio_report([good], floor=0.05, grow=1.0, cap=120)
    assert report.reallocated["meta:sampler"] == pytest.approx(120)


def test_a_diminishing_flag_shrinks_even_a_paying_layer():
    good = _layer("meta:sampler", budget=100, spent=10, return_=1.0, samples=9,
                  diminishing=True)
    report = portfolio_report([good], floor=0.05, shrink=0.5)
    assert report.reallocated["meta:sampler"] == pytest.approx(50)
    assert "diminishing" in report.reasons["meta:sampler"]


def test_non_finite_inputs_are_refused():
    with pytest.raises(ValueError):
        portfolio_report([_layer("bad", budget=100, spent=10, return_=float("nan"),
                                  samples=5)])
    with pytest.raises(ValueError):
        portfolio_report([_layer("bad", budget=float("inf"), spent=10,
                                  return_=0.5, samples=5)])


def test_layers_from_archive_aggregates_per_slot(tmp_path):
    from agentdescent.evolution import RoundInfo
    store = ConsolidationStore(str(tmp_path / "archive"))
    for slot, c, reward, rounds in [("task_sampler", 0.3, 0.8, 5),
                                    ("task_sampler", 0.4, 0.9, 6),
                                    ("selection", 0.5, 0.7, 4)]:
        history = [RoundInfo(round=i, held_out_reward=reward, n_items=1,
                             committed=1, rejected=0)
                   for i in range(rounds)]
        store.archive(
            EvolutionResult(state={"c": f"{c:.1f}"},
                            rendered=f"# Policy parameters\nc: {c:.1f}",
                            final_reward=reward, history=history,
                            ledger_log=[]),
            run_id=f"{slot}-{c}", strategy="ParamSlot",
            config={"dataset": "x"}, artifact_id="meta-rule",
            problem=problem_descriptor(slot, rounds=rounds, n_workers=1))
    layers = layers_from_archive(store)
    by_name = {l.name: l for l in layers}
    assert set(by_name) == {"meta:task_sampler", "meta:selection"}
    sampler = by_name["meta:task_sampler"]
    assert sampler.samples == 2
    assert sampler.return_ == pytest.approx(0.85)      # mean of 0.8, 0.9
    assert sampler.spent == pytest.approx(5.5)         # mean rounds (5 + 6) / 2
    assert sampler.budget == pytest.approx(11)         # total rounds this window
    assert sampler.return_sd == pytest.approx(0.05)    # sd of 0.8, 0.9


def test_rpc_does_not_decay_as_a_slot_accumulates_rules(tmp_path):
    from agentdescent.evolution import RoundInfo
    store = ConsolidationStore(str(tmp_path / "archive"))
    for c, reward, rounds in [("a", 0.8, 6), ("b", 0.8, 6), ("c", 0.8, 6)]:
        history = [RoundInfo(round=i, held_out_reward=reward, n_items=1,
                             committed=1, rejected=0) for i in range(rounds)]
        store.archive(
            EvolutionResult(state={"c": c}, rendered=f"c: {c}",
                            final_reward=reward, history=history, ledger_log=[]),
            run_id=f"r-{c}", strategy="ParamSlot", config={"dataset": "x"},
            artifact_id="meta-rule",
            problem=problem_descriptor("task_sampler", rounds=rounds, n_workers=1))
    [layer] = layers_from_archive(store)
    # 3 rules, 6 rounds each: rpc is the per-rule ratio, unchanged by N
    assert layer.spent == pytest.approx(6.0)
    assert layer.return_ == pytest.approx(0.8)
    assert layer.budget == pytest.approx(18.0)
    assert (layer.return_ / layer.spent) == pytest.approx(0.8 / 6.0)


def test_empty_archive_yields_no_layers(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    assert layers_from_archive(store) == []
