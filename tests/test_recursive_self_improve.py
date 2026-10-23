"""The recursion that evolves the rule moving its own budget (#384 follow-on).

`self_improve` moves the next tick's budget with the fixed `portfolio_report`;
`portfolio_evolve` evolves that rule but never runs the loop. These fixtures run
them alternately and pin that the inner trail and the policy decision both
survive, and that a cold policy round is reported rather than fatal.
"""

from __future__ import annotations

import pytest

from agentdescent import (
    ConsolidationStore,
    LayerBudget,
    RecursiveRound,
    RecursiveSelfImproveReport,
    SelfImproveConfig,
    SourceSlot,
    recursive_self_improve,
    self_improve,
)
from agentdescent.meta import MetaOutcome


def _spec():
    return SourceSlot(initial_value="0")


def _specs():
    return {"selection": _spec(), "task_sampler": _spec()}


def _problem(value, seed):
    r = int(str(value)) / 10.0
    return MetaOutcome(curve=[r], final=r, rollouts=1)


def _val_problem(value, seed):
    r = int(str(value)) / 10.0
    return MetaOutcome(curve=[r], final=r, rollouts=1)


def _domains():
    return {"d": ({"t0": _problem, "t1": _problem},
                  {"v0": _val_problem, "v1": _val_problem})}


def _windows():
    def row(name, ret):
        return LayerBudget(name=name, budget=10.0, spent=5.0,
                           return_=ret, samples=5)
    return [row("meta:selection", 0.4), row("meta:task_sampler", 0.25)]


def _improving(rendered, task, output, reward):
    return "5"


def _policy(rendered, task, output, reward):
    return "grow: 1.00"


def test_the_inner_trail_and_the_policy_decision_both_survive(tmp_path, monkeypatch):
    store = ConsolidationStore(str(tmp_path / "archive"))
    cfg = SelfImproveConfig(
        slots=["selection", "task_sampler"], specs=_specs(),
        domains=_domains(), ticks=2, top_k=2, config={"dataset": "recursive"},
        run_id_prefix="r1")
    import agentdescent.recursive as rec
    real = rec.self_improve
    seen = []

    def spy(store, config, **kwargs):
        seen.append(config.reallocation)
        return real(store, config, **kwargs)

    monkeypatch.setattr(rec, "self_improve", spy)
    report = recursive_self_improve(
        store, cfg, propose=_improving, policy_propose=_policy,
        outer_rounds=2,
        policy_kwargs={"windows": [_windows(), _windows()],
                       "val_windows": [_windows()],
                       "seeds": (0, 1), "val_seeds": (2, 3),
                       "rounds": 1, "n_workers": 1, "repo_path": None})
    assert isinstance(report, RecursiveSelfImproveReport)
    assert len(report.rounds) == 2 and all(
        isinstance(r, RecursiveRound) for r in report.rounds)
    assert report.rounds[0].improve.improved
    assert report.rounds[0].policy_verdict == "adopt"
    assert report.rounds[0].policy_candidate
    assert report.policy.grow == 1.0
    # the next inner round really received the adopted rule
    assert len(seen) == 2 and seen[1] is not None and seen[1].grow == 1.0
    assert report.total_gain > 0 and report.total_policy_gain >= 0


def test_round_one_budget_matches_self_improve(tmp_path):
    cfg = SelfImproveConfig(
        slots=["selection"], specs=_specs(), domains=_domains(), ticks=1,
        top_k=1, config={"dataset": "same"}, run_id_prefix="plain")
    plain = self_improve(
        ConsolidationStore(str(tmp_path / "a")), cfg, propose=_improving)
    recursive = recursive_self_improve(
        ConsolidationStore(str(tmp_path / "b")), cfg, propose=_improving,
        policy_propose=_policy, outer_rounds=1)
    assert plain.ticks[0].budget_after == recursive.rounds[0].improve.ticks[0].budget_after
    assert recursive.rounds[0].policy_verdict == "reject"   # one record: no windows


def test_a_cold_policy_round_is_reported_not_fatal(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    cfg = SelfImproveConfig(
        slots=["selection"], specs=_specs(), domains=_domains(), ticks=0,
        top_k=1, config={"dataset": "cold"}, run_id_prefix="r2")
    report = recursive_self_improve(
        store, cfg, propose=_improving, policy_propose=_policy,
        outer_rounds=1)
    assert report.rounds[0].policy_verdict == "reject"
    assert "windows" in report.rounds[0].reason


def test_it_requires_a_policy_reflector(tmp_path):
    cfg = SelfImproveConfig(slots=["selection"], specs=_specs(),
                            domains=_domains())
    with pytest.raises(ValueError, match="policy_propose"):
        recursive_self_improve(
            ConsolidationStore(str(tmp_path / "a")), cfg,
            propose=_improving, outer_rounds=1)
    with pytest.raises(ValueError, match="outer_rounds"):
        recursive_self_improve(
            ConsolidationStore(str(tmp_path / "b")), cfg,
            propose=_improving, policy_propose=_policy, outer_rounds=0)
