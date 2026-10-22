"""Portfolio evolution has to be right where the framework evolves its own spend.

The reallocation rule is four numbers; the evolved policy must be the *shipped*
rule with different numbers (not a second implementation), a window is a set of
layer rows, and the evolved rule is measured against the fixed baseline on
windows it never trained on. The fixtures pin the surfaces that make this
honest: the default policy equals `portfolio_report`, the projected return
rewards growing the layers that pay, the evolved policy beats the fixed baseline
on unseen windows, an empty archive is refused, and an out-of-bounds proposal is
refused rather than committed.
"""

import pytest

from agentdescent import (
    ConsolidationStore,
    EvolutionResult,
    LayerBudget,
    ReallocationPolicy,
    portfolio_evolve,
    portfolio_problem,
    portfolio_report,
    problem_descriptor,
    reallocation_spec,
    windows_from_archive,
)
from agentdescent.evolution import RoundInfo


def _layer(name, budget, spent, return_, samples):
    return LayerBudget(name=name, budget=budget, spent=spent, return_=return_,
                       samples=samples)


def _window(high=1.0, mid=0.5, low=0.02):
    return [_layer("hi", 10, 10, high, 5),
            _layer("mid", 10, 10, mid, 5),
            _layer("lo", 10, 10, low, 5)]


# -- the policy IS the shipped rule --------------------------------------------


def test_the_default_policy_is_the_shipped_rule():
    rows = [_layer("pay", 10, 10, 1.0, 5), _layer("flat", 10, 10, 0.02, 5),
            _layer("thin", 10, 10, 0.5, 1), _layer("idle", 10, 0, 0.5, 5)]
    assert ReallocationPolicy().apply(rows) == portfolio_report(rows).reallocated


def test_the_projected_return_rewards_growing_the_payers():
    rows = _window()
    default = portfolio_problem(rows)(ReallocationPolicy(), 0).final
    aggressive = portfolio_problem(rows)(
        ReallocationPolicy(grow=2.0), 0).final
    assert aggressive > default


# -- the loop ------------------------------------------------------------------


def test_portfolio_evolve_beats_the_fixed_baseline_on_unseen_windows():
    train = [_window(), _window(high=0.9), _window(high=1.0, low=0.05)]
    val = [_window(high=0.95), _window(high=1.0)]

    def propose(rendered, task, output, reward):
        return "grow: 2.00"          # grow the paying layers harder

    report, result = portfolio_evolve(
        windows=train, val_windows=val, propose=propose,
        seeds=(0, 1, 2), val_seeds=(3, 4), rounds=1, n_workers=1,
        repo_path=None)
    assert report["verdict"] == "adopt"
    assert report["val_gain"] > 0
    assert "grow: 2.0" in report["value"]
    assert result.meta["val_gain"] == pytest.approx(report["val_gain"])


def test_an_out_of_bounds_proposal_is_refused_not_committed():
    train = [_window(), _window()]
    val = [_window()]

    def propose(rendered, task, output, reward):
        return "grow: 999.00"        # outside the spec's bound (0, 5]

    report, _ = portfolio_evolve(
        windows=train, val_windows=val, propose=propose,
        seeds=(0, 1), val_seeds=(2, 3), rounds=1, n_workers=1, repo_path=None)
    # the gate refused the proposal, so the artifact is still the fixed baseline
    assert report["value"] == report["seed_value"]
    assert report["val_gain"] == pytest.approx(0.0)
    assert report["verdict"] == "reject"     # not an adoption, not a no-op


def test_a_custom_meta_reward_drives_the_search():
    # meta_reward must score the rollouts *and* the validation. With the
    # default auc, "grow harder" wins; with the inverted reward it is not a
    # gain, so the search must not commit it.
    train = [_window(), _window(high=0.9)]
    val = [_window(high=0.95)]

    def propose(rendered, task, output, reward):
        return "grow: 2.00"

    def inverted(outcome):
        # still a legal [0, 1] reward, but it prefers the *lower* projection
        return max(0.0, min(1.0, 1.0 - outcome.final))

    default, _ = portfolio_evolve(
        windows=train, val_windows=val, propose=propose,
        seeds=(0, 1), val_seeds=(2, 3), rounds=1, n_workers=1, repo_path=None)
    assert default["verdict"] == "adopt"
    assert "grow: 2.0" in default["value"]

    flipped, _ = portfolio_evolve(
        windows=train, val_windows=val, propose=propose, meta_reward=inverted,
        seeds=(0, 1), val_seeds=(2, 3), rounds=1, n_workers=1, repo_path=None)
    assert "grow: 2.0" not in flipped["value"]
    assert flipped["verdict"] == "reject"


def test_portfolio_evolve_refuses_an_empty_archive(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    with pytest.raises(ValueError):
        portfolio_evolve(store, propose=lambda r, t, o, w: "grow: 1.0")


# -- windows from the archive --------------------------------------------------


def _archive(store, run_id, slot, reward, rounds):
    history = [RoundInfo(round=i, held_out_reward=reward, n_items=1,
                         committed=1, rejected=0) for i in range(rounds)]
    store.archive(
        EvolutionResult(state={"value": "x"}, rendered="x", final_reward=reward,
                        history=history, ledger_log=[]),
        run_id=run_id, strategy="SourceSlot", config={"dataset": "d"},
        artifact_id="meta-rule",
        problem=problem_descriptor(slot, rounds=rounds, n_workers=1))


def test_windows_from_archive_aggregates_per_slot(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive(store, "a1", "selection", 0.9, 5)
    _archive(store, "b1", "task_sampler", 0.2, 5)
    _archive(store, "a2", "selection", 0.8, 5)
    _archive(store, "b2", "task_sampler", 0.3, 5)
    train, val = windows_from_archive(store, windows=2)
    assert train and val
    for window in train + val:
        assert window and all(isinstance(row, LayerBudget) for row in window)
        assert all(row.samples >= 1 for row in window)


def test_the_default_spec_compiles_the_shipped_policy():
    spec = reallocation_spec()
    policy = spec.compile(spec.render(spec.initial()))
    assert isinstance(policy, ReallocationPolicy)
    assert (policy.floor, policy.grow, policy.shrink, policy.min_samples) == \
        (0.05, 0.5, 0.5, 3)
