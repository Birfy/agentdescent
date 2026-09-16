"""Backward compatibility: the new features are extensions, not changes.

The contract this file exists to prove: **a run that configures none of the new
options behaves exactly as it did before they existed.** Every mechanism added
on this branch -- `max_tokens`, `checkpointing`, `stop_on_diminishing_returns`,
`BudgetGovernor`, `CostEfficient` -- must be inert until asked for.

These are behavioural assertions over a deterministic run, not a diff against a
stored golden: the run is seeded and its reward history is a function of the
seed alone, so "same history with and without the feature" is the strongest
statement a test can make here.
"""

from __future__ import annotations

import os
import warnings
from typing import List, Optional

import pytest

from agentdescent.agents import Usage
from agentdescent.evolution import evolve, Task, EvolutionResult


def _tasks(n: int = 8) -> List[Task]:
    return [Task(id=f"t{i}", prompt=f"task {i}") for i in range(n)]


class _Agent:
    """Deterministic agent: same proposals every run, no randomness."""

    def __init__(self) -> None:
        self.usage = Usage()

    def solve(self, rendered: str, task: Task) -> str:
        self.usage.record(prompt_tokens=100, completion_tokens=200)
        return task.id

    def propose(self, rendered: str, task: Task, output: str,
                reward: float) -> Optional[str]:
        self.usage.record(prompt_tokens=100, completion_tokens=200)
        # A deterministic proposal per (task, reward) so the run is repeatable.
        return f"rule-{task.id}-{round(reward, 3)}" if reward < 0.999 else None


def _run(**kwargs) -> EvolutionResult:
    agent = _Agent()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evolve(
            _tasks(8), lambda t, o: 0.5, agent=agent,
            rounds=4, n_workers=1, seed=7, usage=agent.usage, **kwargs)


def _history(r: EvolutionResult) -> list:
    return [(h.round, round(h.held_out_reward, 9), h.committed, h.rejected)
            for h in r.history]


# --- the default run is the pre-branch run ---


def test_no_new_options_is_the_old_path():
    """The default run must not acquire a budget, a checkpoint, or a governor."""
    r = _run()
    assert r.budget is None, "a run with no max_tokens must not report a budget"
    assert r.stop_reason in ("rounds", "patience", "target_reward")
    # The new per-round token field exists but is only meaningful when a usage
    # is shared; here it is shared, so it is populated -- the field being present
    # is not a behaviour change.
    assert hasattr(r.history[0], "tokens")


def test_disabled_flags_equal_no_flags():
    """Passing the new options at their defaults must equal passing none."""
    baseline = _run()
    explicit = _run(max_tokens=None, checkpointing=False,
                    stop_on_diminishing_returns=False)
    assert _history(baseline) == _history(explicit)
    assert baseline.final_reward == explicit.final_reward
    assert baseline.stop_reason == explicit.stop_reason


def test_a_never_reached_budget_does_not_change_the_search():
    """A token budget large enough never to bind must leave quality identical.

    This is the property that makes the budget *additive*: it can end a run
    early, and it can degrade optional spend when it is close, but with room to
    spare it touches nothing.
    """
    baseline = _run()
    budgeted = _run(max_tokens=10**12)
    assert _history(baseline) == _history(budgeted), \
        "a budget that never binds must not change the run"
    assert budgeted.budget is not None      # it *reports* one ...
    assert budgeted.budget["fusion_degraded"] is False
    assert budgeted.budget["self_verify_degraded"] is False


def test_no_checkpoint_files_without_the_flag(tmp_path):
    """No checkpoint is written unless `checkpointing=True`."""
    agent = _Agent()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evolve(_tasks(8), lambda t, o: 0.5, agent=agent,
               rounds=2, n_workers=1, repo_path=str(tmp_path / "repo"),
               usage=agent.usage)
    d = tmp_path / "repo" / "checkpoints"
    assert not d.exists() or not any(d.iterdir()), \
        "a run without checkpointing=True must not write checkpoints"


def test_cost_efficient_is_opt_in():
    """The default selection is SingleHead; a run without a selection policy
    must not construct or consult CostEfficient."""
    from agentdescent.policies import Policies
    from agentdescent.selection import SingleHead, CostEfficient

    # Default policies use SingleHead; this is a structural check that the new
    # policy is not the default anywhere.
    pol = Policies()
    assert pol.selection is None or isinstance(pol.selection, SingleHead)
    assert not isinstance(pol.selection, CostEfficient)


def test_governor_inert_without_a_budget():
    """The governor the engine constructs with no max_tokens says nothing and
    changes nothing."""
    from agentdescent.budget import BudgetGovernor

    g = BudgetGovernor(max_tokens=None, stop_on_diminishing=True)
    assert not g.active
    assert g.remaining_fraction() == 1.0
    assert not g.diminishing_returns()
    assert g.affords_next_round()
    assert g.allow_self_verify()
    assert g.allow_fusion_tournament()
    assert not g.over_budget()
