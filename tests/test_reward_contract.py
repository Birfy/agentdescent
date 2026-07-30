"""The reward contract is [0, 1], and violating it used to fail silently.

The engine treats `reward >= 0.999` as solved and never asks for a proposal, so a
scorer on the wrong scale (accuracy 0-100, say) makes *every* task look solved:
`propose()` is never called, nothing is learned, and the reported final_reward is
a large healthy-looking number. That is the worst kind of bug -- it looks like it
worked.
"""

import warnings

import pytest

from agentdescent.async_evolve import async_evolve
from agentdescent.evolution import Task, evolve


def _tasks(n=10):
    return [Task(id=str(i), prompt="p") for i in range(n)]


def _run(rendered, task):
    return "x"


def _propose(rendered, task, output, reward):
    return "a rule"


def test_out_of_range_reward_is_rejected_with_a_useful_message():
    from agentdescent.evolution import RewardContractError

    with pytest.raises(RewardContractError) as e:
        evolve(_tasks(), lambda t, o: 85.0, run=_run, propose=_propose, rounds=2)
    msg = str(e.value)
    assert "outside [0, 1]" in msg
    assert "0" in msg                       # names the offending task
    assert "Normalise" in msg               # tells them what to do


def test_negative_reward_is_rejected():
    with pytest.raises(ValueError):
        evolve(_tasks(), lambda t, o: -3.0, run=_run, propose=_propose, rounds=2)


def test_non_numeric_reward_is_rejected_clearly():
    from agentdescent.evolution import RewardContractError

    with pytest.raises(RewardContractError) as e:
        evolve(_tasks(), lambda t, o: None, run=_run, propose=_propose, rounds=2)
    assert "must return a number in [0, 1]" in str(e.value)


def test_float_error_just_above_one_is_tolerated():
    """A scorer computing 1.0000000001 must not explode."""
    r = evolve(_tasks(), lambda t, o: 1.0 + 1e-9, run=_run, propose=_propose, rounds=2)
    assert r.final_reward == 1.0


def test_bool_rewards_work():
    r = evolve(_tasks(), lambda t, o: True, run=_run, propose=_propose, rounds=2)
    assert r.final_reward == 1.0


def test_valid_reward_is_unaffected():
    r = evolve(_tasks(), lambda t, o: 0.5, run=_run, propose=_propose, rounds=2)
    assert r.final_reward == 0.5 and r.error is None


def test_async_path_enforces_the_same_contract():
    """The worker called reward() directly, bypassing the held-out check."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            async_evolve(_tasks(), lambda t, o: 85.0, run=_run, propose=_propose,
                         n_workers=2, max_seconds=3.0)
