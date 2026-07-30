"""Caller mistakes must be loud; backend failures must be survivable.

The engine deliberately absorbs a backend failure -- a rate limit, a dead
endpoint -- so a long run keeps its partial results. It must NOT absorb a broken
contract in the caller's own code: that makes the whole run meaningless, and
reporting it through the same channel as a provider outage hides it. All three
contract violations share one base so both engines treat them identically.
"""

import pytest

from agentdescent.aggregator import AggregatorContractError
from agentdescent.evolvable import ContractError
from agentdescent.evolution import (
    AppendRules,
    ProposalContractError,
    RewardContractError,
    Task,
    evolve,
)


def _tasks(n=10):
    return [Task(id=str(i), prompt="q") for i in range(n)]


GOOD_RUN = lambda rendered, task: "good"
GOOD_REWARD = lambda t, o: 1.0 if "good" in o else 0.0
GOOD_PROPOSE = lambda r, t, o, s: "a rule"


def test_all_three_contract_errors_share_one_base():
    for cls in (RewardContractError, ProposalContractError, AggregatorContractError):
        assert issubclass(cls, ContractError)


def test_bad_reward_propagates():
    with pytest.raises(ContractError):
        evolve(_tasks(), lambda t, o: 85.0, run=GOOD_RUN, propose=GOOD_PROPOSE,
               strategy=AppendRules(), rounds=2)


def test_bad_proposal_propagates():
    # the rollout must FAIL, or propose() is never called at all
    with pytest.raises(ContractError):
        evolve(_tasks(), GOOD_REWARD, run=lambda rendered, task: "bad",
               propose=lambda r, t, o, s: 42, strategy=AppendRules(), rounds=2)


class _NoMethods:
    def __init__(self, *a, **k):
        pass


class _StepReturnsNone(_NoMethods):
    def ingest(self, card):
        pass

    def step(self):
        return None


class _StepReturnsGarbage(_NoMethods):
    def ingest(self, card):
        pass

    def step(self):
        return ["not a report"]


@pytest.mark.parametrize("cls", [_StepReturnsNone, _StepReturnsGarbage])
def test_a_custom_aggregator_breaking_its_contract_propagates(cls):
    with pytest.raises(ContractError):
        evolve(_tasks(), GOOD_REWARD, run=GOOD_RUN, propose=GOOD_PROPOSE,
               strategy=AppendRules(), rounds=2,
               aggregator_factory=lambda *a, **k: cls())


def test_an_aggregator_missing_methods_fails_at_construction():
    """Fail before the first rollout, naming what is missing."""
    with pytest.raises(TypeError) as e:
        evolve(_tasks(), GOOD_REWARD, run=GOOD_RUN, propose=GOOD_PROPOSE,
               strategy=AppendRules(), rounds=2,
               aggregator_factory=lambda *a, **k: _NoMethods())
    msg = str(e.value)
    assert "ingest" in msg and "AggregatorProtocol" in msg


def test_a_backend_failure_is_still_absorbed_not_raised():
    """The contrast that makes the distinction worth having."""
    class _Dead:
        def solve(self, rendered, task):
            raise RuntimeError("provider is down")

        def propose(self, rendered, task, output, reward):
            return "x"

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = evolve(_tasks(), GOOD_REWARD, agent=_Dead(),
                     strategy=AppendRules(), rounds=3)
    assert res.error is not None and "provider is down" in res.error


def test_the_async_path_agrees():
    from agentdescent.async_evolve import async_evolve

    with pytest.raises(ContractError):
        async_evolve(_tasks(), lambda t, o: 85.0, run=GOOD_RUN,
                     propose=GOOD_PROPOSE, strategy=AppendRules(),
                     n_workers=2, max_seconds=3.0)
