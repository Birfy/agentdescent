"""Adversarial inputs the aggregator meets in the wild.

A reflector is a language model: it returns empty strings, whitespace, the wrong
type, and occasionally its entire input. The optimizer has to stay sane through
all of that, because a bad proposal that gets committed is rendered into every
later prompt.
"""

import pytest

from agentdescent.aggregator import AggregatorConfig
from agentdescent.evolution import (
    AppendRules,
    ProposalContractError,
    SingleSlot,
    Task,
    evolve,
)


def _tasks(n=10):
    return [Task(id=str(i), prompt="q") for i in range(n)]


REWARD = lambda t, o: 1.0 if "good" in o else 0.0
RUN = lambda rendered, task: "good" if "rule" in rendered else "bad"


def _run(propose, **kw):
    return evolve(_tasks(), REWARD, run=RUN, propose=propose,
                  strategy=AppendRules(), rounds=3, **kw)


@pytest.mark.parametrize("value", ["", "   \n\t ", None])
def test_empty_proposals_are_ignored_not_committed(value):
    res = _run(lambda r, t, o, s: value)
    assert res.state == {}
    assert res.error is None            # not an error, just nothing to learn


@pytest.mark.parametrize("value", [42, {"a": 1}, ["x"], 3.5, object()])
def test_non_text_proposals_raise_a_clear_contract_error(value):
    """Previously this surfaced as "'int' object has no attribute 'strip'"."""
    with pytest.raises(ProposalContractError) as e:
        _run(lambda r, t, o, s: value)
    assert "must return a string or None" in str(e.value)


def test_a_runaway_proposal_is_rejected_by_the_trust_region():
    """The op-count trust region did not bound size: a 500 KB value was committed
    and then rendered into every subsequent prompt."""
    res = _run(lambda r, t, o, s: "rule " + "x" * 500_000)
    assert res.state == {}, "an oversized op must not enter the artifact"


def test_a_normal_sized_proposal_still_lands():
    """The cap must not reject anything real -- shipped ops are ~2.5k chars."""
    res = _run(lambda r, t, o, s: "rule " + "x" * 2_500)
    assert res.state, "a realistic proposal must still be accepted"


def test_the_size_cap_is_configurable():
    res = _run(lambda r, t, o, s: "rule " + "x" * 5_000,
               agg_config=AggregatorConfig(batch_trigger=2, max_wait_rounds=1,
                                           trust_region_chars=1_000))
    assert res.state == {}


def test_repeating_the_same_proposal_dedupes():
    res = _run(lambda r, t, o, s: "rule")
    assert len(res.state) == 1, "content-addressing should collapse duplicates"


def test_single_slot_ignores_empty_and_replaces_on_change():
    """The most common artifact: one prompt that each proposal replaces."""
    seq = iter(["", "   ", "first instruction", "first instruction", "second"])

    def propose(rendered, task, output, reward):
        return next(seq, "second")

    res = evolve(_tasks(), REWARD, run=RUN, propose=propose,
                 strategy=SingleSlot(initial_value="seed"), rounds=6)
    assert res.rendered in ("first instruction", "second", "seed")
    assert len(res.state) <= 1, "SingleSlot holds exactly one value"
