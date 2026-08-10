"""SICA against `MaximeRobeyns/self_improving_coding_agent@ed8275dc`."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentdescent.selection import Archive, Candidate
from examples.sica import sica_self_edit as sica


def _ctx(scores):
    pool = [Candidate("a", i, score=s) for i, s in enumerate(scores)]
    return SimpleNamespace(candidates=tuple(pool), round=1, head=pool[0],
                           n_workers=1)


def test_the_next_base_is_the_best_agent_not_a_sampled_one():
    """`get_best_agent_iteration` takes `idxmax()` of the mean benchmark score,
    and `runner.py` runs `archive.agent_{best_iter}.agent_code`. There is no
    sampling in it.

    The port used `Archive(sampling="performance")`, a softmax over scores in
    [0, 1] at temperature 1 -- which leaves only `exp(1)/exp(0) = 2.7` between
    the best and worst entry. Measured over a four-candidate archive scoring
    0.2 / 0.9 / 0.5 / 0.9, that mode starts from the **worst** agent 8 times in
    40. SICA never would.
    """
    assert sica.build(0).engine.selection.sampling == "best"

    ctx = _ctx([0.2, 0.9, 0.5, 0.9])
    picks = Archive(sampling="best", seed=0).select(ctx, 40)
    assert {p.score for p in picks} == {0.9}

    sampled = [p.score for p in Archive(sampling="performance", seed=0)
               .select(ctx, 40)]
    assert 0.2 in sampled, "the contrast this test exists for has gone"


def test_best_mode_breaks_ties_toward_the_incumbent():
    """`idxmax` returns the first maximum, so a later candidate has to actually
    beat the incumbent rather than merely equal it."""
    ctx = _ctx([0.9, 0.5, 0.9])
    assert Archive(sampling="best").select(ctx, 1)[0].version == 0


def test_best_mode_is_a_declared_archive_mode():
    assert "best" in Archive.MODES
    with pytest.raises(ValueError, match="sampling must be one of"):
        Archive(sampling="greedy")


# -- the AST gate -------------------------------------------------------------


def test_the_gate_admits_a_prompt_that_can_clear_the_domain():
    """A self-edit analogue whose editable surface cannot express a solution is
    Voyager's failure in another shape: three seeds of 0.000 with no invalid
    proposals, against a target the world does not accept.

    This checks the gate *before* the run: the domain needs a prompt that teaches
    integer cents and working-out, and the gate has to let one through.
    """
    teaches = (
        'def agent_prompt(question):\n'
        '    return "Work the problem step by step, writing each calculation on '
        'its own line, then state the final number last.\\n\\n" + question\n'
    )
    functions = sica.compile_policy(teaches, {"agent_prompt": 1})
    from examples._gsm8k_domain import gsm8k_splits
    rendered = sica.policy_prompt(functions["agent_prompt"], gsm8k_splits(0)[0][0])
    assert "step by step" in rendered and "final number" in rendered

    suffix = (
        'def agent_prompt(question):\n'
        '    return "Solve:\\n" + question + "\\nState the final number last."\n'
    )
    assert sica.compile_policy(suffix, {"agent_prompt": 1})


@pytest.mark.parametrize("source,reason", [
    ('def agent_prompt(question):\n    return f"Solve {question}"\n', "JoinedStr"),
    ('def agent_prompt(question):\n    p = "x"\n    return p + question\n', "Assign"),
    ('def agent_prompt(question):\n    return "x" + question.strip()\n', "Call"),
    ('def agent_prompt(q, extra):\n    return "x" + q\n', "arity"),
    ('def other(question):\n    return question\n', "surface"),
    ('def agent_prompt(question):\n    return question\n', None),
])
def test_the_gate_refuses_what_it_declares_it_refuses(source, reason):
    if reason is None:
        assert sica.compile_policy(source, {"agent_prompt": 1})
        return
    with pytest.raises(ValueError):
        sica.compile_policy(source, {"agent_prompt": 1})


def test_an_unparseable_self_edit_costs_its_candidate_and_proposes_nothing():
    """There is no substitute source: a broken edit is a spent candidate, which
    is why `reflective=False` -- a synthesised merge of Python would bypass the
    gate that makes this safe."""
    policy = sica.build(0)
    assert policy.reflective is False
    state = {"policy": sica.SICA_INITIAL_SOURCE}
    assert policy.strategy.to_diff(state, "not python at all", "w", 1, "a") is None
