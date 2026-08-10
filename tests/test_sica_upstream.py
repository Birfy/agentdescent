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
    """There is no substitute source: a broken edit is a spent candidate.

    This used to assert `reflective is False`, pinning the workaround rather
    than the property. The merge is safe now because `ReflectiveFusion` takes
    the strategy's validator and refuses anything that will not compile -- so
    the gate still decides, and the port gets a merge that costs 2.7-2.8x fewer
    calls than the ranking it fell back to.
    """
    policy = sica.build(0)
    assert policy.reflective is True
    state = {"policy": sica.SICA_INITIAL_SOURCE}
    assert policy.strategy.to_diff(state, "not python at all", "w", 1, "a") is None


# -- the domain ---------------------------------------------------------------


def test_gsmhard_splits_are_disjoint_and_the_test_window_is_far_from_training():
    """GSM-Hard ships one split, so the three come from disjoint windows of it.
    The test window walks down from the end while the working windows walk up,
    so a seed's reported tasks sit as far as possible from what it tuned on --
    and the loader refuses outright if they would meet."""
    from examples._gsmhard_domain import gsmhard_splits

    for seed in (0, 1, 2):
        train, held_out, test = gsmhard_splits(seed)
        ids = [{t.id for t in g} for g in (train, held_out, test)]
        assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), seed
        questions = [{t.prompt for t in g} for g in (train, held_out, test)]
        assert not (questions[0] & questions[2]), f"seed {seed} tests what it trained on"


def test_the_answer_key_survives_gsm_hard_s_float_targets():
    """`target` ships as `-9867630.0`: signed, and a float where GSM8K's was a
    positive integer. A pattern written for the latter reads the sign as a
    separator and the `.0` as a different answer."""
    from examples._gsmhard_domain import normalise, parse_answer, score_answer

    assert normalise("-9867630.0") == "-9867630"
    assert normalise("1,200") == "1200"
    assert normalise(72.0) == "72"
    assert parse_answer("...so the total is -9867630.0") == "-9867630"
    assert score_answer("-9867630.0", "the answer is -9867630") == 1.0
    assert score_answer("-9867630.0", "the answer is 9867630") == 0.0
    assert parse_answer("no number here") is None


def test_the_reference_solution_is_never_shown():
    """GSM-Hard ships a `code` field holding the Python that computes the answer.
    Handing it to a method being asked to write a better instruction turns
    instruction search into transcription."""
    from examples._gsmhard_domain import feedback, gsmhard_splits

    train, _, _ = gsmhard_splits(0)
    task = train[0]
    assert "code" not in task.meta
    assert "def solution" not in feedback(task, "1")


def test_a_merge_that_will_not_compile_is_refused_rather_than_committed():
    """`ReflectiveFusion` writes its synthesis straight to the ledger without
    passing `to_diff`, so before it took a validator a model-merged pair of
    Python sources reached the artifact unchecked and failed at every rollout
    that read it. That is why this port declared `reflective=False`, and why it
    could not use a merge that costs 2.7-2.8x fewer calls than the ranking it
    fell back to.

    Returning `None` is already the "fall back to ranking" signal, so a refused
    merge costs exactly what the old behaviour cost and nothing more.
    """
    from agentdescent.fusion import ReflectiveFusion

    broken = ReflectiveFusion(lambda prompt: "def agent_prompt(question: pass",
                              validate=sica._sica_source)
    assert broken._synthesise("seed", ["a", "b"]) is None

    good_source = ('def agent_prompt(question):\n'
                   '    return "Work it step by step.\\n\\n" + question\n')
    fine = ReflectiveFusion(lambda prompt: good_source,
                            validate=sica._sica_source)
    assert fine._synthesise("seed", ["a", "b"])


def test_a_merge_that_breaks_the_ast_gate_is_refused_too():
    """Not only syntax: the gate also refuses calls, assignments and f-strings,
    and a merge is exactly where two valid sources can produce a third that uses
    none of the constructs either of them did."""
    from agentdescent.fusion import ReflectiveFusion

    calls_a_method = ('def agent_prompt(question):\n'
                      '    return "x" + question.strip()\n')
    policy = ReflectiveFusion(lambda prompt: calls_a_method,
                              validate=sica._sica_source)
    assert policy._synthesise("seed", ["a", "b"]) is None


def test_the_runner_hands_the_strategy_validator_to_the_merge():
    import inspect

    from examples import _method_runner

    source = inspect.getsource(_method_runner.run_port)
    assert 'getattr(policy.strategy, "validator", None)' in source
    assert "validate=validate" in source
