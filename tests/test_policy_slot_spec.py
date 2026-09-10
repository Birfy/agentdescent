"""``kind: "policy_slot"`` -- evolving a decision rule of the optimiser from a spec.

The kind exists so that meta-evolution reaches the CLI and the host plugins
without either growing a parallel entry point: it is still an ``evolve()`` call,
so ``plan``, ``evolve``, ``status``, ``show`` and ``apply`` work unchanged.

What is different, and what these tests pin, is that its ``target`` is a *slot*
rather than a path and its ``data`` holds *refs* rather than rows -- an inner
problem is a callable and no row format can express one.
"""
from __future__ import annotations

import pytest

from agentdescent.evolvespec import EvolveSpec, SpecError, compose

PROBLEMS = "examples.metasearch.evolve_search_policy:source_problems"


def _spec(**over):
    d = {"kind": "policy_slot", "target": "selection",
         "data": {"problems": PROBLEMS, "seeds": [0]},
         "agent": {"ref": "agentdescent.agents:echo", "call": True},
         "allow": ["examples"]}
    d.update(over)
    return EvolveSpec.from_dict(d)


def test_it_composes_the_same_call_meta_evolve_makes():
    comp = compose(_spec(data={"problems": PROBLEMS, "seeds": [0, 1]}))
    # one outer task per (problem, seed), and the strategy is the slot itself
    assert [t.id for t in comp.tasks] == ["source:0", "source:1"]
    assert type(comp.kwargs["strategy"]).__name__ == "SourceSlot"
    # L1: the value is a harness, so every merge also passes the oracle
    assert comp.kwargs["blast_radius"] == 0.6
    assert comp.kwargs["self_verify"] is False
    assert callable(comp.kwargs["run"]) and callable(comp.kwargs["propose"])


def test_the_target_is_a_slot_name_and_absolutising_must_not_touch_it():
    """`absolutise` turns every other kind's target into a path.

    It runs wherever the spec is read -- the CLI's cwd, an MCP server's cwd --
    and for `policy_slot` that would silently rewrite `selection` into
    `$PWD/selection`, which is not a slot and fails at compose time with a
    message about slots that names a path.
    """
    assert _spec().absolutise(base="/somewhere/else").target == "selection"


def test_an_unknown_slot_is_refused_by_name():
    with pytest.raises(SpecError, match="evolvable slot"):
        compose(_spec(target="not_a_slot"))


def test_problems_are_required_and_the_message_says_what_to_write():
    with pytest.raises(SpecError, match=r"data\.problems"):
        compose(_spec(data={"seeds": [0]}))


def test_the_text_scorer_default_becomes_this_kind_s_default():
    """`EvolveSpec.score` defaults to "contains", which compares strings.

    A `MetaOutcome` is a curve; `contains` cannot read one. A spec whose author
    never wrote a `score` must not fail on the field they did not write.
    """
    assert compose(_spec()).reward is not None      # "contains" -> auc
    with pytest.raises(SpecError, match="not 'module:attribute'"):
        compose(_spec(score="exact"))


def test_a_meta_reward_short_name_resolves_to_the_function_not_a_factory():
    """`auc` *is* the callable, the opposite of every other ref in a spec."""
    for name in ("auc", "final_reward", "rollouts_to"):
        assert compose(_spec(score=name)).reward is not None


def test_it_says_when_the_gate_cannot_decide_anything():
    """The recurring defect in this line of work, said before the run.

    A gate too small to resolve the effect it judges has failed closed (nothing
    commits) and open (a rule losing on its own family was committed on nine
    held-out instances).
    """
    one = compose(_spec(data={"problems": PROBLEMS, "seeds": [0]}))
    assert any("NO held-out task" in n for n in one.notes)
    few = compose(_spec(data={"problems": PROBLEMS, "seeds": [0, 1, 2]}))
    assert any("held-out task(s)" in n for n in few.notes)


def test_it_asks_whether_the_seed_moves_rather_than_assuming():
    """Seeds are real replicates on a landscape and worthless on a cached,
    deterministic domain. The note has to be a check, not a claim."""
    notes = compose(_spec(data={"problems": PROBLEMS, "seeds": [0, 1]})).notes
    seed_note = next(n for n in notes if n.startswith("seeds="))
    assert "only if it changes the inner run" in seed_note
    assert compose(_spec()).notes[0].startswith("one rollout is a whole inner search")


def test_the_spec_path_and_the_library_path_cannot_drift():
    """`compose` and `meta_evolve` both assemble through `meta_parts`."""
    from agentdescent.meta import meta_parts, priority_selection, slot_reflector

    spec = priority_selection()
    tasks, reward, kwargs = meta_parts(
        {"source": lambda value, seed: None}, slot="selection", spec=spec,
        propose=slot_reflector(lambda p: "", spec))
    composed = compose(_spec()).kwargs
    assert set(kwargs) <= set(composed)
    for k in ("blast_radius", "self_verify", "solved_threshold"):
        assert kwargs[k] == composed[k]


def test_it_runs_end_to_end_and_commits_a_better_rule(monkeypatch):
    """The whole point: a spec goes through the real `evolve()` and improves.

    The stub proposes `greedy`, which genuinely beats the flat-PUCT seed on this
    landscape -- the hand-tuned sweep puts `c -> 0` at +0.0227 -- so a commit
    here is the gate working, not the stub being waved through.
    """
    import agentdescent.evolvespec as es

    monkeypatch.setattr(
        es, "_TEST_STUB",
        lambda prompt: "```python\ndef priority(rank, visits, total, prior, depth, n_nodes):\n"
                       "    return rank\n```",
        raising=False)
    comp = compose(_spec(
        data={"problems": PROBLEMS, "seeds": [0, 1, 2, 3, 4]},
        agent={"ref": "agentdescent.evolvespec:_TEST_STUB", "call": False},
        evolve={"rounds": 2, "n_workers": 1, "held_out_frac": 0.4}))
    result = comp.run()
    assert result.outcomes().get("committed") == 1
    assert "return rank" in result.rendered
