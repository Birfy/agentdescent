"""Fusion that can combine two proposals for the same key.

`fuse_diffs` is a dictionary update, so on an artifact held in one key it cannot
merge anything -- the last writer wins and `DefaultFusion` declines to build a
fused candidate at all. That is not a corner case for GEPA's `InstructionSlot`,
it is every round.

These tests do not assert that a model-written merge is *better*. It is a model
writing text; whether it helps is an empirical question and the tournament is
what answers it. What is pinned here is that it cannot cheat and cannot break a
merge: it enters the same held-out tournament with no privilege, a tie loses, and
every failure path falls back to the shipped behaviour.
"""

import pytest

from agentdescent.evolution import (
    AppendRules, EvolutionResult, KeyedRules, SingleSlot, Task, evolve,
)
from agentdescent.fusion import (
    KeepContradictions, ReflectiveFusion, reflective_merge,
)
from agentdescent.policies import Policies


KEYS = ("alpha", "beta", "gamma", "delta")


def _tasks(n=24):
    return [Task(id=str(i), prompt=f"q{i}",
                 meta={"key": KEYS[i % len(KEYS)], "gold": f"A{i % len(KEYS)}"})
            for i in range(n)]


def _run(rendered, task):
    return task.meta["gold"] if task.meta["gold"] in rendered else "?"


def _reward(task, output):
    return 1.0 if output == task.meta["gold"] else 0.0


def _propose(rendered, task, output, score):
    return f"{rendered}\n{task.meta['gold']}"


def _evolve(**kw):
    kw.setdefault("rounds", 6)
    kw.setdefault("n_workers", 3)
    return evolve(_tasks(), _reward, run=_run, propose=_propose,
                  strategy=SingleSlot(key="instruction"), max_concurrency=1,
                  held_out_frac=0.4, self_verify=False, **kw)


# -- the gap it exists to close ---------------------------------------------


def test_the_shipped_policy_cannot_fuse_a_one_key_artifact():
    """The premise. If this ever stops being true, this module is unnecessary."""
    assert _evolve().fusion_stats().contested == 0


def test_reflective_fusion_gets_a_fused_candidate_into_the_tournament():
    """Same workload, same one key -- now a fused candidate competes."""
    def merger(prompt):
        # A merge that genuinely unions: keep every gold answer it was shown.
        golds = sorted({line.strip() for block in prompt.split("PROPOSAL")[1:]
                        for line in block.splitlines()
                        if line.strip().startswith("A")})
        return "instruction\n" + "\n".join(golds)

    stats = _evolve(policies=Policies(**reflective_merge(merger))).fusion_stats()
    assert stats.contested > 0, "a fused candidate never reached the tournament"


# -- it must not be able to cheat -------------------------------------------


def test_a_synthesis_that_only_ties_does_not_win():
    """`max` keeps the first of equal scores and the synthesised candidate is
    appended last, so it has to strictly beat the field. Without that, a
    held-out set too small to separate anything would credit the model."""
    def echo_first(prompt):
        # Returns something scoring exactly like an existing candidate.
        block = prompt.split("PROPOSAL 1\n---\n", 1)[1].split("\n---", 1)[0]
        return block + "\n"          # differs by whitespace only

    result = _evolve(policies=Policies(**reflective_merge(echo_first)))
    # A tie must never be recorded as a synthesised win.
    for trial in result.fusion_trials:
        if (trial.synthesized_score is not None
                and trial.synthesized_score <= trial.best_single_score):
            assert trial.winner != "synthesized"


def test_an_answer_that_repeats_an_input_is_not_a_merge():
    """It would enter the tournament as a duplicate and, on a tie, be
    indistinguishable from the model having contributed something."""
    fusion = ReflectiveFusion(lambda prompt: "PROPOSAL-ECHO")
    assert fusion._synthesise("x", ["PROPOSAL-ECHO", "other"]) is None


def test_an_oversized_answer_is_refused():
    """The synthesised value skips the trust region, which filters cards before
    the merge -- so the bound has to exist here too."""
    fusion = ReflectiveFusion(lambda prompt: "z" * 50_000, max_chars=100)
    assert fusion._synthesise("x", ["a", "b"]) is None


def test_an_empty_answer_is_refused():
    assert ReflectiveFusion(lambda prompt: "   ")._synthesise("x", ["a", "b"]) is None


# -- it must not be able to break a merge -----------------------------------


def test_a_dead_backend_falls_back_instead_of_raising():
    """Fusion is on the commit path of every round. A merge that dies because a
    model was slow is worse than a merge that did not improve."""
    def broken(prompt):
        raise ConnectionError("Connection error.")

    result = _evolve(policies=Policies(**reflective_merge(broken)))
    assert result.error is None, "a dead merger must not end the run"
    stats = result.fusion_stats()
    assert stats.trials > 0
    assert stats.contested == 0, "nothing was fused, which is the fallback"
    assert stats.synthesis_failed > 0, "and the reason has to be recorded"


def test_the_failure_reason_is_distinguishable_from_never_trying():
    """`contradiction` means no model was asked; `synthesis-failed` means one was
    and could not be used. Same zero in `contested`, opposite fixes."""
    result = _evolve(policies=Policies(
        **reflective_merge(lambda p: (_ for _ in ()).throw(RuntimeError()))))
    reasons = {t.reason for t in result.fusion_trials}
    assert "synthesis-failed" in reasons


# -- partitioning -----------------------------------------------------------


def test_only_the_keys_that_disagree_are_sent_to_the_model():
    """A key every diff agrees on is already handled correctly by the union; a
    model asked to merge values that do not disagree can only make them worse."""
    from agentdescent.evolvable import Diff

    a = Diff(diff_id="a", target="x", ops={"same": "v", "differs": "1"})
    b = Diff(diff_id="b", target="x", ops={"same": "v", "differs": "2"})
    agreed, contested = ReflectiveFusion(lambda p: "")._partition([a, b])
    assert agreed == {"same": "v"}
    assert set(contested) == {"differs"}
    assert sorted(contested["differs"]) == ["1", "2"]


def test_a_partial_synthesis_is_discarded_whole():
    """Committing some workers' contributions and dropping the rest looks like a
    successful merge and is not one."""
    from agentdescent.evolvable import Diff

    calls = {"n": 0}

    def sometimes(prompt):
        calls["n"] += 1
        return "merged" if calls["n"] == 1 else ""      # the second key fails

    fusion = ReflectiveFusion(sometimes)

    class Art:
        id = "x"
        state = {"k1": "a", "k2": "b"}

        def apply(self, diff):
            return self

    fusion.verifier = type("V", (), {"cheap_eval": staticmethod(lambda a: 0.5)})()
    a = Diff(diff_id="a", target="x", ops={"k1": "1", "k2": "3"})
    b = Diff(diff_id="b", target="x", ops={"k1": "2", "k2": "4"})
    fusion.select(Art(), [a, b])
    trial = fusion.trials[-1]
    assert trial.synthesized_score is None
    assert trial.reason == "synthesis-failed"


# -- default is untouched ----------------------------------------------------


def test_installing_nothing_changes_nothing():
    plain = _evolve()
    again = _evolve()
    assert plain.rendered == again.rendered
    assert plain.fusion_stats() == again.fusion_stats()


def test_the_engine_hands_the_policy_its_verifier():
    """A caller cannot supply one -- the verifier is built inside the engine."""
    fusion = ReflectiveFusion(lambda p: "merged")
    assert fusion.verifier is None
    _evolve(policies=Policies(fusion=fusion, conflict=KeepContradictions()))
    assert fusion.verifier is not None


def test_at_least_two_proposals_are_required():
    with pytest.raises(ValueError):
        ReflectiveFusion(lambda p: "", max_proposals=1)


def test_installing_the_fusion_alone_is_the_mistake_it_looks_like():
    """The reason `reflective_merge` returns a pair.

    Conflict resolution runs at step 2 and fusion at step 3, so a fusion policy
    installed on its own is handed one surviving diff on exactly the workloads it
    was written for -- and correctly declines to merge it with itself. It looks
    installed, costs nothing, and does nothing.
    """
    alone = _evolve(policies=Policies(fusion=ReflectiveFusion(lambda p: "merged")))
    assert alone.fusion_stats().contested == 0

    paired = _evolve(policies=Policies(**reflective_merge(lambda p: "merged")))
    assert paired.fusion_stats().contested > 0
