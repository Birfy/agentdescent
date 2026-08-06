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


def test_reflective_fusion_commits_a_union_where_the_default_cannot():
    """Same workload, same one key -- now a union reaches the ledger."""
    def merger(prompt):
        # A merge that genuinely unions: keep every gold answer it was shown.
        golds = sorted({line.strip() for block in prompt.split("PROPOSAL")[1:]
                        for line in block.splitlines()
                        if line.strip().startswith("A")})
        return "instruction\n" + "\n".join(golds)

    stats = _evolve(policies=Policies(**reflective_merge(merger))).fusion_stats()
    assert stats.unranked > 0, "no union was ever built"
    assert stats.contested == 0, "nothing is ranked on this path, by design"


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
    assert stats.unranked == 0, "no union was built"
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
    fusion.bind(type("V", (), {"cheap_eval": staticmethod(lambda a: 0.5)})())
    fusion.select(Art(), [a, b])
    assert fusion.trials[-1].reason == "synthesis-failed"


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
    assert alone.fusion_stats().unranked == 0, "never given more than one diff"

    paired = _evolve(policies=Policies(**reflective_merge(lambda p: "merged")))
    assert paired.fusion_stats().unranked > 0


def test_the_prompt_asks_for_a_union_of_deltas_not_a_rewrite():
    """"Write one version keeping every improvement" invites a fresh composition
    that happens to cover the same ground. "Work out what each proposal changed
    relative to CURRENT, then apply all of those changes" is an operation whose
    result can be checked, and it needs only what a FusionPolicy actually has --
    the current value and the competing ones, never the evidence cards.
    """
    from agentdescent.fusion import MERGE_PROMPT

    filled = MERGE_PROMPT.format(current="C", proposals="PROPOSAL 1\n---\nP\n---\n")
    assert "UNION" in filled
    assert "CHANGED relative to CURRENT" in filled
    assert "{" not in filled and "}" not in filled, "every placeholder is filled"


def test_the_union_is_verified_against_the_thing_it_replaces():
    """`fuse_diffs` on one key is last-writer-wins. This is the contrast the
    whole module exists for, asserted rather than described."""
    from agentdescent.aggregator import fuse_diffs
    from agentdescent.evolvable import Diff

    a = Diff(diff_id="a", target="p", ops={"instruction": "base. CHECK BOTH."})
    b = Diff(diff_id="b", target="p", ops={"instruction": "base. BE SHORT."})
    plain = fuse_diffs([a, b]).ops["instruction"]
    assert ("CHECK BOTH" in plain) != ("BE SHORT" in plain), \
        "a dict update keeps exactly one of them -- that is the gap"

    merged = ReflectiveFusion(
        lambda p: "base. CHECK BOTH. BE SHORT.")._synthesise(
            "base.", ["base. CHECK BOTH.", "base. BE SHORT."])
    assert "CHECK BOTH" in merged and "BE SHORT" in merged


# -- the cost of keeping contradictions --------------------------------------


def test_scoring_a_tournament_is_order_independent():
    """Scoring a candidate must not depend on what was scored before it.

    `learned_eval` used to draw its noise from a shared stream, so the same
    candidate got a different number depending on its position in the tournament
    -- and this policy keeps every contradicting proposal, so tournaments here are
    exactly the ones with several candidates in them. Seeded from the artifact
    instead, a candidate scores the same wherever it sits.
    """
    from agentdescent.verifier import ThreeLayerVerifier

    class Art:
        def __init__(self, text): self.text = text
        def render(self): return self.text

    v = ThreeLayerVerifier(eval_fn=lambda a, t: 0.5, held_out=list(range(8)))
    a, b = Art("one"), Art("two")
    first = [v.cheap_eval(a), v.cheap_eval(b)]
    reversed_order = [v.cheap_eval(b), v.cheap_eval(a)]
    assert first == [reversed_order[1], reversed_order[0]]




# -- the two cost/precision knobs --------------------------------------------


def test_two_arms_of_an_ab_do_not_see_different_noise():
    """The confound that invalidated a real A/B run, pinned.

    The `off` arm scores a pair and drops one; the `on` arm scores every
    contradicting proposal plus the synthesised candidate. Drawing the noise from
    a shared stream meant the two arms consumed different numbers of draws, so
    their streams diverged from the first merge -- the arms differed in the noise
    they saw as well as in the mechanism, and a 0.200 gap on test could not be
    attributed to either.

    Measured before the fix: the same candidate scored 0.5244 / 0.4820 / 0.4909 /
    0.5016 depending only on its position, a spread wider than one task's worth
    of improvement on the ranking subset.
    """
    from agentdescent.verifier import ThreeLayerVerifier

    class Art:
        id = "x"
        version = 1
        def __init__(self, t): self.t = t
        def render(self): return self.t

    target = Art("the candidate")
    others = [Art(f"other-{i}") for i in range(3)]

    scores = []
    for n_before in range(4):
        v = ThreeLayerVerifier(eval_fn=lambda a, t: 0.5, held_out=list(range(20)),
                               rule_subset=8, learned_noise=0.04)
        for o in others[:n_before]:
            v.cheap_eval(o)
        scores.append(v.cheap_eval(target))

    assert len(set(scores)) == 1, (
        f"a candidate's score still depends on its position: {scores}")


def test_the_synthesis_is_one_call_per_tournament_not_per_candidate():
    """The invariant behind the cost figure in the module docstring.

    A union is asked for **once per contested tournament**, however many
    proposals are competing -- so turning this on adds calls proportional to
    merges, not to workers x merges. Counted end-to-end with a stub model, the
    synthesis came to 3% of a HotpotQA run; two earlier claims that it was a
    multiple came from reading a shared `Usage` accumulating across concurrent
    arms, and this is the property that makes the small number believable.
    """
    asked = []
    result = _evolve(policies=Policies(
        **reflective_merge(lambda p: asked.append(p) or "instruction\nmerged")))
    stats = result.fusion_stats()
    assert stats.unranked > 0, "the mechanism has to have fired at all"
    assert len(asked) <= stats.trials, (
        f"{len(asked)} synthesis calls over {stats.trials} tournaments: it is "
        "being asked more than once per merge")


# -- trust_union: the big saving, and what it costs --------------------------


