"""Did merging help, or did it average the improvements away?

`RoundStat.fused` counted how often a fusion was **committed**. That cannot
answer the question: the tournament only ever commits a fusion that won, so the
count is a tally of successes with its denominator missing. These tests pin the
denominators -- how many tournaments were held, how many had a fusion in them at
all, how often it lost, and how badly.

None of them assert that fusion wins. Whether it does is the empirical question
`docs/results.md` is for, and three of the possible answers are useful.
"""

import pytest

from agentdescent.evolution import (
    AppendRules, EvolutionResult, FusionStats, KeyedRules, Task, evolve,
)
from agentdescent.policies import FusionTrial


# -- the record --------------------------------------------------------------


def _trial(**kw):
    base = dict(artifact_id="a", n_candidates=2, best_single_score=0.5,
                baseline_score=0.4, fused_score=0.6, winner="fused")
    base.update(kw)
    return FusionTrial(**base)


def test_gain_is_none_when_no_fusion_was_built():
    """Not zero. A fusion that never ran and a fusion that broke even are
    different facts, and a zero would average into `mean_gain` as if the
    mechanism had been tested."""
    assert _trial(fused_score=None).gain is None
    assert _trial(fused_score=0.5).gain == pytest.approx(0.0)


def test_an_uncontested_run_reports_no_win_rate_rather_than_zero():
    """A rate with an empty denominator printed as 0% reads as 'fusion always
    lost', which is the opposite of 'fusion never ran'."""
    stats = FusionStats.of([_trial(fused_score=None, reason="single-candidate",
                                   winner="single")] * 3)
    assert stats.contested == 0
    assert stats.win_rate is None
    assert "fusion never ran" in stats.summary()
    assert stats.single_candidate == 3


def test_the_two_reasons_a_fusion_is_not_built_are_counted_apart():
    """'Every round had one survivor' and 'the survivors contradicted' point at
    different fixes -- more workers versus a strategy with a key space."""
    stats = FusionStats.of([
        _trial(fused_score=None, reason="single-candidate", winner="single"),
        _trial(fused_score=None, reason="contradiction", winner="single"),
        _trial(fused_score=None, reason="contradiction", winner="single"),
    ])
    assert (stats.single_candidate, stats.contradiction) == (1, 2)


def test_the_losing_tail_is_reported_with_its_worst_case():
    stats = FusionStats.of([
        _trial(fused_score=0.6, best_single_score=0.5),      # +0.1
        _trial(fused_score=0.4, best_single_score=0.5, winner="single"),   # -0.1
        _trial(fused_score=0.2, best_single_score=0.5, winner="single"),   # -0.3
    ])
    assert stats.negative == 2
    assert stats.worst_loss == pytest.approx(-0.3)
    assert stats.mean_loss == pytest.approx(-0.2)
    assert stats.mean_gain == pytest.approx(-0.1)


def test_below_baseline_is_separate_from_merely_losing():
    """Losing to the best single diff is ranking noise. Losing to the artifact
    you started from is the failure the objection actually names."""
    stats = FusionStats.of([
        # ranked below the best single, still an improvement on the baseline
        _trial(fused_score=0.45, best_single_score=0.5, baseline_score=0.4,
               winner="single"),
        # actively harmful
        _trial(fused_score=0.3, best_single_score=0.5, baseline_score=0.4,
               winner="single"),
    ])
    assert stats.negative == 2
    assert stats.below_baseline == 1


def test_ties_are_counted_because_an_empty_tail_can_mean_a_blind_gate():
    """No negative samples at all usually means the cheap layer cannot separate
    the candidates, not that fusion is safe. `ties` is how you tell."""
    stats = FusionStats.of([_trial(fused_score=0.5, best_single_score=0.5,
                                   winner="single")] * 4)
    assert stats.negative == 0 and stats.ties == 4


# -- the wiring --------------------------------------------------------------


KEYS = ("alpha", "beta", "gamma", "delta")


def _tasks(n=24):
    return [Task(id=str(i), prompt=f"q{i}",
                 meta={"key": KEYS[i % len(KEYS)], "gold": f"A{i % len(KEYS)}"})
            for i in range(n)]


def _run(rendered, task):
    lines = rendered.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == f"## {task.meta['key']}" and i + 1 < len(lines):
            return lines[i + 1].strip()
    return "?"


def _reward(task, output):
    return 1.0 if output == task.meta["gold"] else 0.0


def _propose(rendered, task, output, score):
    return f"{task.meta['key']}: {task.meta['gold']}"


def _evolve(**kw):
    # These tests are about the tournament, which is opt-in: the default commits
    # the union straight to the gate and ranks nothing, so `win_rate` has no
    # denominator there by construction. The default's own behaviour is pinned
    # below, under "the default path".
    kw.setdefault("fusion_tournament", True)
    kw.setdefault("rounds", 6)
    kw.setdefault("n_workers", 4)
    return evolve(_tasks(), _reward, run=_run, propose=_propose,
                  strategy=KeyedRules(categories=KEYS), held_out_frac=0.4,
                  max_concurrency=1, **kw)


def test_a_multi_worker_run_records_its_tournaments():
    result = _evolve()
    stats = result.fusion_stats()
    assert stats.trials > 0, "four workers must have produced tournaments"
    assert stats.fused_wins + stats.single_wins + stats.neither == stats.trials


def test_a_single_worker_run_has_nothing_to_fuse():
    """The denominator has to survive the degenerate case: one worker means one
    candidate, so every tournament is uncontested rather than a fusion loss."""
    stats = _evolve(n_workers=1).fusion_stats()
    assert stats.contested == 0
    assert stats.single_candidate == stats.trials
    assert stats.win_rate is None


def test_an_uninstrumented_fusion_policy_reports_zero_trials_not_zero_wins():
    """A replaced `FusionPolicy` keeps no trials, and that must not read as
    evidence about fusion."""
    from agentdescent.aggregator import Aggregator

    class Blind:
        def __init__(self, verifier):
            self.verifier = verifier

        def select(self, artifact, diffs):
            candidate = artifact.apply(diffs[0])
            return diffs[0], candidate, False

    def factory(ledger, verifier, audit, config, policy):
        agg = Aggregator(ledger, verifier, audit, config, staleness_policy=policy)
        agg.fusion_policy = Blind(verifier)
        return agg

    stats = _evolve(aggregator_factory=factory).fusion_stats()
    assert stats.trials == 0
    assert stats.win_rate is None


def test_trials_survive_a_save_and_load_round_trip(tmp_path):
    result = _evolve()
    path = str(tmp_path / "run.json")
    result.save(path)
    assert EvolutionResult.load(path).fusion_stats() == result.fusion_stats()


def test_a_result_written_before_the_field_existed_still_loads(tmp_path):
    import json

    result = _evolve()
    path = str(tmp_path / "old.json")
    result.save(path)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    payload.pop("fusion_trials")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    assert EvolutionResult.load(path).fusion_stats().trials == 0


def test_append_only_strategies_still_produce_contested_tournaments():
    """`AppendRules` never contradicts, so its tournaments are the *upper* bound
    on how often fusion gets to compete -- worth pinning, because a run where
    fusion always wins is usually a run on a domain where it cannot lose."""
    result = evolve(_tasks(), _reward,
                    run=lambda rendered, task: (
                        task.meta["gold"] if task.meta["gold"] in rendered else "?"),
                    propose=lambda rendered, t, o, s: t.meta["gold"],
                    strategy=AppendRules(), rounds=4, n_workers=4,
                    max_concurrency=1, held_out_frac=0.4)
    stats = result.fusion_stats()
    assert stats.contradiction == 0


# -- the default path: union -> gate, nothing ranked -------------------------


def test_the_default_commits_the_union_without_ranking_anything():
    """No `fusion_tournament=`, so no candidate is scored to choose between them.

    The union goes straight to the acceptance gate, which scores it on the full
    held-out set and is the thing that stops a regression either way. What is
    given up is the comparison, and `contested` -- `win_rate`'s denominator --
    has to stay empty to say so."""
    stats = _evolve(fusion_tournament=False).fusion_stats()
    assert stats.unranked > 0, "the default has to build unions, not skip merging"
    assert stats.contested == 0, "nothing was compared, so nothing can be contested"
    assert stats.win_rate is None, "a rate over an empty denominator"


def test_the_default_says_fusion_ran_rather_than_that_it_never_did():
    """`summary()` said "fusion never ran" whenever `contested` was zero, which
    on the default path is every run -- reporting the mechanism as absent on
    exactly the runs that used it."""
    line = _evolve(fusion_tournament=False).fusion_stats().summary()
    assert "never ran" not in line, line
    assert "unranked" in line, line


def test_the_tournament_is_what_produces_a_win_rate():
    """The contrast, and the reason the code stays: `best_single_score` only
    exists where a single was actually scored."""
    stats = _evolve(fusion_tournament=True).fusion_stats()
    assert stats.contested > 0 and stats.win_rate is not None


# -- can the workload exercise the mechanism at all? -------------------------


def test_identical_proposals_are_not_a_fusion():
    """`fuse_diffs` is `ops.update()`, so N copies of one diff "fuse" into that
    diff -- and it was counted as a contested tournament.

    Measured on a mute-backend run before the fix: nine tournaments, every one
    of them two identical `{'value': 'rule-0'}` diffs, all nine in `contested`.
    That is `win_rate`'s denominator, so the statistic issue #72 exists to
    produce was counting non-events. Workers that draw the same task, or
    converge on the same fix, propose the same text; it is not a corner case.
    """
    from agentdescent.evolution import SingleSlot

    stats = evolve(
        _tasks(), _reward,
        run=lambda rendered, t: "?",                     # mute: never improves
        propose=lambda rendered, t, o, s: "same-rule",   # every worker agrees
        strategy=SingleSlot(key="instruction"), rounds=6, n_workers=3,
        max_concurrency=1, held_out_frac=0.4, self_verify=False,
        fusion_tournament=True,
    ).fusion_stats()

    assert stats.trials > 0, "the tournament has to have run at all"
    assert stats.contested == 0, "identical diffs did not fuse into anything"
    assert stats.nothing_to_fuse > 0, \
        "and the reason has to be legible -- not filed under 'contradiction'"


def test_a_single_slot_artifact_can_never_fuse():
    """The check a merge-vs-fork table has to pass before it means anything.

    A strategy that keeps the whole artifact in one key -- GEPA's
    `InstructionSlot` is the shipped example -- makes every pair of worker
    proposals contradict by construction. Conflict resolution collapses them to
    one, so the tournament has a single candidate and no fusion is ever built.
    `merge_of_n` on such a workload is per-round best-of-N *selection*, and
    reporting it as evidence about merging measures the wrong thing.

    `contested == 0` is how a reader finds that out without reading the strategy.
    """
    from agentdescent.evolution import SingleSlot

    stats = evolve(
        _tasks(), _reward,
        run=lambda rendered, t: t.meta["gold"] if t.meta["gold"] in rendered else "?",
        propose=lambda rendered, t, o, s: f"{rendered}\n{t.meta['gold']}",
        strategy=SingleSlot(key="instruction"), rounds=6, n_workers=3,
        max_concurrency=1, held_out_frac=0.4, self_verify=False,
    ).fusion_stats()

    assert stats.trials > 0, "the tournament has to have run at all"
    assert stats.contested == 0, \
        "a one-key artifact cannot produce a fused candidate"


def test_a_multi_key_artifact_can():
    """The contrast, so the test above is about the artifact shape and not about
    some unrelated reason fusion failed to fire."""
    stats = _evolve().fusion_stats()
    assert stats.contested > 0
