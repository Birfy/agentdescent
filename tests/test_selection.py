"""The candidate-selection seam, and the promise that adding it changed nothing.

The whole value of this module on day one is the *absence* of a behaviour change:
`SingleHead` has to be byte-for-byte what the engine already did, or every
measurement in the repository moved under a refactor. So the first test compares
a full run against a recorded one, and the rest pin the properties that make the
other policies trustworthy without a ground truth of their own -- chiefly that
`Beam(1)` computes `SingleHead`'s answer by a different route.

The second thing pinned here is the *refusal*. The ledger holds one live branch,
so a policy that names a different starting point cannot be honoured yet; it
raises rather than being collapsed to the head, because a caller who passes a
beam and watches a run finish has every reason to believe the beam ran.
"""

import pytest

from agentdescent.async_evolve import async_evolve
from agentdescent.evolution import AppendRules, Task, evolve
from agentdescent.evolvable import ContractError
from agentdescent.policies import Policies
from agentdescent.selection import (
    Archive, Beam, Candidate, MCTS, MultiHeadUnsupported, ParetoFrontier,
    SelectionContext, SingleHead, pareto_front,
)


def _c(version, score=None, per_task=None, selected=0, parent=None):
    return Candidate(artifact_id="a", version=version, score=score,
                     per_task=per_task or {}, selected=selected, parent=parent)


def _ctx(*candidates, head=None, n=4):
    head = head or candidates[0]
    return SelectionContext(head=head, candidates=candidates, n_workers=n)


# -- the default is the status quo ------------------------------------------


def _tasks(n=20):
    return [Task(id=str(i), prompt="q", meta={"h": f"H{i % 5}"}) for i in range(n)]


def _run(**kw):
    return evolve(_tasks(), lambda t, o: 1.0 if o == "good" else 0.0,
                  run=lambda rendered, t: "good" if t.meta["h"] in rendered else "bad",
                  propose=lambda rendered, t, o, s: t.meta["h"],
                  strategy=AppendRules(), rounds=6, n_workers=4,
                  max_concurrency=1, held_out_frac=0.4, seed=1, **kw)


def _run_async(**kw):
    return async_evolve(_tasks(), lambda t, o: 1.0 if o == "good" else 0.0,
                        run=lambda rendered, t: "good" if t.meta["h"] in rendered else "bad",
                        propose=lambda rendered, t, o, s: t.meta["h"],
                        # async_ratio=0 resyncs every rollout: this domain scores
                        # in microseconds, so any lag budget above zero has the
                        # workers discarding ~95% of their cards as stale and
                        # warning about it. Staleness is not what these tests are
                        # about, and a warning nobody reads is one nobody reads.
                        strategy=AppendRules(), n_workers=4, async_ratio=0,
                        max_seconds=5.0, target_reward=1.0,
                        held_out_frac=0.4, seed=1, **kw)


def test_the_default_run_is_unchanged():
    """The one thing this module must not do is move a number."""
    plain = _run()
    explicit = _run(policies=Policies(selection=SingleHead()))
    assert plain.rendered == explicit.rendered
    assert plain.final_reward == explicit.final_reward
    assert [h.held_out_reward for h in plain.history] == \
           [h.held_out_reward for h in explicit.history]


def test_beam_of_one_is_the_single_head_path():
    """`Beam` has no independent ground truth, so this is what makes it
    trustworthy: on the shape the engine can run today it must agree exactly."""
    plain = _run()
    beamed = _run(policies=Policies(selection=Beam(1)))
    assert beamed.rendered == plain.rendered
    assert beamed.final_reward == plain.final_reward


@pytest.mark.parametrize("policy", [
    SingleHead(), Beam(1), Beam(4), ParetoFrontier(mode="topk_aggregate"),
    Archive(sampling="uniform"), Archive(sampling="novelty"), MCTS(),
])
def test_every_policy_runs_on_the_single_candidate_path(policy):
    """A policy that cannot cope with an archive of one is not usable at all --
    that is the shape every run starts in."""
    result = _run(policies=Policies(selection=policy))
    assert result.error is None
    assert result.final_reward > 0.0


class _Wanderer:
    """Names a starting point that is not the head. Neither driver can honour it."""

    def __init__(self):
        self.calls = 0

    def select(self, ctx, n):
        self.calls += 1
        return [Candidate(artifact_id=ctx.head.artifact_id,
                          version=ctx.head.version + 7)] * n


def test_a_multi_head_answer_is_refused_rather_than_collapsed():
    with pytest.raises(NotImplementedError) as excinfo:
        _run(policies=Policies(selection=_Wanderer()))
    assert "one live branch" in str(excinfo.value)


# -- both drivers, one answer ------------------------------------------------


def test_the_barrier_free_loop_asks_the_selection_policy():
    """`selection` was listed as wired in `async_evolve` and never consulted.

    The bundle's own check let it through, so the field was accepted and
    dropped -- a caller who passed a beam and read a final reward had every
    reason to believe the beam ran. Counting the calls is the only way to see
    it: with one live candidate every shipped policy answers "the head", so the
    run's numbers are identical whether the policy was asked or not.
    """
    class Counting(SingleHead):
        def __init__(self):
            self.calls = 0

        def select(self, ctx, n):
            self.calls += 1
            return super().select(ctx, n)

    policy = Counting()
    result = _run_async(policies=Policies(selection=policy))
    assert result.error is None
    assert policy.calls >= 1, "async_evolve finished without asking where to start"


def test_a_multi_head_answer_is_refused_on_the_async_path_too():
    """The refusal is the shared one, and it reaches the caller's thread.

    Both halves matter. Raising at all is parity with `evolve`; raising
    `MultiHeadUnsupported` is what stops the merger from filing it under
    "the provider flaked" and retrying until the sweep budget runs out.
    """
    policy = _Wanderer()
    with pytest.raises(MultiHeadUnsupported) as excinfo:
        _run_async(policies=Policies(selection=policy))
    assert "one live branch" in str(excinfo.value)
    assert policy.calls >= 1


def test_the_refusal_stays_catchable_as_what_it_used_to_be():
    """`MultiHeadUnsupported` gained a base; it must not have lost one.

    Callers catch `NotImplementedError` -- the type this raised before it had a
    name, and the type the sync test above still asks for.
    """
    assert issubclass(MultiHeadUnsupported, NotImplementedError)
    assert issubclass(MultiHeadUnsupported, ContractError)


# -- the policies themselves -------------------------------------------------


def test_single_head_gives_every_worker_the_head():
    head = _c(3, score=0.5)
    assert SingleHead().select(_ctx(head), 4) == [head] * 4


def test_beam_spreads_workers_over_its_best_k():
    best, mid, worst = _c(1, 0.9), _c(2, 0.5), _c(3, 0.1)
    chosen = Beam(2).select(_ctx(best, mid, worst, head=best), 4)
    assert {c.version for c in chosen} == {1, 2}
    assert [c.version for c in chosen] == [1, 2, 1, 2], "round-robin, not all on one"


def test_an_unscored_candidate_is_explored_not_ranked_last():
    """`score=None` means unmeasured. Sorting it as the worst is how a beam
    collapses onto one line of descent and stops being a beam."""
    scored, fresh = _c(1, 0.9), _c(2, None)
    chosen = Beam(1).select(_ctx(scored, fresh, head=scored), 2)
    assert all(c.version == 2 for c in chosen)


def test_a_beam_narrower_than_one_is_refused():
    with pytest.raises(ValueError):
        Beam(0)


def test_pareto_keeps_the_specialist_that_aggregate_ranking_drops():
    """The difference between GEPA's rule and EvoSkill's, in one assertion."""
    allrounder = _c(1, 0.6, {"t1": 0.6, "t2": 0.6})
    specialist = _c(2, 0.5, {"t1": 1.0, "t2": 0.0})
    ctx = _ctx(allrounder, specialist, head=allrounder, n=2)

    per_instance = ParetoFrontier(mode="per_instance").select(ctx, 2)
    assert {c.version for c in per_instance} == {1, 2}

    topk = ParetoFrontier(mode="topk_aggregate", k=1).select(ctx, 2)
    assert {c.version for c in topk} == {1}, "aggregate ranking drops the specialist"


def test_per_instance_pareto_refuses_to_silently_become_aggregate_ranking():
    """Falling back would report EvoSkill's rule under GEPA's name, which is the
    exact confusion this class exists to remove."""
    with pytest.raises(ValueError, match="per_task"):
        ParetoFrontier(mode="per_instance").select(_ctx(_c(1, 0.6), _c(2, 0.5)), 2)


def test_pareto_mode_is_checked_at_construction():
    with pytest.raises(ValueError):
        ParetoFrontier(mode="whatever")


def test_pareto_front_treats_a_missing_task_score_as_zero():
    """Explicit: skipping the task instead would let a candidate dominate by
    having been measured on less."""
    full = _c(1, per_task={"t1": 0.5, "t2": 0.5})
    partial = _c(2, per_task={"t1": 0.9})
    front = pareto_front([full, partial], tasks=["t1", "t2"])
    assert {c.version for c in front} == {1, 2}


def test_archive_novelty_moves_off_a_candidate_it_has_already_used():
    """Without the novelty term an archive is a greedy hill climb wearing a
    different name."""
    # The temperature is low and the gap wide on purpose: with a near-tie the
    # softmax is close to uniform and forty draws would be measuring noise.
    stale = _c(1, 0.9, selected=50)
    fresh = _c(2, 0.3, selected=0)
    ctx = _ctx(stale, fresh, head=stale, n=40)

    greedy = Archive(sampling="performance", temperature=0.2, seed=0).select(ctx, 40)
    assert sum(c.version == 1 for c in greedy) > sum(c.version == 2 for c in greedy), \
        "performance sampling must prefer the better candidate"

    picks = Archive(sampling="novelty", temperature=0.2, seed=0).select(ctx, 40)
    assert sum(c.version == 2 for c in picks) > sum(c.version == 1 for c in picks), \
        "novelty must move off a candidate that has been the parent 50 times"


def test_archive_sampling_is_reproducible():
    ctx = _ctx(_c(1, 0.9), _c(2, 0.5), _c(3, 0.7), n=8)
    a = [c.version for c in Archive(seed=3).select(ctx, 8)]
    b = [c.version for c in Archive(seed=3).select(ctx, 8)]
    assert a == b, "a seeded comparison is meaningless if the archive drifts"


@pytest.mark.parametrize("kwargs", [{"sampling": "nope"}, {"temperature": 0.0}])
def test_archive_arguments_are_checked(kwargs):
    with pytest.raises(ValueError):
        Archive(**kwargs)


def test_mcts_tries_an_unvisited_candidate_first():
    visited = _c(1, 0.9, selected=10)
    unvisited = _c(2, None, selected=0)
    chosen = MCTS().select(_ctx(visited, unvisited, head=visited, n=1), 1)
    assert chosen[0].version == 2


def test_mcts_spreads_a_batch_across_distinct_arms():
    """Sending every worker to one arm makes a batch worth one rollout of
    information, which is the opposite of what a batch is for."""
    ctx = _ctx(_c(1, 0.9, selected=5), _c(2, 0.8, selected=5),
               _c(3, 0.7, selected=5), n=3)
    chosen = MCTS().select(ctx, 3)
    assert len({c.version for c in chosen}) == 3
