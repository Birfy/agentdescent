"""The ports' published selection rules, now shipped -- and provably the same.

Two ports kept a hand-written selection rule rather than using the shipped
policy, each for a stated reason: GEPA samples its frontier by win frequency
where `ParetoFrontier` walked it round-robin, and DGM weights by
``sigmoid(10*(s-0.5))`` where `Archive('novelty')` uses a temperature softmax.
Both comments said the same thing -- "close enough to look right and wrong
enough to change a measured run".

The fix is not to pick one. It is to make the shipped policy able to express the
published rule, so the fidelity difference is an argument a result can record
instead of a class a reader has to find. These tests are what makes that
claimable: they compare the shipped mode against the rule it replaces, on the
same seeded rng, and a mismatch anywhere is a port whose measured numbers moved.

The rng is the part that is easy to get wrong and impossible to see. Both ports
draw from a *shared, stateful* `random.Random` that their aggregator owns, and
both spend exactly one `choices` call per selection. A shipped policy that
re-seeded per call, or drew ``k=n`` where upstream draws ``k=1``, would agree on
the distribution and disagree on every actual run.
"""

import math
import random

import pytest

from agentdescent.selection import (
    Archive, Candidate, ParetoFrontier, SelectionContext,
    pareto_win_frequency, sigmoid_novelty_weights,
)


def _cand(version, score=None, per_task=None, selected=0):
    return Candidate(artifact_id="a", version=version, score=score,
                     state={"k": f"v{version}"}, per_task=per_task or {},
                     selected=selected)


# -- GEPA: Algorithm 2 --------------------------------------------------------


ROWS = [
    [0.9, 0.1, 0.4],     # a specialist: wins instance 0
    [0.3, 0.9, 0.5],     # a specialist: wins instance 1
    [0.4, 0.4, 0.9],     # a specialist: wins instance 2
    [0.2, 0.2, 0.2],     # dominated by everything, wins nothing
    [0.9, 0.9, 0.9],     # wins all three
]
TASKS = ["t0", "t1", "t2"]


def _ctx_from_rows(rows):
    cands = tuple(_cand(i, score=sum(r) / len(r),
                        per_task=dict(zip(TASKS, r)))
                  for i, r in enumerate(rows))
    return SelectionContext(head=cands[0], candidates=cands, n_workers=4)


@pytest.mark.parametrize("rows", [
    ROWS,
    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    [[1.0, 0.0], [0.0, 1.0]],
    [[0.5, 0.5], [0.5, 0.5]],                 # identical rows: both win both
    [[0.2, 0.2], [0.9, 0.9], [0.9, 0.9]],     # a tie at the top
    [[0.0]],
])
def test_the_shipped_frontier_is_the_ports_frontier(rows):
    """Against GEPA's own implementation, not against a transcription of it."""
    from examples.gepa.gepa_prompt_evolution import pareto_frontier

    assert pareto_win_frequency(rows) == pareto_frontier(rows)


def test_the_frontier_is_the_instance_winners_not_the_undominated_set():
    """`pareto_win_frequency` and `pareto_front` are different sets, on purpose.

    ``[0.5, 0.5]`` is dominated by neither specialist -- so plain Pareto keeps
    it -- and ties the best on nothing, so Algorithm 2 never admits it. A class
    that ran one under the other's name would develop a candidate GEPA would
    not, which is the whole reason the port kept its own copy.
    """
    rows = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    kept, freq = pareto_win_frequency(rows)
    assert kept == {0, 1} and freq == {0: 1, 1: 1}

    cands = tuple(_cand(i, per_task=dict(zip(["t0", "t1"], r)))
                  for i, r in enumerate(rows))
    from agentdescent.selection import pareto_front
    plain = pareto_front(cands, tasks=["t0", "t1"])
    assert {c.version for c in plain} == {0, 1, 2}


def test_a_dominated_row_is_pruned_and_every_survivor_carries_weight():
    kept, freq = pareto_win_frequency(ROWS)
    assert 3 not in kept, "a strictly dominated row must be pruned"
    assert all(freq[c] > 0 for c in kept), "a survivor with no win has no weight"


def _gepa_reference(rows, rng):
    """GEPA's inline rule, transcribed: frontier, then one weighted draw."""
    kept, freq = pareto_win_frequency(rows)
    cands = [c for c in sorted(kept) if freq[c] > 0]
    if not cands:
        return max(range(len(rows)), key=lambda c: sum(rows[c]))
    return rng.choices(cands, weights=[freq[c] for c in cands], k=1)[0]


def test_shipped_win_frequency_draws_what_the_port_drew():
    """Same rng object, same call order, same parent -- 200 consecutive draws.

    One shared stream stepped in lockstep, not two streams compared by
    distribution: a policy that drew the right parent from the wrong offset
    would pass a distribution test and change every seeded run.
    """
    rows = [[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]]
    ctx = _ctx_from_rows(rows)
    theirs, ours = random.Random(7), random.Random(7)
    policy = ParetoFrontier(mode="win_frequency", rng=ours)
    for _ in range(200):
        expected = _gepa_reference(rows, theirs)
        got = policy.select(ctx, 4)
        assert [c.version for c in got] == [expected] * 4


def test_win_frequency_spends_one_draw_per_batch_not_one_per_worker():
    """Algorithm 2 selects *a* parent per iteration; every worker mutates it.

    Drawing per worker would consume `n` numbers where upstream consumes one,
    so the streams diverge from the second selection onwards even though the
    first parent matches.
    """
    ctx = _ctx_from_rows([[0.9, 0.1], [0.1, 0.9]])
    rng = random.Random(3)
    ParetoFrontier(mode="win_frequency", rng=rng).select(ctx, 8)
    spent = random.Random(3)
    spent.choices([0, 1], weights=[1, 1], k=1)
    assert rng.random() == spent.random(), "more than one draw was consumed"


def test_win_frequency_needs_per_task_rows():
    ctx = SelectionContext(head=_cand(1, 0.5), candidates=(_cand(1, 0.5), _cand(2, 0.6)))
    with pytest.raises(ValueError, match="Algorithm 2"):
        ParetoFrontier(mode="win_frequency").select(ctx, 1)


def test_identical_rows_all_survive_with_equal_weight():
    """Not the degenerate case, which is the easy thing to assume.

    Two identical rows tie the best on every instance, so both are instance
    winners and neither strictly dominates the other: the frontier is both, at
    equal weight, and the draw is a coin flip. The aggregate fallback is for a
    matrix with no instances at all, which `select` rejects before reaching it.
    """
    kept, freq = pareto_win_frequency([[0.5, 0.5], [0.5, 0.5]])
    assert kept == {0, 1} and freq == {0: 2, 1: 2}


# -- DGM: choose_selfimproves -------------------------------------------------


def _dgm_reference_weights(scores, children):
    """DGM's inline formula, transcribed from `DGM_outer.py:score_child_prop`."""
    raw = []
    for s, c in zip(scores, children):
        raw.append((1.0 / (1.0 + math.exp(-10.0 * (s - 0.5)))) * (1.0 / (1.0 + c)))
    total = sum(raw) or 1.0
    return [r / total for r in raw]


@pytest.mark.parametrize("scores,children", [
    ([0.9, 0.5, 0.1], [0, 0, 0]),
    ([0.9, 0.9], [0, 5]),
    ([0.0, 0.0], [0, 0]),            # the all-zero vector: divide by 1, not 0
    ([1.0], [0]),
    ([0.5, 0.5, 0.5], [3, 1, 0]),
])
def test_shipped_weights_are_the_ports_weights(scores, children):
    assert sigmoid_novelty_weights(scores, children) == \
           _dgm_reference_weights(scores, children)


def test_sigmoid_novelty_draws_what_the_port_drew():
    """Same rng object, same single `choices(..., k=n)` call, same parents."""
    pool = [_cand(0, 0.9, selected=0), _cand(1, 0.5, selected=2),
            _cand(2, 0.1, selected=0), _cand(3, 0.7, selected=1)]
    ctx = SelectionContext(head=pool[0], candidates=tuple(pool), n_workers=3)
    theirs, ours = random.Random(5), random.Random(5)
    policy = Archive(sampling="sigmoid_novelty", rng=ours)
    weights = _dgm_reference_weights([c.score for c in pool],
                                     [c.selected for c in pool])
    for _ in range(200):
        expected = [c.version for c in theirs.choices(pool, weights=weights, k=3)]
        assert [c.version for c in policy.select(ctx, 3)] == expected


def test_sigmoid_novelty_is_not_novelty():
    """If the two agreed, the port would have had no reason to hand-write one.

    A sigmoid at gain 10 is nearly a step at 0.5; a temperature-1 softmax over
    [0, 1] spans a factor of e. They disagree most where an archive lives.
    """
    pool = [_cand(0, 0.9), _cand(1, 0.1)]
    ctx = SelectionContext(head=pool[0], candidates=tuple(pool))
    dgm = Archive(sampling="sigmoid_novelty", rng=random.Random(0))
    soft = Archive(sampling="novelty", rng=random.Random(0))
    dgm_picks = [dgm.select(ctx, 1)[0].version for _ in range(400)]
    soft_picks = [soft.select(ctx, 1)[0].version for _ in range(400)]
    # DGM's sigmoid nearly excludes the 0.1 candidate; the softmax does not.
    assert dgm_picks.count(1) < soft_picks.count(1) / 2


def test_an_unscored_candidate_is_zero_here_not_the_archive_mean():
    """`novelty` gives an unscored candidate 0.5; `sigmoid_novelty` gives it 0.

    0.5 is the sigmoid's inflection point, so borrowing the other mode's
    convention would place an unevaluated agent at exactly the median weight --
    the most favourable spot on the curve, for a candidate nothing is known
    about.
    """
    pool = [_cand(0, 1.0), _cand(1, None)]
    ctx = SelectionContext(head=pool[0], candidates=tuple(pool))
    weights = sigmoid_novelty_weights([1.0, 0.0], [0, 0])
    rng, mirror = random.Random(2), random.Random(2)
    got = Archive(sampling="sigmoid_novelty", rng=rng).select(ctx, 5)
    expected = mirror.choices(pool, weights=weights, k=5)
    assert [c.version for c in got] == [c.version for c in expected]
