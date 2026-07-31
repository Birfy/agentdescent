"""Tensor parallelism must actually be tensor-parallel -- and must not cost more
than it buys.

TP's promise is that each worker owns a *disjoint section of the artifact*, which
is what makes the merge a conflict-free union. Two things went wrong, and the
second was invisible because the tests only asserted the first.

**Enforcement without a key space.** `plan()` sharded **task ids** through
`section_of`, while `evolve()` enforced the section against the **artifact keys**
the resulting diff wrote. Two unrelated key spaces: a worker's legal tasks said
nothing about its legal edits, so most of every worker's proposals were discarded
before reaching the aggregator -- measured at 75-88% -- with no report, no
`MergeReport`, and nothing in `outcomes()`.

**Assertions that could not fail.** The old tests asserted `res.state` was
truthy, which one surviving key out of four satisfies. They passed on exactly the
behaviour above. So the tests here assert *throughput* and *coverage*, not
liveness: how many proposals reached the aggregator, and whether every section is
reachable at all.
"""

import warnings

import pytest

from agentdescent.aggregator import Aggregator
from agentdescent.evolution import AppendRules, KeyedRules, SingleSlot, Task, evolve
from agentdescent.parallel import DataParallel, PipelineParallel, TensorParallel

CATS = ["alpha", "beta", "gamma", "delta"]


def _reward(task, output):
    return 1.0 if output == "yes" else 0.0


def _tasks(n=24):
    return [Task(id=f"t{i}", prompt="q") for i in range(n)]


class _Counting(Aggregator):
    """Counts what actually reached the optimizer -- the quantity TP must preserve."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ingested = 0

    def ingest(self, card):
        self.ingested += 1
        super().ingest(card)


def _counting_factory():
    holder = {}

    def factory(ledger, verifier, audit, config, policy):
        agg = _Counting(ledger, verifier, audit, config, staleness_policy=policy)
        holder["agg"] = agg
        return agg
    factory.holder = holder
    return factory


class _ProposesEveryCategory:
    """Every worker proposes across all four categories, in turn.

    The section guard is then the only thing deciding which land -- which is what
    TP is for. The old fixture claimed to "give each worker its own category" but
    keyed the category off the task id, so the mapping it asserted never existed.
    """

    def solve(self, rendered, task):
        return "yes" if "rule" in rendered else "no"

    def propose(self, rendered, task, output, reward):
        return f"{CATS[int(task.id[1:]) % len(CATS)]}: rule"


def _run(parallel, agent=None, strategy=None, n=24, rounds=8):
    factory = _counting_factory()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = evolve(_tasks(n), _reward,
                     agent=agent or _ProposesEveryCategory(),
                     strategy=strategy or KeyedRules(categories=CATS),
                     parallel=parallel, rounds=rounds, n_workers=4,
                     max_concurrency=4, aggregator_factory=factory)
    return res, factory.holder["agg"].ingested


# -- the guarantee: enforcement holds ------------------------------------------


def test_only_the_owning_worker_may_edit_a_section():
    class _AlwaysEditsAlpha:
        def solve(self, rendered, task):
            return "yes" if "a rule" in rendered else "no"

        def propose(self, rendered, task, output, reward):
            return "alpha: a rule"

    tp = TensorParallel(n_sections=4, keys=CATS)
    owner = f"w{tp.section_map()['alpha']}"
    res, _ = _run(tp, agent=_AlwaysEditsAlpha(), rounds=6)
    authors = [line for line in res.ledger_log if "alpha" in line]
    assert authors, "the owning worker's edit should still land"
    for line in authors:
        assert owner in line, f"an out-of-section worker committed: {line}"


# -- what the old tests could not see: throughput and coverage -----------------


def test_tensor_parallel_delivers_proposals_at_a_comparable_rate_to_dp():
    """The section guard must gate *who* edits a key, not throw the work away.

    Every worker proposes across all four categories and owns one of them, so
    about a quarter of each worker's proposals are legal. What must not happen is
    the old behaviour, where legality was decided by an unrelated hash and the
    delivered fraction collapsed to ~12%.
    """
    _, dp_ingested = _run(DataParallel())
    _, tp_ingested = _run(TensorParallel(n_sections=4, keys=CATS))
    assert dp_ingested > 0, "premise: DP delivers proposals at all"
    assert tp_ingested >= dp_ingested * 0.2, (
        f"TP delivered {tp_ingested} proposals against DP's {dp_ingested}; the "
        "section guard is discarding work rather than routing it")


def test_routing_tasks_to_their_own_section_costs_nothing_in_throughput():
    """The point of `route=`: a worker only sees tasks whose edits it may make.

    Without it a worker is handed a data-parallel shard and most of what it
    proposes lands outside its own section -- honest, reported, but wasteful. With
    it, TP delivers every proposal DP would, while keeping the disjoint-section
    guarantee that makes the merge a union.
    """
    def route(task_id):
        return CATS[int(task_id[1:]) % len(CATS)]

    _, dp = _run(DataParallel())
    _, unrouted = _run(TensorParallel(n_sections=4, keys=CATS))
    res, routed = _run(TensorParallel(n_sections=4, keys=CATS, route=route))
    assert routed == dp, f"routed TP delivered {routed}, DP delivered {dp}"
    assert routed > unrouted, "routing should recover the proposals TP was rejecting"
    assert res.outcomes().get("section-violation", 0) == 0, \
        "a routed run should produce no out-of-section proposals at all"


def test_every_section_is_reachable():
    """No section may own nothing.

    `section_of` is a hash bucket: on these four categories it put two in one
    section and left a third owning nothing, so the worker holding it could never
    commit anything at all. A partition makes every section reachable.
    """
    owners = TensorParallel(n_sections=4, keys=CATS).section_map()
    assert sorted(owners.values()) == [0, 1, 2, 3], \
        f"sections are not a partition of the key space: {owners}"


def test_section_violations_are_reported_not_swallowed():
    """A dropped proposal must be visible in `outcomes()`.

    Otherwise a TP run that discards most of its work looks exactly like one whose
    reflector had nothing useful to say -- and those need opposite fixes.
    """
    class _AlwaysWritesOneCategory:
        def solve(self, rendered, task):
            return "yes" if "rule" in rendered else "no"

        def propose(self, rendered, task, output, reward):
            return "alpha: a rule"          # only one worker may write this

    res, _ = _run(TensorParallel(n_sections=4, keys=CATS),
                  agent=_AlwaysWritesOneCategory(), rounds=6)
    assert res.outcomes().get("section-violation", 0) > 0, (
        "workers were blocked every round and nothing said so; "
        f"outcomes()={res.outcomes()}")


# -- incompatible pairings fail up front, not one silent diff at a time --------


def test_append_rules_cannot_be_tensor_parallelised():
    """Content-addressed keys have no fixed key space, so TP cannot partition them.

    This was the default pairing, and it silently discarded ~88% of proposals.
    """
    with pytest.raises(ValueError) as excinfo:
        _run(TensorParallel(n_sections=4), strategy=AppendRules())
    msg = str(excinfo.value)
    assert "AppendRules" in msg and "key space" in msg
    assert "DataParallel" in msg or "KeyedRules" in msg, "the error must say what to do"


def test_more_sections_than_keys_is_refused():
    """A section owning nothing means a worker that can never commit."""
    with pytest.raises(ValueError) as excinfo:
        _run(TensorParallel(n_sections=4), strategy=SingleSlot(initial_value="v"))
    assert "SingleSlot" in str(excinfo.value)
    assert "1 key" in str(excinfo.value)


def test_pipeline_parallel_is_refused_rather_than_silently_ignored():
    """`WorkUnit.stage` was never read, so PP degraded to redundant DP."""
    with pytest.raises(ValueError) as excinfo:
        _run(PipelineParallel(stages=["a", "b", "c"]))
    msg = str(excinfo.value)
    assert "PipelineParallel" in msg
    assert "PipelineChain" in msg, "the error must point at the usable primitive"


def test_data_parallel_is_unrestricted():
    """The section check must apply only to TP -- DP has no sections."""
    res, _ = _run(DataParallel())
    assert res.state, "DP should still let any worker edit any key"
    assert "section-violation" not in res.outcomes()
