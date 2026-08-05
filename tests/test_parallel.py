import pytest

from agentdescent.domains.router import RouterSkill
from agentdescent.evolvable import Diff
from agentdescent.parallel import (
    assign_key_sections,
    PipelineChain,
    SectionViolation,
    TensorParallelMerge,
    assign_sections,
    section_of,
    shard_round_robin,
)


# -- DP ----------------------------------------------------------------------


def test_dp_sharding_is_disjoint_and_complete():
    shards = shard_round_robin(list(range(10)), 3)
    assert sum(len(s) for s in shards) == 10
    flat = sorted(x for s in shards for x in s)
    assert flat == list(range(10))


# -- TP ----------------------------------------------------------------------


def test_section_assignment_is_stable():
    assert section_of("kw01", 4) == section_of("kw01", 4)
    assigns = assign_sections(["w0", "w1", "w2", "w3", "w4"], 4)
    assert set(assigns.values()) <= {0, 1, 2, 3}


def test_tp_merge_is_conflict_free_union():
    base = RouterSkill("hot", table={})
    tp = TensorParallelMerge(n_sections=3)
    # build one diff per section, each only touching keys in its own section.
    section_diffs = []
    keys_by_section = {0: [], 1: [], 2: []}
    for i in range(30):
        kw = f"kw{i:02d}"
        keys_by_section[section_of(kw, 3)].append(kw)
    for sec, keys in keys_by_section.items():
        ops = {k: f"label{sec}" for k in keys[:3]}
        section_diffs.append((sec, Diff(f"d{sec}", "hot", ops)))
    merged, ok = tp.merge(base, section_diffs)
    assert ok
    # union of all section ops landed.
    assert sum(len(d.ops) for _, d in section_diffs) == len(merged.table)


def test_tp_rejects_out_of_section_edit():
    tp = TensorParallelMerge(n_sections=3)
    # find a key that is NOT in section 0.
    bad_key = next(f"kw{i:02d}" for i in range(50) if section_of(f"kw{i:02d}", 3) != 0)
    with pytest.raises(SectionViolation):
        tp.validate(Diff("d", "hot", {bad_key: "x"}), section=0)


# -- PP ----------------------------------------------------------------------


def test_pp_blames_earliest_failing_stage():
    chain = PipelineChain(["lit-review", "mol-engine", "hpc-submit"])
    # downstream failed but so did an upstream stage -> blame flows upstream.
    blame = chain.blame({"lit-review": False, "mol-engine": True, "hpc-submit": False})
    assert blame == "lit-review"
    # only the downstream stage failed -> it owns the blame.
    blame = chain.blame({"lit-review": True, "mol-engine": True, "hpc-submit": False})
    assert blame == "hpc-submit"
    # all green -> nobody to blame.
    assert chain.blame({s: True for s in chain.stages}) is None


def test_pp_upstream_lookup():
    chain = PipelineChain(["a", "b", "c"])
    assert chain.upstream_of("c") == ["a", "b"]
    assert chain.upstream_of("a") == []


# -- pluggable parallel strategies (DP / TP / PP + custom) -------------------

from agentdescent.parallel import (
    assign_key_sections,
    DataParallel,
    TensorParallel,
    PipelineParallel,
    ParallelStrategy,
    WorkUnit,
)

KEYS = [f"kw{i:02d}" for i in range(24)]


def test_data_parallel_partitions_keys_disjointly():
    plan = DataParallel().plan(4, round_index=0, keys=KEYS)
    assert len(plan) == 4
    seen = [k for u in plan for k in u.keys]
    assert sorted(seen) == sorted(KEYS)          # a partition: covers all, no dup
    assert len(seen) == len(set(seen))


def test_data_parallel_rotates_ownership_across_rounds():
    a = {u.worker: set(u.keys) for u in DataParallel().plan(4, 0, KEYS)}
    b = {u.worker: set(u.keys) for u in DataParallel().plan(4, 1, KEYS)}
    assert a != b                                # ownership rotates by round


def test_tensor_parallel_shards_tasks_and_assigns_sections_independently():
    """`plan` receives TASK ids; the section is about the ARTIFACT.

    These used to be the same call: `plan` filtered task ids through
    `section_of`, and `evolve` then enforced the section against the artifact keys
    the resulting diff wrote. Two unrelated key spaces, so a worker's legal tasks
    said nothing about its legal edits.
    """
    plan = TensorParallel(n_sections=4, keys=list("abcd")).plan(4, 0, KEYS)
    assert [u.section for u in plan] == [0, 1, 2, 3], "each worker owns one section"
    # tasks are sharded data-parallel: every task assigned exactly once
    seen = [k for u in plan for k in u.keys]
    assert sorted(seen) == sorted(KEYS) and len(seen) == len(set(seen))
    # ...and no worker is starved of tasks just because of how its section hashed
    assert all(u.keys for u in plan), "a worker with no tasks cannot contribute"


def test_sections_partition_the_declared_key_space():
    """Every section non-empty, disjoint, and covering -- a partition, not a hash.

    `section_of` is `stable_hash(key) % n`, a hash *bucket*. On the four categories
    the TP tests use it put two keys in one section and left another owning
    nothing at all, so the worker holding it could never commit a single edit.
    """
    keys = ["alpha", "beta", "gamma", "delta"]
    mapping = assign_key_sections(keys, 4)
    assert sorted(mapping) == sorted(keys), "every key is assigned"
    assert sorted(mapping.values()) == [0, 1, 2, 3], "every section owns exactly one"
    # balanced when the split is uneven, and stable across calls
    uneven = assign_key_sections(["a", "b", "c", "d", "e"], 2)
    sizes = sorted(list(uneven.values()).count(s) for s in set(uneven.values()))
    assert sizes == [2, 3], f"sections should differ by at most one, got {sizes}"
    assert uneven == assign_key_sections(["e", "d", "c", "b", "a"], 2), \
        "assignment must not depend on input order"


def test_pipeline_parallel_assigns_stages():
    pp = PipelineParallel(stages=["a", "b", "c"])
    plan = pp.plan(3, 0, KEYS)
    assert [u.stage for u in plan] == [0, 1, 2]
    assert all(u.keys == KEYS for u in plan)     # every stage sees all keys
    assert pp.chain().stages == ["a", "b", "c"]


def test_custom_strategy_is_structural():
    class Blocks:
        name = "blocks"
        def plan(self, n_workers, round_index, keys):
            keys = list(keys)
            size = (len(keys) + n_workers - 1) // n_workers
            return [WorkUnit(worker=i, keys=keys[i * size:(i + 1) * size])
                    for i in range(n_workers)]

    assert isinstance(Blocks(), ParallelStrategy)
    plan = Blocks().plan(3, 0, KEYS)
    assert sorted(k for u in plan for k in u.keys) == sorted(KEYS)


# ---------------------------------------------------------------------------
# ClusterParallel — UCB over task clusters, and the channel that makes it possible
# ---------------------------------------------------------------------------


def _cluster_tasks(n=18, k=3):
    from agentdescent.evolution import Task

    return [Task(id=f"c{i % k}-{i}", prompt=f"q{i}", meta={"gold": str(i)})
            for i in range(n)]


def _cluster_of(task_id):
    return task_id.split("-")[0]


def test_a_lease_hands_a_worker_one_whole_cluster():
    """The point of leasing clusters rather than sharding tasks: a worker sees a
    coherent slice of the distribution, not one task from each of them."""
    from agentdescent.parallel import ClusterParallel

    cp = ClusterParallel(cluster_of=_cluster_of)
    units = cp.plan(3, 0, [t.id for t in _cluster_tasks()])
    assert units and all(u.keys for u in units)
    for unit in units:
        assert len({_cluster_of(k) for k in unit.keys}) == 1, (
            f"worker {unit.worker} was handed tasks from several clusters: {unit.keys}")


def test_feedback_reaches_the_scheduler():
    """`plan` was a pure function of its arguments, so a strategy could not learn
    from the rollouts it dispatched -- which is why UCB over clusters lived only
    in the reference runtime."""
    from agentdescent.parallel import ClusterParallel

    cp = ClusterParallel(cluster_of=_cluster_of)
    units = cp.plan(3, 0, [t.id for t in _cluster_tasks()])
    before = dict(cp.scheduler.clusters["c0"].__dict__)

    cp.observe(units[0], "c0-0", 0.0)          # a failure: room to learn
    after = cp.scheduler.clusters["c0"]
    assert after.n_evidence > before["n_evidence"]
    assert after.recent_value > before["recent_value"], (
        "a failing cluster did not become more attractive to the scheduler")


def test_a_solved_cluster_loses_its_pull():
    """The difficulty filter: a cluster that always passes carries no gradient."""
    from agentdescent.parallel import ClusterParallel

    cp = ClusterParallel(cluster_of=_cluster_of)
    units = cp.plan(3, 0, [t.id for t in _cluster_tasks()])
    for _ in range(30):
        cp.observe(units[0], "c0-0", 1.0)      # solved, every time
    solved = cp.scheduler.clusters["c0"]
    assert solved.pass_rate > 0.9
    assert cp.scheduler._difficulty_weight(solved) < \
        cp.scheduler._difficulty_weight(cp.scheduler.clusters["c1"]), (
        "an all-pass cluster is still weighted like one that still fails")


def test_an_unknown_task_id_is_ignored_rather_than_raising():
    """`cluster_of` is caller code over ids the caller supplied; a stray one must
    not take the round down from inside a worker thread."""
    from agentdescent.parallel import ClusterParallel

    cp = ClusterParallel(cluster_of=_cluster_of)
    units = cp.plan(3, 0, [t.id for t in _cluster_tasks()])
    cp.observe(units[0], "nope-999", 0.0)      # no such cluster


def test_evolve_drives_it_end_to_end():
    """The hook has to be load-bearing in the round body, not just callable."""
    import warnings

    from agentdescent.evolution import AppendRules, evolve
    from agentdescent.parallel import ClusterParallel

    tasks = _cluster_tasks(n=18, k=3)
    cp = ClusterParallel(cluster_of=_cluster_of)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evolve(
            tasks, lambda t, o: 1.0 if o == t.meta["gold"] else 0.0,
            run=lambda rendered, t: t.meta["gold"] if t.id in rendered else "?",
            propose=lambda r, t, o, s: t.id, strategy=AppendRules(),
            parallel=cp, rounds=6, n_workers=3, max_concurrency=3,
            held_out_frac=0.4)

    assert result.error is None
    assert cp.scheduler is not None, "evolve() never asked the strategy to plan"
    total = sum(c.n_evidence for c in cp.scheduler.clusters.values())
    assert total > 0, (
        "no rollout outcome reached the strategy -- the observe hook is not wired")


def test_a_strategy_without_the_hook_is_untouched():
    """`observe` is optional; DataParallel and TensorParallel do not define it."""
    from agentdescent.parallel import DataParallel, TensorParallel

    assert not hasattr(DataParallel(), "observe")
    assert not hasattr(TensorParallel(n_sections=2, keys=["a", "b"]), "observe")
