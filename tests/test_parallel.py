import pytest

from concordia.domains.router import RouterSkill
from concordia.evolvable import Diff
from concordia.parallel import (
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

from concordia.parallel import (
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


def test_tensor_parallel_keys_stay_in_their_section():
    plan = TensorParallel(n_sections=4).plan(4, 0, KEYS)
    for u in plan:
        assert all(section_of(k, 4) == u.section for k in u.keys)
    # union covers every key exactly once (disjoint sections)
    seen = [k for u in plan for k in u.keys]
    assert sorted(seen) == sorted(KEYS) and len(seen) == len(set(seen))


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
