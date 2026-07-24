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
