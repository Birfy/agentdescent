"""Multi-artifact evolution: ``extra_artifacts=``, PP stages, and atomic contracts.

The library's single-artifact heritage is the constraint these tests pin down.
Before this feature three things were impossible at once -- ``evolve()``
registered exactly one artifact, ``PipelineParallel`` was refused (it needs one
artifact per stage and a single-artifact run has nowhere for the stages to live),
and a contract-breaking change could not land through the engine at all (the
ledger's ``_assert_contract`` refuses it, and nothing routed through
``commit_atomic``). The tests here assert all three seams are now real:
registration, cross-artifact diff production by the worker loop, and the atomic
adaptation transaction.
"""

import warnings

import pytest

from agentdescent.evolution import (
    AppendRules,
    EvolvingArtifact,
    Task,
    evolve,
)
from agentdescent.evolvable import Contract, Diff
from agentdescent.ledger import Ledger
from agentdescent.parallel import PipelineParallel


def _reward(task, output):
    return 1.0 if output == "correct" else 0.0


def _ledger(tmp_path, extra_ids=()):
    """A scratch ledger with an 'artifact' plus the given extra artifact ids."""
    def serialize(a):
        return {"state": a.state, "blast_radius": a.blast_radius}

    def deserialize(aid, version, state):
        return EvolvingArtifact(aid, state.get("state", {}), version,
                                state.get("blast_radius", 0.2))

    lg = Ledger(str(tmp_path / "repo"), serialize, deserialize)
    lg.register(EvolvingArtifact("artifact", {"k": "v0"}, blast_radius=0.2))
    for aid in extra_ids:
        if aid != "artifact":
            lg.register(EvolvingArtifact(aid, {}, blast_radius=0.2))
    return lg


def _tasks(n=12):
    return [Task(id=f"t{i}", prompt=f"p{i}") for i in range(n)]


class _Reflector:
    """Fails t0 until a clue is in the artifact, then proposes that clue."""

    def solve(self, rendered, task):
        return "correct" if "clue" in rendered or task.id != "t0" else "wrong"

    def propose(self, rendered, task, output, reward):
        return "t0 clue" if task.id == "t0" else None


def _run(*, extra_artifacts, parallel=None, rounds=4, n_workers=2,
         agent=None, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evolve(_tasks(), _reward, agent=agent or _Reflector(), rounds=rounds,
                      n_workers=n_workers, max_concurrency=n_workers,
                      strategy=AppendRules(), extra_artifacts=extra_artifacts,
                      parallel=parallel, **kw)


class _BreakingRules(AppendRules):
    """AppendRules, but every proposal is a semver-major (contract-breaking) diff.

    Shipped strategies never mark a diff ``contract_breaking`` -- the flag is
    the caller's declaration that a change is a deliberate interface break, and
    only a custom strategy knows when to make it. This is that declaration, so
    the tests can reach the atomic adaptation transaction end to end."""

    def to_diff(self, state, proposal, author, base_version, target):
        d = super().to_diff(state, proposal, author, base_version, target)
        if d is not None:
            d = Diff(d.diff_id, d.target, dict(d.ops), contract_breaking=True,
                     author=d.author)
        return d


# -- registration -------------------------------------------------------------


def test_extra_artifacts_are_registered_in_the_ledger():
    extra = {"skill-b": EvolvingArtifact("skill-b", {}, blast_radius=0.2)}
    res = _run(extra_artifacts=extra)
    commits = [line for line in res.ledger_log if "register skill-b" in line]
    assert commits, "the extra artifact must be registered into the ledger"


def test_single_artifact_default_behaviour_is_unchanged():
    res = _run(extra_artifacts=None)
    assert res.state, "the primary artifact should still evolve"
    assert res.final_reward == 1.0


# -- cross-artifact diffs -----------------------------------------------------


def test_worker_loop_proposes_against_extra_artifacts():
    extra = {"skill-b": EvolvingArtifact("skill-b", {}, blast_radius=0.2)}
    res = _run(extra_artifacts=extra)
    merged = [line for line in res.ledger_log if "-> skill-b" in line]
    assert merged, "a proposal must target the extra artifact and be merged"
    assert any("-> artifact" in line for line in res.ledger_log), (
        "the primary must still receive proposals in a multi-artifact run")


def test_extra_artifacts_carry_their_own_strategy():
    from agentdescent.strategies import SingleSlot

    class _SlotReflector(_Reflector):
        def propose(self, rendered, task, output, reward):
            return "the one true clue" if task.id == "t0" else None

    extra = {"slot-b": EvolvingArtifact(
        "slot-b", {}, blast_radius=0.2, strategy=SingleSlot())}
    res = _run(extra_artifacts=extra, agent=_SlotReflector())
    merged = [line for line in res.ledger_log if "-> slot-b" in line]
    assert merged


# -- PipelineParallel with extra_artifacts ------------------------------------


def test_pipeline_parallel_is_allowed_with_extra_artifacts():
    extra = {"stage-b": EvolvingArtifact("stage-b", {}, blast_radius=0.2)}
    res = _run(extra_artifacts=extra,
               parallel=PipelineParallel(stages=["artifact", "stage-b"]))
    assert any("-> stage-b" in line for line in res.ledger_log), (
        "PP's second stage must have an artifact to propose against")


def test_pipeline_parallel_without_extra_artifacts_is_still_refused():
    with pytest.raises(ValueError) as excinfo:
        _run(extra_artifacts=None,
             parallel=PipelineParallel(stages=["artifact"]))
    assert "PipelineParallel" in str(excinfo.value)


def test_pipeline_parallel_stages_must_name_registered_artifacts():
    extra = {"stage-b": EvolvingArtifact("stage-b", {}, blast_radius=0.2)}
    with pytest.raises(ValueError) as excinfo:
        _run(extra_artifacts=extra,
             parallel=PipelineParallel(stages=["artifact", "ghost-stage"]))
    assert "ghost-stage" in str(excinfo.value)


# -- the atomic adaptation transaction ----------------------------------------


def test_contract_breaking_diff_lands_atomically():
    """A contract-breaking diff with no declared dependents lands atomically
    through `commit_atomic` -- the sanctioned route for a deliberate contract
    change, bypassing `_assert_contract`'s refusal."""
    # No `depends_on`: this breaking change has nothing it must land with.
    extra = {"b": EvolvingArtifact("b", {}, blast_radius=0.2,
                                   contract=Contract(
                                       input_schema="text", output_schema="text",
                                       major=1))}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = evolve(_tasks(), _reward, agent=_Reflector(), rounds=3,
                     n_workers=2, max_concurrency=2, strategy=_BreakingRules(),
                     extra_artifacts=extra)
    log = res.ledger_log
    assert any("-> artifact" in line and "contract" in line for line in log), (
        "a contract-breaking diff must be committed atomically, with a message "
        "that says so")


def test_contract_breaking_with_dependents_but_no_adapters_is_refused(tmp_path):
    """A contract-breaking change to an artifact with declared dependents must
    land *with* its adapted dependents in the same atomic commit. If none are
    supplied, the merge is refused (`missing-adapters`) rather than half-applied
    -- committing just the breaking artifact would leave the dependents
    registered against a superseded contract."""
    from agentdescent.aggregator import Aggregator, AggregatorConfig, _Candidate
    from agentdescent.scheduler import AuditScheduler
    from agentdescent.verifier import ThreeLayerVerifier, VerifierBudget

    lg = _ledger(tmp_path, extra_ids=("artifact", "b"))
    agg = Aggregator(lg, ThreeLayerVerifier(
        eval_fn=lambda a, t: 0.5, held_out=[1, 2, 3],
        budget=VerifierBudget()), AuditScheduler(),
        AggregatorConfig(batch_trigger=1))
    # The engine wires the reverse dependency graph: "artifact" depends on "b".
    agg.contract_dependents = {"artifact": ["b"]}
    snap = lg.snapshot(Ledger.DEV)
    base = snap.get("artifact")
    cand = base.apply(Diff(
        diff_id="d", target="artifact", ops={"k": "v"},
        author="w", contract_breaking=True))
    c = _Candidate(
        artifact_id="artifact", artifact=base, candidate=cand,
        diff=Diff(diff_id="d", target="artifact", ops={"k": "v"},
                  author="w", contract_breaking=True),
        cards=[], survivor_cards=[], head=snap.version,
        fused=False, considered=1, survived=1, discarded=0, conflicts=0,
        base_counts=(2.0, 0.0), cand_counts=(3.0, 0.0),
        base_cheap=1.0, cand_cheap=1.0)
    report = agg._decide(c)
    assert report.category == "missing-adapters", (
        f"breaking change with dependents but no adapters must be refused, "
        f"got {report.category}")
    assert report.committed_version is None, "nothing may commit"
    assert lg.head_version(Ledger.DEV).get("artifact") == 1, (
        "the breaking change must not land without its adapters")


def test_contract_breaking_with_adapters_commits_both_atomically(tmp_path):
    """Supplying the adapted dependents makes the atomic adaptation transaction
    go through: candidate + adapters land in one `commit_atomic`, dependents
    are re-registered against the new contract."""
    from agentdescent.aggregator import Aggregator, AggregatorConfig, _Candidate
    from agentdescent.scheduler import AuditScheduler
    from agentdescent.verifier import ThreeLayerVerifier, VerifierBudget
    lg = _ledger(tmp_path, extra_ids=("artifact", "b"))
    agg = Aggregator(lg, ThreeLayerVerifier(
        eval_fn=lambda a, t: 0.5, held_out=[1, 2, 3],
        budget=VerifierBudget()), AuditScheduler(),
        AggregatorConfig(batch_trigger=1))
    agg.contract_dependents = {"artifact": ["b"]}
    invalidated = []
    agg.on_contract_change = lambda aid: invalidated.append(aid)
    snap = lg.snapshot(Ledger.DEV)
    base = snap.get("artifact")
    b_old = snap.get("b")
    cand = base.apply(Diff(
        diff_id="d", target="artifact", ops={"k": "v"},
        author="w", contract_breaking=True))
    b_adapted = b_old.apply(Diff(
        diff_id="adapter", target="b", ops={"k2": "v2"}, author="w"))
    c = _Candidate(
        artifact_id="artifact", artifact=base, candidate=cand,
        diff=Diff(diff_id="d", target="artifact", ops={"k": "v"},
                  author="w", contract_breaking=True),
        cards=[], survivor_cards=[], head=snap.version,
        fused=False, considered=1, survived=1, discarded=0, conflicts=0,
        adapters=[b_adapted],
        base_counts=(2.0, 0.0), cand_counts=(3.0, 0.0),
        base_cheap=1.0, cand_cheap=1.0)
    report = agg._decide(c)
    assert report.committed_version is not None, "with adapters, it commits"
    after = lg.snapshot(Ledger.DEV)
    assert after.version["artifact"] == 2 and after.version["b"] == 2, (
        "candidate and adapter must land in the same atomic commit")
    assert invalidated == ["artifact"], (
        "dependents must be re-measured after the breaking change")


def test_contract_dependents_are_re_measured_after_a_breaking_change(tmp_path):
    """After a breaking commit on the primary, dependent 'b' loses its cached
    scores (they were measured under the superseded contract). The engine's
    `invalidate_dependents` (wired as `aggregator.on_contract_change`) evicts
    them by the dependent's render."""
    from agentdescent.evalcache import MemoryCache
    from agentdescent.evolution import _Engine

    cache = MemoryCache()
    lg = _ledger(tmp_path, extra_ids=("artifact", "b"))
    b_art = lg.snapshot(Ledger.DEV).get("b")
    key = (b_art.render(), "t0", "")
    cache.get_or_eval(key, lambda: 0.7)

    runtime = type("Runtime", (), {})()   # a stand-in _Runtime with a cache
    runtime.cache = cache
    eng = _Engine.__new__(_Engine)
    eng.ledger = lg
    eng.runtime = runtime
    eng.contract_dependents = {"artifact": ["b"]}
    eng.invalidate_dependents("artifact")

    assert key not in cache._values, (
        "the dependent artifact's cached evaluation must be evicted after a "
        "contract-breaking commit on what it depends on")


def test_commit_atomic_is_all_or_nothing(tmp_path):
    def serialize(a): return {"state": a.state, "blast_radius": a.blast_radius}

    def deserialize(aid, version, state):
        return EvolvingArtifact(aid, state.get("state", {}), version,
                                state.get("blast_radius", 0.2))

    led = Ledger(str(tmp_path / "repo"), serialize, deserialize)
    a = EvolvingArtifact("a", {"v": "1"}, blast_radius=0.2)
    b = EvolvingArtifact("b", {"v": "2"}, blast_radius=0.2,
                         contract=Contract(depends_on=("a",)))
    led.register(a)
    led.register(b)
    base = led.head_version(Ledger.DEV)
    a_new = EvolvingArtifact("a", {"v": "1", "extra": "x"}, blast_radius=0.2,
                             contract=Contract(major=2))
    b_new = EvolvingArtifact("b", {"v": "2"}, blast_radius=0.2,
                             contract=Contract(depends_on=("a",)))
    _, vv = led.commit_atomic([a_new, b_new], base)
    assert vv["a"] == 2 and vv["b"] == 2
