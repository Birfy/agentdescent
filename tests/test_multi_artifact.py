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
    extra = {"b": EvolvingArtifact(
        "b", {}, blast_radius=0.2,
        contract=Contract(input_schema="text", output_schema="text",
                          major=1, depends_on=("artifact",)))}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = evolve(_tasks(), _reward, agent=_Reflector(), rounds=3,
                     n_workers=2, max_concurrency=2, strategy=_BreakingRules(),
                     extra_artifacts=extra)
    log = res.ledger_log
    assert any("-> artifact" in line and "contract" in line for line in log), (
        "a contract-breaking diff must be committed atomically, with a message "
        "that says so")


def test_contract_dependents_are_re_measured_after_a_breaking_change():
    """After a breaking commit on the primary, dependent 'b' loses its cached
    scores (they were measured under the superseded contract)."""
    from agentdescent.evalcache import MemoryCache

    cache = MemoryCache()
    b = EvolvingArtifact("b", {"k": "v"}, blast_radius=0.2,
                         contract=Contract(depends_on=("artifact",)))
    key = (b.render(), "t0", "")
    cache.get_or_eval(key, lambda: 0.7)

    extra = {"b": EvolvingArtifact(
        "b", {"k": "v"}, blast_radius=0.2,
        contract=Contract(depends_on=("artifact",)))}

    from agentdescent.policies import Policies
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evolve(_tasks(), _reward, agent=_Reflector(), rounds=3,
               n_workers=2, max_concurrency=2, strategy=_BreakingRules(),
               extra_artifacts=extra, policies=Policies(eval_cache=cache))

    # the dependent b was re-measured: a hit would return the stale 0.7
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
