"""A finished loop must land in console, lineage, telemetry and QD once."""

import pytest

from agentdescent.durable_runs import SqliteRunStore
from agentdescent.lineage import LineageGraph
from agentdescent.loop_kernel import (
    LoopConfig, LoopGeneration, LoopKernel, LoopObservation, LoopReport,
)
from agentdescent.ops_console import OperationsConsole
from agentdescent.ops_fabric import OpsFabric
from agentdescent.quality_diversity import QualityDiversityArchive
from agentdescent.rsi_telemetry import RSITelemetry


def _loop(tmp_path):
    kernel = LoopKernel(
        SqliteRunStore(str(tmp_path / "runs.sqlite3")),
        LoopConfig("t1", "ws1", max_generations=3, target=0.9))
    run_id = kernel.submit()

    def step(context):
        return LoopObservation(
            0.7 + 0.1 * context.generation, cost=1.0,
            artifact_id=f"a{context.generation}",
            evidence={"slot": "skill"})

    return kernel.run(run_id, "worker", step)


def test_record_lands_in_all_read_models_and_is_idempotent(tmp_path):
    qd = QualityDiversityArchive((2, 2))
    qd.add("a0", 0.5, (0, 0))
    qd.add("a1", 0.9, (1, 1))
    telemetry = RSITelemetry()
    graph = LineageGraph()
    fabric = OpsFabric(OperationsConsole(), lineage=graph,
                       telemetry=telemetry, qd=qd)
    assert fabric.lineage is graph
    report = _loop(tmp_path)

    record = fabric.record(report, tenant="t1", workspace="ws1",
                           tags=("rsi",))
    assert record.run.status == "completed"
    assert record.run.reward == pytest.approx(0.9)
    assert record.lineage.nodes == 3
    assert fabric.lineage.path("a0", "a1") == ("a0", "a1")
    snapshot = fabric.snapshot()
    assert snapshot.console.runs == 1
    assert snapshot.console.qd is not None
    assert snapshot.console.qd.filled == 2
    assert snapshot.lineage.nodes == 3
    assert snapshot.telemetry.spans >= 1
    assert snapshot.recorded == 1

    spans = len(telemetry.spans)
    again = fabric.record(report, tenant="t1", workspace="ws1")
    assert again == record
    assert len(telemetry.spans) == spans
    assert fabric.snapshot().recorded == 1
    assert len(fabric.lineage) == 3
    assert fabric.snapshot().to_dict()["recorded"] == 1


def test_invalid_inputs_are_refused(tmp_path):
    fabric = OpsFabric(OperationsConsole())
    with pytest.raises(TypeError):
        fabric.record("run", tenant="t", workspace="w")
    report = _loop(tmp_path)
    with pytest.raises(ValueError):
        fabric.record(report, tenant="", workspace="w")
    with pytest.raises(ValueError):
        fabric.record(report, tenant="t", workspace="")
    with pytest.raises(TypeError):
        OpsFabric(object())


def _gen(index, artifact, *, accepted=True, reward=0.5):
    return LoopGeneration(
        generation=index, seed=index, action="accept" if accepted else "reject",
        reason="", reward=reward, cost=1.0, risk=0.0, gain=None,
        artifact_id=artifact, stop=False, applied=accepted)


def test_partial_then_terminal_re_record_updates_the_console(tmp_path):
    fabric = OpsFabric(OperationsConsole())
    partial = LoopReport(
        run_id="r1", status="running", generations=(_gen(0, "a0"),),
        best_reward=0.1, total_cost=1.0, accepted=1, stopped_reason="",
        reason="", detail="")
    fabric.record(partial, tenant="t", workspace="w")
    assert fabric.snapshot().console.by_status.get("running") == 1
    assert len(fabric.lineage) == 1

    complete = LoopReport(
        run_id="r1", status="completed",
        generations=(_gen(0, "a0"), _gen(1, "a1", reward=0.9)),
        best_reward=0.9, total_cost=2.0, accepted=2, stopped_reason="target",
        reason="", detail="")
    fabric.record(complete, tenant="t", workspace="w")
    run = fabric.snapshot().console.top_runs[0]
    assert run.status == "completed"
    assert run.reward == pytest.approx(0.9)
    assert len(fabric.lineage) == 2
    assert fabric.snapshot().recorded == 1


def test_unknown_status_is_refused_and_rejected_children_are_not_parents():
    fabric = OpsFabric(OperationsConsole())
    bad = LoopReport(
        run_id="x", status="weird", generations=(), best_reward=None,
        total_cost=0.0, accepted=0, stopped_reason="", reason="", detail="")
    with pytest.raises(ValueError):
        fabric.record(bad, tenant="t", workspace="w")

    report = LoopReport(
        run_id="r2", status="completed",
        generations=(_gen(0, "a0"), _gen(1, "bad", accepted=False),
                     _gen(2, "a2")),
        best_reward=0.5, total_cost=3.0, accepted=2, stopped_reason="",
        reason="", detail="")
    fabric.record(report, tenant="t", workspace="w")
    assert fabric.lineage.parents("bad") == ("a0",)
    assert fabric.lineage.parents("a2") == ("a0",)
