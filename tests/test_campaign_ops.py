"""A campaign publishes every finished loop into one control-room report."""

import json

import pytest

from agentdescent.campaign_ops import CampaignOps, CampaignOpsReport
from agentdescent.durable_runs import SqliteRunStore
from agentdescent.loop_kernel import (
    LoopConfig,
    LoopGeneration,
    LoopKernel,
    LoopObservation,
    LoopReport,
)
from agentdescent.ops_console import OperationsConsole
from agentdescent.ops_fabric import OpsFabric
from agentdescent.rsi_telemetry import RSITelemetry


def _run(kernel, tag):
    run_id = kernel.submit()

    def step(context):
        return LoopObservation(
            0.5 + 0.1 * context.generation, cost=1.0,
            artifact_id=f"{tag}-{context.generation}")

    return kernel.run(run_id, "worker", step)


def _campaign(tmp_path):
    store = SqliteRunStore(str(tmp_path / "runs.sqlite3"))
    kernel = LoopKernel(
        store, LoopConfig("t1", "ws1", max_generations=2))
    first = _run(kernel, "first")
    second = _run(kernel, "second")
    telemetry = RSITelemetry()
    fabric = OpsFabric(OperationsConsole(), telemetry=telemetry)
    campaign = CampaignOps(fabric)
    report = campaign.publish([first, second], tenant="t1", workspace="ws1")
    return store, telemetry, fabric, campaign, report, first, second


def _gen(index, action="accept", *, cost=1.0):
    return LoopGeneration(
        generation=index, seed=index, action=action, reason="",
        reward=0.5 if action == "accept" else 0.1, cost=cost, risk=0.0,
        gain=None, artifact_id=f"a{index}", stop=False,
        applied=action == "accept")


def _report(run_id, generations):
    accepted = sum(1 for row in generations if row.accepted)
    return LoopReport(
        run_id=run_id, status="completed", generations=tuple(generations),
        best_reward=0.5, total_cost=sum(row.cost for row in generations),
        accepted=accepted, stopped_reason="max_generations", reason="",
        detail="")


def test_publish_lands_both_loops_in_every_read_model(tmp_path):
    store, telemetry, fabric, _, report, first, second = _campaign(tmp_path)
    generations = len(first.generations) + len(second.generations)

    assert isinstance(report, CampaignOpsReport)
    assert len(report.records) == 2
    assert {run.run_id for run in fabric.console.list_runs().items} == {
        first.run_id, second.run_id}
    assert report.snapshot.console.runs == 2
    assert report.snapshot.lineage.nodes == generations
    assert report.snapshot.telemetry.spans == 2
    assert len(telemetry.spans) == 2
    assert report.metrics.decisions == generations
    assert report.metrics.accepted == generations
    assert report.alerts == ()
    assert first.run_id in report.detail
    assert second.run_id in report.detail
    store.close()


def test_publish_is_idempotent_for_the_same_loops(tmp_path):
    store, telemetry, fabric, campaign, report, first, second = _campaign(
        tmp_path)
    spans = len(telemetry.spans)

    again = campaign.publish([first, second], tenant="t1", workspace="ws1")

    assert again.records == report.records
    assert again.metrics == report.metrics
    assert fabric.snapshot().recorded == 2
    assert fabric.snapshot().console.runs == 2
    assert fabric.snapshot().lineage.nodes == report.snapshot.lineage.nodes
    assert len(telemetry.spans) == spans
    store.close()


def test_min_acceptance_alert_fires_below_the_bar():
    fabric = OpsFabric(OperationsConsole())
    campaign = CampaignOps(fabric)
    loop = _report("r1", [_gen(0), _gen(1, "reject")])

    report = campaign.publish(
        [loop], tenant="t1", workspace="ws1", min_acceptance=0.75)

    assert report.metrics.acceptance_rate == pytest.approx(0.5)
    assert any("below" in alert for alert in report.alerts)


def test_max_cost_alert_fires_when_calls_were_spent_without_acceptance():
    fabric = OpsFabric(OperationsConsole())
    campaign = CampaignOps(fabric)
    loop = _report("r1", [_gen(0, "reject", cost=3.0),
                          _gen(1, "reject", cost=3.0)])

    report = campaign.publish(
        [loop], tenant="t1", workspace="ws1", max_cost_per_accepted=2.0)

    assert report.metrics.accepted == 0
    assert report.metrics.calls == 6
    assert any("no accepted decisions after 6 calls" in alert
               for alert in report.alerts)


def test_invalid_campaigns_are_refused_before_recording():
    fabric = OpsFabric(OperationsConsole())
    campaign = CampaignOps(fabric)
    loop = _report("r1", [_gen(0)])

    with pytest.raises(TypeError):
        CampaignOps(object())
    with pytest.raises(ValueError):
        campaign.publish([], tenant="t", workspace="w")
    with pytest.raises(TypeError):
        campaign.publish("bad", tenant="t", workspace="w")
    with pytest.raises(TypeError):
        campaign.publish([object()], tenant="t", workspace="w")
    with pytest.raises(ValueError):
        campaign.publish([loop], tenant="", workspace="w")
    with pytest.raises(ValueError):
        campaign.publish([loop], tenant="t", workspace=" ")
    with pytest.raises(ValueError):
        campaign.publish([loop], tenant="t", workspace="w",
                         min_acceptance=2.0)
    with pytest.raises(ValueError):
        campaign.publish([loop], tenant="t", workspace="w",
                         max_cost_per_accepted=-1.0)
    with pytest.raises(ValueError):
        campaign.publish(
            [_report("r2", [_gen(0, cost=-1.0)])],
            tenant="t", workspace="w")
    assert fabric.snapshot().console.runs == 0


def test_to_dict_round_trips_through_json(tmp_path):
    store, _, _, _, report, first, second = _campaign(tmp_path)

    loaded = json.loads(json.dumps(report.to_dict()))

    assert len(loaded["records"]) == 2
    assert {row["run"]["run_id"] for row in loaded["records"]} == {
        first.run_id, second.run_id}
    assert loaded["snapshot"]["console"]["runs"] == 2
    assert loaded["snapshot"]["telemetry"]["spans"] == 2
    assert loaded["metrics"]["decisions"] == report.metrics.decisions
    assert loaded["alerts"] == []
    assert loaded["detail"] == report.detail
    store.close()
