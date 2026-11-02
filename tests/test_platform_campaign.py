"""Multi-cycle campaigns must stop, share learning and fail closed."""

import json

import pytest

from agentdescent.artifact_graph import ArtifactGraph
from agentdescent.capability_foundry import CapabilityFoundry
from agentdescent.compute_policy import (
    ComputeAction, ComputeController, ComputeOption,
)
from agentdescent.continual_meta import ContinualMetaLearner
from agentdescent.durable_runs import SqliteRunStore
from agentdescent.experience import ExperienceEngine, ExperienceStore
from agentdescent.loop_kernel import LoopObservation
from agentdescent.model_router import ModelRouter, ModelSpec
from agentdescent.platform import PlatformConfig, RSIPlatform
from agentdescent.platform_campaign import (
    CampaignConfig, PlatformCampaign, PlatformCampaignReport,
)
from agentdescent.proofs import ProofInvariant, ProofRunner
from agentdescent.promotion import PromotionRules
from agentdescent.safe_selfmod import SafeSelfModifier


def _platform(tmp_path, *, with_foundry=True, with_experience=True,
              **overrides):
    router = ModelRouter([ModelSpec("m", cost=1)],
                         cost_weight=0.0, exploration=0.0)
    compute = ComputeController([
        ComputeOption("mcts", ComputeAction.MCTS, calls=1,
                      quality_prior=(5, 2)),
    ], router, exploration=0.0)
    foundry = None
    if with_foundry:
        foundry = CapabilityFoundry(
            SafeSelfModifier(set()), graph=ArtifactGraph(),
            prover=ProofRunner([
                ProofInvariant("has-source", lambda art: bool(art.source)),
            ]))
    experience = None
    if with_experience:
        experience = ExperienceEngine(
            ContinualMetaLearner(["mcts", "direct"], dim=3,
                                 learning_rate=0.5, min_gain=0.0,
                                 replay_size=16, seed=0),
            ExperienceStore())
    values = dict(max_generations=1, target=None)
    values.update(overrides)
    platform = RSIPlatform(
        SqliteRunStore(str(tmp_path / "runs.sqlite3")),
        PlatformConfig("tenant-a", "ws-1", **values),
        compute=compute, foundry=foundry, experience=experience,
        promotion_rules=PromotionRules())
    return platform, compute, foundry


def _seed_step(context, decision):
    reward = 0.5 + 0.1 * context.seed
    return LoopObservation(reward, cost=decision.expected_cost,
                           artifact_id=f"a{context.seed}")


def _flat_step(context, decision):
    return LoopObservation(0.5, cost=decision.expected_cost,
                           artifact_id=f"a{context.generation}")


def test_target_stop_runs_until_target_with_unique_runs(tmp_path):
    platform, _, _ = _platform(tmp_path)
    campaign = PlatformCampaign(
        platform, CampaignConfig(max_cycles=5, target=0.65, patience=3))
    report = campaign.run(step=_seed_step)

    assert isinstance(report, PlatformCampaignReport)
    assert len(report.cycles) == 3
    assert len({cycle.run_id for cycle in report.cycles}) == 3
    assert report.control.stopped_reason == "target"
    assert report.best_cycle == 2
    assert report.complete
    assert "best_cycle=2" in report.detail
    assert "stopped=target" in report.detail


def test_patience_stop_after_flat_rewards(tmp_path):
    platform, _, _ = _platform(tmp_path)
    report = PlatformCampaign(
        platform, CampaignConfig(max_cycles=5, patience=2)).run(
            step=_flat_step)

    assert len(report.cycles) == 3
    assert report.control.stopped_reason == "patience"
    assert report.best_cycle == 0
    assert report.complete


def test_incomplete_cycle_stops_campaign_as_error(tmp_path):
    platform, _, _ = _platform(tmp_path)
    report = PlatformCampaign(
        platform, CampaignConfig(max_cycles=3)).run(
            step=_seed_step,
            capability=lambda _loop, _promotion: None)

    assert len(report.cycles) == 1
    assert not report.cycles[0].complete
    assert report.control.stopped_reason == "error"
    assert "incomplete cycle" in report.control.reason
    assert not report.complete


def test_exception_stops_campaign_as_error(tmp_path):
    platform, _, _ = _platform(tmp_path)

    def broken_evidence(_loop, _best):
        raise RuntimeError("promotion evidence exploded")

    report = PlatformCampaign(
        platform, CampaignConfig(max_cycles=3)).run(
            step=_seed_step, evidence=broken_evidence)

    assert report.cycles == ()
    assert report.control.stopped_reason == "error"
    assert "promotion evidence exploded" in report.control.reason
    assert report.best_cycle is None
    assert not report.complete


def test_experience_store_is_shared_across_cycles(tmp_path):
    platform, _, _ = _platform(tmp_path)
    before = len(platform.experience.store)
    report = PlatformCampaign(
        platform, CampaignConfig(max_cycles=3)).run(step=_seed_step)

    assert len(report.cycles) == 3
    assert len(platform.experience.store) == before + 3
    run_ids = {row.run_id.split(":generation:")[0]
               for row in platform.experience.store.rows}
    assert run_ids == {cycle.run_id for cycle in report.cycles}


def test_invalid_campaign_config_and_bad_platform_are_refused(tmp_path):
    for kwargs in (
            {"max_cycles": 0},
            {"max_cycles": True},
            {"patience": 0},
            {"patience": True},
            {"target": float("nan")},
            {"target": float("inf")},
            {"max_cost": 0},
            {"max_cost": -1},
            {"min_gain": -0.1},
            {"base_seed": True},
            {"base_seed": 1.5}):
        with pytest.raises(ValueError):
            CampaignConfig(**kwargs)

    platform, _, _ = _platform(tmp_path)
    with pytest.raises(TypeError):
        PlatformCampaign(object())
    with pytest.raises(TypeError):
        PlatformCampaign(platform, config=object())
    with pytest.raises(TypeError):
        PlatformCampaign(platform).run(step=object())


def test_report_to_dict_round_trips_through_json(tmp_path):
    platform, _, _ = _platform(tmp_path)
    report = PlatformCampaign(
        platform, CampaignConfig(max_cycles=3, target=0.65)).run(
            step=_seed_step)

    data = json.loads(json.dumps(report.to_dict()))
    assert data["best_cycle"] == report.best_cycle
    assert data["complete"] is True
    assert data["control"]["stopped_reason"] == "target"
    assert len(data["cycles"]) == 3
    assert data["cycles"][0]["run_id"] == report.cycles[0].run_id
