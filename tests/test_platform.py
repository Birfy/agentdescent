"""The platform must make the five layers one complete cycle, not five islands."""

import json

import pytest

from agentdescent.artifact_graph import ArtifactGraph
from agentdescent.capability_foundry import CapabilityFoundry, CapabilityRequest
from agentdescent.compute_policy import (
    ComputeAction, ComputeController, ComputeOption,
)
from agentdescent.continual_meta import ContinualMetaLearner
from agentdescent.deliberation import DeliberationReport
from agentdescent.durable_runs import SqliteRunStore
from agentdescent.evalops import EvalResult
from agentdescent.experience import ExperienceEngine, ExperienceRecord, ExperienceStore
from agentdescent.loop_kernel import LoopObservation, LoopReport
from agentdescent.model_router import ModelRouter, ModelSpec
from agentdescent.platform import PlatformConfig, RSIPlatform
from agentdescent.proofs import ProofInvariant, ProofRunner
from agentdescent.promotion import PromotionRequest, PromotionRules
from agentdescent.safe_selfmod import ModificationSpec, SafeSelfModifier
from agentdescent.safety_controller import canary
from agentdescent.tool_synthesis import ToolSpec, ToolValidation

_TOOL = "def f():\n    return 1\n"


def _platform(tmp_path, *, with_foundry=True, with_experience=True, **overrides):
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
    values = dict(max_generations=3, target=0.85)
    values.update(overrides)
    platform = RSIPlatform(
        SqliteRunStore(str(tmp_path / "runs.sqlite3")),
        PlatformConfig("tenant-a", "ws-1", **values),
        compute=compute, foundry=foundry, experience=experience,
        promotion_rules=PromotionRules())
    return platform, compute, foundry


def _step(context, decision):
    reward = min(1.0, 0.8 + 0.05 * context.generation)
    return LoopObservation(
        reward, cost=decision.expected_cost,
        artifact_id=f"a{context.generation}",
        evidence={"action": decision.action,
                  "rewards": {"mcts": reward, "direct": 0.4}})


def _evidence(loop, best):
    incumbent = {f"t{i}": 0.5 for i in range(6)}
    candidate = {key: best.reward for key in incumbent}
    result = EvalResult(
        eval_id="e", dataset_id="d", version=1, content_hash="h", n=20,
        score=best.reward, baseline=0.5, delta=best.reward - 0.5,
        ci=(0.1, 1.0), passed=True, cost=0.0, per_task={}, detail="ok")
    return PromotionRequest(
        artifact_id=best.artifact_id, eval_result=result, contaminated=False,
        canary=canary(incumbent, candidate), risk=0.1)


def _capability(loop, promotion):
    tool = ToolSpec("demo-tool", "demo", _TOOL)
    return CapabilityRequest(
        gap_id="gap", tool=tool, validation=ToolValidation(True, 0.9),
        review=DeliberationReport("approve", 0.8, 0.9, (), "consensus"),
        modification=ModificationSpec(tool.name, tool.source, "tool"),
        before=0.5, after=0.7, canary=0.65)


def test_one_cycle_composes_all_five_layers(tmp_path):
    platform, compute, foundry = _platform(tmp_path)
    run_id = platform.submit()
    report = platform.run_cycle(
        run_id, "worker-1", step=_step, evidence=_evidence,
        capability=_capability)

    assert report.complete
    assert report.loop.terminal
    assert report.loop.best_reward == pytest.approx(0.85)
    assert [row.generation for row in report.loop.generations] == [0, 1]
    assert all(row.evidence["action"] == "mcts"
               for row in report.loop.generations)
    assert report.promotion is not None
    assert report.promotion.action == "promote"
    assert report.foundry is not None
    assert report.foundry.status == "promoted"
    assert report.experience is not None
    assert report.experience.accepted
    assert report.experience.reason == "improved"
    assert compute.report().pulls == 2
    assert len(foundry.graph) == 1
    data = report.to_dict()
    assert json.loads(json.dumps(data))["complete"] is True
    assert data["experience"]["anchor_after"] is not None
    assert report.self_validated
    assert "experience=self-validation" in report.detail
    assert all(row.evidence["compute_action"] == "mcts"
               for row in report.loop.generations)


def test_experience_guard_uses_pre_cycle_history_when_anchors_absent(tmp_path):
    platform, _, _ = _platform(tmp_path)
    old = ExperienceRecord(
        "old-run", (0.8, 0.2, 0.5), {"mcts": 1.0, "direct": 0.0},
        cost=0.5, accepted=True, artifact_id="old")
    platform.experience.ingest(old)
    captured = {}
    original = platform.experience.update

    def spy(*, validation, anchors):
        captured["anchors"] = list(anchors)
        return original(validation=validation, anchors=anchors)

    platform.experience.update = spy
    run_id = platform.submit()
    report = platform.run_cycle(run_id, "worker-1", step=_step)
    assert [row.run_id for row in captured["anchors"]] == ["old-run"]
    assert report.experience is not None
    assert report.complete


def test_only_configured_layers_are_required(tmp_path):
    platform, _, _ = _platform(
        tmp_path, with_foundry=False, with_experience=False)
    run_id = platform.submit()
    report = platform.run_cycle(run_id, "worker-1", step=_step)
    assert report.complete
    assert report.promotion is None
    assert report.foundry is None
    assert report.experience is None


def test_missing_foundry_or_none_capability_is_not_silent(tmp_path):
    platform, _, _ = _platform(tmp_path, with_foundry=False)
    run_id = platform.submit()
    with pytest.raises(RuntimeError):
        platform.run_cycle(run_id, "worker-1", step=_step,
                           capability=_capability)

    (tmp_path / "second").mkdir()
    platform, _, _ = _platform(tmp_path / "second", with_foundry=True)
    run_id = platform.submit()
    report = platform.run_cycle(
        run_id, "worker-1", step=_step, evidence=_evidence,
        capability=lambda _loop, _promotion: None)
    assert not report.complete
    assert report.promotion is not None
    assert report.foundry is None


def test_invalid_config_and_types_are_refused(tmp_path):
    with pytest.raises(ValueError):
        PlatformConfig("", "ws")
    with pytest.raises(ValueError):
        PlatformConfig("t", "ws", max_generations=0)
    with pytest.raises(ValueError):
        PlatformConfig("t", "ws", risk_limit=2)
    with pytest.raises(TypeError):
        RSIPlatform(SqliteRunStore(str(tmp_path / "x.sqlite3")),
                    PlatformConfig("t", "ws"), compute=object())



def _rejecting_evidence(loop, best):
    incumbent = {f"t{i}": 0.5 for i in range(6)}
    result = EvalResult(
        eval_id="e", dataset_id="d", version=1, content_hash="h", n=20,
        score=best.reward, baseline=0.5, delta=best.reward - 0.5,
        ci=(0.1, 1.0), passed=True, cost=0.0, per_task={}, detail="ok")
    return PromotionRequest(
        artifact_id=best.artifact_id, eval_result=result, contaminated=True,
        canary=canary(incumbent, {key: best.reward for key in incumbent}),
        risk=0.1)


def test_rejected_promotion_gates_the_foundry(tmp_path):
    platform, _, foundry = _platform(tmp_path, max_generations=2, target=0.85)
    run_id = platform.submit()
    report = platform.run_cycle(
        run_id, "worker-1", step=_step, evidence=_rejecting_evidence,
        capability=_capability)
    assert report.promotion is not None
    assert report.promotion.action == "reject"
    assert report.foundry is None
    assert "foundry:promotion_not_accepted" in report.skipped
    assert len(foundry.graph) == 0
    assert report.complete


def test_nonterminal_cycle_is_not_cached_and_runs_later(tmp_path):
    platform, _, _ = _platform(tmp_path)
    run_id = platform.submit()
    real_run = platform.kernel.run
    calls = {"n": 0}

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return LoopReport(
                run_id=run_id, status="running", generations=(),
                best_reward=None, total_cost=0.0, accepted=0,
                stopped_reason="", reason="", detail="")
        return real_run(*args, **kwargs)

    platform.kernel.run = fake_run
    first = platform.run_cycle(
        run_id, "worker-1", step=_step, evidence=_evidence,
        capability=_capability)
    assert not first.complete
    assert first.promotion is None and first.foundry is None
    assert run_id not in platform._processed

    second = platform.run_cycle(
        run_id, "worker-1", step=_step, evidence=_evidence,
        capability=_capability)
    assert second.complete
    assert second.promotion is not None
    assert run_id in platform._processed


def test_terminal_run_returns_cached_post_loop_decisions(tmp_path):
    platform, compute, foundry = _platform(tmp_path)
    run_id = platform.submit()
    first = platform.run_cycle(
        run_id, "worker-1", step=_step, evidence=_evidence,
        capability=_capability)
    graph_size = len(foundry.graph)
    version = platform.experience.learner.version
    second = platform.run_cycle(
        run_id, "worker-1", step=_step, evidence=_evidence,
        capability=_capability)
    assert second.foundry == first.foundry
    assert second.experience == first.experience
    assert len(foundry.graph) == graph_size
    assert platform.experience.learner.version == version
    assert compute.report().pulls == 2


def test_terminal_run_without_a_cache_refuses_rerun(tmp_path):
    platform, _, _ = _platform(tmp_path)
    run_id = platform.submit()

    def unchanged(context, decision):
        return LoopObservation(0.5, cost=decision.expected_cost,
                               artifact_id="a0")

    platform.run_cycle(run_id, "worker-1", step=unchanged)
    fresh = RSIPlatform(
        platform.store, platform.config, compute=platform.compute,
        foundry=platform.foundry, experience=platform.experience)
    with pytest.raises(RuntimeError):
        fresh.run_cycle(run_id, "worker-1", step=unchanged)
