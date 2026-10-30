"""The newly exposed loop/promotion/compute layer must compose end to end."""

import pytest

import agentdescent
from agentdescent import (
    CapabilityFoundry,
    ComputeAction,
    ComputeContext,
    ComputeController,
    ComputeOption,
    ExperienceEngine,
    LoopConfig,
    LoopKernel,
    LoopObservation,
    PromotionRequest,
    PromotionRules,
    promote,
)
from agentdescent.durable_runs import SqliteRunStore
from agentdescent.evalops import EvalResult
from agentdescent.model_router import ModelRouter, ModelSpec
from agentdescent.safety_controller import canary


def test_loop_compute_and_promotion_share_one_surface(tmp_path):
    # Top-level names are the same objects, not copies or re-implementations.
    assert LoopKernel is agentdescent.loop_kernel.LoopKernel
    assert promote is agentdescent.promotion.promote
    assert ComputeController is agentdescent.compute_policy.ComputeController
    assert CapabilityFoundry is agentdescent.capability_foundry.CapabilityFoundry
    assert ExperienceEngine is agentdescent.experience.ExperienceEngine

    router = ModelRouter([ModelSpec("m", cost=1)],
                         cost_weight=0.0, exploration=0.0)
    compute = ComputeController([
        ComputeOption("direct", ComputeAction.DIRECT, calls=1),
    ], router, exploration=0.0)
    store = SqliteRunStore(str(tmp_path / "runs.sqlite3"))
    kernel = LoopKernel(store, LoopConfig(
        "tenant-a", "ws-1", max_generations=2, target=0.6))
    run_id = kernel.submit()
    seen = []

    def step(context):
        decision = compute.choose(ComputeContext(
            task_id=f"g{context.generation}", difficulty=0.3))
        seen.append((decision.action, decision.model))
        reward = 0.5 + 0.1 * context.generation
        compute.observe(decision, reward=reward)
        return LoopObservation(reward, cost=decision.expected_cost,
                               artifact_id=f"a{context.generation}")

    report = kernel.run(run_id, "worker-1", step)
    assert report.accepted == 2
    assert report.best_reward == pytest.approx(0.6)
    assert report.stopped_reason == "target"
    assert seen == [("direct", "m"), ("direct", "m")]
    assert compute.report().pulls == 2
    store.close()

    incumbent = {f"t{i}": 0.5 for i in range(6)}
    candidate = {key: 0.6 for key in incumbent}
    eval_result = EvalResult(
        eval_id="e", dataset_id="d", version=1, content_hash="h", n=20,
        score=0.7, baseline=0.5, delta=0.2, ci=(0.1, 0.3), passed=True,
        cost=0.0, per_task={}, detail="ok")
    decision = promote(PromotionRequest(
        artifact_id="a1", eval_result=eval_result, contaminated=False,
        canary=canary(incumbent, candidate), risk=0.1),
        PromotionRules(require_eval=True, require_contamination=True,
                       require_canary=True, require_risk=True))
    assert decision.accepted
    assert decision.action == "promote"
    assert decision.evidence_hash
