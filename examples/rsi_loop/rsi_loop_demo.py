"""Five-layer RSI demo: loop, compute, promotion, foundry, experience.

Runs entirely offline and deterministically, in a temporary directory:

* ``LoopKernel`` runs generations with durable checkpoints;
* ``ComputeController`` picks direct vs PRM-rerank vs search per generation;
* ``promote`` turns an eval + canary report into a promotion decision;
* ``CapabilityFoundry`` admits a synthesized tool through review, safety, proof
  and registration;
* ``ExperienceEngine`` recalls the loop's experience and updates the meta-policy
  behind an anchor guard.

    python -m examples.rsi_loop.rsi_loop_demo
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

from agentdescent import (
    ComputeAction,
    ComputeContext,
    ComputeController,
    ComputeOption,
    DecisionLog,
    LoopConfig,
    LoopKernel,
    LoopObservation,
    PromotionRequest,
    PromotionRules,
    canary,
    promote,
)
from agentdescent.artifact_graph import ArtifactGraph
from agentdescent.capability_foundry import (
    CapabilityFoundry,
    CapabilityRequest,
)
from agentdescent.continual_meta import ContinualMetaLearner
from agentdescent.deliberation import DeliberationReport
from agentdescent.durable_runs import SqliteRunStore
from agentdescent.evalops import EvalResult
from agentdescent.experience import (
    ExperienceEngine,
    ExperienceRecord,
    ExperienceStore,
)
from agentdescent.model_router import ModelRouter, ModelSpec
from agentdescent.proofs import ProofInvariant, ProofRunner
from agentdescent.rsi_telemetry import RSITelemetry
from agentdescent.safe_selfmod import ModificationSpec, SafeSelfModifier
from agentdescent.tool_synthesis import ToolSpec, ToolValidation

_TOOL_SOURCE = "def parse_csv(text):\n    return text.split(',')\n"


def _loop(root: Path) -> Dict[str, Any]:
    router = ModelRouter([ModelSpec("worker", cost=1)],
                         cost_weight=0.0, exploration=0.0)
    compute = ComputeController([
        ComputeOption("direct", ComputeAction.DIRECT, calls=1,
                      quality_prior=(3, 1)),
        ComputeOption("prm", ComputeAction.PRM_RERANK, calls=2,
                      quality_prior=(4, 2), min_difficulty=0.3),
        ComputeOption("mcts", ComputeAction.MCTS, calls=4,
                      quality_prior=(5, 2), min_difficulty=0.6),
    ], router, exploration=0.0, seed=7)
    log = DecisionLog(str(root / "decisions.jsonl"))
    telemetry = RSITelemetry()
    kernel = LoopKernel(
        SqliteRunStore(str(root / "runs.sqlite3")),
        LoopConfig("demo", "ws-1", max_generations=3, target=0.9),
        decision_log=log, telemetry=telemetry)
    run_id = kernel.submit()
    applied = []

    def step(context):
        decision = compute.choose(ComputeContext(
            task_id=f"generation-{context.generation}", difficulty=0.8))
        # "search" buys a better candidate than a single direct answer.
        base = 0.7 if decision.action != ComputeAction.DIRECT else 0.5
        reward = min(1.0, base + 0.1 * context.generation)
        compute.observe(decision, reward=reward)
        return LoopObservation(
            reward, cost=decision.expected_cost,
            artifact_id=f"skill-{context.generation}",
            evidence={"action": decision.action, "model": decision.model})

    def apply(context, observation):
        applied.append(observation.artifact_id)

    report = kernel.run(run_id, "demo-worker", step, apply=apply)
    return {
        "run_id": run_id,
        "status": report.status,
        "stopped_reason": report.stopped_reason,
        "best_reward": report.best_reward,
        "generations": [row.to_dict() for row in report.generations],
        "applied": applied,
        "compute": compute.report().detail,
        "decisions": log.count(),
        "spans": telemetry.summary().spans,
        "artifacts": [row.artifact_id for row in report.generations],
        "_report": report,
    }


def _promotion(best: float) -> Dict[str, Any]:
    incumbent = {f"t{i}": 0.5 for i in range(6)}
    candidate = {key: max(0.5, best) for key in incumbent}
    eval_result = EvalResult(
        eval_id="demo-eval", dataset_id="demo", version=1, content_hash="demo",
        n=20, score=best, baseline=0.5, delta=best - 0.5,
        ci=(0.05, 1.0), passed=True, cost=0.0, per_task={}, detail="demo")
    decision = promote(PromotionRequest(
        artifact_id="skill-best", eval_result=eval_result, contaminated=False,
        canary=canary(incumbent, candidate), risk=0.1),
        PromotionRules())
    return decision.to_dict()


def _foundry() -> Dict[str, Any]:
    graph = ArtifactGraph()
    tool = ToolSpec("csv-tool", "parse a CSV line", _TOOL_SOURCE)
    modification = ModificationSpec(tool.name, tool.source, "tool")
    foundry = CapabilityFoundry(
        SafeSelfModifier(set()),
        prover=ProofRunner([
            ProofInvariant("has-source", lambda art: bool(art.source)),
        ]),
        graph=graph)
    outcome = foundry.admit(CapabilityRequest(
        gap_id="demo-csv-failures", tool=tool,
        validation=ToolValidation(True, 0.9),
        review=DeliberationReport("approve", 0.8, 0.9, (), "consensus"),
        modification=modification, before=0.5, after=0.7, canary=0.65))
    data = outcome.to_dict()
    data["registered"] = outcome.artifact_id in graph
    return data


def _experience(generations) -> Dict[str, Any]:
    learner = ContinualMetaLearner(
        ["direct", "search"], dim=2, learning_rate=0.5,
        min_gain=0.0, replay_size=16, seed=0)
    engine = ExperienceEngine(learner, ExperienceStore())
    rows = []
    for generation in generations:
        features = (0.8, 0.2)
        rewards = {"direct": 0.2, "search": 1.0}
        record = ExperienceRecord(
            f"demo-{generation['generation']}", features, rewards,
            cost=generation["cost"],
            accepted=generation["accepted"],
            artifact_id=generation["artifact_id"])
        engine.ingest(record)
        rows.append(record)
    update = engine.update(validation=rows[:1], anchors=rows[:1])
    return {
        "rows": engine.report().rows,
        "replay": engine.report().learner.replay,
        "accepted": update.accepted,
        "rolled_back": update.rolled_back,
        "anchor_before": update.anchor_before,
        "anchor_after": update.anchor_after,
        "reason": update.reason,
    }


def main() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        loop = _loop(root)
        report = loop.pop("_report")
        best = report.best_reward or 0.0
        result = {
            "loop": loop,
            "promotion": _promotion(best),
            "foundry": _foundry(),
            "experience": _experience(loop["generations"]),
        }
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False))
