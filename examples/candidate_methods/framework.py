"""AgentDescent-native execution shared by the eleven candidate methods."""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from agentdescent.aggregator import AggregatorConfig
from agentdescent.agents import Usage
from agentdescent.async_evolve import async_evolve
from agentdescent.evolution import EvolutionResult, Task, evolve
from agentdescent.evolvable import Diff

from .runtime import MODES, Recorder, compact_events, usage_dict


ARTIFACT_KEY = "artifact"
QUALITY_TARGET = 0.75

Run = Callable[[str, Task], str]
Reward = Callable[[Task, str], float]
Propose = Callable[[str, Task, str, float], Optional[str]]
Validator = Callable[[str], str]


@dataclass
class ValidatedSlot:
    """A single replacement slot with method-specific parsing and safety gates."""

    initial_value: str
    validator: Validator
    invalid_proposals: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def initial(self) -> Dict[str, str]:
        return {ARTIFACT_KEY: self.initial_value}

    def render(self, state: Dict[str, str]) -> str:
        return state.get(ARTIFACT_KEY, self.initial_value)

    def keys(self) -> Sequence[str]:
        return (ARTIFACT_KEY,)

    def to_diff(
        self,
        state: Dict[str, str],
        proposal: str,
        author: str,
        base_version: int,
        target: str,
    ) -> Optional[Diff]:
        try:
            value = self.validator(proposal)
        except (KeyError, TypeError, ValueError):
            with self._lock:
                self.invalid_proposals += 1
            return None
        if not value or value == state.get(ARTIFACT_KEY):
            return None
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
        return Diff(
            diff_id=f"{author}:{digest}:{base_version}",
            target=target,
            ops={ARTIFACT_KEY: value},
            author=author,
        )


@dataclass
class PortSpec:
    """One algorithm's actors, artifact representation, and disjoint datasets."""

    name: str
    fidelity: str
    strategy: ValidatedSlot
    train_tasks: Sequence[Task]
    held_out_tasks: Sequence[Task]
    test_tasks: Sequence[Task]
    run: Run
    reward: Reward
    propose: Propose
    proposal_calls_per_candidate: int
    notes: Sequence[str]

    @property
    def engine_tasks(self) -> List[Task]:
        return list(self.train_tasks) + list(self.held_out_tasks)

    @property
    def held_out_frac(self) -> float:
        return len(self.held_out_tasks) / len(self.engine_tasks)

    def evaluate(self, rendered: str, tasks: Sequence[Task]) -> float:
        if not tasks:
            return 0.0
        return sum(self.reward(task, self.run(rendered, task)) for task in tasks) / len(tasks)


class ProposalLimiter:
    """Reserve exactly N algorithm proposals even if async workers overshoot."""

    def __init__(self, propose: Propose, limit: int) -> None:
        self.propose = propose
        self.limit = limit
        self.claimed = 0
        self._lock = threading.Lock()

    def __call__(
        self,
        rendered: str,
        task: Task,
        output: str,
        reward: float,
    ) -> Optional[str]:
        with self._lock:
            if self.claimed >= self.limit:
                return None
            self.claimed += 1
        return self.propose(rendered, task, output, reward)


@dataclass
class FrameworkMethodResult:
    algorithm: str
    fidelity: str
    mode: str
    seed: int
    wall_seconds: float
    engine_wall_seconds: float
    baseline_quality: float
    final_quality: float
    baseline_validation_quality: float
    validation_quality: float
    quality_target: float
    time_to_quality_s: Optional[float]
    candidates: int
    accepted: int
    stale_considered: int
    stale_discarded: int
    invalid_candidates: int
    model_usage: Dict[str, float]
    actor_usage: Dict[str, float]
    phase_summary: Dict[str, Dict[str, float]]
    events: List[Dict[str, object]]
    framework: Dict[str, object]
    budget: Dict[str, object]
    notes: Sequence[str]

    @property
    def quality_gain(self) -> float:
        return self.final_quality - self.baseline_quality

    @property
    def target_reached(self) -> bool:
        return self.time_to_quality_s is not None

    def compact(self) -> Dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "fidelity": self.fidelity,
            "mode": self.mode,
            "seed": self.seed,
            "wall_seconds": self.wall_seconds,
            "engine_wall_seconds": self.engine_wall_seconds,
            "baseline_quality": self.baseline_quality,
            "final_quality": self.final_quality,
            "quality_gain": self.quality_gain,
            "baseline_validation_quality": self.baseline_validation_quality,
            "validation_quality": self.validation_quality,
            "quality_target": self.quality_target,
            "target_reached": self.target_reached,
            "time_to_quality_s": self.time_to_quality_s,
            "candidates": self.candidates,
            "accepted": self.accepted,
            "stale_considered": self.stale_considered,
            "stale_discarded": self.stale_discarded,
            "stale_rate": (
                self.stale_discarded / self.stale_considered
                if self.stale_considered
                else 0.0
            ),
            "invalid_candidates": self.invalid_candidates,
            "usage": self.model_usage,
            "actor_usage": self.actor_usage,
            "phase_summary": self.phase_summary,
            "events": self.events,
            "framework": self.framework,
            "budget": self.budget,
            "notes": list(self.notes),
        }


def _accepted(result: EvolutionResult) -> int:
    return int(result.outcomes().get("committed", 0))


def run_port(
    spec: PortSpec,
    recorder: Recorder,
    *,
    mode: str,
    seed: int,
    workers: int = 2,
    candidate_budget: int = 2,
    async_ratio: int = 1,
    max_seconds: float = 300.0,
    shutdown_grace: float = 120.0,
) -> FrameworkMethodResult:
    """Run one method through AgentDescent with an equal candidate budget."""
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if workers < 2 or candidate_budget < workers or candidate_budget % workers:
        raise ValueError("candidate_budget must be divisible by workers and at least workers")
    if len(spec.train_tasks) < workers or len(spec.held_out_tasks) < 4:
        raise ValueError("ports require enough train tasks and at least four held-out tasks")

    initial = spec.strategy.render(spec.strategy.initial())
    started = time.monotonic()
    baseline_quality = spec.evaluate(initial, spec.test_tasks)
    baseline_validation = spec.evaluate(initial, spec.held_out_tasks)
    engine_started = time.monotonic()
    actor_usage = Usage()
    proposal_limiter = ProposalLimiter(spec.propose, candidate_budget)
    common = {
        "run": spec.run,
        "propose": proposal_limiter,
        "strategy": spec.strategy,
        "artifact_id": f"candidate-{spec.name}",
        "initial_state": spec.strategy.initial(),
        "blast_radius": 0.6,
        "n_workers": workers,
        "self_verify": False,
        "held_out_frac": spec.held_out_frac,
        "solved_threshold": 1.1,
        "eval_concurrency": 1 if mode == "serial" else workers,
        "agg_config": AggregatorConfig(batch_trigger=1, max_wait_rounds=1),
        "shuffle": False,
        "seed": seed,
        "usage": actor_usage,
        "verbose": False,
    }
    if mode == "async_pipeline":
        result = async_evolve(
            spec.engine_tasks,
            spec.reward,
            async_ratio=async_ratio,
            max_iters=candidate_budget,
            max_seconds=max_seconds,
            shutdown_grace=shutdown_grace,
            **common,
        )
    else:
        result = evolve(
            spec.engine_tasks,
            spec.reward,
            rounds=candidate_budget // workers,
            max_concurrency=1 if mode == "serial" else workers,
            **common,
        )
    engine_wall = time.monotonic() - engine_started
    if result.error:
        raise RuntimeError(result.error)
    final_quality = spec.evaluate(result.rendered, spec.test_tasks)
    wall_seconds = time.monotonic() - started

    engine_ttq = result.time_to_quality(QUALITY_TARGET)
    if baseline_validation >= QUALITY_TARGET:
        time_to_quality = engine_started - started
    elif engine_ttq is None:
        time_to_quality = None
    else:
        time_to_quality = engine_started - started + engine_ttq

    proposal_calls = sum(
        1 for event in recorder.events if event.phase.startswith("proposal:")
    )
    expected_proposal_calls = candidate_budget * spec.proposal_calls_per_candidate
    budget = {
        "reserved_candidates": candidate_budget,
        "observed_candidates": proposal_limiter.claimed,
        "observed_rollouts": result.rollouts,
        "expected_proposal_calls": expected_proposal_calls,
        "observed_proposal_calls": proposal_calls,
        "matched": (
            proposal_limiter.claimed == candidate_budget
            and proposal_calls == expected_proposal_calls
        ),
    }
    history = [
        {
            "round": row.round,
            "held_out_reward": row.held_out_reward,
            "elapsed_s": row.elapsed_s,
            "rollouts": row.rollouts,
            "committed": row.committed,
            "rejected": row.rejected,
            "reasons": row.reasons,
        }
        for row in result.history
    ]
    return FrameworkMethodResult(
        algorithm=spec.name,
        fidelity=spec.fidelity,
        mode=mode,
        seed=seed,
        wall_seconds=wall_seconds,
        engine_wall_seconds=engine_wall,
        baseline_quality=baseline_quality,
        final_quality=final_quality,
        baseline_validation_quality=baseline_validation,
        validation_quality=result.final_reward,
        quality_target=QUALITY_TARGET,
        time_to_quality_s=time_to_quality,
        candidates=proposal_limiter.claimed,
        accepted=_accepted(result),
        stale_considered=result.stale_considered,
        stale_discarded=result.stale_discarded,
        invalid_candidates=spec.strategy.invalid_proposals,
        model_usage=usage_dict(recorder.usage),
        actor_usage=usage_dict(actor_usage),
        phase_summary=recorder.phase_summary(),
        events=compact_events(recorder.events),
        framework={
            "runtime": "async_evolve" if mode == "async_pipeline" else "evolve",
            "max_concurrency": (
                workers
                if mode in ("sync_parallel", "async_pipeline")
                else 1
            ),
            "baseline_reward": baseline_validation,
            "final_reward": result.final_reward,
            "stop_reason": result.stop_reason,
            "outcomes": result.outcomes(),
            "rollouts": result.rollouts,
            "rollout_seconds": result.rollout_seconds,
            "eval_seconds": result.eval_seconds,
            "stale_considered": result.stale_considered,
            "stale_discarded": result.stale_discarded,
            "retired_workers": result.retired_workers,
            "history": history,
        },
        budget=budget,
        notes=spec.notes,
    )
