"""The single runner behind every MethodPolicy port.

This is the only module that touches the execution plane: it owns the recorder,
phase naming, budgets, mode dispatch, and the merge configuration. A
:class:`~examples._method_policy.MethodPolicy` supplies everything else.

Merging is configured for **model-merged unions instead of ranking**: batches
are sized to the worker count and, for text-valued artifacts,
:func:`agentdescent.fusion.reflective_merge` is installed so contradicting
proposals are synthesised by one model call and gated by one held-out
evaluation -- rather than each candidate paying its own ranking evaluation.
"""

from __future__ import annotations

import argparse
import random
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence

from agentdescent.aggregator import AggregatorConfig
from agentdescent.agents import Usage
from agentdescent.async_evolve import async_evolve
from agentdescent.staleness import get_policy
from agentdescent.evolution import EvolutionResult, Task, evolve
from agentdescent.fusion import reflective_merge

from ._common import add_standard_args, completion_for, confirm
from ._measure import MODES, PhasedLLM, Recorder, compact_events, usage_dict
from ._method_policy import MethodPolicy, PopulationContext
from ._population import population_factory


QUALITY_TARGET = 0.75


class ProposalLimiter:
    """Reserve exactly N algorithm proposals even if async workers overshoot.

    The slot is claimed before the proposal call, so a provider error mid-call
    burns a candidate; that is deliberate -- it surfaces as a budget mismatch
    rather than a silent extra call.
    """

    def __init__(self, propose: Callable, limit: int) -> None:
        self.propose = propose
        self.limit = limit
        self.claimed = 0
        self._lock = threading.Lock()

    def __call__(self, rendered: str, task: Task, output: str, reward: float):
        with self._lock:
            if self.claimed >= self.limit:
                return None
            self.claimed += 1
        return self.propose(rendered, task, output, reward)


@dataclass
class MethodRunResult:
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


def _batch(tasks, size: int, seed: int):
    """A random batch of the training split, or all of it when `size` covers it.

    Seeded from the run's seed and the requested size, so a run is reproducible
    while still resampling as the size changes -- the sampling is part of the
    algorithm for a method whose fitness is defined on a batch, not a caching
    detail.
    """
    if size <= 0 or size >= len(tasks):
        return list(tasks)
    # An int, not a tuple: `Random(tuple)` seeds by hashing, which Python 3.9
    # deprecates and which is `PYTHONHASHSEED`-dependent -- the batch would
    # differ between processes and the run would stop being reproducible.
    return random.Random(seed * 1_000_003 + size * 1009 + len(tasks)).sample(
        list(tasks), size)


def _default_population(ctx: PopulationContext):
    """The shared archive: keep every committed head, ask `selection` who is next."""
    return population_factory(
        ctx.selection, ctx.artifact_id, conflict=ctx.conflict,
        fusion=ctx.fusion, acceptance=ctx.acceptance)


def _evaluate(policy: MethodPolicy, llm: PhasedLLM, rendered: str,
              tasks: Sequence[Task]) -> float:
    if not tasks:
        return 0.0
    return sum(
        policy.reward(task, policy.solve(llm, rendered, task)) for task in tasks
    ) / len(tasks)


def _fusion_summary(result: EvolutionResult) -> Dict[str, object]:
    stats = getattr(result, "fusion_stats", None)
    if not callable(stats):
        return {}
    try:
        report = stats()
    except Exception:  # noqa: BLE001 - diagnostics must not fail a run
        return {}
    if hasattr(report, "_asdict"):
        return dict(report._asdict())
    return dict(report) if isinstance(report, dict) else {"raw": repr(report)}


def run_port(
    policy: MethodPolicy,
    recorder: Recorder,
    *,
    mode: str,
    seed: int,
    workers: int = 2,
    candidate_budget: int = 2,
    async_ratio: int = 1,
    max_seconds: float = 300.0,
    shutdown_grace: float = 120.0,
    staleness: str = "guarded",
) -> MethodRunResult:
    """Run one method through AgentDescent with an equal candidate budget."""
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if workers < 2 or candidate_budget < workers or candidate_budget % workers:
        raise ValueError("candidate_budget must be divisible by workers and at least workers")
    if len(policy.train_tasks) < workers or len(policy.held_out_tasks) < 4:
        raise ValueError("ports require enough train tasks and at least four held-out tasks")

    run_llm = PhasedLLM(recorder, f"run:{policy.name}")
    prop_llm = PhasedLLM(recorder, f"proposal:{policy.name}")
    eval_llm = PhasedLLM(recorder, f"eval:{policy.name}")
    merge_llm = PhasedLLM(recorder, f"fusion:{policy.name}")
    # A population layer that seeds itself -- PromptBreeder generates its N
    # initial units -- spends model calls that are not proposals. Recording them
    # under `proposal:` would blow the budget check that makes these rows
    # comparable, and hide initialisation cost inside the search's cost.
    init_llm = PhasedLLM(recorder, f"init:{policy.name}")

    initial = policy.strategy.render(policy.strategy.initial())
    started = time.monotonic()
    baseline_quality = _evaluate(policy, eval_llm, initial, policy.test_tasks)
    baseline_validation = _evaluate(policy, eval_llm, initial, policy.held_out_tasks)
    engine_started = time.monotonic()
    actor_usage = Usage()
    limiter = ProposalLimiter(
        lambda rendered, task, output, reward: policy.propose(
            prop_llm, rendered, task, output, reward),
        candidate_budget,
    )
    engine = policy.engine
    if policy.reflective:
        engine = engine.merged_with(
            **reflective_merge(lambda prompt: merge_llm(prompt, unit="merge")))
    aggregator_factory = None
    if engine.selection is not None:
        # The engine's own selection seam is single-head degenerate, so a
        # declared selection policy runs through the sanctioned optimizer exit
        # instead: PopulationAggregator keeps the archive and commits parent
        # switches (the GEPA/DGM pattern, generalised). The factory path
        # bypasses the bundle's decision fields, so they travel through the
        # factory and are stripped from the bundle -- carried in both places,
        # one copy would be silently ignored.
        pop_ctx = PopulationContext(
            selection=engine.selection,
            artifact_id=f"candidate-{policy.name}",
            conflict=engine.conflict, fusion=engine.fusion,
            acceptance=engine.acceptance,
            llm=lambda prompt, **kw: init_llm(prompt, **kw),
            # The *training* split, which is where PromptBreeder's Algorithm 1
            # puts its tournament: "evaluate the fitness of both units on a
            # random batch of training data". The held-out score the shared
            # archive uses is the acceptance gate's signal, a different question,
            # and using it to rank a population is a departure worth not making
            # silently.
            fitness=lambda state, batch: _evaluate(
                policy, eval_llm, policy.strategy.render(state),
                _batch(policy.train_tasks, batch, seed)),
            train_size=len(policy.train_tasks),
            seed=seed)
        aggregator_factory = (policy.population or _default_population)(pop_ctx)
        engine = replace(engine, selection=None, conflict=None, fusion=None,
                         acceptance=None)
    common = {
        "run": lambda rendered, task: policy.solve(run_llm, rendered, task),
        "propose": limiter,
        "strategy": policy.strategy,
        "artifact_id": f"candidate-{policy.name}",
        "initial_state": policy.strategy.initial(),
        "blast_radius": 0.6,
        "n_workers": workers,
        "self_verify": policy.self_verify,
        "held_out_frac": policy.held_out_frac,
        "solved_threshold": 1.1,
        "eval_concurrency": 1 if mode == "serial" else workers,
        # Batches sized to the worker count so concurrent proposals actually
        # meet in one merge -- with batch_trigger=1 every batch is a single
        # card and no merge machinery (union or reflective) ever runs.
        "agg_config": AggregatorConfig(batch_trigger=workers, max_wait_rounds=1),
        "shuffle": False,
        "seed": seed,
        "usage": actor_usage,
        "verbose": False,
        "policies": engine,
        "aggregator_factory": aggregator_factory,
        # A discarded card is a candidate from the budget spent on nothing, and
        # the budget is what makes these rows comparable. `full` rebases onto the
        # current head and leaves the acceptance gate as the verification, which
        # it already is.
        "staleness_policy": get_policy(staleness),
    }
    if mode == "async_pipeline":
        result = async_evolve(
            policy.engine_tasks,
            policy.reward,
            async_ratio=async_ratio,
            max_iters=candidate_budget,
            max_seconds=max_seconds,
            shutdown_grace=shutdown_grace,
            **common,
        )
    else:
        result = evolve(
            policy.engine_tasks,
            policy.reward,
            rounds=candidate_budget // workers,
            max_concurrency=1 if mode == "serial" else workers,
            **common,
        )
    engine_wall = time.monotonic() - engine_started
    if result.error:
        raise RuntimeError(result.error)
    final_quality = _evaluate(policy, eval_llm, result.rendered, policy.test_tasks)
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
    fusion_calls = sum(
        1 for event in recorder.events if event.phase.startswith("fusion:")
    )
    expected_proposal_calls = candidate_budget * policy.proposal_calls_per_candidate
    budget = {
        "reserved_candidates": candidate_budget,
        "observed_candidates": limiter.claimed,
        "observed_rollouts": result.rollouts,
        "expected_proposal_calls": expected_proposal_calls,
        "observed_proposal_calls": proposal_calls,
        "observed_fusion_calls": fusion_calls,
        # <= rather than ==: a method with a native early stop (Self-Refine's
        # "it is correct") can only under-spend its reserved calls; spending
        # more than reserved is still a mismatch.
        "matched": (
            limiter.claimed == candidate_budget
            and proposal_calls <= expected_proposal_calls
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
    return MethodRunResult(
        algorithm=policy.name,
        fidelity=policy.fidelity,
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
        candidates=limiter.claimed,
        accepted=int(result.outcomes().get("committed", 0)),
        stale_considered=result.stale_considered,
        stale_discarded=result.stale_discarded,
        invalid_candidates=policy.invalid_proposals,
        model_usage=usage_dict(recorder.usage),
        actor_usage=usage_dict(actor_usage),
        phase_summary=recorder.phase_summary(),
        events=compact_events(recorder.events),
        framework={
            "runtime": "async_evolve" if mode == "async_pipeline" else "evolve",
            "max_concurrency": (
                workers if mode in ("sync_parallel", "async_pipeline") else 1
            ),
            "reflective_merge": policy.reflective,
            "self_verify": policy.self_verify,
            "baseline_reward": baseline_validation,
            "final_reward": result.final_reward,
            "stop_reason": result.stop_reason,
            "outcomes": result.outcomes(),
            "fusion": _fusion_summary(result),
            "rollouts": result.rollouts,
            "rollout_seconds": result.rollout_seconds,
            "eval_seconds": result.eval_seconds,
            "stale_considered": result.stale_considered,
            "stale_discarded": result.stale_discarded,
            "retired_workers": result.retired_workers,
            "history": history,
        },
        budget=budget,
        notes=policy.notes,
    )


def standard_main(build: Callable[[int], MethodPolicy],
                  argv: Optional[Sequence[str]] = None) -> int:
    """The shared ``main`` for one method's folder module.

    ``--dry-run`` prints the plan with zero network access; a live run drives
    the method through :func:`run_port` once, in the mode picked by
    ``--serial`` / ``--async`` (default: synchronous parallel).
    """
    parser = argparse.ArgumentParser()
    add_standard_args(parser, model_default="glm-5.2")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--staleness", default="guarded",
                        choices=["guarded", "reflective", "full"],
                        help="what to do with a diff proposed against a head the "
                             "merger has since moved (agentdescent.staleness)")
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help=("sampling temperature. Every other port in this repository takes "
              "one; this runner passed none, so these eleven ran at the API "
              "default. Measured on deepseek-v4-flash over the money domain, a "
              "prompt that works the arithmetic out scores 1.000 / 0.958 / 0.875 "
              "at 0.0 / 0.7 / 1.0, so it is second-order next to what the prompt "
              "says -- but it is not nothing, and it was not reportable"))
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args(argv)

    policy = build(args.seed)
    # `--budget-rollouts` is `add_standard_args`' name for the quantity these
    # methods call `--candidates`: the total number of proposals the run may
    # spend. It was declared for every port and read by none of them here, so a
    # sweep pinning the budget got the 2-candidate default and a row that
    # measured nothing it asked for. One quantity, two vocabularies -- mapped
    # rather than duplicated, the way OpenEvolve maps it onto `iterations`.
    if getattr(args, "budget_rollouts", None):
        args.candidates = args.budget_rollouts
    mode = ("serial" if args.serial
            else "async_pipeline" if args.asynchronous
            else "sync_parallel")
    if args.dry_run:
        print(f"{policy.name} [{policy.fidelity}] mode={mode} "
              f"candidates={args.candidates} workers={args.workers} "
              f"proposal calls={args.candidates * policy.proposal_calls_per_candidate} "
              f"reflective_merge={policy.reflective} self_verify={policy.self_verify}")
        for note in policy.notes:
            print(f"  - {note}")
        print("[dry-run] no dataset or model API was accessed.")
        return 0

    if not confirm(args):
        return 0
    usage = Usage()
    recorder = Recorder(
        completion_for(args, usage=usage, max_tokens=args.max_tokens,
                       timeout=args.timeout, temperature=args.temperature),
        usage,
    )
    outcome = run_port(
        policy, recorder, mode=mode, seed=args.seed, workers=args.workers,
        candidate_budget=args.candidates, max_seconds=args.max_seconds,
        staleness=args.staleness,
    )
    print(f"{policy.name}/{mode}: quality {outcome.baseline_quality:.3f} -> "
          f"{outcome.final_quality:.3f}, validation "
          f"{outcome.baseline_validation_quality:.3f} -> {outcome.validation_quality:.3f}, "
          f"accepted={outcome.accepted}/{outcome.candidates} "
          f"invalid={outcome.invalid_candidates} "
          f"wall={outcome.wall_seconds:.1f}s engine={outcome.engine_wall_seconds:.1f}s "
          f"calls={outcome.model_usage['calls']} budget={outcome.budget['matched']}")
    if policy.report is not None:
        detail = policy.report()
        if detail:
            print(detail)
    return 0
