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
import json
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

from ._common import (add_standard_args, completion_for, confirm,
                      is_openai_compatible)
from ._measure import MODES, PhasedLLM, Recorder, compact_events, usage_dict
from ._method_policy import MethodPolicy, PopulationContext


QUALITY_TARGET = 0.75

#: The lag budget these ports run at when ``--async-ratio`` is not given.
#:
#: A deliberate 1 rather than ``add_standard_args``' 3, because **two** entry
#: points reach :func:`run_port` -- this command line and
#: ``bench.candidate_methods`` -- and a lag budget they disagree on is a trap:
#: the same nominal configuration would run two different searches depending on
#: which one launched it. ``run_port``'s own signature says 1 and
#: ``bench.candidate_methods`` defaults to 1, so the parser says 1 too. The 3
#: stays where it was measured, on the seven benchmark ports.
#:
#: Stated here, in the parser's help, and in `docs/self-evolution-examples.md`,
#: because a default that changes what a run measures is not an implementation
#: detail -- the budget bounds both how far a worker's snapshot may drift behind
#: head and how many cards may sit un-merged ahead of the merger. Fifteen rows in
#: ``bench/results/`` recorded 2 and ran at this default, because the flag was
#: dropped before it reached the runtime; :func:`run_config` is why that can no
#: longer happen unnoticed.
DEFAULT_ASYNC_RATIO = 1


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
    async_ratio: int = DEFAULT_ASYNC_RATIO,
    max_seconds: float = 300.0,
    shutdown_grace: float = 120.0,
    staleness: str = "guarded",
    reflective: Optional[bool] = None,
    eval_concurrency: Optional[int] = None,
    eval_cache: str = "",
) -> MethodRunResult:
    """Run one method through AgentDescent with an equal candidate budget.

    ``eval_concurrency`` bounds the gate's evaluations, and ``None`` keeps this
    runner's own rule -- one in ``serial``, the worker count otherwise. It is a
    wall-clock knob rather than a result knob, but it is not free of meaning
    either: the serial arm is the *control*, and scoring its gate concurrently
    is what ``bench/matrix_run.py`` found had already made one control arm
    partly parallel. So an explicit value is honoured in every mode, and a
    missing one never turns the control on.

    ``eval_cache`` is a directory. Empty means the in-process default; a path
    installs :class:`~agentdescent.evalcache.FileCache`, so two processes stop
    paying twice for the same gate -- and a rerun returns the first run's
    numbers, which is what you want while sizing a configuration and exactly
    what you do not want when the question is run-to-run variance. Either way
    the choice is recorded in ``framework``.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if workers < 2 or candidate_budget < workers or candidate_budget % workers:
        raise ValueError("candidate_budget must be divisible by workers and at least workers")
    if len(policy.train_tasks) < workers or len(policy.held_out_tasks) < 4:
        raise ValueError("ports require enough train tasks and at least four held-out tasks")
    if eval_concurrency is not None and eval_concurrency < 1:
        raise ValueError("eval_concurrency must be at least 1")

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
    # `reflective=None` means "whatever the method declares", which is the
    # fidelity statement: AFlow's contested workflow fields are model-merged
    # because its optimizer rewrites a whole workflow, and Voyager's are not
    # because its library overwrites a key. An explicit True/False is a
    # deliberate override, for a control run that needs to vary exactly this.
    #
    # It used to be `policy.reflective` alone while `--reflective-merge` sat in
    # the shared parser being ignored, so a run could pass the flag, print
    # `reflective_merge=True` because the *policy* said so, and be identical to
    # a run that did not pass it. A control experiment built on that flag
    # measured nothing, and one in this series did.
    use_reflective = policy.reflective if reflective is None else bool(reflective)
    if use_reflective:
        # A strategy that validates its proposals must validate the *merge* too.
        # `ReflectiveFusion`'s synthesis reaches the ledger without passing
        # `to_diff`, so an artifact with a shape -- Python behind an AST gate --
        # would otherwise take a merged value that fails at every rollout that
        # reads it. Handing the validator over lets those ports merge at all;
        # anything it rejects falls back to ranking, which is what they used to
        # do for every diff.
        validate = getattr(policy.strategy, "validator", None)
        engine = engine.merged_with(
            **reflective_merge(lambda prompt: merge_llm(prompt, unit="merge"),
                               validate=validate))
    aggregator_factory = None
    if engine.selection is not None and policy.population is not None:
        # A *custom* population layer still needs the optimizer exit: it is a
        # `PopulationAggregator` subclass with the method's own algorithm in it
        # (PromptBreeder's tournament evaluates and replaces, which a
        # `SelectionPolicy` cannot do). The plain case no longer comes through
        # here -- `Policies(selection=...)` installs the shared layer in the
        # engine itself, so the runner stopped carrying a rule the engine now
        # owns.
        #
        # The factory path bypasses the bundle's decision fields, so they travel
        # through the factory and are stripped from the bundle: carried in both
        # places, one copy would be silently ignored -- and a declared policy
        # alongside a factory is now refused outright.
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
        aggregator_factory = policy.population(pop_ctx)
        engine = replace(engine, selection=None, conflict=None, fusion=None,
                         acceptance=None)
    if eval_cache:
        # Merged onto the bundle rather than built as a fresh one: the bundle is
        # what this runner already hands to both engines, and `_common`'s
        # `eval_cache_kwargs` returns `Policies(eval_cache=...)`, which would
        # have replaced everything the method declared.
        from agentdescent.evalcache import FileCache
        engine = engine.merged_with(eval_cache=FileCache(eval_cache))
    # `--serial` is the upstream algorithm's own loop, and the published loop
    # scores one task at a time; an explicit flag overrides that, in every mode,
    # and is reported below so a row says which control it had.
    gate_concurrency = (
        (1 if mode == "serial" else workers) if eval_concurrency is None
        else eval_concurrency)
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
        "eval_concurrency": gate_concurrency,
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
            "reflective_merge": use_reflective,
            # The three settings that used to be unrecordable because nothing
            # could set them. `async_ratio` only means something without a
            # barrier, so it is `None` in the other two modes rather than a
            # number a reader would compare across arms; `eval_cache` is a
            # boolean because the directory is a local path, and what a reader
            # needs is that the gate may not have been re-measured at all.
            "async_ratio": async_ratio if mode == "async_pipeline" else None,
            "eval_concurrency": gate_concurrency,
            "eval_cache": bool(eval_cache),
            # The one field a run could not state about itself, and the one that
            # made fifteen recorded config blocks impossible to attribute: a
            # block saying `staleness: full` could not have come from
            # `bench.candidate_methods`, which never passes it, and the run's own
            # record did not say either way.
            "staleness": staleness,
            "acceptance": (type(policy.engine.acceptance).__name__
                           if policy.engine.acceptance is not None else "default"),
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


#: What a run has to state about itself for its numbers to be reproducible.
#: The key names are the ones ``bench/results/*.json`` config blocks already use,
#: so a block is copied out of a run rather than remembered about it.
CONFIG_FIELDS = (
    "arm", "seed", "budget_rollouts", "workers", "async_ratio", "staleness",
    "reflective_merge", "self_verify", "eval_concurrency", "eval_cache",
    "temperature", "thinking", "provider", "model",
)


def run_config(args, policy: MethodPolicy, mode: str, *,
               reflective: Optional[bool] = None) -> Dict[str, object]:
    """The resolved configuration of the run about to happen, as a dict.

    Every ``bench/results/*.json`` config block was typed by hand beside the
    numbers, and the numbers themselves were transcribed from the line
    :func:`standard_main` prints -- which states what a run *reached* and nothing
    about how it was set up. Fifteen of those blocks then recorded
    ``async_ratio: 2`` for runs that took the runner's default of 1, because
    ``--async-ratio`` was dropped before it reached the runtime and nothing in
    the run's own output contradicted the number the author had asked for.

    So the run states its own configuration, in the key names those blocks use,
    and it is printed next to the results line. A block that disagrees with the
    run is now a block somebody retyped instead of copying.

    ``async_ratio`` is ``None`` outside ``async_pipeline`` for the reason
    ``framework`` gives: a lag budget means nothing with a round barrier in the
    way, and a number printed anyway is one a reader would compare across arms.
    """
    on = policy.reflective if reflective is None else bool(reflective)
    return {
        "arm": mode,
        "seed": args.seed,
        "budget_rollouts": args.candidates,
        "workers": args.workers,
        "async_ratio": args.async_ratio if mode == "async_pipeline" else None,
        "staleness": args.staleness,
        "reflective_merge": on,
        "self_verify": policy.self_verify,
        "eval_concurrency": ((1 if mode == "serial" else args.workers)
                             if args.eval_concurrency is None
                             else args.eval_concurrency),
        "eval_cache": bool(args.eval_cache),
        "temperature": args.temperature,
        # `--no-thinking` only reaches an Anthropic-shaped endpoint;
        # `completion_for` drops it on the OpenAI-compatible path rather than
        # guessing a per-vendor equivalent, so the run must not claim otherwise.
        "thinking": ("disabled" if getattr(args, "no_thinking", False)
                     and not is_openai_compatible(args) else "default"),
        "provider": args.provider,
        "model": args.model,
    }


def _reflective_override(args, policy: MethodPolicy) -> Optional[bool]:
    """`True`/`False` when a flag says so, `None` to keep the method's own.

    Both flags at once is refused rather than silently resolved: a run that asked
    for the merge and against it has not said what it wants, and picking one is
    how a control arm ends up measuring the other.
    """
    on = bool(getattr(args, "reflective_merge", False))
    off = bool(getattr(args, "no_reflective_merge", False))
    if on and off:
        raise SystemExit(
            "--reflective-merge and --no-reflective-merge are contradictory; "
            "pass neither to use the method's own declaration")
    if on:
        return True
    if off:
        return False
    return None


def build_parser(
    extra_args: Optional[Callable[[argparse.ArgumentParser], None]] = None,
) -> argparse.ArgumentParser:
    """The command line every MethodPolicy port accepts.

    Split out of :func:`standard_main` so that "which flags does this declare"
    and "which flags does this honour" can be checked against each other --
    ``tests/test_method_runner_flags.py`` enumerates this parser and requires
    every declared flag to name where it is read. A parser built inside a
    function body can only be inspected by running the function, which is how
    four flags sat here unread across all eleven ports.

    ``--val-cap`` is deliberately **not** offered: these methods freeze their
    three splits in ``build()``, before the parser is ever consulted, so there
    is no gate split left to cap. Refusing it is loud; accepting it and moving
    nothing is the failure this module exists to prevent.
    """
    parser = argparse.ArgumentParser()
    add_standard_args(parser, model_default="glm-5.2",
                      async_ratio_default=DEFAULT_ASYNC_RATIO,
                      # `None`, not 8: an explicit flag wins, and without one the
                      # runner keeps one-at-a-time gate scoring in `--serial`.
                      eval_concurrency_default=None,
                      include_val_cap=False)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument(
        "--no-reflective-merge", action="store_true",
        help=("turn the model merge off for a method that declares it. The "
              "shared `--reflective-merge` turns it on for a method that does "
              "not; without either, the method's own declaration stands, and "
              "that declaration is a fidelity statement rather than a knob"))
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
    if extra_args is not None:
        extra_args(parser)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser


def standard_main(build: Callable[..., MethodPolicy],
                  argv: Optional[Sequence[str]] = None,
                  extra_args: Optional[Callable[[argparse.ArgumentParser], None]] = None,
                  build_kwargs: Optional[Callable[[argparse.Namespace], dict]] = None) -> int:
    """The shared ``main`` for one method's folder module.

    ``--dry-run`` prints the plan with zero network access; a live run drives
    the method through :func:`run_port` once, in the mode picked by
    ``--serial`` / ``--async`` (default: synchronous parallel).

    ``extra_args`` and ``build_kwargs`` are the seam for a method that has a
    switch of its own. Gödel Agent's ``--gateless`` was documented in five
    places -- its module docstring, its README, `algo-godel-agent.md`,
    `acceptance-policies.md` and `matrix-overview.md` -- while the parser
    rejected it outright, because `build` took a keyword nothing could reach.
    A method-specific flag needs somewhere to be declared, or it becomes prose.

    The reverse failure is the one this function shipped for longer:
    ``--async-ratio``, ``--eval-concurrency`` and ``--eval-cache`` were declared
    by the shared parser and never handed to :func:`run_port`, so a run could
    pass all three and be byte-identical to one that passed none. They are
    threaded below, and every flag :func:`build_parser` declares is now covered
    by ``tests/test_method_runner_flags.py``.
    """
    parser = build_parser(extra_args)
    args = parser.parse_args(argv)
    extra = dict(build_kwargs(args)) if build_kwargs is not None else {}

    policy = build(args.seed, **extra)
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
    override = _reflective_override(args, policy)
    merge_on = policy.reflective if override is None else override
    # What `run_port` will resolve `--eval-concurrency` to, computed here so the
    # plan states a number rather than "whatever the runner decides".
    gate_concurrency = (
        (1 if mode == "serial" else args.workers)
        if args.eval_concurrency is None else args.eval_concurrency)
    if args.dry_run:
        print(f"{policy.name} [{policy.fidelity}] mode={mode} "
              f"candidates={args.candidates} workers={args.workers} "
              f"proposal calls={args.candidates * policy.proposal_calls_per_candidate} "
              f"reflective_merge={merge_on}{'' if override is None else ' (overridden)'} "
              f"self_verify={policy.self_verify} "
              f"eval_concurrency={gate_concurrency}"
              # A lag budget means nothing with a barrier in the way, so it is
              # printed only where it applies. A flag absent from the plan is a
              # flag nobody checks, which is how these three went unnoticed.
              + (f" async_ratio={args.async_ratio}"
                 if mode == "async_pipeline" else "")
              + (" eval_cache=on" if args.eval_cache else "")
              # A switch that changes the acceptance rule belongs in the run's
              # own record, not only in the flag that set it.
              + (f" acceptance={type(policy.engine.acceptance).__name__}"
                 if policy.engine.acceptance is not None else ""))
        for note in policy.notes:
            print(f"  - {note}")
        # `build` has already run, and a method whose domain is a real dataset
        # fetches it there. Claiming otherwise was true of the hand-written
        # fixtures and stopped being true the moment a port moved to GSM8K.
        print("[dry-run] no model API was accessed; the dataset is loaded and "
              "cached by `build`.")
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
        staleness=args.staleness, reflective=_reflective_override(args, policy),
        async_ratio=args.async_ratio, eval_concurrency=args.eval_concurrency,
        eval_cache=args.eval_cache,
    )
    print(f"{policy.name}/{mode}: quality {outcome.baseline_quality:.3f} -> "
          f"{outcome.final_quality:.3f}, validation "
          f"{outcome.baseline_validation_quality:.3f} -> {outcome.validation_quality:.3f}, "
          f"accepted={outcome.accepted}/{outcome.candidates} "
          f"invalid={outcome.invalid_candidates} "
          f"wall={outcome.wall_seconds:.1f}s engine={outcome.engine_wall_seconds:.1f}s "
          f"calls={outcome.model_usage['calls']} budget={outcome.budget['matched']}")
    # The line above is what every `bench/results/*.json` cell was transcribed
    # from -- `.3f` qualities, `.1f` seconds, integer calls, which is why not one
    # of 45 cells carries more precision than that. It says what the run reached
    # and nothing about how it was set up, so the config block beside those cells
    # was typed from memory and fifteen of them recorded a lag budget the run
    # never had. This one is copyable.
    print("config: " + json.dumps(run_config(args, policy, mode,
                                             reflective=override)))
    if policy.report is not None:
        detail = policy.report()
        if detail:
            print(detail)
    return 0
