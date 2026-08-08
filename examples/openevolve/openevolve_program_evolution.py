"""OpenEvolve function minimization, ported onto the AgentDescent engine.

Upstream reference
------------------
Repository: https://github.com/algorithmicsuperintelligence/openevolve
Commit: 411fb59c886c18704caaffb611e17cf9e7d824d2
Example: examples/function_minimization

Preserved from that revision:

* Python source is the genome and the model mutates a selected parent.
* Fitness uses the upstream function-minimization evaluator weights.
* Parent selection mixes exploitation with exploration.
* Each island owns a MAP-Elites grid over complexity and code diversity.
* Children stay on their target island and elites migrate in a ring.

Intentional differences:

* The small genome is rewritten in full instead of SEARCH/REPLACE diffs.
* Candidate execution is deterministic, budgeted, and isolated with Bubblewrap.
* AgentDescent supplies workers, the ledger, evidence cards, staleness handling,
  synchronous barriers, and the barrier-free async runtime. OpenEvolve's process
  controller and database are therefore represented by a Strategy and Aggregator.
* Feature bins use fixed log-length boundaries and insertion-time token-Jaccard
  diversity instead of OpenEvolve's evolving min/max scaling state.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from agentdescent.agents import Completion, Usage
from agentdescent.aggregator import (
    AggregatorConfig,
    AggregatorProtocol,
    MergeOutcome,
    MergeReport,
)
from agentdescent.async_evolve import async_evolve
from agentdescent.evolution import EvolutionResult, EvolvingArtifact, Task, evolve
from agentdescent.evolvable import Diff, EvidenceCard, vv_staleness
from agentdescent.governance import classify
from agentdescent.ledger import CASConflict, Ledger
from agentdescent.staleness import StaleAction, get_policy

from examples._common import (
    add_standard_args,
    completion_for,
    confirm,
    worker_count,
    is_openai_compatible,
    report_engine,
)
from examples.openevolve._openevolve_support import (
    INITIAL_PROGRAM,
    UPSTREAM_COMMIT,
    Program,
    code_distance,
    combined_metrics,
    evaluate_source,
    extract_program,
    mutation_prompt,
    program_id,
)


ARTIFACT_ID = "openevolve_program"
ARCHIVE_UPDATED = "archive-updated"
NO_VALID_CANDIDATES = "no-valid-candidates"
DEFAULT_OUTPUT = Path("openevolve-agentdescent-result.json")
Evaluator = Callable[..., Tuple[bool, Dict[str, Any], str, List[Dict[str, Any]]]]


def _require_api_environment(provider: str) -> None:
    required = {
        "openai": ("OPENAI_API_KEY",),
        "glm": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    }.get(provider, ())
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"missing environment variable(s): {', '.join(missing)}")


def _make_completion(args: argparse.Namespace, usage: Usage) -> Completion:
    """Dispatch through the shared helper, branching only for the GLM extra.

    ``thinking`` is one-sided -- it reaches an OpenAI-compatible GLM endpoint and
    nothing else -- so it is the one option assembled here rather than passed to
    both factories, exactly as ADAS does for its OpenAI-only ``--timeout``.
    """
    options: Dict[str, Any] = {}
    if (
        is_openai_compatible(args)
        and args.thinking != "default"
        and (args.provider == "glm" or args.model.lower().startswith("glm"))
    ):
        options["thinking"] = {"type": args.thinking}
    return completion_for(
        args,
        usage=usage,
        max_tokens=args.max_tokens,
        timeout=args.api_timeout,
        temperature=args.temperature,
        **options,
    )


def _usage_dict(usage: Usage) -> Dict[str, Any]:
    return {
        "calls": usage.calls,
        "failures": usage.failures,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "model_seconds": round(usage.seconds, 6),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def framework_score(metrics: Dict[str, Any]) -> float:
    """Map OpenEvolve's [0, 1.5] combined score into AgentDescent's [0, 1]."""
    return max(0.0, min(1.0, float(metrics.get("combined_score") or 0.0) / 1.5))


def _trial_framework_score(trial: Dict[str, Any]) -> float:
    return framework_score(combined_metrics([trial])) if trial.get("success") else 0.0


def evaluate_program_metrics(
    code: str,
    *,
    trials: int,
    budget: int,
    seed: int,
    timeout: float,
    max_length: int,
    evaluator: Evaluator = evaluate_source,
) -> Tuple[bool, Dict[str, Any], str]:
    valid, metrics, error, trial_rows = evaluator(
        code,
        trials=trials,
        budget=budget,
        seed=seed,
        timeout=timeout,
        max_length=max_length,
    )
    metrics = dict(metrics)
    metrics["framework_reward"] = (
        sum(_trial_framework_score(row) for row in trial_rows) / trials if trials else 0.0
    )
    return valid, metrics, error


def _program_summary(program: Program) -> Dict[str, Any]:
    """Keep result files compact: metrics and lineage, not generated source text."""
    return {
        "program_id": program.program_id,
        "iteration": program.iteration,
        "island": program.island,
        "parent_id": program.parent_id,
        "change_summary": program.change_summary,
        "metrics": program.metrics,
        "valid": program.valid,
        "error": program.error,
        "code_chars": len(program.code),
    }


class EpsilonGreedy:
    """OpenEvolve's in-pool parent rule at the standard selection seam.

    A :class:`~agentdescent.selection.SelectionPolicy`: with probability
    ``exploitation_ratio`` take the pool's best-fitness member (first maximal
    wins, as the inline ``max`` did), otherwise draw uniformly. It shares the
    archive's rng, and consumes it in the same order as the inline rule --
    one ``random()`` then at most one ``choice`` -- so seeded runs pick
    identical parents. Islands, MAP-Elites cells, and ring migration are the
    upstream mechanism itself and stay on the archive.
    """

    def __init__(self, rng, exploitation_ratio: float) -> None:
        self.rng = rng
        self.exploitation_ratio = exploitation_ratio

    def select(self, ctx, n: int):
        candidates = list(ctx.candidates)
        if not candidates:
            return [ctx.head] * n
        if self.rng.random() < self.exploitation_ratio:
            pick = max(candidates, key=lambda c: c.score or 0.0)
        else:
            pick = self.rng.choice(candidates)
        return [pick] * n


@dataclass
class OpenEvolveArchive:
    """Thread-safe MAP-Elites and island state shared by proposers and merger."""

    archive_size: int = 20
    num_islands: int = 3
    feature_bins: int = 4
    exploitation_ratio: float = 0.7
    migration_interval: int = 4
    max_code_length: int = 20_000
    rng_seed: int = 0
    candidate_limit: Optional[int] = None
    programs: Dict[str, Program] = field(default_factory=dict)
    history: List[Program] = field(default_factory=list)
    island_cells: List[Dict[Tuple[int, int], str]] = field(default_factory=list)
    island_generations: List[int] = field(default_factory=list)
    best_id: Optional[str] = None
    baseline_id: Optional[str] = None
    migrations: int = 0
    _next_iteration: int = 1
    _last_migration_generation: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.rng_seed)
        self.island_cells = [dict() for _ in range(self.num_islands)]
        self.island_generations = [0 for _ in range(self.num_islands)]

    def _fitness(self, program: Program) -> float:
        return float(program.metrics.get("combined_score") or 0.0)

    def _archive_ids_locked(self) -> List[str]:
        ids = {pid for cells in self.island_cells for pid in cells.values()}
        ranked = sorted(ids, key=lambda pid: self._fitness(self.programs[pid]), reverse=True)
        return ranked[: self.archive_size]

    def _feature_cell_locked(self, code: str) -> Tuple[int, int]:
        length_ratio = math.log1p(len(code)) / math.log1p(self.max_code_length)
        complexity = min(self.feature_bins - 1, int(length_ratio * self.feature_bins))
        references = [program.code for program in self.programs.values() if program.valid]
        diversity = (
            sum(code_distance(code, other) for other in references) / len(references)
            if references
            else 0.0
        )
        diversity_bin = min(self.feature_bins - 1, int(diversity * self.feature_bins))
        return complexity, diversity_bin

    def add_program(self, program: Program, *, baseline: bool = False) -> bool:
        """Evaluate one MAP-Elites insertion; return whether its cell changed."""
        with self._lock:
            if program.program_id in self.programs:
                return False
            self.programs[program.program_id] = program
            self.history.append(program)
            if baseline:
                self.baseline_id = program.program_id
            if not program.valid:
                return False

            island = program.island % self.num_islands
            cell = self._feature_cell_locked(program.code)
            incumbent_id = self.island_cells[island].get(cell)
            changed = incumbent_id is None or self._fitness(program) > self._fitness(
                self.programs[incumbent_id]
            )
            if changed:
                self.island_cells[island][cell] = program.program_id
            self.island_generations[island] += 1

            if self.best_id is None or self._fitness(program) > self._fitness(
                self.programs[self.best_id]
            ):
                self.best_id = program.program_id

            if (
                max(self.island_generations) - self._last_migration_generation
                >= self.migration_interval
            ):
                self._migrate_locked()
            return changed

    def _migrate_locked(self) -> None:
        """Copy each island's best cell elite to its clockwise neighbor."""
        outgoing: List[Tuple[int, Tuple[int, int], str]] = []
        for island, cells in enumerate(self.island_cells):
            if not cells:
                continue
            elite_id = max(cells.values(), key=lambda pid: self._fitness(self.programs[pid]))
            target = (island + 1) % self.num_islands
            cell = self._feature_cell_locked(self.programs[elite_id].code)
            outgoing.append((target, cell, elite_id))
        for target, cell, elite_id in outgoing:
            incumbent_id = self.island_cells[target].get(cell)
            if incumbent_id is None or self._fitness(self.programs[elite_id]) > self._fitness(
                self.programs[incumbent_id]
            ):
                self.island_cells[target][cell] = elite_id
                self.migrations += 1
        self._last_migration_generation = max(self.island_generations)

    def select_parent(self) -> Optional[Tuple[int, int, Program, Program, Program]]:
        """Reserve an iteration and take one consistent archive snapshot."""
        with self._lock:
            if self.best_id is None:
                raise RuntimeError("OpenEvolve archive has not been seeded")
            iteration = self._next_iteration
            if self.candidate_limit is not None and iteration > self.candidate_limit:
                return None
            self._next_iteration += 1
            island = (iteration - 1) % self.num_islands
            island_ids = list(dict.fromkeys(self.island_cells[island].values()))
            archive_ids = self._archive_ids_locked()
            pool_ids = island_ids or archive_ids
            if not pool_ids:
                raise RuntimeError("OpenEvolve archive has no parent candidates")
            # The epsilon-greedy pick at the standard seam (version = pool index).
            from agentdescent.selection import Candidate, SelectionContext
            if not hasattr(self, "_selection") or self._selection is None:
                self._selection = EpsilonGreedy(self._rng, self.exploitation_ratio)
            rows = [Candidate(artifact_id="openevolve", version=i,
                              score=self._fitness(self.programs[pid]))
                    for i, pid in enumerate(pool_ids)]
            sel_ctx = SelectionContext(head=rows[0], candidates=tuple(rows),
                                       n_workers=1)
            parent_id = pool_ids[self._selection.select(sel_ctx, 1)[0].version]
            parent = self.programs[parent_id]
            best = self.programs[self.best_id]
            inspiration_id = max(
                archive_ids,
                key=lambda pid: code_distance(parent.code, self.programs[pid].code),
            )
            return iteration, island, parent, best, self.programs[inspiration_id]

    def best(self) -> Program:
        with self._lock:
            if self.best_id is None:
                raise RuntimeError("OpenEvolve archive has no best program")
            return self.programs[self.best_id]

    def baseline(self) -> Program:
        with self._lock:
            if self.baseline_id is None:
                raise RuntimeError("OpenEvolve archive has no baseline program")
            return self.programs[self.baseline_id]

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            archive_ids = self._archive_ids_locked()
            return {
                "programs_evaluated": len(self.history),
                "valid_programs": sum(program.valid for program in self.history),
                "archive_ids": archive_ids,
                "island_cell_counts": [len(cells) for cells in self.island_cells],
                "island_generations": list(self.island_generations),
                "migrations": self.migrations,
                "history": [_program_summary(program) for program in self.history],
            }


class OpenEvolveStrategy:
    """Represent one executable program and parse model mutations into Diffs."""

    def initial(self) -> Dict[str, str]:
        return {
            "code": INITIAL_PROGRAM,
            "program_id": program_id(INITIAL_PROGRAM),
            "change_summary": "uniform random search baseline",
        }

    def render(self, state: Dict[str, str]) -> str:
        return state.get("code", INITIAL_PROGRAM)

    def keys(self) -> Sequence[str]:
        return ("code", "program_id", "change_summary", "parent_id", "island")

    def to_diff(
        self,
        state: Dict[str, str],
        proposal: str,
        author: str,
        base_version: int,
        target: str,
    ) -> Optional[Diff]:
        try:
            payload = json.loads(proposal)
            code = str(payload["code"]).strip()
            iteration = int(payload["iteration"])
            island = int(payload["island"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if not code:
            return None
        pid = program_id(code)
        return Diff(
            diff_id=f"{author}:{pid}:{iteration}:{base_version}",
            target=target,
            ops={
                "code": code,
                "program_id": pid,
                "change_summary": str(payload.get("change_summary") or ""),
                "parent_id": str(payload.get("parent_id") or ""),
                "island": str(island),
                "iteration": str(iteration),
            },
            author=author,
        )


def make_propose(
    archive: OpenEvolveArchive,
    complete: Completion,
    *,
    objective_budget: int,
    held_out_trials: int,
) -> Callable[[str, Task, str, float], Optional[str]]:
    """Adapt OpenEvolve parent selection and mutation to the engine actor API."""

    def propose(rendered: str, task: Task, output: str, reward: float) -> Optional[str]:
        selection = archive.select_parent()
        if selection is None:
            return None
        iteration, island, parent, best, inspiration = selection
        raw = complete(
            mutation_prompt(
                parent,
                best,
                inspiration,
                iteration=iteration,
                budget=objective_budget,
                trials=held_out_trials,
            )
        ).strip()
        code, summary = extract_program(raw)
        return json.dumps(
            {
                "code": code,
                "change_summary": summary,
                "iteration": iteration,
                "island": island,
                "parent_id": parent.program_id,
            },
            separators=(",", ":"),
        )

    return propose


def make_run(
    *,
    objective_budget: int,
    candidate_timeout: float,
    max_code_length: int,
    evaluator: Evaluator = evaluate_source,
) -> Callable[[str, Task], str]:
    """Turn one deterministic evaluator seed into an AgentDescent rollout."""

    def run(rendered: str, task: Task) -> str:
        valid, metrics, error, trials = evaluator(
            rendered,
            trials=1,
            budget=objective_budget,
            seed=int(task.meta["seed"]),
            timeout=candidate_timeout,
            max_length=max_code_length,
        )
        return json.dumps(
            {"valid": valid, "metrics": metrics, "error": error, "trials": trials},
            separators=(",", ":"),
        )

    return run


def reward_program(task: Task, output: str) -> float:
    try:
        payload = json.loads(output)
        if not payload.get("valid"):
            return 0.0
        return framework_score(payload["metrics"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 0.0


class OpenEvolveAggregator(AggregatorProtocol):
    """OpenEvolve's MAP-Elites archive as an AgentDescent merge optimizer."""

    def __init__(
        self,
        ledger: Ledger,
        verifier: Any,
        archive: OpenEvolveArchive,
        config: AggregatorConfig,
        staleness_policy: Any,
        *,
        objective_budget: int,
        candidate_timeout: float,
        max_code_length: int,
        evaluator: Evaluator = evaluate_source,
        artifact_id: str = ARTIFACT_ID,
    ) -> None:
        self.ledger = ledger
        self.verifier = verifier
        self.archive = archive
        self.config = config
        self.staleness_policy = staleness_policy or get_policy("guarded")
        self.objective_budget = objective_budget
        self.candidate_timeout = candidate_timeout
        self.max_code_length = max_code_length
        self.evaluator = evaluator
        self.artifact_id = artifact_id
        self.cards: List[EvidenceCard] = []
        self._cards_lock = threading.Lock()
        self._seeded = False
        self.meter = None

    def _held_out_seed(self) -> Tuple[int, int]:
        seeds = sorted(int(task.meta["seed"]) for task in self.verifier.held_out)
        if not seeds:
            raise RuntimeError("OpenEvolve needs held-out evaluator seeds")
        expected = list(range(seeds[0], seeds[0] + len(seeds)))
        if seeds != expected:
            raise ValueError("OpenEvolve held-out seeds must be contiguous")
        return seeds[0], len(seeds)

    def _evaluate(self, code: str) -> Tuple[bool, Dict[str, Any], str]:
        seed, trials = self._held_out_seed()
        return evaluate_program_metrics(
            code,
            trials=trials,
            budget=self.objective_budget,
            seed=seed,
            timeout=self.candidate_timeout,
            max_length=self.max_code_length,
            evaluator=self.evaluator,
        )

    def seed(self) -> None:
        if self._seeded:
            return
        self._seeded = True
        head = self.ledger.snapshot(Ledger.DEV).get(self.artifact_id)
        code = head.state.get("code", INITIAL_PROGRAM)
        valid, metrics, error = self._evaluate(code)
        if not valid:
            raise RuntimeError(f"initial OpenEvolve program failed: {error}")
        self.archive.add_program(
            Program(
                program_id(code),
                0,
                0,
                None,
                code,
                "uniform random search baseline",
                metrics,
                valid,
                error,
            ),
            baseline=True,
        )

    def ingest(self, card: EvidenceCard) -> None:
        with self._cards_lock:
            self.cards.append(card)

    def _staleness_filter(
        self, head_version: Dict[str, int], cards: Sequence[EvidenceCard]
    ) -> Tuple[List[EvidenceCard], List[EvidenceCard]]:
        survivors: List[EvidenceCard] = []
        discarded: List[EvidenceCard] = []
        for card in cards:
            eta = vv_staleness(head_version, card.base_version)
            alpha = 0 if card.diff.contract_breaking else self.config.alpha_tail
            action = self.staleness_policy.decide(eta, alpha, card.diff.contract_breaking)
            if action is StaleAction.DISCARD:
                discarded.append(card)
            else:
                # REBASE is safe here because every survivor is fully re-evaluated
                # against the current held-out seeds before it can enter the archive.
                survivors.append(card if eta == 0 else card.rebased_onto(head_version))
        if self.meter is not None:
            self.meter.add("stale_considered", len(cards))
            self.meter.add("stale_discarded", len(discarded))
        return survivors, discarded

    def step(self) -> List[MergeReport]:
        self.seed()
        with self._cards_lock:
            cards, self.cards = self.cards, []
        if not cards:
            return []

        snapshot = self.ledger.snapshot(Ledger.DEV)
        head = snapshot.get(self.artifact_id)
        base_vv = {self.artifact_id: snapshot.version.get(self.artifact_id, 0)}
        survivors, discarded = self._staleness_filter(base_vv, cards)
        valid_candidates = 0
        archive_updates = 0
        for card in survivors:
            ops = card.diff.ops
            code = ops.get("code", "")
            valid, metrics, error = self._evaluate(code)
            program = Program(
                program_id(code),
                int(ops.get("iteration", "0")),
                int(ops.get("island", "0")),
                ops.get("parent_id") or None,
                code,
                ops.get("change_summary", ""),
                metrics,
                valid,
                error,
            )
            archive_updates += int(self.archive.add_program(program))
            valid_candidates += int(valid)

        best = self.archive.best()
        accepted: Optional[Diff] = None
        committed_version: Optional[int] = None
        category = ARCHIVE_UPDATED
        if not survivors:
            category = MergeOutcome.ALL_STALE.value
        elif not valid_candidates:
            category = NO_VALID_CANDIDATES
        if best.code != head.state.get("code"):
            accepted = Diff(
                diff_id=f"archive-best:{best.program_id}:{head.version}",
                target=self.artifact_id,
                ops={
                    "code": best.code,
                    "program_id": best.program_id,
                    "change_summary": best.change_summary,
                    "parent_id": best.parent_id or "",
                    "island": str(best.island),
                },
                author="openevolve-archive",
            )
            try:
                _, committed_version = self.ledger.commit(
                    head.apply(accepted),
                    base_vv,
                    branch=Ledger.DEV,
                    message="openevolve: commit best MAP-Elites program",
                )
                category = MergeOutcome.COMMITTED.value
            except CASConflict:
                accepted = None
                category = MergeOutcome.CAS_CONFLICT.value

        return [
            MergeReport(
                self.artifact_id,
                accepted,
                False,
                len(cards),
                len(survivors),
                len(discarded),
                0,
                float(best.metrics.get("combined_score") or 0.0),
                committed_version,
                (
                    f"valid={valid_candidates}/{len(survivors)} "
                    f"cell_updates={archive_updates} "
                    f"best={best.metrics.get('combined_score', 0.0):.6f}"
                ),
                category,
            )
        ]


@dataclass
class OpenEvolveRun:
    mode: str
    result: EvolutionResult
    archive: OpenEvolveArchive
    wall_seconds: float
    baseline_test_metrics: Dict[str, Any] = field(default_factory=dict)
    best_test_metrics: Dict[str, Any] = field(default_factory=dict)

    def summary(self, quality_target: Optional[float] = None) -> Dict[str, Any]:
        baseline = self.archive.baseline()
        best = self.archive.best()
        payload: Dict[str, Any] = {
            "mode": self.mode,
            "wall_seconds": self.wall_seconds,
            "baseline": _program_summary(baseline),
            "best": _program_summary(best),
            "framework": {
                "baseline_reward": baseline.metrics.get("framework_reward", 0.0),
                "final_reward": self.result.final_reward,
                "quality_gain": self.result.final_reward
                - float(baseline.metrics.get("framework_reward", 0.0)),
                "stop_reason": self.result.stop_reason,
                "error": self.result.error,
                "outcomes": self.result.outcomes(),
                "rollouts": self.result.rollouts,
                "rollout_seconds": self.result.rollout_seconds,
                "eval_seconds": self.result.eval_seconds,
                "stale_considered": self.result.stale_considered,
                "stale_discarded": self.result.stale_discarded,
                "stale_rate": self.result.stale_rate(),
                "retired_workers": self.result.retired_workers,
                "cache_hits": self.result.cache_hits,
                "cache_misses": self.result.cache_misses,
                "history": [
                    {
                        "round": row.round,
                        "held_out_reward": row.held_out_reward,
                        "elapsed_s": row.elapsed_s,
                        "rollouts": row.rollouts,
                        "committed": row.committed,
                        "rejected": row.rejected,
                        "reasons": row.reasons,
                    }
                    for row in self.result.history
                ],
            },
            "actor_usage": _usage_dict(self.result.usage),
            "archive": self.archive.summary(),
            "test": {
                "baseline": self.baseline_test_metrics,
                "best": self.best_test_metrics,
                "combined_score_gain": (
                    float(self.best_test_metrics.get("combined_score") or 0.0)
                    - float(self.baseline_test_metrics.get("combined_score") or 0.0)
                ),
            },
        }
        if quality_target is not None:
            payload["quality_target"] = quality_target
            payload["target_reached"] = self.result.final_reward >= quality_target
            payload["time_to_quality_s"] = self.result.time_to_quality(quality_target)
            payload["rollouts_to_quality"] = self.result.cost_to_quality(quality_target)
        return payload


def build_tasks(count: int, seed: int) -> List[Task]:
    return [
        Task(
            id=f"objective-seed-{seed + index}",
            prompt=f"Evaluate the search program with RNG seed {seed + index}.",
            meta={"seed": seed + index},
        )
        for index in range(count)
    ]


def run_agentdescent_openevolve(
    complete: Completion,
    *,
    mode: str,
    iterations: int = 6,
    workers: int = 3,
    task_count: int = 12,
    test_trials: int = 6,
    held_out_frac: float = 0.5,
    objective_budget: int = 200,
    archive_size: int = 20,
    islands: int = 3,
    feature_bins: int = 4,
    exploitation_ratio: float = 0.7,
    migration_interval: int = 4,
    candidate_timeout: float = 15.0,
    max_code_length: int = 20_000,
    async_ratio: int = 1,
    max_seconds: float = 300.0,
    shutdown_grace: float = 120.0,
    seed: int = 0,
    usage: Optional[Usage] = None,
    evaluator: Evaluator = evaluate_source,
    verbose: bool = False,
) -> OpenEvolveRun:
    """Run one fixed-candidate-budget serial, sync, or async experiment."""
    if mode not in ("serial", "sync", "async"):
        raise ValueError("mode must be serial, sync, or async")
    if iterations < 1 or workers < 1 or iterations % workers:
        raise ValueError("iterations must be positive and divisible by workers")
    if task_count < 8:
        raise ValueError("task_count must be >= 8 so the held-out set has at least four seeds")
    if not 0.0 < held_out_frac < 1.0:
        raise ValueError("held_out_frac must be in (0, 1)")
    if not 0.0 <= exploitation_ratio <= 1.0:
        raise ValueError("exploitation_ratio must be in [0, 1]")

    tasks = build_tasks(task_count, seed)
    held_out_trials = task_count - max(1, round(task_count * (1 - held_out_frac)))
    archive = OpenEvolveArchive(
        archive_size=archive_size,
        num_islands=islands,
        feature_bins=feature_bins,
        exploitation_ratio=exploitation_ratio,
        migration_interval=migration_interval,
        max_code_length=max_code_length,
        rng_seed=seed,
        candidate_limit=iterations,
    )
    strategy = OpenEvolveStrategy()
    run = make_run(
        objective_budget=objective_budget,
        candidate_timeout=candidate_timeout,
        max_code_length=max_code_length,
        evaluator=evaluator,
    )
    propose = make_propose(
        archive,
        complete,
        objective_budget=objective_budget,
        held_out_trials=held_out_trials,
    )

    def factory(ledger, verifier, audit, config, policy):
        aggregator = OpenEvolveAggregator(
            ledger,
            verifier,
            archive,
            config,
            policy,
            objective_budget=objective_budget,
            candidate_timeout=candidate_timeout,
            max_code_length=max_code_length,
            evaluator=evaluator,
        )
        aggregator.seed()
        return aggregator

    started = time.monotonic()
    common = {
        "run": run,
        "propose": propose,
        "strategy": strategy,
        "blast_radius": 0.6,
        "artifact_id": ARTIFACT_ID,
        "n_workers": workers,
        "self_verify": False,
        "eval_concurrency": workers,
        "held_out_frac": held_out_frac,
        "solved_threshold": 1.0,
        "usage": usage,
        "aggregator_factory": factory,
        "verbose": verbose,
    }
    if mode == "async":
        result = async_evolve(
            tasks,
            reward_program,
            async_ratio=async_ratio,
            max_seconds=max_seconds,
            max_iters=iterations,
            shutdown_grace=shutdown_grace,
            **common,
        )
    else:
        result = evolve(
            tasks,
            reward_program,
            rounds=iterations // workers,
            max_concurrency=1 if mode == "serial" else workers,
            **common,
        )
    if verbose:
        report_engine(result)
    wall_seconds = time.monotonic() - started
    test_seed = seed + task_count
    _, baseline_test, _ = evaluate_program_metrics(
        archive.baseline().code,
        trials=test_trials,
        budget=objective_budget,
        seed=test_seed,
        timeout=candidate_timeout,
        max_length=max_code_length,
        evaluator=evaluator,
    )
    _, best_test, _ = evaluate_program_metrics(
        archive.best().code,
        trials=test_trials,
        budget=objective_budget,
        seed=test_seed,
        timeout=candidate_timeout,
        max_length=max_code_length,
        evaluator=evaluator,
    )
    return OpenEvolveRun(
        mode,
        result,
        archive,
        wall_seconds,
        baseline_test,
        best_test,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_standard_args(parser, model_default="glm-5.2", max_seconds_default=300.0)
    parser.set_defaults(provider="openai", async_ratio=1)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--test-trials", type=int, default=6)
    parser.add_argument("--held-out-frac", type=float, default=0.5)
    parser.add_argument("--objective-budget", type=int, default=200)
    parser.add_argument("--archive-size", type=int, default=20)
    parser.add_argument("--islands", type=int, default=3)
    parser.add_argument("--feature-bins", type=int, default=4)
    parser.add_argument("--exploitation-ratio", type=float, default=0.7)
    parser.add_argument("--migration-interval", type=int, default=4)
    parser.add_argument("--candidate-timeout", type=float, default=15.0)
    parser.add_argument("--max-code-length", type=int, default=20_000)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--thinking", choices=("disabled", "enabled", "default"), default="disabled")
    parser.add_argument("--api-timeout", type=float, default=180.0)
    parser.add_argument(
        "--shutdown-grace",
        type=float,
        default=120.0,
        help="wait for already-started async model calls so end-to-end cost is complete",
    )
    parser.add_argument("--quality-target", type=float)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    # --serial collapses this to the upstream algorithm's own semantics:
    # one worker, nothing to merge. Applied to args so the printed plan,
    # the cost estimate and the run cannot disagree about what ran.
    args.workers = worker_count(args, args.workers)
    # This port already had the property the other six were missing: `iterations`
    # is the total candidate budget and `rounds = iterations // workers`, so work
    # is fixed and workers only change how it is divided. `--budget-rollouts` is
    # therefore the same quantity under the shared name, not a second knob -- and
    # mapping it here is what lets one matrix row be produced the same way as the
    # other six.
    if args.budget_rollouts:
        args.iterations = args.budget_rollouts
    mode = "async" if args.asynchronous else ("serial" if args.serial else "sync")
    print("Algorithm: OpenEvolve program evolution on AgentDescent")
    print("Evaluator: pinned function-minimization combined score, Bubblewrap isolated")
    print(
        f"Plan     : mode={mode}, model={args.model}, iterations={args.iterations}, "
        f"workers={args.workers}, temperature={args.temperature}"
    )
    artifact = EvolvingArtifact(ARTIFACT_ID, blast_radius=0.6)
    print(
        f"Governance: generated program blast_radius={artifact.blast_radius} "
        f"-> {classify(artifact).name}"
    )
    if args.dry_run:
        print("[dry-run] no API, dataset, or sandbox process was accessed.")
        return 0

    _require_api_environment(args.provider)
    if not confirm(args):
        return 0

    model_usage = Usage()
    actor_usage = Usage()
    complete = _make_completion(args, model_usage)
    run = run_agentdescent_openevolve(
        complete,
        mode=mode,
        iterations=args.iterations,
        workers=args.workers,
        task_count=args.tasks,
        test_trials=args.test_trials,
        held_out_frac=args.held_out_frac,
        objective_budget=args.objective_budget,
        archive_size=args.archive_size,
        islands=args.islands,
        feature_bins=args.feature_bins,
        exploitation_ratio=args.exploitation_ratio,
        migration_interval=args.migration_interval,
        candidate_timeout=args.candidate_timeout,
        max_code_length=args.max_code_length,
        async_ratio=args.async_ratio,
        max_seconds=args.max_seconds,
        shutdown_grace=args.shutdown_grace,
        seed=args.seed,
        usage=actor_usage,
        verbose=True,
    )
    payload = {
        "experiment": "OpenEvolve on AgentDescent",
        "status": "completed" if run.result.error is None else "partial",
        "completed_at": _utc_now(),
        "upstream_commit": UPSTREAM_COMMIT,
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in ("output", "yes")
        },
        "observation": run.summary(args.quality_target),
        "model_usage": _usage_dict(model_usage),
    }
    _write_json(args.output, payload)
    best_path = args.output.with_name(f"{args.output.stem}-best.py")
    best_path.write_text(run.archive.best().code.rstrip() + "\n", encoding="utf-8")
    print(
        f"completed: reward={run.archive.baseline().metrics['framework_reward']:.4f} -> "
        f"{run.result.final_reward:.4f}, wall={run.wall_seconds:.2f}s, "
        f"model_calls={model_usage.calls}, output={args.output}"
    )
    return 0 if run.result.error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
