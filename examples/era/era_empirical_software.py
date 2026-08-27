"""ERA empirical-software search, ported onto the AgentDescent engine.

Upstream reference
------------------
Repository: https://github.com/google-research/era
Commit: b836730b5c000526af95116b1d0e2c60c8cf0a10
Example: implementation/playground_s3e1.py, on implementation/futs.py

Preserved from that revision:

* Flat UCB tree search: every node is selectable, the exploitation term is a
  normalised rank rather than a raw score, and the prior is uniform.
* `puct = rank_score + c_puct * (1/N) * sqrt(total_visits) / (1 + visits)`,
  with visits backpropagated up the parent chain and `c_puct = 1.0`.
* A node is appended for every expansion, including one whose program failed --
  a failure scores `-inf`, it does not vanish.
* The Kaggle Playground S3E1 regression task, its 80/20 head/tail split, its
  `train_and_predict(train_path, test_path)` contract, RMSE, and the mutation
  prompt with its four speed constraints.

Intentional differences:

* Upstream ships **no sandbox** -- `sandbox.py` is an abstract class whose
  `run` raises "Must provide a sandbox for executing untrusted code". This port
  supplies one (Bubblewrap on Linux, Seatbelt on macOS) plus an AST gate, and
  says plainly that with scikit-learn on the allowlist the sandbox, not the
  gate, is the boundary.
* Upstream scores its whole 20% tail and reports that same number. Here the
  tail is cut into shards and the last few are never shown to the search, so
  the reported figure is on rows the optimiser could not see.
* AgentDescent supplies workers, the ledger, evidence cards, staleness handling,
  and the barrier-free runtime. `futs.search`'s loop body therefore becomes a
  Strategy plus an Aggregator, and its selection rule is the shipped
  `selection.FlatPuct`.
* Upstream is serial. With N workers in flight, a visit is reserved at
  *selection* time instead of after execution -- which is what the serial loop
  would have seen had those expansions already been dispatched, and is
  identical to upstream at one worker.
* The loop is parameterised by a :class:`~examples.era._era_domain.Domain` so a
  second ERA task can run on it without a second copy of `futs.search`.
  :func:`s3e1_domain` is the default and is upstream's task;
  `era_hard_integrals.py` supplies the other one. Nothing about the search
  differs between them -- the domain carries the seed program, the evaluator,
  the prompt and the name of the metric, and that is all.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import threading
import time
import warnings
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
from agentdescent.selection import Candidate, FlatPuct, SelectionContext
from agentdescent.staleness import StaleAction, get_policy

from examples._common import (
    add_standard_args,
    completion_for,
    confirm,
    report_engine,
    worker_count,
)
from examples.era._era_domain import Domain
from examples.era._era_support import (
    INITIAL_PROGRAM,
    UPSTREAM_COMMIT,
    Program,
    Splits,
    data_preview,
    evaluate_source,
    extract_program,
    framework_score,
    mutation_prompt,
    prepare_splits,
    program_id,
    sandbox_backend,
    with_intact_replies,
)


#: Mutation replies that parsed to nothing, for the warning in `make_propose`.
_EMPTY_PROPOSALS = {"n": 0}

ARTIFACT_ID = "era_program"
TREE_UPDATED = "tree-updated"
NO_VALID_CANDIDATES = "no-valid-candidates"
DEFAULT_OUTPUT = Path("era-agentdescent-result.json")


def _require_api_environment(provider: str) -> None:
    required = {
        "openai": ("OPENAI_API_KEY",),
        "glm": ("OPENAI_API_KEY", "OPENAI_BASE_URL"),
    }.get(provider, ())
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"missing environment variable(s): {', '.join(missing)}")


def _make_completion(args: argparse.Namespace, usage: Usage) -> Completion:
    options: Dict[str, Any] = {}
    if args.thinking != "default":
        # One-sided, exactly as in the OpenEvolve port: it reaches an
        # OpenAI-compatible endpoint and nothing else. Left on, a reasoning
        # model spends its whole token budget on hidden thinking and the reply
        # arrives empty -- which this port records as a node that scored -inf,
        # indistinguishable from a program that genuinely did not run.
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
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _finite(value: Any) -> Optional[float]:
    """`-inf` is upstream's failure sentinel and is not valid strict JSON.

    `json.dump` writes it as the bare token `-Infinity`, which Python reads back
    and most other parsers reject -- so a result file carrying one is readable
    by exactly the tool that wrote it. Failure is already recorded as
    ``valid: false``; the score field carries `null` instead.
    """
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@dataclass
class Node:
    """`futs.Node`, with the program payload this port carries alongside it."""

    index: int
    parent_index: Optional[int]
    program: Program
    score: float
    num_visits: int = 0
    #: The domain's metric, reported under its own name. A node summary that
    #: called every metric `rmse` would misreport the second task as the first.
    metric_key: str = "rmse"

    def summary(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "parent_index": self.parent_index,
            "program_id": self.program.program_id,
            "iteration": self.program.iteration,
            "change_summary": self.program.change_summary,
            "score": _finite(self.score),
            self.metric_key: self.program.metrics.get(self.metric_key),
            "num_visits": self.num_visits,
            "valid": self.program.valid,
            "error": self.program.error,
            "code_chars": len(self.program.code),
        }


@dataclass
class EraTree:
    """`futs.search`'s node list, made safe for N workers to expand at once.

    The only behavioural difference from upstream is *when* a visit is counted:
    upstream backpropagates after `execute_fn` returns, this reserves at
    selection. With one proposal in flight nothing else can observe the tree
    between those two points, so the visit counts every selection sees are
    identical -- `tests/test_era_example.py` pins that against a transcription
    of upstream's loop.
    """

    c_puct: float = 1.0
    candidate_limit: Optional[int] = None
    metric_key: str = "rmse"
    nodes: List[Node] = field(default_factory=list)
    _next_iteration: int = 1
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _policy: FlatPuct = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._policy = FlatPuct(self.c_puct)

    def seed(self, program: Program, score: float) -> Node:
        with self._lock:
            if self.nodes:
                return self.nodes[0]
            root = Node(0, None, program, score, metric_key=self.metric_key)
            self.nodes.append(root)
            return root

    def _backpropagate_locked(self, node: Node) -> None:
        """`futs.backpropagate_visit` -- the node, then every ancestor."""
        node.num_visits += 1
        if node.parent_index is not None:
            self._backpropagate_locked(self.nodes[node.parent_index])

    def select_parent(self) -> Optional[Tuple[int, Node]]:
        with self._lock:
            if not self.nodes:
                raise RuntimeError("the ERA tree has not been seeded")
            iteration = self._next_iteration
            if self.candidate_limit is not None and iteration > self.candidate_limit:
                return None
            self._next_iteration += 1
            rows = tuple(
                Candidate(
                    artifact_id=ARTIFACT_ID,
                    version=node.index,
                    score=node.score,
                    selected=node.num_visits,
                    parent=node.parent_index,
                )
                for node in self.nodes
            )
            ctx = SelectionContext(head=rows[0], candidates=rows, n_workers=1)
            chosen = self.nodes[self._policy.select(ctx, 1)[0].version]
            self._backpropagate_locked(chosen)
            return iteration, chosen

    def add_node(self, program: Program, score: float, parent_index: Optional[int]) -> Node:
        """Append an expansion. A failed program is a node too, scoring -inf."""
        with self._lock:
            if parent_index is None or not 0 <= parent_index < len(self.nodes):
                parent_index = 0
            node = Node(len(self.nodes), parent_index, program, score, num_visits=1,
                        metric_key=self.metric_key)
            self.nodes.append(node)
            return node

    def best(self) -> Node:
        with self._lock:
            if not self.nodes:
                raise RuntimeError("the ERA tree is empty")
            return max(self.nodes, key=lambda node: node.score)

    def root(self) -> Node:
        with self._lock:
            return self.nodes[0]

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            valid = [node for node in self.nodes if node.program.valid]
            depths = []
            for node in self.nodes:
                depth, cursor = 0, node
                while cursor.parent_index is not None:
                    cursor = self.nodes[cursor.parent_index]
                    depth += 1
                depths.append(depth)
            return {
                "nodes": len(self.nodes),
                "valid_nodes": len(valid),
                "max_depth": max(depths) if depths else 0,
                "root_visits": self.nodes[0].num_visits if self.nodes else 0,
                "c_puct": self.c_puct,
                "tree": [node.summary() for node in self.nodes],
            }


class EraStrategy:
    """One executable program, and the parser that turns a reply into a Diff."""

    def __init__(self, domain: Optional[Domain] = None) -> None:
        self.domain = domain

    @property
    def _seed_program(self) -> str:
        return self.domain.initial_program if self.domain else INITIAL_PROGRAM

    def initial(self) -> Dict[str, str]:
        code = self._seed_program
        return {
            "code": code,
            "program_id": program_id(code),
            "change_summary": (
                self.domain.initial_summary if self.domain
                else "LinearRegression baseline"),
        }

    def render(self, state: Dict[str, str]) -> str:
        return state.get("code", self._seed_program)

    def keys(self) -> Sequence[str]:
        return ("code", "program_id", "change_summary", "parent_id", "parent_index")

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
            parent_index = int(payload["parent_index"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        # An empty reply still becomes a diff, and so still becomes a node that
        # the gate scores `-inf`. Upstream's `generate_fn` returns
        # `Solution("")` in that case and `futs.search` appends the node
        # regardless -- dropping it here would shrink the rank denominator and
        # raise the `1/N` prior for every later iteration, which is a different
        # search. It can never reach the ledger: `-inf` is never the best node.
        pid = program_id(code)
        return Diff(
            diff_id=f"{author}:{pid}:{iteration}:{base_version}",
            target=target,
            ops={
                "code": code,
                "program_id": pid,
                "change_summary": str(payload.get("change_summary") or ""),
                "parent_id": str(payload.get("parent_id") or ""),
                "parent_index": str(parent_index),
                "iteration": str(iteration),
            },
            author=author,
        )


def make_propose(
    tree: EraTree,
    complete: Completion,
    *,
    preview: str = "",
    candidate_timeout: float = 60.0,
    prompt_for: Optional[Callable[[Program], str]] = None,
) -> Callable[[str, Task, str, float], Optional[str]]:
    """`PlaygroundGenerator.__call__`, behind the engine's actor API.

    ``prompt_for`` is the domain's mutation prompt; without one this is the
    Kaggle task's, which is what upstream ships.
    """
    write_prompt = prompt_for or (
        lambda program: mutation_prompt(
            program, preview=preview, timeout=candidate_timeout))

    def propose(rendered: str, task: Task, output: str, reward: float) -> Optional[str]:
        selection = tree.select_parent()
        if selection is None:
            return None
        iteration, parent = selection
        raw = complete(write_prompt(parent.program)).strip()
        code, summary = extract_program(raw)
        if not code:
            _EMPTY_PROPOSALS["n"] += 1
            if _EMPTY_PROPOSALS["n"] in (1, 5, 25):
                warnings.warn(
                    f"{_EMPTY_PROPOSALS['n']} mutation repl(y/ies) came back empty, "
                    f"so no program could be parsed. On a reasoning model that is "
                    f"the token budget going to hidden thinking: pass "
                    f"--thinking disabled, and raise --max-tokens -- this port "
                    f"rewrites the whole program on every expansion.",
                    RuntimeWarning, stacklevel=2)
        return json.dumps(
            {
                "code": code,
                "change_summary": summary,
                "iteration": iteration,
                "parent_index": parent.index,
                "parent_id": parent.program.program_id,
            },
            separators=(",", ":"),
        )

    return propose


def make_run(
    *,
    splits: Optional[Splits] = None,
    candidate_timeout: float = 60.0,
    max_code_length: int = 20_000,
    evaluate: Optional[Callable[[str, Sequence[int]], Tuple[bool, Dict[str, Any], str]]] = None,
) -> Callable[[str, Task], str]:
    """One shard of the held-out split is one AgentDescent rollout.

    ``evaluate`` is the domain's sandboxed scorer; without one this scores the
    Kaggle task, which is what upstream ships.
    """
    score_shards = evaluate or (
        lambda code, shards: evaluate_source(
            code, splits=splits, shards=shards,
            timeout=candidate_timeout, max_length=max_code_length))

    def run(rendered: str, task: Task) -> str:
        valid, metrics, error = score_shards(rendered, (int(task.meta["shard"]),))
        return json.dumps(
            {"valid": valid, "metrics": metrics, "error": error},
            separators=(",", ":"),
            default=str,
        )

    return run


def make_reward(score: Callable[[Dict[str, Any]], float]) -> Callable[[Task, str], float]:
    """The engine's `reward`, reading the runner payload `make_run` writes."""

    def reward(task: Task, output: str) -> float:
        try:
            payload = json.loads(output)
            if not payload.get("valid"):
                return 0.0
            return score(payload["metrics"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return 0.0

    return reward


#: The Kaggle task's reward, which is upstream's.
reward_program = make_reward(framework_score)


class EraTreeAggregator(AggregatorProtocol):
    """FUTS's expand-execute-append step as an AgentDescent merge optimizer."""

    def __init__(
        self,
        ledger: Ledger,
        verifier: Any,
        tree: EraTree,
        config: AggregatorConfig,
        staleness_policy: Any,
        *,
        splits: Optional[Splits] = None,
        candidate_timeout: float = 60.0,
        max_code_length: int = 20_000,
        domain: Optional[Domain] = None,
        artifact_id: str = ARTIFACT_ID,
    ) -> None:
        self.ledger = ledger
        self.verifier = verifier
        self.tree = tree
        self.config = config
        self.staleness_policy = staleness_policy or get_policy("guarded")
        self.splits = splits
        self.candidate_timeout = candidate_timeout
        self.max_code_length = max_code_length
        self.domain = domain
        self.artifact_id = artifact_id
        self.cards: List[EvidenceCard] = []
        self._cards_lock = threading.Lock()
        self._seeded = False
        self.meter = None

    def _held_out_shards(self) -> Tuple[int, ...]:
        shards = tuple(sorted(int(task.meta["shard"]) for task in self.verifier.held_out))
        if not shards:
            raise RuntimeError("ERA needs held-out shards to score a node")
        return shards

    def _evaluate(self, code: str) -> Tuple[bool, Dict[str, Any], str]:
        if self.domain is not None:
            return self.domain.evaluate(code, self._held_out_shards())
        return evaluate_source(
            code,
            splits=self.splits,
            shards=self._held_out_shards(),
            timeout=self.candidate_timeout,
            max_length=self.max_code_length,
        )

    def _reward(self, metrics: Dict[str, Any]) -> float:
        return (self.domain.reward if self.domain else framework_score)(metrics)

    def seed(self) -> None:
        if self._seeded:
            return
        self._seeded = True
        seed_program = self.domain.initial_program if self.domain else INITIAL_PROGRAM
        summary = (self.domain.initial_summary if self.domain
                   else "LinearRegression baseline")
        head = self.ledger.snapshot(Ledger.DEV).get(self.artifact_id)
        code = head.state.get("code", seed_program)
        valid, metrics, error = self._evaluate(code)
        if not valid:
            # Upstream prints the initial score and carries on even if it is
            # -inf. Refusing here instead: a root that cannot run means the
            # sandbox or the data is broken, and every child would inherit it.
            raise RuntimeError(f"the initial ERA program failed to run: {error}")
        self.tree.seed(
            Program(program_id(code), 0, None, code, summary, metrics, valid, error),
            float(metrics["score"]),
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
                # REBASE is safe: a survivor is fully re-executed against the
                # held-out shards before it can become a node, and its place in
                # the tree is its parent index, which the head cannot move.
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
        for card in survivors:
            ops = card.diff.ops
            code = ops.get("code", "")
            valid, metrics, error = self._evaluate(code)
            program = Program(
                program_id(code),
                int(ops.get("iteration", "0")),
                ops.get("parent_id") or None,
                code,
                ops.get("change_summary", ""),
                metrics,
                valid,
                error,
            )
            # `float(metrics["score"])` is -inf on failure, which is upstream's
            # own sentinel -- the node is appended either way.
            self.tree.add_node(program, float(metrics["score"]),
                               int(ops.get("parent_index", "0")))
            valid_candidates += int(valid)

        best = self.tree.best()
        accepted: Optional[Diff] = None
        committed_version: Optional[int] = None
        category = TREE_UPDATED
        if not survivors:
            category = MergeOutcome.ALL_STALE.value
        elif not valid_candidates:
            category = NO_VALID_CANDIDATES
        if best.program.code != head.state.get("code"):
            accepted = Diff(
                diff_id=f"tree-best:{best.program.program_id}:{head.version}",
                target=self.artifact_id,
                ops={
                    "code": best.program.code,
                    "program_id": best.program.program_id,
                    "change_summary": best.program.change_summary,
                    "parent_id": best.program.parent_id or "",
                    "parent_index": str(best.parent_index or 0),
                },
                author="era-tree",
            )
            try:
                _, committed_version = self.ledger.commit(
                    head.apply(accepted),
                    base_vv,
                    branch=Ledger.DEV,
                    message="era: commit best node in the FUTS tree",
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
                self._reward(best.program.metrics),
                committed_version,
                (
                    f"valid={valid_candidates}/{len(survivors)} "
                    f"nodes={len(self.tree.nodes)} "
                    f"best_{self.tree.metric_key}="
                    f"{best.program.metrics.get(self.tree.metric_key)}"
                ),
                category,
            )
        ]


@dataclass
class EraRun:
    mode: str
    result: EvolutionResult
    tree: EraTree
    splits: Optional[Splits]
    wall_seconds: float
    baseline_test_metrics: Dict[str, Any] = field(default_factory=dict)
    best_test_metrics: Dict[str, Any] = field(default_factory=dict)
    domain: Optional[Domain] = None

    def _data_summary(self) -> Dict[str, Any]:
        if self.domain is not None:
            return dict(self.domain.data_summary)
        return {
            "train_rows": self.splits.train_rows,
            "scoring_shards": len(self.splits.shard_paths),
            "rows_per_shard": self.splits.rows(0),
        }

    def summary(self, quality_target: Optional[float] = None) -> Dict[str, Any]:
        root = self.tree.root()
        best = self.tree.best()
        reward = self.domain.reward if self.domain else framework_score
        metric = self.tree.metric_key
        payload: Dict[str, Any] = {
            "mode": self.mode,
            "wall_seconds": self.wall_seconds,
            "baseline": root.summary(),
            "best": best.summary(),
            "data": self._data_summary(),
            "framework": {
                "baseline_reward": reward(root.program.metrics),
                "final_reward": self.result.final_reward,
                "quality_gain": self.result.final_reward - reward(root.program.metrics),
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
            "tree": self.tree.summary(),
            "test": {
                "baseline": {**self.baseline_test_metrics,
                             "score": _finite(self.baseline_test_metrics.get("score"))},
                "best": {**self.best_test_metrics,
                         "score": _finite(self.best_test_metrics.get("score"))},
                f"{metric}_gain": (
                    self.domain.gain(self.baseline_test_metrics.get(metric),
                                     self.best_test_metrics.get(metric))
                    if self.domain is not None else
                    float(self.baseline_test_metrics.get(metric) or 0.0)
                    - float(self.best_test_metrics.get(metric) or 0.0)
                ),
            },
        }
        if quality_target is not None:
            payload["quality_target"] = quality_target
            payload["target_reached"] = self.result.final_reward >= quality_target
            payload["time_to_quality_s"] = self.result.time_to_quality(quality_target)
            payload["rollouts_to_quality"] = self.result.cost_to_quality(quality_target)
        return payload


def s3e1_domain(
    splits: Splits,
    *,
    candidate_timeout: float = 60.0,
    max_code_length: int = 20_000,
    shards: int = 8,
    test_shards: int = 4,
) -> Domain:
    """Upstream's own task, expressed as a :class:`Domain`.

    Every field here was a module-level constant or a direct call before the
    second task arrived; collecting them changes nothing about what runs.
    """
    preview = data_preview(splits)
    return Domain(
        name="Kaggle Playground Series S3E1 (synthetic California housing), RMSE",
        entrypoint="train_and_predict",
        metric_key="rmse",
        metric_better="lower",
        initial_program=INITIAL_PROGRAM,
        initial_summary="LinearRegression baseline",
        evaluate=lambda code, shard_ids: evaluate_source(
            code, splits=splits, shards=shard_ids,
            timeout=candidate_timeout, max_length=max_code_length),
        reward=framework_score,
        prompt=lambda program: mutation_prompt(
            program, preview=preview, timeout=candidate_timeout),
        task_prompt=lambda index: (
            f"Score the regression program on held-out shard {index}."),
        test_shards=tuple(range(shards, shards + test_shards)),
        data_summary={
            "train_rows": splits.train_rows,
            "scoring_shards": len(splits.shard_paths),
            "rows_per_shard": splits.rows(0),
        },
    )


def build_tasks(count: int, seed: int = 0,
                prompt_for: Optional[Callable[[int], str]] = None) -> List[Task]:
    """One task per scoring shard. `seed` is accepted for the shared CLI only."""
    describe = prompt_for or (
        lambda index: f"Score the regression program on held-out shard {index}.")
    return [
        Task(id=f"shard-{index}", prompt=describe(index), meta={"shard": index})
        for index in range(count)
    ]


def run_agentdescent_era(
    complete: Completion,
    *,
    mode: str,
    iterations: int = 6,
    workers: int = 3,
    shards: int = 8,
    test_shards: int = 4,
    train_rows: int = 0,
    held_out_frac: float = 0.5,
    c_puct: float = 1.0,
    candidate_timeout: float = 60.0,
    max_code_length: int = 20_000,
    async_ratio: int = 1,
    max_seconds: float = 1800.0,
    shutdown_grace: float = 120.0,
    seed: int = 0,
    usage: Optional[Usage] = None,
    staleness: str = "guarded",
    splits: Optional[Splits] = None,
    domain: Optional[Domain] = None,
    eval_concurrency: Optional[int] = None,
    verbose: bool = False,
) -> EraRun:
    """Run one fixed-expansion-budget serial, sync, or async experiment.

    ``domain=None`` is upstream's Kaggle task; anything else runs the identical
    search over that domain's programs, evaluator and prompt.
    """
    if mode not in ("serial", "sync", "async"):
        raise ValueError("mode must be serial, sync, or async")
    if iterations < 1 or workers < 1 or iterations % workers:
        raise ValueError("iterations must be positive and divisible by workers")
    if not 0.0 < held_out_frac < 1.0:
        raise ValueError("held_out_frac must be in (0, 1)")

    if domain is None:
        splits = splits or prepare_splits(
            shards=shards, test_shards=test_shards, train_rows=train_rows)
        domain = s3e1_domain(splits, candidate_timeout=candidate_timeout,
                             max_code_length=max_code_length,
                             shards=shards, test_shards=test_shards)
    tasks = build_tasks(shards, seed, domain.task_prompt)
    tree = EraTree(c_puct=c_puct, candidate_limit=iterations,
                   metric_key=domain.metric_key)
    strategy = EraStrategy(domain)
    run = make_run(evaluate=domain.evaluate)
    propose = make_propose(tree, complete, prompt_for=domain.prompt)

    def factory(ledger, verifier, audit, config, policy):
        aggregator = EraTreeAggregator(
            ledger,
            verifier,
            tree,
            config,
            policy,
            splits=splits,
            candidate_timeout=candidate_timeout,
            max_code_length=max_code_length,
            domain=domain,
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
        # This port's own rule is "as many as there are workers", because a
        # held-out evaluation here is a sandboxed process rather than an API
        # call. `--eval-concurrency` overrides it when given; it used to be
        # parsed, recorded in the result file and then ignored, which made the
        # recorded configuration disagree with the run.
        "eval_concurrency": eval_concurrency if eval_concurrency else workers,
        "held_out_frac": held_out_frac,
        "solved_threshold": 1.0,
        "usage": usage,
        "aggregator_factory": factory,
        "verbose": verbose,
        # A discarded card is a whole trained-and-scored program. The tree is
        # append-only and a node's place in it is its parent index, so a late
        # arrival is still a legitimate expansion of the parent it was drawn
        # from -- there is nothing for it to be stale against.
        "staleness_policy": get_policy(staleness),
    }
    reward = make_reward(domain.reward)
    if mode == "async":
        result = async_evolve(
            tasks,
            reward,
            async_ratio=async_ratio,
            max_seconds=max_seconds,
            max_iters=iterations,
            shutdown_grace=shutdown_grace,
            **common,
        )
    else:
        result = evolve(
            tasks,
            reward,
            rounds=iterations // workers,
            max_concurrency=1 if mode == "serial" else workers,
            **common,
        )
    if verbose:
        report_engine(result)
    wall_seconds = time.monotonic() - started

    _, baseline_test, _ = domain.evaluate(tree.root().program.code, domain.test_shards)
    _, best_test, _ = domain.evaluate(tree.best().program.code, domain.test_shards)
    return EraRun(mode, result, tree, splits, wall_seconds, baseline_test, best_test,
                  domain)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_standard_args(parser, model_default="glm-5.2", max_seconds_default=1800.0,
                      eval_concurrency_default=None)
    parser.set_defaults(provider="openai", async_ratio=1)
    parser.add_argument("--staleness", default="guarded",
                        choices=["guarded", "reflective", "full"],
                        help=("what to do with an expansion proposed against a "
                              "head the merger has since moved. The tree is "
                              "append-only, so `full` is the honest default for "
                              "a comparison and `guarded` the conservative one"))
    parser.add_argument("--iterations", type=int, default=6,
                        help="FUTS expansions in total (upstream's num_iterations)")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--shards", type=int, default=8,
                        help="scoring shards cut from upstream's 20%% tail")
    parser.add_argument("--test-shards", type=int, default=4,
                        help="further shards the search never sees")
    parser.add_argument("--train-rows", type=int, default=0,
                        help=("cap the 80%% training file; 0 is upstream's full "
                              "split. A speed knob, and a difficulty one -- a "
                              "capped run is not comparable to an uncapped one"))
    parser.add_argument("--held-out-frac", type=float, default=0.5)
    parser.add_argument("--c-puct", type=float, default=1.0,
                        help="upstream's exploration constant (futs.search default)")
    parser.add_argument("--candidate-timeout", type=float, default=60.0,
                        help="upstream's Sandbox(timeout_seconds=60)")
    parser.add_argument("--max-code-length", type=int, default=20_000)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--thinking", choices=("disabled", "enabled", "default"),
                        default="disabled")
    parser.add_argument("--api-timeout", type=float, default=180.0)
    parser.add_argument(
        "--reply-attempts", type=int, default=4,
        help=("redraws allowed when a reply arrives damaged -- unparsable, or "
              "holding characters Python source cannot hold. A *badly written* "
              "program is never redrawn; it becomes a node scoring -inf, as "
              "upstream requires. 1 disables the guard (see "
              "examples.era._era_support.reply_is_intact)"))
    parser.add_argument("--shutdown-grace", type=float, default=120.0)
    parser.add_argument("--quality-target", type=float)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.workers = worker_count(args, args.workers)
    # `iterations` is the total expansion budget and `rounds = iterations //
    # workers`, so work is fixed and workers only change how it is divided.
    # `--budget-rollouts` is therefore the same quantity under the shared name.
    if args.budget_rollouts:
        args.iterations = args.budget_rollouts
    if getattr(args, "reflective_merge", False):
        raise SystemExit(
            "--reflective-merge is not supported by the ERA port: a candidate is "
            "a whole program rather than a delta, and fusing a round's "
            "expansions into one would delete tree nodes that FUTS selects from. "
            "Use --staleness {guarded,reflective,full}.")
    mode = "async" if args.asynchronous else ("serial" if args.serial else "sync")
    print("Algorithm: ERA Flat UCB tree search (FUTS) on AgentDescent")
    print("Evaluator: Kaggle Playground S3E1 RMSE, "
          f"{sandbox_backend() or 'NO SANDBOX -- this run will fail'} isolated")
    print(
        f"Plan     : mode={mode}, model={args.model}, iterations={args.iterations}, "
        f"workers={args.workers}, c_puct={args.c_puct}, temperature={args.temperature}"
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
    damage: Dict[str, int] = {}
    complete = with_intact_replies(
        _make_completion(args, model_usage),
        attempts=max(1, args.reply_attempts), counter=damage)
    splits = prepare_splits(
        shards=args.shards, test_shards=args.test_shards, train_rows=args.train_rows)
    print(f"Data     : train={splits.train_rows} rows, "
          f"{args.shards}+{args.test_shards} shards of {splits.rows(0)} rows")
    run = run_agentdescent_era(
        complete,
        mode=mode,
        iterations=args.iterations,
        workers=args.workers,
        shards=args.shards,
        test_shards=args.test_shards,
        train_rows=args.train_rows,
        held_out_frac=args.held_out_frac,
        c_puct=args.c_puct,
        candidate_timeout=args.candidate_timeout,
        max_code_length=args.max_code_length,
        async_ratio=args.async_ratio,
        staleness=args.staleness,
        max_seconds=args.max_seconds,
        shutdown_grace=args.shutdown_grace,
        seed=args.seed,
        usage=actor_usage,
        eval_concurrency=args.eval_concurrency,
        splits=splits,
        verbose=True,
    )
    payload = {
        "experiment": "ERA on AgentDescent",
        "status": "completed" if run.result.error is None else "partial",
        "completed_at": _utc_now(),
        "upstream_commit": UPSTREAM_COMMIT,
        "config": {
            key: value for key, value in vars(args).items()
            if key not in ("output", "yes")
        },
        "observation": run.summary(args.quality_target),
        "model_usage": _usage_dict(model_usage),
        "reply_damage": {
            "drawn": damage.get("drawn", 0),
            "damaged": damage.get("damaged", 0),
            "attempts_allowed": max(1, args.reply_attempts),
        },
    }
    _write_json(args.output, payload)
    best_path = args.output.with_name(f"{args.output.stem}-best.py")
    best_path.write_text(run.tree.best().program.code.rstrip() + "\n", encoding="utf-8")
    baseline_rmse = run.baseline_test_metrics.get("rmse")
    best_rmse = run.best_test_metrics.get("rmse")
    print(
        f"completed: test RMSE {baseline_rmse} -> {best_rmse}, "
        f"nodes={len(run.tree.nodes)}, wall={run.wall_seconds:.2f}s, "
        f"model_calls={model_usage.calls}, output={args.output}"
    )
    return 0 if run.result.error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
