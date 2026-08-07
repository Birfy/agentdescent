"""Harness self-evolution, faithful port: **Darwin Godel Machine (DGM)**.

Paper : "Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents",
        Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, Jeff Clune, 2025
        (arXiv:2505.22954).
Repo  : https://github.com/jennyzzt/dgm
Dataset: **SWE-bench Verified** (real coding-agent benchmark; instance metadata
        loaded from HF `princeton-nlp/SWE-bench_Verified`).

DGM is the archetypal *harness* self-evolution: a coding agent that **edits its
own codebase** (tools, prompts, workflow), keeping every variant it discovers in
an open-ended **archive**. This port reproduces the DGM_outer.py loop faithfully
(traced from the repo):

  * **Keep-all archive** of agents (stepping stones retained, not just the best).
  * **Parent selection** = `score_child_prop` (`DGM_outer.py:choose_selfimproves`):
    ``p_i proportional to sigmoid(10*(score-0.5)) * 1/(1+children_i)`` -- favour
    high performers, discount already-explored parents (open-endedness). Reused
    exactly (`dgm_parent_weights`) and unit-tested.
  * **Self-modification**: a parent inspects its own evaluation logs (which SWE
    instances it failed), diagnoses a weakness, and proposes "the next feature to
    implement" on its own harness -> a child agent.
  * **Staged empirical validation** (`swe_bench/subsets/`): small=10, escalate to
    medium=50 iff score > `test_more_threshold=0.4`. That is the whole ladder in
    `DGM_outer.py`'s self-improve loop; `big.json` (140) belongs to the separate
    full-evaluation path gated by the archive-relative `full_eval_threshold`,
    which this surrogate does not model.

It runs **through `evolve()`** like ACE/GEPA: a `HarnessStrategy` turns a proposed
capability into a `Diff`, and a custom `aggregator_factory`
(`DGMArchiveAggregator`) is the keep-all archive + parent selection. The agent
artifact is a **harness** -> AgentDescent's **L1** governance layer
(`blast_radius=0.6`, printed via `classify`): harness changes are high-blast-
radius and, in the full system, oracle-gated.

**Honesty boundary (you accepted a mechanism-only harness example).** DGM's real
objective runs each candidate patch inside the **SWE-bench Docker harness**
(per-task containers, real test suites, ~1.5 h timeouts, arbitrary code
execution). That is out of scope for a dependency-free example, so the objective
here is a **transparent surrogate**: each real SWE instance is assigned a latent
required-capability set, and an agent "resolves" it when its self-added
capabilities cover that set. The DGM *algorithm* (archive, selection, staged
escalation, keep-all) is therefore fully runnable and testable offline; the
*scores* are simulated, not SWE-bench results. Pass a real `evaluate_fn` to
`run_dgm` to plug in the actual Docker harness.

    python -m examples.dgm.dgm_self_improve                 # runs offline (surrogate objective)
    python -m examples.dgm.dgm_self_improve --dry-run       # plan only, zero network
    python -m examples.dgm.dgm_self_improve --generations 12 --archive keep_all
    python -m examples.dgm.dgm_self_improve --model claude-haiku-4-5   # LLM proposes modifications
"""

from __future__ import annotations

import threading

import argparse
import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from agentdescent.aggregator import AggregatorProtocol, MergeReport
from agentdescent.dataloader import Dataset, hf_rows, split_dataset
from agentdescent.evolvable import Diff, EvidenceCard
from agentdescent.evolution import EvolvingArtifact, Task, evolve
from agentdescent.governance import classify
from agentdescent.ledger import CASConflict, Ledger
from examples._common import (add_standard_args, completion_for, confirm,
                              worker_count,
                              budget_kwargs)

SWEBENCH = ("princeton-nlp/SWE-bench_Verified", "test", "default")   # (dataset, split, config)

# DGM staged-eval subset sizes (swe_bench/subsets/{small,medium,big}.json).
STAGE_SMALL, STAGE_MEDIUM, STAGE_BIG = 10, 50, 140
TEST_MORE_THRESHOLD = 0.4

# A pool of coding-agent harness features a DGM variant might add to itself.
CAPABILITY_POOL = [
    "regression-test-runner", "multi-file-edit", "lint-before-submit",
    "context-retrieval", "patch-validation", "error-log-analysis",
    "dependency-resolver", "retry-on-failure", "test-case-generation",
    "diff-minimization", "stack-trace-parser", "type-checker",
]


# ===========================================================================
# The DGM parent-selection rule (DGM_outer.py:choose_selfimproves)
# ===========================================================================


def dgm_parent_weights(scores: List[float], children: List[int]) -> List[float]:
    """``p_i proportional to sigmoid(10*(score-0.5)) * 1/(1+children_i)``."""
    raw = []
    for s, c in zip(scores, children):
        sig = 1.0 / (1.0 + math.exp(-10.0 * (s - 0.5)))
        nov = 1.0 / (1.0 + c)
        raw.append(sig * nov)
    total = sum(raw) or 1.0
    return [r / total for r in raw]


def choose_selfimproves(archive: List["Agent"], k: int,
                        rng: random.Random) -> List[int]:
    """Sample k parent indices ~ the DGM weights (with replacement, as in repo)."""
    scores = [a.score for a in archive]
    children = [a.children for a in archive]
    weights = dgm_parent_weights(scores, children)
    return rng.choices(range(len(archive)), weights=weights, k=k)


# ===========================================================================
# The self-improving coding agent (its harness = its editable "codebase")
# ===========================================================================


@dataclass
class Agent:
    """A DGM variant: a harness with a set of self-added capabilities + lineage."""

    capabilities: Tuple[str, ...]
    score: float = 0.0
    parent: Optional[int] = None
    children: int = 0
    generation: int = 0

    def key(self) -> str:
        return "|".join(sorted(self.capabilities))


def initial_agent() -> Agent:
    """DGM starts from one seed agent (a minimal coding harness)."""
    return Agent(capabilities=("context-retrieval",))


# ===========================================================================
# Self-modification: analyse own failures -> propose the "next feature"
# ===========================================================================


def _failed_capabilities(agent: Agent, instances: List[dict]) -> List[str]:
    """Capabilities most often required by the instances this agent failed."""
    from collections import Counter
    have = set(agent.capabilities)
    missing: Counter = Counter()
    for inst in instances:
        req = required_capabilities(inst["instance_id"])
        if not req <= have:
            missing.update(req - have)
    return [cap for cap, _ in missing.most_common()]


def propose_modification(agent: Agent, instances: List[dict], rng: random.Random,
                         complete: Optional[Callable[[str], str]] = None
                         ) -> Optional[str]:
    """Diagnose a weakness from eval logs and propose ONE new capability.

    Deterministic by default (offline); LLM-driven if `complete` is given."""
    candidates = _failed_capabilities(agent, instances) or \
        [c for c in CAPABILITY_POOL if c not in agent.capabilities]
    candidates = [c for c in candidates if c not in agent.capabilities]
    if not candidates:
        return None
    if complete is not None:
        prompt = (
            "You are a self-improving SWE coding agent. Your harness has these "
            f"capabilities: {list(agent.capabilities)}. On failed tasks these "
            f"capabilities were most often missing: {candidates[:6]}. Propose the "
            "SINGLE next capability to implement. Reply with exactly one item from "
            f"this list: {candidates[:6]}.")
        reply = complete(prompt).strip().lower()
        for c in candidates:
            if c in reply:
                return c
    return candidates[0]


# ===========================================================================
# Objective: SURROGATE for the SWE-bench Docker harness (clearly labelled)
# ===========================================================================


def required_capabilities(instance_id: str) -> set:
    """A deterministic latent capability set per real SWE instance (surrogate).

    Stands in for actually running the task's test suite in Docker; keeps the
    objective reproducible and offline while depending on the real instance id."""
    h = int(hashlib.sha1(instance_id.encode()).hexdigest(), 16)
    n = 1 + (h % 3)                       # each task "needs" 1-3 capabilities
    picks = []
    for i in range(n):
        picks.append(CAPABILITY_POOL[(h >> (i * 5)) % len(CAPABILITY_POOL)])
    return set(picks)


def surrogate_resolved(agent: Agent, instance_id: str) -> bool:
    return required_capabilities(instance_id) <= set(agent.capabilities)


def make_surrogate_evaluator() -> Callable[[Agent, List[dict]], float]:
    def evaluate_fn(agent: Agent, instances: List[dict]) -> float:
        if not instances:
            return 0.0
        return sum(surrogate_resolved(agent, i["instance_id"]) for i in instances) \
            / len(instances)
    return evaluate_fn


def staged_evaluate(agent: Agent, evaluate_fn: Callable[[Agent, List[dict]], float],
                    small: List[dict], medium: List[dict],
                    big: Optional[List[dict]] = None) -> float:
    """Shallow-then-deep evaluation, as `DGM_outer.py` runs it.

    Upstream's self-improve loop escalates **small -> medium and stops**::

        swe_issues_sm  = load_json_file("./swe_bench/subsets/small.json")    # 10
        swe_issues_med = load_json_file("./swe_bench/subsets/medium.json")   # 50
        test_more_threshold = 0.4
        ...
        self_improve(test_task_list=swe_issues_sm,
                     test_more_threshold=test_more_threshold,
                     test_task_list_more=swe_issues_med,
                     full_eval_threshold=get_full_eval_threshold(output_dir, archive))

    ``big.json`` (140) is **not** a third rung on this ladder: it belongs to the
    separate full-evaluation path, gated by ``full_eval_threshold``, which is
    *archive-relative* rather than the fixed 0.4. This used to run it as a third
    rung on the same 0.4, which changed what ``agent.score`` means -- a high
    scorer's score became a 140-instance number while a low scorer's stayed a
    10-instance one, and both then fed the same ``dgm_parent_weights`` sigmoid. The
    surrogate objective here does not model an archive-relative full eval, so the
    ladder stops where upstream's loop stops; ``big`` is accepted and ignored.
    """
    score = evaluate_fn(agent, small)
    if score > TEST_MORE_THRESHOLD and medium:
        score = evaluate_fn(agent, medium)
    return score


# ===========================================================================
# Running DGM THROUGH evolve() -- harness strategy + keep-all archive optimizer
# ===========================================================================
#
# Like ACE/GEPA, DGM plugs into `evolve()`: a `Strategy` turns a proposed
# capability into a `Diff` on the coding-agent harness, and a custom
# `aggregator_factory` is the keep-all archive with DGM's parent selection --
# it sets the dev head to the sampled parent, so `evolve()`'s next round mutates
# *that* open-endedly (not the greedy best). The objective (`run`) is the
# transparent surrogate; swap in a real Docker harness via `evaluate_fn`.


def _caps_of(rendered: str) -> set:
    return {c for c in rendered.split(",") if c}


class HarnessStrategy:
    """The artifact is the harness's capability set; a proposal adds one capability."""

    def initial(self):
        return {"capabilities": ",".join(sorted(initial_agent().capabilities))}

    def render(self, state):
        return state.get("capabilities", "")

    def to_diff(self, state, proposal, author, base_version, target):
        cap = proposal.strip()
        current = _caps_of(state.get("capabilities", ""))
        if not cap or cap in current:
            return None
        new = ",".join(sorted(current | {cap}))
        return Diff(diff_id=f"{author}:{cap}:{base_version}", target=target,
                    ops={"capabilities": new}, author=author)


@dataclass
class DGMContext:
    rng: random.Random
    evaluate_fn: Callable[[Agent, List[dict]], float]
    archive_mode: str = "keep_all"
    archive: List[Agent] = field(default_factory=list)
    seen: set = field(default_factory=set)
    seed_score: float = 0.0
    best_score: float = 0.0


class DGMArchiveAggregator(AggregatorProtocol):
    """DGM's optimizer: keep-all archive + staged eval + sigmoid×novelty selection."""

    def __init__(self, ledger: Ledger, verifier, ctx: DGMContext,
                 artifact_id: str = "coding_agent"):
        self.ledger = ledger
        self.verifier = verifier
        self.ctx = ctx
        self.aid = artifact_id
        self.cards: List[EvidenceCard] = []
        self._lock = threading.Lock()   # ingest: workers; step: one thread
        self.head_index = 0
        self._seeded = False

    def _instances(self) -> List[dict]:
        return [{"instance_id": t.meta["instance_id"]} for t in self.verifier.held_out]

    def _staged(self, caps: set) -> float:
        insts = self._instances()
        return staged_evaluate(Agent(tuple(sorted(caps))), self.ctx.evaluate_fn,
                               insts[:STAGE_SMALL], insts[:STAGE_MEDIUM], insts[:STAGE_BIG])

    def ingest(self, card: EvidenceCard) -> None:
        # ingest runs on worker threads, step on one: see AggregatorProtocol.
        with self._lock:
            self.cards.append(card)

    def step(self) -> List[MergeReport]:
        snap = self.ledger.snapshot(Ledger.DEV)
        head = snap.get(self.aid)
        base_vv = {self.aid: snap.version.get(self.aid, 0)}
        if not self._seeded:
            seed_caps = _caps_of(head.state.get("capabilities", ""))
            seed_agent = Agent(tuple(sorted(seed_caps)), parent=None, generation=0)
            seed_agent.score = self._staged(seed_caps)
            self.ctx.archive.append(seed_agent)
            self.ctx.seen.add(seed_agent.key())
            self.ctx.seed_score = self.ctx.best_score = seed_agent.score
            self.head_index = 0
            self._seeded = True

        parent_idx = self.head_index
        parent = self.ctx.archive[parent_idx]
        with self._lock:
            cards, self.cards = self.cards, []
        for card in cards:                                 # each card = a self-modification
            child_caps = _caps_of(card.diff.ops["capabilities"])
            child = Agent(tuple(sorted(child_caps)), parent=parent_idx,
                          generation=parent.generation + 1)
            if child.key() in self.ctx.seen:
                continue
            child.score = self._staged(child_caps)
            if self.ctx.archive_mode == "keep_better" and child.score <= parent.score:
                continue
            self.ctx.archive.append(child)                 # keep-all: stepping stones stay
            self.ctx.seen.add(child.key())
            self.ctx.best_score = max(self.ctx.best_score, child.score)
        parent.children += 1                               # this parent was explored

        # DGM parent selection: sample the next head ~ sigmoid(perf) x 1/(1+children).
        self.head_index = choose_selfimproves(self.ctx.archive, 1, self.ctx.rng)[0]
        target = ",".join(sorted(self.ctx.archive[self.head_index].capabilities))
        committed = None
        if target != head.state.get("capabilities"):
            try:
                _, committed = self.ledger.commit(
                    head.apply(Diff(diff_id="dgm-select", target=self.aid,
                                    ops={"capabilities": target}, author="dgm")),
                    base_vv, branch=Ledger.DEV, message="dgm: select parent")
            except CASConflict:
                committed = None
        return [MergeReport(self.aid, None, False, len(cards), len(cards), 0, 0,
                            self.ctx.best_score, committed,
                            f"archive={len(self.ctx.archive)} best={self.ctx.best_score:.3f}")]


@dataclass
class DGMResult:
    archive: List[Agent]
    best: Agent
    seed_score: float
    best_score: float


def run_dgm(instances: List[dict], generations: int = 12,
            selfimprove_size: int = 2, archive_mode: str = "keep_all",
            evaluate_fn: Optional[Callable[[Agent, List[dict]], float]] = None,
            complete: Optional[Callable[[str], str]] = None, seed: int = 0,
            asynchronous: bool = False, async_ratio: int = 3, max_seconds: float = 20.0,
            max_rollouts: Optional[int] = None,
            verbose: bool = False) -> DGMResult:
    """Drive DGM through `evolve()` (SWE instances split into trigger/held-out)."""
    evaluate_fn = evaluate_fn or make_surrogate_evaluator()
    tasks = [Task(id=inst["instance_id"], prompt=inst["instance_id"],
                  meta={"instance_id": inst["instance_id"]}) for inst in instances]
    ctx = DGMContext(rng=random.Random(seed), evaluate_fn=evaluate_fn,
                     archive_mode=archive_mode)

    def run(rendered, task):
        return ("resolved" if required_capabilities(task.meta["instance_id"]) <= _caps_of(rendered)
                else "unresolved")

    def reward(task, output):
        return 1.0 if output == "resolved" else 0.0

    def propose(rendered, task, output, score):
        agent = Agent(tuple(sorted(_caps_of(rendered))))
        return propose_modification(agent, [{"instance_id": task.meta["instance_id"]}],
                                    ctx.rng, complete)

    def factory(ledger, verifier, audit, config, policy):
        return DGMArchiveAggregator(ledger, verifier, ctx, artifact_id="coding_agent")

    evolve(tasks, reward, run=run, propose=propose, strategy=HarnessStrategy(),
           blast_radius=0.6, artifact_id="coding_agent", rounds=generations,
           n_workers=selfimprove_size,
           max_concurrency=1 if asynchronous else selfimprove_size,
           asynchronous=asynchronous, async_ratio=async_ratio,
           max_seconds=max_seconds if asynchronous else None,
           held_out_frac=0.5, aggregator_factory=factory, verbose=verbose,
           max_rollouts=max_rollouts)
    best = max(ctx.archive, key=lambda a: a.score)
    return DGMResult(ctx.archive, best, ctx.seed_score, best.score)


# ===========================================================================
# Dataset: SWE-bench Verified instance metadata (real ids for the subsets)
# ===========================================================================


def download_swebench(limit: int) -> List[dict]:
    dataset, split, config = SWEBENCH
    return [{"instance_id": r["instance_id"], "repo": r["repo"]}
            for r in hf_rows(dataset, split, config=config, limit=limit)]


def load_dataset(limit: int, seed: int = 0, ratios=(0.5, 0.25, 0.25)) -> Dataset:
    """SWE-bench Verified instances split into train / val (staged eval) / test."""
    return split_dataset(download_swebench(limit), ratios=ratios, seed=seed,
                         name="SWE-bench Verified")


# ===========================================================================
# main
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_standard_args(
        p, model_default=None, max_seconds_default=15.0,
        model_help="optional: let an LLM propose self-modifications (else deterministic)")
    p.add_argument("--generations", type=int, default=12)
    p.add_argument("--selfimprove-size", type=int, default=2)
    p.add_argument("--archive", default="keep_all", choices=["keep_all", "keep_better"])
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    # --serial collapses this to the upstream algorithm's own semantics:
    # one worker, nothing to merge. Applied to args so the printed plan,
    # the cost estimate and the run cannot disagree about what ran.
    args.selfimprove_size = worker_count(args, args.selfimprove_size)

    print("Algorithm: Darwin Godel Machine (DGM) -- harness self-evolution")
    print("Dataset  : SWE-bench Verified (real instance ids; surrogate objective)")
    print(f"Plan     : model={args.model or 'deterministic'}, "
          f"generations={args.generations}, archive={args.archive}")
    if args.asynchronous:
        print(f"Async    : {args.selfimprove_size} agents, barrier-free (async_ratio="
              f"{args.async_ratio}, max {args.max_seconds:.0f}s); "
              "archive rebases/discards stale")
    else:
        print(f"Parallel : {args.selfimprove_size} agents self-modify concurrently each "
              "generation (synchronous DP; the archive merge is the barrier)")
    print("\nObjective: SURROGATE (capability-cover) -- real DGM runs SWE-bench in "
          "Docker.\n           The archive + selection + staged escalation are faithful.")
    if args.dry_run:
        print("Data     : deferred (dry-run performs no network access)")
        print("\n[dry-run] plan only; no dataset or model API was accessed.")
        return

    ds = load_dataset(STAGE_BIG, seed=args.seed)
    art = EvolvingArtifact("coding_agent", blast_radius=0.6)
    ntr, nva, nte = ds.sizes()
    print(f"Governance: coding-agent harness blast_radius={art.blast_radius} "
          f"-> {classify(art).name} (harness changes are high-blast-radius)")
    print(f"Loaded   : {len(ds)} SWE-bench Verified instances; "
          f"{ntr} train / {nva} val (staged eval) / {nte} test")
    print(f"Example  : {ds.train[0]['instance_id']} ({ds.train[0]['repo']})")
    complete = None
    if args.model:
        if not confirm(args):
            return
        complete = completion_for(args)
        try:
            complete("Reply with the single word: ok")
        except Exception as e:  # noqa: BLE001
            print(f"\nLLM unreachable ({type(e).__name__}: {e}); using deterministic proposals.")
            complete = None

    print(f"\nRunning DGM ({args.archive}, selfimprove_size={args.selfimprove_size}, "
          f"L1 harness)...\n")
    result = run_dgm(ds.trainval, generations=args.generations,
                     selfimprove_size=args.selfimprove_size, archive_mode=args.archive,
                     **budget_kwargs(args),
                     complete=complete, seed=args.seed, asynchronous=args.asynchronous,
                     async_ratio=args.async_ratio, max_seconds=args.max_seconds, verbose=True)

    test_score = make_surrogate_evaluator()(result.best, ds.test)
    print("\n=== best self-improved agent ===")
    print(f"capabilities: {list(result.best.capabilities)}")
    print(f"lineage     : generation {result.best.generation}")
    print(f"\nval resolve-rate : {result.seed_score:.3f} -> {result.best_score:.3f}")
    print(f"test resolve-rate: {test_score:.3f}  (held out, never seen by selection)")
    print(f"archive size (keep-all): {len(result.archive)} agents")


if __name__ == "__main__":
    main()
