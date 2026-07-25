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
    medium=50 iff score > `test_more_threshold=0.4`, then big=140 for top agents.

The agent artifact is a **harness** -> Concordia's **L1** governance layer
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

    python -m examples.dgm_self_improve                 # runs offline (surrogate objective)
    python -m examples.dgm_self_improve --generations 12 --archive keep_all
    python -m examples.dgm_self_improve --model claude-haiku-4-5   # LLM proposes modifications
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from concordia.agents import claude, openai_compatible
from concordia.dataloader import hf_rows
from concordia.evolvable import Contract, Evolvable
from concordia.governance import classify

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
                    small: List[dict], medium: List[dict], big: List[dict]
                    ) -> float:
    """Shallow-then-deep evaluation (DGM: small always; escalate past 0.4)."""
    score = evaluate_fn(agent, small)
    if score > TEST_MORE_THRESHOLD and medium:
        score = evaluate_fn(agent, medium)
        if score > TEST_MORE_THRESHOLD and big:
            score = evaluate_fn(agent, big)
    return score


# ===========================================================================
# Governance artifact (the coding-agent harness = L1)
# ===========================================================================


@dataclass
class HarnessArtifact(Evolvable):
    id: str = "coding_agent"
    blast_radius: float = 0.6           # a self-modifying harness -> L1
    version: int = 1
    contract: Contract = field(default_factory=lambda: Contract("task", "patch", 1))

    def render(self) -> str: return ""
    def diff(self, other): ...
    def apply(self, diff): ...
    def cheap_eval(self, evidence): return 0.0
    def full_eval(self, task_set): return {}


# ===========================================================================
# The DGM outer loop (DGM_outer.py:main)
# ===========================================================================


@dataclass
class DGMResult:
    archive: List[Agent]
    best: Agent
    seed_score: float
    best_score: float


def run_dgm(instances: List[dict], generations: int = 12,
            selfimprove_size: int = 2, archive_mode: str = "keep_all",
            evaluate_fn: Optional[Callable[[Agent, List[dict]], float]] = None,
            complete: Optional[Callable[[str], str]] = None,
            seed: int = 0, verbose: bool = False) -> DGMResult:
    rng = random.Random(seed)
    evaluate_fn = evaluate_fn or make_surrogate_evaluator()
    small = instances[:STAGE_SMALL]
    medium = instances[:STAGE_MEDIUM]
    big = instances[:STAGE_BIG]

    seed_agent = initial_agent()
    seed_agent.score = staged_evaluate(seed_agent, evaluate_fn, small, medium, big)
    archive: List[Agent] = [seed_agent]
    seen = {seed_agent.key()}
    if verbose:
        print(f"  seed agent {list(seed_agent.capabilities)} score={seed_agent.score:.3f}")

    for g in range(generations):
        parents = choose_selfimproves(archive, selfimprove_size, rng)
        for pi in parents:
            parent = archive[pi]
            parent.children += 1                       # bookkeeping on the parent
            new_cap = propose_modification(parent, small, rng, complete)
            if new_cap is None:
                continue
            child = Agent(capabilities=tuple(sorted(set(parent.capabilities) | {new_cap})),
                          parent=pi, generation=g + 1)
            # DGM validity gate: a child must still be a functioning code editor.
            if child.key() in seen:
                continue
            child.score = staged_evaluate(child, evaluate_fn, small, medium, big)
            if archive_mode == "keep_better" and child.score <= parent.score:
                continue
            archive.append(child)
            seen.add(child.key())
            if verbose:
                print(f"  gen {g}: parent#{pi}{list(parent.capabilities)} +{new_cap} "
                      f"-> score={child.score:.3f}")

    best = max(archive, key=lambda a: a.score)
    return DGMResult(archive, best, seed_agent.score, best.score)


# ===========================================================================
# Dataset: SWE-bench Verified instance metadata (real ids for the subsets)
# ===========================================================================


def download_swebench(limit: int) -> List[dict]:
    dataset, split, config = SWEBENCH
    return [{"instance_id": r["instance_id"], "repo": r["repo"]}
            for r in hf_rows(dataset, split, config=config, limit=limit)]


# ===========================================================================
# main
# ===========================================================================


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generations", type=int, default=12)
    p.add_argument("--selfimprove-size", type=int, default=2)
    p.add_argument("--archive", default="keep_all", choices=["keep_all", "keep_better"])
    p.add_argument("--provider", default="claude", choices=["claude", "glm"])
    p.add_argument("--model", default=None,
                   help="optional: let an LLM propose self-modifications (else deterministic)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="load dataset + show plan only")
    args = p.parse_args()

    print("Algorithm: Darwin Godel Machine (DGM) -- harness self-evolution")
    print("Dataset  : SWE-bench Verified (real instance ids; surrogate objective)")
    instances = download_swebench(STAGE_BIG)
    art = HarnessArtifact()
    print(f"Governance: coding-agent harness blast_radius={art.blast_radius} "
          f"-> {classify(art).name} (harness changes are high-blast-radius)")
    print(f"Loaded   : {len(instances)} SWE-bench Verified instances "
          f"(stages {STAGE_SMALL}/{STAGE_MEDIUM}/{STAGE_BIG})")
    print(f"Example  : {instances[0]['instance_id']} ({instances[0]['repo']})")
    print("\nObjective: SURROGATE (capability-cover) -- real DGM runs SWE-bench in "
          "Docker.\n           The archive + selection + staged escalation are faithful.")

    if args.dry_run:
        print("\n[dry-run] not running the loop.")
        return

    complete = None
    if args.model:
        complete = (openai_compatible(model=args.model) if args.provider == "glm"
                    else claude(model=args.model))
        try:
            complete("Reply with the single word: ok")
        except Exception as e:  # noqa: BLE001
            print(f"\nLLM unreachable ({type(e).__name__}: {e}); using deterministic proposals.")
            complete = None

    print(f"\nRunning DGM ({args.archive}, selfimprove_size={args.selfimprove_size}, "
          f"L1 harness)...\n")
    result = run_dgm(instances, generations=args.generations,
                     selfimprove_size=args.selfimprove_size, archive_mode=args.archive,
                     complete=complete, seed=args.seed, verbose=True)

    print("\n=== best self-improved agent ===")
    print(f"capabilities: {list(result.best.capabilities)}")
    print(f"lineage     : generation {result.best.generation}")
    print(f"\nsurrogate resolve-rate: {result.seed_score:.3f} -> {result.best_score:.3f}")
    print(f"archive size (keep-all): {len(result.archive)} agents")


if __name__ == "__main__":
    main()
