"""Skill self-evolution, faithful port: **SkillOpt** (a.k.a. the ReflACT loop).

Paper : "SkillOpt: Executive Strategy for Self-Evolving Agent Skills",
        Yifan Yang et al., 2025 (arXiv:2605.23904).
Repo  : https://github.com/microsoft/SkillOpt   (PyPI: `skillopt`)
Dataset: **SearchQA** (`lucadiliello/searchqa`), the lightest of SkillOpt's six
        benchmarks -- single-turn text QA, deterministic EM/F1, no tools.

SkillOpt trains the **skill document as the external state of a frozen agent**,
"with the same discipline that makes weight-space optimization reproducible."
The repo's per-step loop (ReflACT) and its four load-bearing invariants are all
reproduced here, faithful to the code (traced from `skillopt/engine/trainer.py`,
`optimizer/skill.py`, `evaluation/gate.py`, `optimizer/scheduler.py`):

  1. **Bounded string edits on ONE markdown doc.** The optimizer returns a patch
     of ops from `{append, insert_after, replace, delete}` (`apply_patch` below,
     matching `optimizer/skill.py`). The doc is the whole trainable state and is
     injected into the frozen agent by prompt concatenation -- zero extra
     deployment calls.
  2. **Strict held-out accept gate.** A candidate is accepted only if it
     *strictly improves* the held-out validation hard-EM over the **current**
     skill (`evaluation/gate.py`, default `gate_metric=hard`). This is greedy
     hill-climbing -- the same acceptance shape as AgentDescent's `evolve()`.
  3. **Textual learning-rate budget.** An integer cap on edits per step
     (`optimizer/scheduler.py`) -- AgentDescent's `trust_region_ops` analogue.
  4. **Rejected-edit buffer.** Rejected edits are remembered within the epoch and
     fed back to the optimizer so it stops re-proposing them (`_format_step_buffer`)
     -- AgentDescent's "settled evidence survives" (aggregator §3.3), made explicit.

It runs **through `evolve()`** exactly like ACE/GEPA: a custom `SkillDocStrategy`
turns the analyst's edit patch into a `Diff`, and a custom `aggregator_factory`
(`StrictGateAggregator`) is the strict-EM gate + LR budget + rejected-edit
buffer. A skill doc is an **L2** artifact (`blast_radius=0.2`, via `classify`).
The epoch-level *slow-update* and *meta-skill* stabilisers are optional in the
repo and omitted from this minimal-but-faithful slice (noted, not hidden).

    python -m examples.skillopt.skillopt_skill_training --dry-run        # plan only, zero network
    python -m examples.skillopt.skillopt_skill_training --model claude-haiku-4-5
    python -m examples.skillopt.skillopt_skill_training --steps 8 --lr 4
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import string
import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from agentdescent.agents import Usage
from agentdescent.aggregator import AggregatorProtocol, MergeReport
from agentdescent.dataloader import Dataset, hf_rows, split_dataset
from agentdescent.evolvable import Diff, EvidenceCard
from agentdescent.evolution import EvolvingArtifact, Task, evolve, rule_id
from agentdescent.governance import classify
from agentdescent.ledger import CASConflict, Ledger
from agentdescent.staleness import get_policy
from examples._common import (add_standard_args, completion_for, confirm,
                              worker_count,
                              budget_kwargs, report_engine)

SEARCHQA = ("lucadiliello/searchqa", "default")   # (dataset, config)
Completion = Callable[[str], str]

# The repo's seed skill (skillopt/envs/searchqa/skills/initial.md).
SEED_SKILL = ("# Question Answering Skill\n"
              "(No learned rules yet. Rules will be added through the reflection process.)")

# The optimizer lives in the package registry, so a spec can name
# `"algorithm": "skillopt"` and run the same implementation (#384).
from agentdescent.algorithms.skillopt import (  # noqa: E402
    LRScheduler,
    SkillDocStrategy,
    SkillOptContext,
    StrictGateAggregator,
    StrictImprovement,
    apply_patch,
    make_propose,
    propose_patch,
    valid_edit,
)

_ROLLOUT_TMPL = ("## Skill\n{skill}\n\n"
                 "Answer the question using the context. Put ONLY the final "
                 "answer inside <answer></answer>.\n\n"
                 "Context: {context}\n\nQuestion: {question}")


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    return (m.group(1) if m else text).strip()


def em_score(pred: str, golds: List[str]) -> float:
    p = _normalize(pred)
    return 1.0 if any(p == _normalize(g) for g in golds) else 0.0


def f1_score(pred: str, golds: List[str]) -> float:
    def f1(a: str, b: str) -> float:
        at, bt = _normalize(a).split(), _normalize(b).split()
        if not at or not bt:
            return float(at == bt)
        common = collections.Counter(at) & collections.Counter(bt)
        n = sum(common.values())
        if n == 0:
            return 0.0
        prec, rec = n / len(at), n / len(bt)
        return 2 * prec * rec / (prec + rec)
    return max((f1(pred, g) for g in golds), default=0.0)


@dataclass
class Rollout:
    complete: Completion

    def answer(self, skill: str, ex: dict) -> str:
        text = self.complete(_ROLLOUT_TMPL.format(
            skill=skill, context=ex["context"][:2000], question=ex["question"]))
        return _extract_answer(text)


def eval_hard_em(rollout: Rollout, skill: str, examples: List[dict]) -> float:
    return sum(em_score(rollout.answer(skill, ex), ex["answers"])
               for ex in examples) / max(1, len(examples))


# ===========================================================================
# Running SkillOpt THROUGH evolve() -- strategy (edits) + strict-gate optimizer
# ===========================================================================
#
# SkillOpt plugs into the framework exactly like ACE/GEPA: a custom `Strategy`
# turns the analyst's edit patch into a `Diff` on the one-slot skill document,
# and a custom `aggregator_factory` implements the strict held-out gate + the
# rejected-edit buffer + the learning-rate schedule. The shared optimizer state
# (buffer, LR budget, edit registry, stats) lives in one `SkillOptContext` that
# both the propose step and the aggregator close over.


@dataclass
class SkillOptResult:
    skill: str
    seed_em: Optional[float]
    best_em: Optional[float]
    accepted: int
    rejected: int
    history: List[float]
    #: Carried through from the underlying
    #: :class:`~agentdescent.evolution.EvolutionResult`: without them a run that
    #: ended on a backend failure reported its "seed -> best" line exactly like a
    #: converged one, which is what `error` exists to distinguish.
    error: Optional[str] = None
    stop_reason: str = "rounds"


def run_skillopt(complete: Completion, train: List[dict], val: List[dict],
                 steps: int = 8, lr: int = 4, minibatch: int = 4,
                 lr_mode: str = "cosine", seed: int = 0, asynchronous: bool = False,
                 async_ratio: int = 3, max_seconds: float = 30.0,
                 max_rollouts: Optional[int] = None, staleness: str = "guarded",
                 merge_round=None, eval_concurrency: int = 8,
                 verbose: bool = False) -> SkillOptResult:
    """Drive SkillOpt through `evolve()` (`val` becomes the held-out gate set)."""
    def to_task(i, ex):
        return Task(id=f"sq{i}", prompt=ex["question"],
                    meta={"context": ex["context"], "answers": ex["answers"]})

    tasks = [to_task(i, ex) for i, ex in enumerate(list(train) + list(val))]
    rollout = Rollout(complete)
    ctx = SkillOptContext(scheduler=LRScheduler(max_lr=lr, min_lr=max(1, lr // 2),
                                                total_steps=steps, mode=lr_mode), budget=lr)

    def run(rendered, task):
        return rollout.answer(rendered, {"context": task.meta["context"],
                                         "question": task.prompt,
                                         "answers": task.meta["answers"]})

    def reward(task, output):
        return em_score(output, task.meta["answers"])

    def factory(ledger, verifier, audit, config, policy):
        return StrictGateAggregator(ledger, verifier, ctx,
                                    artifact_id="skill_document",
                                    merge_round=merge_round)

    result = evolve(tasks, reward, run=run, propose=make_propose(ctx, complete),
                    strategy=SkillDocStrategy(ctx, seed=SEED_SKILL), initial_state={"skill": SEED_SKILL},
                    blast_radius=0.2, artifact_id="skill_document", rounds=steps,
                    n_workers=minibatch,
                    max_concurrency=1 if asynchronous else minibatch,
                    asynchronous=asynchronous, async_ratio=async_ratio,
                    max_seconds=max_seconds if asynchronous else None,
                    held_out_frac=len(val) / max(1, len(tasks)),
                    eval_concurrency=eval_concurrency,
                    # A discarded card is a whole optimizer step thrown away.
                    # `full` rebases onto the current head and leaves the strict
                    # gate as the only verification -- which it already is: the
                    # candidate is scored on the full held-out split before it
                    # can commit, so nothing unverified reaches the document.
                    staleness_policy=get_policy(staleness),
                    aggregator_factory=factory, verbose=verbose,
                    max_rollouts=max_rollouts)
    if verbose:
        report_engine(result)
    return SkillOptResult(result.rendered, ctx.seed_em, ctx.best_em,
                          ctx.accepted, ctx.rejected,
                          [h.held_out_reward for h in result.history],
                          error=result.error, stop_reason=result.stop_reason)


# ===========================================================================
# Dataset: SearchQA, loaded dependency-free
# ===========================================================================


def download_searchqa(split: str, limit: int) -> List[dict]:
    dataset, config = SEARCHQA
    out: List[dict] = []
    for r in hf_rows(dataset, split, config=config, limit=limit):
        answers = r["answers"] if isinstance(r["answers"], list) else [r["answers"]]
        out.append({"question": r["question"], "context": r["context"],
                    "answers": [str(a) for a in answers if a]})
    return out


def load_dataset(n_train: int, n_val: int, seed: int = 0, hard: bool = False,
                 rollout: "Rollout" = None, hard_passes: int = 3) -> Dataset:
    """SearchQA using its native `train` split, and splitting `validation` into
    the val (gate) and test halves.

    ``hard=True`` first runs the seed skill over a wider pool and keeps only the
    questions it gets **wrong**. Measured with ``deepseek-v4-flash``, plain
    SearchQA scores 0.900 out of the box -- the model already knows the answers,
    so a skill document has nothing to add and the strict gate correctly accepts
    no edit. That is the right behaviour and a useless demonstration; this keeps
    the dataset and drops the items carrying no signal.
    """
    train = download_searchqa("train", n_train)
    pool_rows = download_searchqa("validation", n_val)
    if hard and rollout is not None:
        from agentdescent.dataloader import select_hard
        scorer = lambda ex: em_score(rollout.answer(SEED_SKILL, ex), ex["answers"])
        before = len(train) + len(pool_rows)
        # `hard_passes` measurements, and an item has to fail all of them. One
        # pass selects the answers the model got unlucky on, and the gate then
        # re-measures them back to correct: at `passes=1` this filter produced a
        # val split the seed skill scored **1.000** on -- a ceiling, where a
        # strict `candidate > current` gate can accept nothing by construction.
        train = select_hard(train, scorer, passes=hard_passes)
        pool_rows = select_hard(pool_rows, scorer, passes=hard_passes)
        print(f"Hard mode: kept {len(train) + len(pool_rows)} of {before} items the "
              f"seed skill answers incorrectly")
    pool = split_dataset(pool_rows, ratios=(0.0, 0.5, 0.5), seed=seed, name="SearchQA")
    return Dataset(train=train, val=pool.val, test=pool.test, name="SearchQA")


# ===========================================================================
# main
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    add_standard_args(p, max_seconds_default=40.0)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--lr", type=int, default=4, help="max edits/step (learning rate)")
    p.add_argument("--lr-mode", default="cosine", choices=["constant", "linear", "cosine"])
    p.add_argument("--staleness", default="guarded",
                   choices=["guarded", "reflective", "full"],
                   help=("what to do with an edit patch proposed against a doc "
                         "the merger has since moved: `guarded` rebases inside "
                         "the --async-ratio band and discards beyond it, `full` "
                         "rebases regardless and leaves the strict gate as the "
                         "only verification"))
    p.add_argument("--minibatch", type=int, default=4)
    p.add_argument("--train", type=int, default=40)
    p.add_argument("--val", type=int, default=20)
    p.add_argument("--hard", action="store_true",
                   help="keep only questions the seed skill answers wrong -- plain "
                        "SearchQA is already ~0.900 for a strong model, so there is "
                        "nothing for a skill document to add")
    p.add_argument("--hard-passes", type=int, default=3,
                   help="baseline measurements an item must fail before --hard "
                        "keeps it; 1 selects unlucky answers rather than hard ones")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    # --serial collapses this to the upstream algorithm's own semantics:
    # one worker, nothing to merge. Applied to args so the printed plan,
    # the cost estimate and the run cannot disagree about what ran.
    args.minibatch = worker_count(args, args.minibatch)

    print("Algorithm: SkillOpt / ReflACT -- skill-document self-evolution")
    print("Dataset  : SearchQA (single-turn text QA, EM/F1)")
    print(f"\nPlan     : model={args.model}, steps={args.steps}, lr={args.lr} "
          f"({args.lr_mode})")
    if args.asynchronous:
        print(f"Async    : {args.minibatch} workers, barrier-free "
              f"(async_ratio={args.async_ratio}, max {args.max_seconds:.0f}s); "
              "the strict gate rebases/discards stale edits")
    else:
        print(f"Parallel : {args.minibatch} workers run concurrently each step "
              f"(synchronous DP; the strict-gate merge is the barrier)")
    if args.dry_run:
        print("Data     : deferred (dry-run performs no network access)")
        print("\n[dry-run] plan only; no dataset or model API was accessed.")
        return

    ds = load_dataset(args.train, args.val, seed=args.seed)
    art = EvolvingArtifact("skill_document", blast_radius=0.2)
    ntr, nva, nte = ds.sizes()
    print(f"Governance: skill doc blast_radius={art.blast_radius} -> {classify(art).name}")
    print(f"Splits   : {ntr} train / {nva} val / {nte} test")
    print("\nSeed skill document:")
    print("  " + SEED_SKILL.replace("\n", "\n  "))
    print("\nExample problem:")
    print("  Q:", ds.train[0]["question"][:150])
    print("  A:", ds.train[0]["answers"])

    calls = args.steps * (args.minibatch + 1 + nva) + nte
    print(f"Budget   : up to ~{calls} model calls (rollouts dominate)")

    if not confirm(args):
        return

    usage = Usage()                       # what the run actually costs
    completion = completion_for(args, usage=usage)
    if args.hard:
        # Needs the model, which is built above, so the dataset is re-selected here
        # rather than at load time.
        ds = load_dataset(args.train, args.val, seed=args.seed, hard=True,
                          hard_passes=args.hard_passes,
                          rollout=Rollout(completion))
        ntr, nva, nte = ds.sizes()
        print(f"Splits   : {ntr} train / {nva} val / {nte} test  (hard subset)")
    try:
        completion("Reply with the single word: ok")
    except Exception as e:  # noqa: BLE001
        print(f"\nCould not reach the model ({type(e).__name__}: {e}).")
        print("For --provider glm set OPENAI_BASE_URL + OPENAI_API_KEY; "
              "for claude set ANTHROPIC_API_KEY (or `ant auth login`).")
        return

    # `--reflective-merge` was declared by `add_standard_args` and never read.
    # `StrictGateAggregator` implements `AggregatorProtocol` itself, so the
    # engine's fusion policy never reaches it -- GEPA hands its aggregator the
    # merger explicitly, and so does this one now.
    merge_round = None
    if args.reflective_merge:
        from agentdescent.fusion import ReflectiveFusion
        merge_round = ReflectiveFusion(completion)
        print("NOTE: --reflective-merge scores ONE fused patch per step instead "
              "of one candidate per worker -- which is upstream's shape (a step "
              "emits one patch of up to `lr` edits). The strict gate is "
              "unchanged; what it scores is.")

    print("\nTraining skill document (ReflACT: edits + strict gate + LR + buffer, L2)...\n")
    result = run_skillopt(completion, ds.train, ds.val, steps=args.steps, lr=args.lr,
                          minibatch=args.minibatch, lr_mode=args.lr_mode, seed=args.seed,
                          asynchronous=args.asynchronous, async_ratio=args.async_ratio,
                          max_seconds=args.max_seconds, staleness=args.staleness,
                          merge_round=merge_round,
                          eval_concurrency=args.eval_concurrency, verbose=True,
                          **budget_kwargs(args))

    test_em = eval_hard_em(Rollout(completion), result.skill, ds.test)
    print("\n=== trained skill document ===")
    print(result.skill)
    seed = ("not measured (no patch reached the gate)"
            if result.seed_em is None else f"{result.seed_em:.3f}")
    best = "—" if result.best_em is None else f"{result.best_em:.3f}"
    print(f"\nval hard-EM : {seed} -> {best}")
    print(f"test hard-EM: {test_em:.3f}  (held out, never seen by the gate)")
    print(f"edits accepted / rejected: {result.accepted} / {result.rejected}")
    print(f"stopped     : {result.stop_reason}")
    if result.error:
        print(f"WARNING: the run did not finish cleanly -- {result.error}")
    print(f"model usage: {usage.summary()}")


if __name__ == "__main__":
    main()
