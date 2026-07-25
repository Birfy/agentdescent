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
     hill-climbing -- the same acceptance shape as Concordia's `evolve()`.
  3. **Textual learning-rate budget.** An integer cap on edits per step
     (`optimizer/scheduler.py`) -- Concordia's `trust_region_ops` analogue.
  4. **Rejected-edit buffer.** Rejected edits are remembered within the epoch and
     fed back to the optimizer so it stops re-proposing them (`_format_step_buffer`)
     -- Concordia's "settled evidence survives" (aggregator §3.3), made explicit.

It runs **through `evolve()`** exactly like ACE/GEPA: a custom `SkillDocStrategy`
turns the analyst's edit patch into a `Diff`, and a custom `aggregator_factory`
(`StrictGateAggregator`) is the strict-EM gate + LR budget + rejected-edit
buffer. A skill doc is an **L2** artifact (`blast_radius=0.2`, via `classify`).
The epoch-level *slow-update* and *meta-skill* stabilisers are optional in the
repo and omitted from this minimal-but-faithful slice (noted, not hidden).

    python -m examples.skillopt_skill_training --dry-run        # dataset + seed doc, no API
    python -m examples.skillopt_skill_training --model claude-haiku-4-5
    python -m examples.skillopt_skill_training --steps 8 --lr 4
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import string
import sys
import threading
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

from concordia.agents import claude, openai_compatible
from concordia.aggregator import AggregatorProtocol, MergeReport
from concordia.dataloader import Dataset, hf_rows, split_dataset
from concordia.evolvable import Diff, EvidenceCard
from concordia.evolution import EvolvingArtifact, Task, evolve, rule_id
from concordia.governance import classify
from concordia.ledger import CASConflict, Ledger

SEARCHQA = ("lucadiliello/searchqa", "default")   # (dataset, config)
Completion = Callable[[str], str]

# The repo's seed skill (skillopt/envs/searchqa/skills/initial.md).
SEED_SKILL = ("# Question Answering Skill\n"
              "(No learned rules yet. Rules will be added through the reflection process.)")

EDIT_OPS = ("append", "insert_after", "replace", "delete")


# ===========================================================================
# The bounded skill-document edits (faithful to optimizer/skill.py)
# ===========================================================================


def apply_patch(doc: str, edits: List[dict]) -> str:
    """Apply the four bounded ops to the skill document, in order.

    Semantics mirror `optimizer/skill.py:apply_patch_with_report`:
    * ``append``       -> add ``content`` at the document tail;
    * ``insert_after`` -> insert ``content`` right after the line containing
                          ``target`` (fallback: append if target not found);
    * ``replace``      -> first occurrence of ``target`` -> ``content`` (skip if
                          target absent);
    * ``delete``       -> remove first occurrence of ``target`` (skip if absent).
    """
    for edit in edits:
        op = edit.get("op")
        content = edit.get("content", "")
        target = edit.get("target", "")
        if op == "append":
            doc = doc.rstrip("\n") + "\n" + content
        elif op == "insert_after":
            idx = doc.find(target)
            if target and idx != -1:
                line_end = doc.find("\n", idx)
                if line_end == -1:
                    line_end = len(doc)
                doc = doc[:line_end] + "\n" + content + doc[line_end:]
            else:
                doc = doc.rstrip("\n") + "\n" + content
        elif op == "replace":
            if target and target in doc:
                doc = doc.replace(target, content, 1)
        elif op == "delete":
            if target and target in doc:
                doc = doc.replace(target, "", 1)
    return doc


def valid_edit(edit: dict) -> bool:
    if not isinstance(edit, dict) or edit.get("op") not in EDIT_OPS:
        return False
    if edit["op"] in ("insert_after", "replace", "delete") and not edit.get("target"):
        return False
    if edit["op"] in ("append", "insert_after", "replace") and "content" not in edit:
        return False
    return True


# ===========================================================================
# The learning-rate scheduler (faithful to optimizer/scheduler.py)
# ===========================================================================


@dataclass
class LRScheduler:
    """Integer edit-budget per step -- the 'textual learning rate'."""

    max_lr: int = 4
    min_lr: int = 2
    total_steps: int = 8
    mode: str = "cosine"
    _t: int = 0

    def step(self) -> int:
        import math
        if self.mode == "constant":
            budget = self.max_lr
        elif self.mode == "linear":
            frac = self._t / max(1, self.total_steps - 1)
            budget = round(self.max_lr + (self.min_lr - self.max_lr) * frac)
        else:  # cosine (repo default)
            frac = self._t / max(1, self.total_steps - 1)
            cos = 0.5 * (1 + math.cos(math.pi * frac))
            budget = round(self.min_lr + (self.max_lr - self.min_lr) * cos)
        self._t += 1
        return max(self.min_lr, budget)


# ===========================================================================
# The optimizer (analyst): scored rollouts -> a bounded edit patch
# ===========================================================================

_ANALYST_TMPL = """You are optimising a QA agent's skill document. Below are \
failed cases (the agent's answer did not match the gold answer). Analyse the \
patterns ACROSS these failures and propose a small set of GENERAL edits to the \
skill document that would fix them and similar cases -- not a fix for any single \
case.

Current skill document:
\"\"\"
{skill}
\"\"\"

Failed cases:
{failures}
{buffer}
Return ONLY a JSON object:
{{"reasoning": "...", "edits": [
  {{"op": "append", "content": "<new rule>"}},
  {{"op": "insert_after", "target": "<exact existing text>", "content": "<text>"}},
  {{"op": "replace", "target": "<exact existing text>", "content": "<text>"}},
  {{"op": "delete", "target": "<exact existing text>"}}
]}}
Use AT MOST {budget} edits. Prefer `append` for new rules."""


def _format_failures(failures: List[Tuple[str, str, str]]) -> str:
    out = []
    for i, (q, pred, gold) in enumerate(failures[:8]):
        out.append(f"[{i}] Q: {q[:200]}\n    agent: {pred[:120]}\n    gold : {gold[:120]}")
    return "\n".join(out)


def _format_buffer(rejected: List[dict]) -> str:
    if not rejected:
        return ""
    lines = [json.dumps(e) for e in rejected[-6:]]
    return ("\nPrevious edits this epoch that were REJECTED (did not improve "
            "validation -- do NOT propose these again):\n" + "\n".join(lines) + "\n")


def propose_patch(complete: Completion, skill: str,
                  failures: List[Tuple[str, str, str]],
                  rejected: List[dict], budget: int) -> List[dict]:
    """The analyst LLM returns a validated, budget-capped list of edits."""
    prompt = _ANALYST_TMPL.format(skill=skill, failures=_format_failures(failures),
                                  buffer=_format_buffer(rejected), budget=budget)
    text = complete(prompt)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    edits = [e for e in obj.get("edits", []) if valid_edit(e)]
    return edits[:budget]


# ===========================================================================
# Rollout + SearchQA scoring (hard = EM, soft = token F1)
# ===========================================================================

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
class SkillOptContext:
    scheduler: LRScheduler
    budget: int
    rejected_buffer: List[dict] = field(default_factory=list)
    edits_by_diff: dict = field(default_factory=dict)
    seed_em: float = 0.0
    best_em: float = 0.0
    accepted: int = 0
    rejected: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)   # workers run concurrently


class SkillDocStrategy:
    """The artifact is one markdown skill doc; a proposal is an edit patch."""

    def __init__(self, ctx: SkillOptContext):
        self.ctx = ctx

    def initial(self):
        return {"skill": SEED_SKILL}

    def render(self, state):
        return state.get("skill", SEED_SKILL)

    def to_diff(self, state, proposal, author, base_version, target):
        try:
            edits = [e for e in json.loads(proposal).get("edits", []) if valid_edit(e)]
        except (json.JSONDecodeError, TypeError, AttributeError):
            return None
        if not edits:
            return None
        new_doc = apply_patch(state.get("skill", SEED_SKILL), edits)
        if new_doc == state.get("skill"):
            return None
        did = f"{author}:{rule_id(new_doc)}:{base_version}"
        with self.ctx.lock:                                 # concurrent workers
            self.ctx.edits_by_diff[did] = edits             # for the rejected buffer
        return Diff(diff_id=did, target=target, ops={"skill": new_doc}, author=author)


def make_propose(ctx: SkillOptContext, complete: Completion):
    """The analyst: turn one failed rollout into a budget-capped edit patch."""
    def propose(rendered, task, output, score):
        failures = [(task.prompt, output, task.meta["answers"][0])]
        edits = propose_patch(complete, rendered, failures, ctx.rejected_buffer, ctx.budget)
        return json.dumps({"edits": edits}) if edits else None
    return propose


class StrictGateAggregator(AggregatorProtocol):
    """SkillOpt's optimizer: strict held-out EM gate + rejected-edit buffer + LR."""

    def __init__(self, ledger: Ledger, verifier, ctx: SkillOptContext,
                 artifact_id: str = "skill_document"):
        self.ledger = ledger
        self.verifier = verifier
        self.ctx = ctx
        self.aid = artifact_id
        self.cards: List[EvidenceCard] = []
        self.current_em = None

    def ingest(self, card: EvidenceCard) -> None:
        self.cards.append(card)

    def step(self) -> List[MergeReport]:
        snap = self.ledger.snapshot(Ledger.DEV)
        head = snap.get(self.aid)
        base_vv = {self.aid: snap.version.get(self.aid, 0)}
        if self.current_em is None:                        # gate baseline = seed doc
            self.current_em = head.score(self.verifier.held_out)
            self.ctx.seed_em = self.ctx.best_em = self.current_em

        cards, self.cards = self.cards, []
        best = None
        for card in cards:                                 # pick the best strict improver
            em = head.apply(card.diff).score(self.verifier.held_out)
            if em > self.current_em and (best is None or em > best[1]):
                best = (card, em)

        report_diff = committed = None
        if best is not None:
            card, em = best
            try:
                _, committed = self.ledger.commit(
                    head.apply(card.diff), base_vv, branch=Ledger.DEV,
                    message=f"skillopt accept {card.diff.diff_id}")
                self.current_em = em
                self.ctx.best_em = max(self.ctx.best_em, em)
                self.ctx.accepted += 1
                self.ctx.rejected_buffer.clear()           # accepted -> reset buffer
                report_diff = card.diff
            except CASConflict:
                committed = None
        # every non-accepted candidate's edits are remembered in-epoch.
        for card in cards:
            if best is None or card is not best[0]:
                self.ctx.rejected += 1
                self.ctx.rejected_buffer.extend(self.ctx.edits_by_diff.get(card.diff.diff_id, []))

        self.ctx.budget = self.ctx.scheduler.step()        # advance the LR schedule
        return [MergeReport(self.aid, report_diff, False, len(cards), len(cards), 0, 0,
                            self.current_em, committed,
                            f"lr={self.ctx.budget} val_EM={self.current_em:.3f}")]


@dataclass
class SkillOptResult:
    skill: str
    seed_em: float
    best_em: float
    accepted: int
    rejected: int
    history: List[float]


def run_skillopt(complete: Completion, train: List[dict], val: List[dict],
                 steps: int = 8, lr: int = 4, minibatch: int = 4,
                 lr_mode: str = "cosine", seed: int = 0, asynchronous: bool = False,
                 async_ratio: int = 3, max_seconds: float = 30.0,
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
        return StrictGateAggregator(ledger, verifier, ctx, artifact_id="skill_document")

    result = evolve(tasks, reward, run=run, propose=make_propose(ctx, complete),
                    strategy=SkillDocStrategy(ctx), initial_state={"skill": SEED_SKILL},
                    blast_radius=0.2, artifact_id="skill_document", rounds=steps,
                    n_workers=minibatch, max_concurrency=minibatch,
                    asynchronous=asynchronous, async_ratio=async_ratio, max_seconds=max_seconds,
                    held_out_frac=len(val) / max(1, len(tasks)),
                    aggregator_factory=factory, verbose=verbose)
    return SkillOptResult(result.rendered, ctx.seed_em, ctx.best_em,
                          ctx.accepted, ctx.rejected,
                          [h.held_out_reward for h in result.history])


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


def load_dataset(n_train: int, n_val: int, seed: int = 0) -> Dataset:
    """SearchQA using its native `train` split, and splitting `validation` into
    the val (gate) and test halves."""
    train = download_searchqa("train", n_train)
    pool = split_dataset(download_searchqa("validation", n_val), ratios=(0.0, 0.5, 0.5),
                         seed=seed, name="SearchQA")
    return Dataset(train=train, val=pool.val, test=pool.test, name="SearchQA")


# ===========================================================================
# main
# ===========================================================================


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", default="claude", choices=["claude", "glm"])
    p.add_argument("--model", default="claude-haiku-4-5")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--lr", type=int, default=4, help="max edits/step (learning rate)")
    p.add_argument("--lr-mode", default="cosine", choices=["constant", "linear", "cosine"])
    p.add_argument("--minibatch", type=int, default=4)
    p.add_argument("--train", type=int, default=40)
    p.add_argument("--val", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--async", dest="asynchronous", action="store_true",
                   help="run barrier-free (async_evolve)")
    p.add_argument("--async-ratio", type=int, default=3, help="staleness lag budget")
    p.add_argument("--max-seconds", type=float, default=40.0, help="async wall-clock budget")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()

    print("Algorithm: SkillOpt / ReflACT -- skill-document self-evolution")
    print("Dataset  : SearchQA (single-turn text QA, EM/F1)")
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
    print(f"\nPlan     : model={args.model}, steps={args.steps}, lr={args.lr} ({args.lr_mode})")
    if args.asynchronous:
        print(f"Async    : {args.minibatch} workers, barrier-free (async_ratio={args.async_ratio}, "
              f"max {args.max_seconds:.0f}s); the strict gate rebases/discards stale edits")
    else:
        print(f"Parallel : {args.minibatch} workers run concurrently each step "
              f"(synchronous DP; the strict-gate merge is the barrier)")
    print(f"Budget   : up to ~{calls} model calls (rollouts dominate)")

    if args.dry_run:
        print("\n[dry-run] not calling the API. Drop --dry-run to train the skill.")
        return

    if not args.yes and sys.stdin.isatty():
        if input("\nProceed with real API calls? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted.")
            return

    completion = (openai_compatible(model=args.model) if args.provider == "glm"
                  else claude(model=args.model))
    try:
        completion("Reply with the single word: ok")
    except Exception as e:  # noqa: BLE001
        print(f"\nCould not reach the model ({type(e).__name__}: {e}).")
        print("For --provider glm set OPENAI_BASE_URL + OPENAI_API_KEY; "
              "for claude set ANTHROPIC_API_KEY (or `ant auth login`).")
        return

    print("\nTraining skill document (ReflACT: edits + strict gate + LR + buffer, L2)...\n")
    result = run_skillopt(completion, ds.train, ds.val, steps=args.steps, lr=args.lr,
                          minibatch=args.minibatch, lr_mode=args.lr_mode, seed=args.seed,
                          asynchronous=args.asynchronous, async_ratio=args.async_ratio,
                          max_seconds=args.max_seconds, verbose=True)

    test_em = eval_hard_em(Rollout(completion), result.skill, ds.test)
    print("\n=== trained skill document ===")
    print(result.skill)
    print(f"\nval hard-EM : {result.seed_em:.3f} -> {result.best_em:.3f}")
    print(f"test hard-EM: {test_em:.3f}  (held out, never seen by the gate)")
    print(f"edits accepted / rejected: {result.accepted} / {result.rejected}")


if __name__ == "__main__":
    main()
