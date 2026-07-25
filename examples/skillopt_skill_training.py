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

This maps onto Concordia's provider layer (`concordia.agents`) and governance
(a skill doc is an **L2** artifact -> `blast_radius=0.2`, printed via `classify`).
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
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

from concordia.agents import claude, openai_compatible
from concordia.dataloader import hf_rows
from concordia.evolvable import Contract, Evolvable
from concordia.governance import classify

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
# Governance artifact (skill doc = L2)
# ===========================================================================


@dataclass
class SkillArtifact(Evolvable):
    id: str = "skill_document"
    blast_radius: float = 0.2
    version: int = 1
    contract: Contract = field(default_factory=lambda: Contract("task", "text", 1))

    def render(self) -> str: return ""
    def diff(self, other): ...
    def apply(self, diff): ...
    def cheap_eval(self, evidence): return 0.0
    def full_eval(self, task_set): return {}


# ===========================================================================
# The SkillOpt training loop (ReflACT, minimal faithful slice)
# ===========================================================================


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
                 lr_mode: str = "cosine", seed: int = 0,
                 verbose: bool = False) -> SkillOptResult:
    import random
    rng = random.Random(seed)
    rollout = Rollout(complete)
    scheduler = LRScheduler(max_lr=lr, min_lr=max(1, lr // 2),
                            total_steps=steps, mode=lr_mode)

    skill = SEED_SKILL
    current_em = eval_hard_em(rollout, skill, val)      # gate baseline
    best_em = seed_em = current_em
    rejected_buffer: List[dict] = []
    history = [current_em]
    accepted = rejected = 0
    if verbose:
        print(f"  seed skill: val hard-EM = {current_em:.3f}")

    for t in range(steps):
        budget = scheduler.step()
        # rollout on a train minibatch; collect failures (hard == 0).
        batch = rng.sample(train, min(minibatch, len(train)))
        failures = []
        for ex in batch:
            pred = rollout.answer(skill, ex)
            if em_score(pred, ex["answers"]) == 0.0:
                failures.append((ex["question"], pred, ex["answers"][0]))
        if not failures:
            history.append(current_em)
            continue

        edits = propose_patch(complete, skill, failures, rejected_buffer, budget)
        if not edits:
            history.append(current_em)
            continue
        candidate = apply_patch(skill, edits)

        # strict accept gate: candidate must beat CURRENT on held-out val EM.
        cand_em = eval_hard_em(rollout, candidate, val)
        if cand_em > current_em:
            skill, current_em = candidate, cand_em
            best_em = max(best_em, cand_em)
            rejected_buffer.clear()
            accepted += 1
            tag = "accept"
        else:
            rejected_buffer.extend(edits)              # remembered in-epoch
            rejected += 1
            tag = "reject"
        history.append(current_em)
        if verbose:
            print(f"  step {t}: lr={budget} edits={len(edits)} "
                  f"cand_EM={cand_em:.3f} -> {tag} (cur={current_em:.3f})")

    return SkillOptResult(skill, seed_em, best_em, accepted, rejected, history)


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
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()

    print("Algorithm: SkillOpt / ReflACT -- skill-document self-evolution")
    print("Dataset  : SearchQA (single-turn text QA, EM/F1)")
    train = download_searchqa("train", args.train)
    val = download_searchqa("validation", args.val)
    art = SkillArtifact()
    print(f"Governance: skill doc blast_radius={art.blast_radius} -> {classify(art).name}")
    print(f"Loaded   : {len(train)} train / {len(val)} val")
    print("\nSeed skill document:")
    print("  " + SEED_SKILL.replace("\n", "\n  "))
    print("\nExample problem:")
    print("  Q:", train[0]["question"][:150])
    print("  A:", train[0]["answers"])

    calls = args.steps * (args.minibatch + 1 + args.val) + args.val
    print(f"\nPlan     : model={args.model}, steps={args.steps}, lr={args.lr} ({args.lr_mode})")
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
    result = run_skillopt(completion, train, val, steps=args.steps, lr=args.lr,
                          minibatch=args.minibatch, lr_mode=args.lr_mode,
                          seed=args.seed, verbose=True)

    print("\n=== trained skill document ===")
    print(result.skill)
    print(f"\nval hard-EM: {result.seed_em:.3f} -> {result.best_em:.3f}")
    print(f"edits accepted / rejected: {result.accepted} / {result.rejected}")


if __name__ == "__main__":
    main()
