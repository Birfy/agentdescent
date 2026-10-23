"""SkillOpt's strict gate and textual learning rate, shipped as a named
algorithm (#384).

The mechanism is the paper's optimizer: a single markdown skill document,
bounded edit patches (`append` / `insert_after` / `replace` / `delete`), a
strict full-held-out gate that commits only a strict improvement, an
in-epoch rejected-edit buffer, and an integer edit budget annealed by a
cosine/linear schedule. The example keeps its SearchQA domain, prompts and
CLI and imports this implementation, so there is one copy of the algorithm.

Use it from a spec as ``"algorithm": "skillopt"`` (``kind: text``). The
spec's ``target`` is the seed skill document and its ``reflect`` model drives
the analyst (the patch proposer).
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..agents import Completion
from ..aggregator import AggregatorProtocol, MergeReport
from ..evolvable import Diff, EvidenceCard
from ..evolution import rule_id
from ..ledger import CASConflict, Ledger
from . import Algorithm, BuildContext, Overrides, Param, register

__all__ = ["LRScheduler", "SkillDocStrategy", "SkillOptContext",
           "StrictGateAggregator", "StrictImprovement", "apply_patch",
           "make_propose", "strict_gate_factory", "valid_edit"]



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


_ANALYST_TMPL = """You are optimising a skill document. Below are \
failed cases (the agent's answer did not match the reference answer). Analyse the \
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


@dataclass
class SkillOptContext:
    scheduler: LRScheduler
    budget: int
    rejected_buffer: List[dict] = field(default_factory=list)
    edits_by_diff: dict = field(default_factory=dict)
    #: ``None`` until the gate has measured the seed document, which happens
    #: inside ``step()`` -- and ``step()`` only runs when a card reaches the
    #: merger. A run that proposes nothing therefore never measures its own
    #: baseline, and a 0.0 default printed ``val hard-EM : 0.000 -> 0.000`` for
    #: that case: indistinguishable from a seed document that answers nothing.
    seed_em: Optional[float] = None
    best_em: Optional[float] = None
    accepted: int = 0
    rejected: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)   # workers run concurrently


class SkillDocStrategy:
    """The artifact is one markdown skill doc; a proposal is an edit patch."""

    def __init__(self, ctx: SkillOptContext, seed: str = ""):
        self.ctx = ctx
        self.seed = seed

    def initial(self):
        return {"skill": self.seed}

    def render(self, state):
        return state.get("skill", self.seed)

    def to_diff(self, state, proposal, author, base_version, target):
        try:
            edits = [e for e in json.loads(proposal).get("edits", []) if valid_edit(e)]
        except (json.JSONDecodeError, TypeError, AttributeError):
            return None
        if not edits:
            return None
        new_doc = apply_patch(state.get("skill", self.seed), edits)
        if new_doc == state.get("skill"):
            return None
        did = f"{author}:{rule_id(new_doc)}:{base_version}"
        with self.ctx.lock:                                 # concurrent workers
            self.ctx.edits_by_diff[did] = edits             # for the rejected buffer
        return Diff(diff_id=did, target=target, ops={"skill": new_doc}, author=author)


def make_propose(ctx: SkillOptContext, complete: Completion):
    """The analyst: turn one failed rollout into a budget-capped edit patch."""
    def propose(rendered, task, output, score):
        gold = task.meta.get("answers") or [task.meta.get("gold", "")]
        failures = [(task.prompt, output, gold[0])]
        edits = propose_patch(complete, rendered, failures, ctx.rejected_buffer, ctx.budget)
        return json.dumps({"edits": edits}) if edits else None
    return propose


class StrictImprovement:
    """SkillOpt's gate at the standard acceptance seam.

    An :class:`~agentdescent.policies.AcceptancePolicy` over genuine
    ``MergeContext`` records: commit only a candidate whose **full held-out**
    EM strictly beats the current document's. No Beta draw, no annealing --
    SkillOpt's rule is deliberately harsher than the shipped statistical gate,
    and naming it at the seam records that as a decision rather than a loop
    detail.
    """

    def accept(self, ctx):
        from agentdescent.policies import AcceptDecision, MergeContext
        base = MergeContext.rate(ctx.base_counts)
        cand = MergeContext.rate(ctx.cand_counts)
        improved = cand > base
        return AcceptDecision(
            accept=improved,
            category="" if improved else "strict-gate",
            detail=f"val EM {base:.3f} -> {cand:.3f}",
            observed_delta=cand - base)


class StrictGateAggregator(AggregatorProtocol):
    """SkillOpt's optimizer: strict held-out EM gate + rejected-edit buffer + LR."""

    def __init__(self, ledger: Ledger, verifier, ctx: SkillOptContext,
                 artifact_id: str = "skill_document",
                 acceptance: Optional[StrictImprovement] = None,
                 merge_round=None):
        self.ledger = ledger
        self.verifier = verifier
        self.ctx = ctx
        self.acceptance = acceptance or StrictImprovement()
        self.aid = artifact_id
        self.cards: List[EvidenceCard] = []
        self._lock = threading.Lock()   # ingest: workers; step: one thread
        self.current_em = None
        #: Fuse the sweep's edit patches into ONE before the gate scores them
        #: (`--reflective-merge`). Closer to upstream than the default, not
        #: further from it: ReflACT's step emits **one patch of up to `lr`
        #: edits** and evaluates it once (`engine/trainer.py`'s `ranked_patch`),
        #: where this loop scores one candidate per worker and takes the best.
        #: The gate is untouched either way -- still `cand > current` on the
        #: full held-out split -- but N held-out sweeps become one, and the
        #: surviving edits go forward together instead of all but one being
        #: dropped.
        self.merge_round = merge_round
        if merge_round is not None and hasattr(merge_round, "bind"):
            merge_round.bind(verifier)

    def ingest(self, card: EvidenceCard) -> None:
        # ingest runs on worker threads, step on one: see AggregatorProtocol.
        with self._lock:
            self.cards.append(card)

    def step(self) -> List[MergeReport]:
        snap = self.ledger.snapshot(Ledger.DEV)
        head = snap.get(self.aid)
        base_vv = {self.aid: snap.version.get(self.aid, 0)}
        if self.current_em is None:                        # gate baseline = seed doc
            self.current_em = head.score(self.verifier.held_out)
            self.ctx.seed_em = self.ctx.best_em = self.current_em

        with self._lock:
            cards, self.cards = self.cards, []
        best = None
        n = max(1, len(self.verifier.held_out))
        # One fused patch per sweep instead of one candidate per worker. The
        # strict gate is unchanged; what it scores is.
        diffs = [card.diff for card in cards]
        if self.merge_round is not None and len(diffs) > 1:
            merged, _, _ = self.merge_round.select(head, diffs)
            if merged is not None:
                diffs = [merged]
        for diff in diffs:                                 # pick the best strict improver
            from agentdescent.policies import MergeContext
            candidate = head.apply(diff)
            # `current_em` is the bar: this gate takes strict improvers only, so
            # a candidate that provably cannot reach it is rejected whatever the
            # unscored tail says. `score_bounded` stops there instead of buying
            # the rest -- and a candidate that *does* clear the bar is never cut,
            # so `em` is exact on every path that commits or is recorded.
            em = candidate.score_bounded(self.verifier.held_out, self.current_em)
            decision = self.acceptance.accept(MergeContext(
                artifact=head, candidate=candidate, cards=list(cards),
                base_counts=(self.current_em * n, (1 - self.current_em) * n),
                cand_counts=(em * n, (1 - em) * n), diff=diff))
            if decision.accept and (best is None or em > best[1]):
                best = (diff, em)

        report_diff = committed = None
        if best is not None:
            diff, em = best
            try:
                _, committed = self.ledger.commit(
                    head.apply(diff), base_vv, branch=Ledger.DEV,
                    message=f"skillopt accept {diff.diff_id}")
                self.current_em = em
                self.ctx.best_em = em if self.ctx.best_em is None else max(self.ctx.best_em, em)
                self.ctx.accepted += 1
                self.ctx.rejected_buffer.clear()           # accepted -> reset buffer
                report_diff = diff
            except CASConflict:
                committed = None
        # Every non-accepted candidate's edits are remembered in-epoch, which is
        # upstream's `step_buffer`: "use it to avoid repeating ineffective edits".
        # Compared by diff id, not object identity: under `--reflective-merge`
        # the accepted candidate is a *fused* diff that is none of the cards, so
        # an identity test would file every contributing edit as rejected -- and
        # then feed the optimizer a list telling it not to re-propose the edits
        # it had just accepted.
        accepted_ids = ({best[0].diff_id} if best is not None else set())
        contributing = ({c.diff.diff_id for c in cards}
                        if best is not None and best[0].diff_id not in
                        {c.diff.diff_id for c in cards} else set())
        for card in cards:
            if card.diff.diff_id in accepted_ids or card.diff.diff_id in contributing:
                continue
            self.ctx.rejected += 1
            self.ctx.rejected_buffer.extend(self.ctx.edits_by_diff.get(card.diff.diff_id, []))

        self.ctx.budget = self.ctx.scheduler.step()        # advance the LR schedule
        return [MergeReport(self.aid, report_diff, False, len(cards), len(cards), 0, 0,
                            self.current_em, committed,
                            f"lr={self.ctx.budget} val_EM={self.current_em:.3f}",
                            category=("skillopt-accept" if committed
                                      else "strict-gate"))]


def strict_gate_factory(ctx: SkillOptContext, *, artifact_id: str = "skill_document",
                        acceptance: Optional[StrictImprovement] = None):
    """The `aggregator_factory=` evolve() expects for SkillOpt."""
    def factory(ledger, verifier, audit, config, policy):
        return StrictGateAggregator(ledger, verifier, ctx,
                                    artifact_id=artifact_id,
                                    acceptance=acceptance)
    return factory


def _read_seed(target: str) -> str:
    from ..evolvespec import _read_text_target
    try:
        return _read_text_target(target)
    except Exception:  # noqa: BLE001 - compose will report an unreadable target
        return ""


def _build(ctx: BuildContext, p: Dict[str, object]) -> Overrides:
    scheduler = LRScheduler(max_lr=int(p["max_lr"]),
                            min_lr=int(p["min_lr"]),
                            total_steps=int(p["total_steps"]),
                            mode=str(p["mode"]))
    context = SkillOptContext(scheduler=scheduler, budget=int(p["max_lr"]))
    seed = _read_seed(str(getattr(ctx.spec, "target", "")))
    knobs: Dict[str, object] = {
        "strategy": SkillDocStrategy(context, seed=seed),
        "initial_state": {"skill": seed},
        "aggregator_factory": strict_gate_factory(
            context, artifact_id=ctx.spec.artifact_id()),
    }
    if callable(ctx.reflect):
        knobs["propose"] = make_propose(context, ctx.reflect)
    return Overrides(
        knobs=knobs,
        notes=["algorithm skillopt: a strict full-held-out gate (not the shipped "
               "Beta gate), bounded edits and a cosine edit budget; the seed is "
               "the spec's target document."])


register(Algorithm(
    name="skillopt",
    summary="SkillOpt's skill-document training: bounded edit patches, a strict "
            "held-out gate, a rejected-edit buffer and an annealed textual "
            "learning rate.",
    kinds=("text",),
    fidelity="mechanism_microport",
    params=(
        Param("max_lr", "int", 4, "largest edit budget per step", minimum=1),
        Param("min_lr", "int", 2, "smallest edit budget per step", minimum=1),
        Param("total_steps", "int", 8, "steps the budget anneals over", minimum=1),
        Param("mode", "str", "cosine", "anneal schedule",
              choices=("cosine", "linear", "constant")),
    ),
    build=_build,
    source="SkillOpt: skill-document training; see docs/algo-skillopt.md",
    extra_calls="one analyst call per failed rollout; the strict gate scores each "
                "candidate over the full held-out set",
    doc="algo-skillopt.md"))
