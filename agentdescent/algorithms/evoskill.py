"""EvoSkill's bounded skill-library frontier, shipped as a named algorithm (#384).

The mechanism is the two-role induction loop: the Proposer analyses a batch of
failures, the Generator writes one new/edited `SKILL.md`, and a bounded top-K
frontier keeps the best skill libraries (parent = the best member, committed to
dev so the next round extends it). The example keeps its OfficeQA domain, prompts
and CLI and imports this implementation.

Use it from a spec as ``"algorithm": "evoskill"`` (``kind: skill_dir``,
``agent_dir`` or ``plugin``). The spec's target directory is the initial skill library and its
``reflect`` model drives the Proposer/Generator.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..agents import Completion
from ..aggregator import AggregatorProtocol, MergeReport
from ..evolvable import Diff, EvidenceCard
from ..filetree import load_tree, parse_tree
from ..ledger import CASConflict, Ledger
from ..selection import Candidate, SelectionContext
from ..treestrategy import FileTree
from . import Algorithm, BuildContext, Overrides, Param, register

#: Only genuine failures induce a skill.
PASS_THRESHOLD = 0.8

__all__ = ["EvoSkillContext", "Frontier", "FrontierBest", "SkillLibraryTree",
           "TopKFrontierAggregator", "frontier_aggregator_factory", "make_propose",
           "propose_and_generate", "render_skills", "skill_name", "skill_path",
           "skills_from_rendered", "skills_of"]


_PROPOSER_TMPL = """You are the Skill Proposer. The agent failed these cases. \
Analyse the patterns ACROSS them and identify ONE general improvement (not a fix \
for a single case).

Existing skills: {skill_names}

Failures:
{failures}
{feedback}
Reply with ONE line: `create <skill-name>` for a new skill, or `edit <skill-name>` \
to revise an existing one. Then, on the next lines, a one-sentence justification."""

_GENERATOR_TMPL = """You are the Skill Generator. Write ONE reusable skill file \
for the agent, addressing this improvement:

{justification}

{existing}
Output a SKILL.md body: a short imperative title line, then 2-5 concise, GENERAL
bullet rules for the task. Output only the skill text (no code fences)."""


def render_skills(skills: Dict[str, str]) -> str:
    if not skills:
        return "(no learned skills yet)"
    return "\n\n".join(f"### skill: {name}\n{body}" for name, body in sorted(skills.items()))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "skill"


def propose_and_generate(complete: Completion, skills: Dict[str, str],
                         failures: List[Tuple[str, str, str]],
                         feedback: List[str]) -> Optional[Tuple[str, str]]:
    """Two model calls: Proposer picks the improvement, Generator writes the skill."""
    fail_block = "\n".join(
        f"[{i}] Q: {q[:160]}\n    agent: {p[:100]}\n    gold: {g}"
        for i, (q, p, g) in enumerate(failures))
    fb = ("\nPreviously DISCARDED proposals (do not repeat):\n" +
          "\n".join(f"- {x}" for x in feedback[-5:]) + "\n") if feedback else ""
    decision = complete(_PROPOSER_TMPL.format(
        skill_names=", ".join(sorted(skills)) or "(none)",
        failures=fail_block, feedback=fb)).strip()
    m = re.match(r"\s*(create|edit)\s+([\w\- ]+)", decision, re.IGNORECASE)
    action = (m.group(1).lower() if m else "create")
    name = _slug(m.group(2)) if m else _slug(decision.split("\n")[0])
    existing = (f"You are EDITING the existing skill (preserve its intent):\n"
                f"{skills.get(name,'')}\n" if action == "edit" and name in skills else "")
    body = complete(_GENERATOR_TMPL.format(
        justification=decision, existing=existing)).strip()
    return (name, body) if body else None


@dataclass
class Frontier:
    max_size: int = 5      # src/registry/manager.py:update_frontier's default
    members: List[Tuple[Dict[str, str], float]] = field(default_factory=list)

    def update(self, skills: Dict[str, str], score: float) -> bool:
        """Admit if room; else replace the worst iff strictly greater."""
        if len(self.members) < self.max_size:
            self.members.append((skills, score))
            return True
        worst_i = min(range(len(self.members)), key=lambda i: self.members[i][1])
        if score > self.members[worst_i][1]:
            self.members[worst_i] = (skills, score)
            return True
        return False

    def select_parent(self) -> Tuple[Dict[str, str], float]:
        return max(self.members, key=lambda m: m[1])       # strategy="best"


class FrontierBest:
    """EvoSkill's parent rule at the standard selection seam.

    A :class:`~agentdescent.selection.SelectionPolicy`: pick the frontier's
    best-scoring member (upstream ``strategy="best"``). Local rather than the
    shipped ``Beam(1)`` so the tie-break stays byte-identical to the inline
    ``max`` it replaces (first maximal member wins, in admission order).
    """

    def select(self, ctx, n: int):
        candidates = list(ctx.candidates)
        if not candidates:
            return [ctx.head] * n
        best = max(candidates, key=lambda c: c.score or 0.0)
        return [best] * n

@dataclass
class EvoSkillContext:
    docs: Dict[str, str]
    frontier: Frontier
    feedback: List[str] = field(default_factory=list)
    recent_failures: List[Tuple[str, str, str]] = field(default_factory=list)
    #: ``None`` until the frontier has actually measured the base agent, which
    #: only happens inside ``step()`` -- and ``step()`` only runs when a card
    #: reaches the merger. A run that never induces a skill therefore never
    #: measures its own baseline, and defaulting this to 0.0 printed
    #: ``val score : 0.000 -> 0.000`` for that case: indistinguishable from a
    #: base agent that scores nothing, which is what it looked like. Measured on
    #: FinQA, the same configuration reports 0.727 as soon as one card lands.
    seed_score: Optional[float] = None
    best_score: Optional[float] = None
    eval_concurrency: int = 8     # parallelise the aggregator's held-out (val) re-eval
    batch_size: int = 4           # failures per induction BATCH (repo is batch-level, not per-trajectory)
    val_every: int = 3            # async SGD: validate the head every N applied skill updates (else roll back)
    last_propose_at: int = 0      # index into recent_failures at the last emitted skill
    lock: threading.Lock = field(default_factory=threading.Lock)   # workers run concurrently


#: Where a skill lives inside the artifact. The library was always a set of
#: `SKILL.md` files -- until now it only existed in memory, keyed by name. Keyed
#: by *path* it is a real directory, which is what lets a tool-using agent read
#: one skill at a time instead of carrying the whole library in every prompt.
SKILL_ROOT = "skills"


def skill_path(name: str) -> str:
    return f"{SKILL_ROOT}/{_slug(name)}/SKILL.md"


def skill_name(path: str) -> str:
    parts = path.split("/")
    return parts[1] if len(parts) >= 3 and parts[0] == SKILL_ROOT else path


def skills_of(state: Dict[str, str]) -> Dict[str, str]:
    """`{path: body}` (the artifact) -> `{name: body}` (what the algorithm reads)."""
    return {skill_name(p): body for p, body in state.items()}


def skills_from_rendered(rendered: str) -> Dict[str, str]:
    """The artifact as the algorithm wants it, from what `run`/`propose` are handed."""
    return skills_of(parse_tree(rendered))


class SkillLibraryTree(FileTree):
    """The artifact is a **directory** of `SKILL.md` files -- one per skill.

    The same library EvoSkill always evolved; what changed is that a state key is
    now a real path (`skills/<name>/SKILL.md`), so the library can be materialised
    into an agent's workspace and read file by file (`--backend claude-code`)
    instead of being inlined in every prompt.

    Two deliberate departures from stock :class:`~agentdescent.treestrategy.FileTree`:

    * ``to_diff`` keeps the repo's ``name :: body`` proposal protocol rather than
      FileTree's ``<EDITS>`` JSON. What is faithful about EvoSkill is the
      two-role Proposer/Generator induction, not the separator between a name and
      a body -- switching protocols would change the Generator's prompt and with
      it the thing being measured.
    * ``max_files_per_diff=1``: the repo induces exactly one skill per iteration.
    """

    def __init__(self, initial_files: Optional[Dict[str, str]] = None,
                 editable: Tuple[str, ...] = ("**",),
                 frozen: Tuple[str, ...] = ()) -> None:
        super().__init__(initial_files=dict(initial_files or {}),
                         editable=tuple(editable), frozen=tuple(frozen),
                         max_files_per_diff=1)

    def to_diff(self, state, proposal, author, base_version, target):
        name, _, body = proposal.partition(" :: ")
        name, body = name.strip(), body.strip()
        if not (name and body):
            return None
        path = skill_path(name)
        if not self.writable(path):
            return None
        if state.get(path) == body:
            return None
        return Diff(diff_id=f"{author}:{name}:{base_version}", target=target,
                    ops={path: body}, author=author)


def make_propose(ctx: EvoSkillContext, complete: Completion, generate=None):
    """Failure-driven skill induction -- BATCH-level, faithful to the repo.

    The repo induces one SKILL.md **per iteration from a batch of failures**
    (the Proposer analyses their shared *pattern*), not one skill per failed
    trajectory.  So we accumulate a batch of ``batch_size`` failures (shared
    across the concurrent workers) and only then emit a single skill -- keeping
    the top-K frontier validating a handful of skills, not dozens.  The
    parallelism (concurrent workers + concurrent val eval) is the *acceleration*;
    the batch->one-skill->validate->frontier flow stays the repo's."""
    def propose(rendered, task, output, score):
        if score >= PASS_THRESHOLD:                        # only learn from genuine failures (<0.8)
            return None
        with ctx.lock:                                     # workers share one growing batch
            gold = task.meta.get("gold") or task.meta.get("answer", "")
            ctx.recent_failures.append((task.prompt, output, gold))
            if len(ctx.recent_failures) - ctx.last_propose_at < ctx.batch_size:
                return None                                # batch not full yet -> keep collecting
            ctx.last_propose_at = len(ctx.recent_failures)
            batch = list(ctx.recent_failures[-ctx.batch_size:])
        skills = skills_from_rendered(rendered)            # Proposer + Generator over the batch
        proposed = (generate or propose_and_generate)(
            complete, skills, batch, ctx.feedback)
        if not proposed:
            return None
        name, body = proposed
        return f"{name} :: {body}"
    return propose


class TopKFrontierAggregator(AggregatorProtocol):
    """EvoSkill's optimizer: the bounded top-K aggregate frontier."""

    def __init__(self, ledger: Ledger, verifier, ctx: EvoSkillContext,
                 artifact_id: str = "skill_library",
                 selection: Optional[FrontierBest] = None,
                 merge_round=None):
        self.ledger = ledger
        self.verifier = verifier
        self.ctx = ctx
        self.selection = selection or FrontierBest()
        self.aid = artifact_id
        self.cards: List[EvidenceCard] = []
        self._lock = threading.Lock()   # ingest: workers; step: one thread
        self._seeded = False
        #: Fuse the sweep's proposals into ONE candidate before the frontier sees
        #: them (`--reflective-merge`). Off by default, because it changes what
        #: the frontier is offered: upstream evaluates each child on the full
        #: validation split and admits per candidate, so N workers cost N val
        #: sweeps. That per-candidate evaluation is this port's dominant cost and
        #: the only thing a wider run adds -- which is what makes fusing them a
        #: lever worth having and a departure worth naming. The admission rule
        #: itself is untouched; what it admits is not.
        self.merge_round = merge_round
        if merge_round is not None and hasattr(merge_round, "bind"):
            merge_round.bind(verifier)

    def ingest(self, card: EvidenceCard) -> None:
        # ingest runs on worker threads, step on one: see AggregatorProtocol.
        with self._lock:
            self.cards.append(card)

    def _eval(self, artifact) -> float:
        """Score an artifact on the held-out (val) set -- **concurrently**, so the
        acceptance check parallelises like the workers (agent-based rewards are the
        bottleneck otherwise). The runtime cache is thread-safe."""
        tasks = self.verifier.held_out
        if not tasks:
            return 0.0
        n = min(self.ctx.eval_concurrency, len(tasks))
        if n <= 1:
            return artifact.score(tasks)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n) as pool:
            scores = list(pool.map(lambda t: artifact.score([t]), tasks))
        return sum(scores) / len(scores)

    def step(self) -> List[MergeReport]:
        snap = self.ledger.snapshot(Ledger.DEV)
        head = snap.get(self.aid)
        base_vv = {self.aid: snap.version.get(self.aid, 0)}
        if not self._seeded:
            self.ctx.seed_score = self._eval(head)
            self.ctx.best_score = self.ctx.seed_score
            self.ctx.frontier.update(dict(head.state), self.ctx.seed_score)
            self._seeded = True

        with self._lock:
            cards, self.cards = self.cards, []
        # One fused candidate per sweep instead of one per worker. `update_frontier`
        # is unchanged -- bounded top-K on mean validation accuracy, admit on room
        # or strictly-greater -- and so is the parent draw; only the number of
        # children offered to it changes, from N to 1. The saving is the whole
        # point: admission costs a full validation sweep *per child*, so the cost
        # of a wider run is the part fusing removes.
        diffs = [card.diff for card in cards]
        if self.merge_round is not None and len(diffs) > 1:
            merged, _, _ = self.merge_round.select(head, diffs)
            if merged is not None:
                diffs = [merged]
        for diff in diffs:
            candidate = head.apply(diff)
            score = self._eval(candidate)
            admitted = self.ctx.frontier.update(dict(candidate.state), score)
            self.ctx.best_score = max(self.ctx.best_score, score)
            self.ctx.feedback.append(
                f"{skill_name(list(diff.ops)[0])}: "
                f"{'admitted' if admitted else 'discarded'} (val {score:.3f})")

        # strategy="best" at the standard seam: frontier members become
        # Candidates (version = member index) and FrontierBest picks the parent.
        rows = [Candidate(artifact_id=self.aid, version=i, state=dict(state),
                          score=score)
                for i, (state, score) in enumerate(self.ctx.frontier.members)]
        sel_ctx = SelectionContext(head=rows[0] if rows else None,
                                   candidates=tuple(rows), n_workers=1)
        parent_state = dict(self.selection.select(sel_ctx, 1)[0].state)
        report_diff = committed = None
        if parent_state != head.state:
            try:
                _, committed = self.ledger.commit(
                    head.apply(Diff(diff_id="frontier", target=self.aid,
                                    ops=dict(parent_state), author="evoskill")),
                    base_vv, branch=Ledger.DEV, message="evoskill: select best frontier member")
            except CASConflict:
                committed = None
        return [MergeReport(self.aid, report_diff, False, len(cards), len(cards), 0, 0,
                            self.ctx.best_score, committed,
                            f"frontier={len(self.ctx.frontier.members)} best={self.ctx.best_score:.3f}")]

def frontier_aggregator_factory(ctx: EvoSkillContext, *,
                                artifact_id: str = "skill_library",
                                max_frontier: int = 5,
                                eval_concurrency: int = 8):
    ctx.frontier.max_size = int(max_frontier)
    ctx.eval_concurrency = int(eval_concurrency)

    holder = {}

    def factory(ledger, verifier, audit, config, policy):
        agg = TopKFrontierAggregator(ledger, verifier, ctx,
                                     artifact_id=artifact_id)
        holder["agg"] = agg
        return agg
    factory.holder = holder  # type: ignore[attr-defined]
    return factory


def _build(ctx: BuildContext, p: Dict[str, object]) -> Overrides:
    try:
        initial = load_tree(os.path.abspath(os.path.expanduser(ctx.spec.target)))
    except Exception:  # noqa: BLE001 - compose reports an unreadable target first
        initial = {}
    context = EvoSkillContext(docs={}, frontier=Frontier(),
                              batch_size=int(p["batch_size"]))
    factory = frontier_aggregator_factory(
        context, artifact_id=ctx.spec.artifact_id(),
        max_frontier=int(p["max_frontier"]),
        eval_concurrency=int(p["eval_concurrency"]))
    knobs: Dict[str, object] = {
        "strategy": SkillLibraryTree(
            initial_files=initial,
            editable=tuple(getattr(ctx.spec, "editable", ()) or ("**",)),
            frozen=tuple(getattr(ctx.spec, "frozen", ()) or ())),
        "aggregator_factory": factory,
    }
    if callable(ctx.reflect):
        knobs["propose"] = make_propose(context, ctx.reflect)
    return Overrides(
        knobs=knobs,
        notes=["algorithm evoskill: a bounded top-K frontier over skill libraries; "
               "each admitted change is re-scored on the full held-out set and the "
               "best member becomes the next parent."])


register(Algorithm(
    name="evoskill",
    summary="EvoSkill's two-role skill induction: Proposer analyses a failure "
            "batch, Generator writes one SKILL.md, and a bounded top-K frontier "
            "keeps the best skill libraries.",
    kinds=("skill_dir", "agent_dir", "plugin"),
    fidelity="mechanism_microport",
    params=(
        Param("max_frontier", "int", 5, "frontier size", minimum=1),
        Param("batch_size", "int", 4, "failures per induction batch", minimum=1),
        Param("eval_concurrency", "int", 8,
              "held-out re-evaluations run in parallel", minimum=1),
    ),
    build=_build,
    source="EvoSkill: skill discovery (see docs/algo-evoskill.md)",
    extra_calls="two model calls per induced skill; one full held-out re-eval per "
                "admitted candidate",
    doc="algo-evoskill.md"))
