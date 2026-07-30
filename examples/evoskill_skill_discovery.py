"""Skill self-evolution, faithful port: **EvoSkill** (Automated Skill Discovery).

Paper : "EvoSkill: Automated Skill Discovery for Coding Agents / Multi-Agent
        Systems", Salaheddin Alzubi et al., 2026 (arXiv:2603.02766).
Repo  : https://github.com/sentient-agi/EvoSkill
Dataset: **OfficeQA** (grounded reasoning over U.S. Treasury Bulletins) -- the
        repo's flagship, with a deterministic offline numeric scorer.

This port is faithful to what the **code actually does** (traced from
`src/loop/runner.py`, `src/registry/manager.py`, `src/evaluation/reward.py`),
which differs from some paper-level claims -- flagged inline:

  * **Failure-driven skill induction.** Each iteration samples train items, runs
    the base agent, and collects failures (an item is a FAILURE when its
    multi-tolerance score < 0.8, `runner.py:319`). A **Skill Proposer** analyses
    the failure *patterns* ("a GENERAL improvement, not a fix for any single
    case") and a **Skill Generator** writes/edits one `SKILL.md` skill file.
  * **Bounded top-K aggregate frontier (NOT per-instance Pareto).** Despite the
    paper's framing, `registry/manager.py:update_frontier` keeps a bounded
    leaderboard on a single scalar (mean validation accuracy): admit if the
    frontier has room, else replace the worst member iff strictly greater. The
    parent for the next iteration is selected from it (default: `best`).
  * **Skill = a folder of Markdown files** with YAML frontmatter; the active skill
    set is injected into the agent by concatenation.

It runs **through `evolve()`** like ACE/GEPA: a `SkillLibraryStrategy` turns a
proposed `SKILL.md` into a `Diff`, and a custom `aggregator_factory`
(`TopKFrontierAggregator`) is the bounded top-K frontier. A skill is an **L2**
artifact -> `blast_radius=0.2` (via `classify`).

    python -m examples.evoskill_skill_discovery --dry-run     # dataset + scorer, no API
    python -m examples.evoskill_skill_discovery --model claude-haiku-4-5

Dataset caveat (you asked to use the real OfficeQA): the full set is HF-**gated**
(`databricks/officeqa`, needs `HF_TOKEN`); absent that this loads the repo's
**bundled 12-row sample** from GitHub. EvoSkill's agent uses Read/Grep tools over
the Treasury `.txt` docs; here a dependency-free keyword line-retriever stands in
for those tools. With one non-tool LLM on 272 KB bulletins accuracy is low -- the
value is the faithful *loop*, not a leaderboard number.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from agentdescent.agents import claude, openai_compatible
from agentdescent.aggregator import AggregatorProtocol, MergeReport
from agentdescent.dataloader import Dataset, fetch_text, load_gated_hf, split_dataset
from agentdescent.evolvable import Diff, EvidenceCard
from agentdescent.evolution import EvolvingArtifact, Task, evolve
from agentdescent.governance import classify
from agentdescent.ledger import CASConflict, Ledger

RAW = "https://raw.githubusercontent.com/sentient-agi/EvoSkill/main/examples/officeqa/data"
Completion = Callable[[str], str]

# runner.py:79 -- the exact tolerance ladder and the 0.8 pass threshold.
TOLERANCE_LEVELS = [0.05, 0.01, 0.1, 0.0, 0.025]
PASS_THRESHOLD = 0.8
UNITS = {"trillion": 1e12, "billion": 1e9, "million": 1e6, "thousand": 1e3}


# ===========================================================================
# Faithful numeric scorer (src/evaluation/reward.py + runner._score_multi_tolerance)
# ===========================================================================


def extract_numbers(text: str) -> List[float]:
    """Numbers with unit-aware scaling (million/billion/...), commas stripped."""
    out: List[float] = []
    for m in re.finditer(r"(-?\d[\d,]*\.?\d*)\s*(trillion|billion|million|thousand)?",
                         text, re.IGNORECASE):
        raw = m.group(1).replace(",", "")
        if raw in ("", "-", "."):
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        unit = (m.group(2) or "").lower()
        out.append(val * UNITS.get(unit, 1.0))
        if unit:                      # also keep the bare figure (unit ambiguity)
            out.append(val)
    return out


def fuzzy_match_answer(ground_truth: str, predicted: str, tolerance: float) -> bool:
    """Unit-aware numeric tolerance match (single- and multi-number answers)."""
    if not (ground_truth and predicted):
        return False
    gt = extract_numbers(ground_truth)
    pred = extract_numbers(predicted)
    if not gt:
        return _norm(ground_truth) in _norm(predicted)      # text fallback
    if not pred:
        return False

    def close(a: float, b: float) -> bool:
        if a == 0:
            return b == 0
        return abs(a - b) / abs(a) <= tolerance

    gt_singles = _dedup(gt)
    if len(_primary_numbers(ground_truth)) > 1:             # multi-number list
        return all(any(close(g, p) for p in pred) for g in _primary_numbers(ground_truth))
    return any(close(gt_singles[0], p) for p in pred)


def _primary_numbers(text: str) -> List[float]:
    """Numbers as written (no unit duplication) -- for the multi-number check."""
    return [float(m.replace(",", "")) for m in re.findall(r"-?\d[\d,]*\.?\d*", text)
            if m not in ("", "-", ".")]


def _dedup(xs: List[float]) -> List[float]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def score_multi_tolerance(predicted: str, ground_truth: str) -> float:
    """Weighted average over the tolerance ladder (runner._score_multi_tolerance).

    weight = 1 / (1 + 20 * tolerance): stricter tolerances dominate."""
    if not str(predicted or "").strip():
        return 0.0
    weighted, total = 0.0, 0.0
    for tol in TOLERANCE_LEVELS:
        w = 1.0 / (1.0 + 20.0 * tol)
        weighted += w * (1.0 if fuzzy_match_answer(ground_truth, predicted, tol) else 0.0)
        total += w
    return weighted / total


# ===========================================================================
# Base agent: skills + retrieved doc context -> answer
# ===========================================================================


def retrieve_context(doc: str, question: str, n_lines: int = 40) -> str:
    """Keyword line-retriever standing in for EvoSkill's Read/Grep tools."""
    keys = set(re.findall(r"[a-z]{4,}", question.lower()))
    lines = doc.splitlines()
    scored = sorted(range(len(lines)),
                    key=lambda i: -len(keys & set(re.findall(r"[a-z]{4,}", lines[i].lower()))))
    keep = sorted(scored[:n_lines])
    return "\n".join(lines[i] for i in keep if lines[i].strip())


def render_skills(skills: Dict[str, str]) -> str:
    if not skills:
        return "(no learned skills yet)"
    return "\n\n".join(f"### skill: {name}\n{body}" for name, body in sorted(skills.items()))


_AGENT_TMPL = (
    "You are a financial-analysis agent answering questions from U.S. Treasury "
    "Bulletin excerpts.\n\nLearned skills:\n{skills}\n\n"
    "Document excerpt:\n{context}\n\nQuestion: {question}\n\n"
    "Reason carefully, then end with `Answer: <value>`.")
# (the base agent that consumes the skills is `agent_answer`, defined with the
# evolve() wiring below.)


# ===========================================================================
# Skill Proposer + Skill Generator (two-role, faithful to agent_profiles/*)
# ===========================================================================

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
Output a SKILL.md body: a short imperative title line, then 2-5 concise, GENERAL \
bullet rules for answering U.S. Treasury Bulletin questions. Output only the skill \
text (no code fences)."""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "skill"


def propose_and_generate(complete: Completion, skills: Dict[str, str],
                         failures: List[Tuple[str, str, str]],
                         feedback: List[str]) -> Optional[Tuple[str, str]]:
    """Returns (skill_name, skill_md) or None. Two LLM calls (proposer, generator)."""
    fail_block = "\n".join(                       # analyse the whole failure BATCH's shared pattern
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
    justification = decision

    existing = (f"You are EDITING the existing skill (preserve its intent):\n"
                f"{skills.get(name,'')}\n" if action == "edit" and name in skills else "")
    body = complete(_GENERATOR_TMPL.format(
        justification=justification, existing=existing)).strip()
    return (name, body) if body else None


# ===========================================================================
# Bounded top-K frontier (registry/manager.py:update_frontier)
# ===========================================================================


@dataclass
class Frontier:
    max_size: int = 3
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


# ===========================================================================
# Dataset: OfficeQA (real, HF-gated) with a bundled-sample fallback
# ===========================================================================


def _load_bundled_sample() -> List[dict]:
    text = fetch_text(f"{RAW}/officeqa_sample.csv", cache_subdir="officeqa",
                      filename="officeqa_sample.csv")
    return list(csv.DictReader(text.splitlines()))


def load_officeqa() -> List[dict]:
    """Full OfficeQA if HF_TOKEN grants access; else the bundled 12-row sample."""
    rows = load_gated_hf("databricks/officeqa", "test") or _load_bundled_sample()
    # normalise the fields the loop needs.
    out = []
    for r in rows:
        out.append({"uid": r.get("uid", ""), "question": r["question"],
                    "answer": str(r["answer"]),
                    "source_files": r.get("source_files", ""),
                    "difficulty": r.get("difficulty", "easy")})
    return out


def fetch_docs(items: List[dict]) -> Dict[str, str]:
    docs: Dict[str, str] = {}
    for name in sorted({it["source_files"] for it in items if it["source_files"]}):
        try:
            docs[name] = fetch_text(f"{RAW}/treasury_bulletins/{name}",
                                    cache_subdir="officeqa", filename=name)
        except Exception:  # noqa: BLE001 - a missing doc shouldn't abort loading
            continue
    return docs


def stratified_split(items: List[dict], train_ratio: float = 0.5,
                     seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Category-stratified train/val split (data_utils.stratified_split)."""
    import random
    by_cat: Dict[str, List[dict]] = {}
    for it in items:
        by_cat.setdefault(it["difficulty"], []).append(it)
    rng = random.Random(seed)
    train, val = [], []
    for cat, group in by_cat.items():
        rng.shuffle(group)
        cut = max(1, int(len(group) * train_ratio))
        train.extend(group[:cut])
        val.extend(group[cut:] or group[:1])
    return train, val


# ===========================================================================
# Running EvoSkill THROUGH evolve() -- skill-library strategy + top-K frontier
# ===========================================================================
#
# Like ACE/GEPA, EvoSkill plugs into `evolve()`: a `Strategy` turns a proposed
# `SKILL.md` into a `Diff` on the skill library, and a custom `aggregator_factory`
# is the bounded top-K frontier (parent = the best member, which the aggregator
# makes the dev head so the next round extends it). Shared state (frontier,
# feedback history, recent failures, docs) lives in one `EvoSkillContext`.


@dataclass
class EvoSkillContext:
    docs: Dict[str, str]
    frontier: Frontier
    feedback: List[str] = field(default_factory=list)
    recent_failures: List[Tuple[str, str, str]] = field(default_factory=list)
    seed_score: float = 0.0
    best_score: float = 0.0
    eval_concurrency: int = 8     # parallelise the aggregator's held-out (val) re-eval
    batch_size: int = 4           # failures per induction BATCH (repo is batch-level, not per-trajectory)
    val_every: int = 3            # async SGD: validate the head every N applied skill updates (else roll back)
    last_propose_at: int = 0      # index into recent_failures at the last emitted skill
    lock: threading.Lock = field(default_factory=threading.Lock)   # workers run concurrently


def _parse_rendered_skills(rendered: str) -> Dict[str, str]:
    skills: Dict[str, str] = {}
    for chunk in rendered.split("### skill: ")[1:]:
        name, _, body = chunk.partition("\n")
        skills[name.strip()] = body.strip()
    return skills


def agent_answer(complete: Completion, docs: Dict[str, str],
                 rendered_skills: str, task: Task) -> str:
    doc = docs.get(task.meta["source_files"], "")
    context = retrieve_context(doc, task.prompt) if doc else "(document unavailable)"
    text = complete(_AGENT_TMPL.format(skills=rendered_skills, context=context,
                                       question=task.prompt))
    m = re.search(r"answer\s*[:=]\s*(.+)", text, re.IGNORECASE)
    return (m.group(1) if m else text).strip()


class SkillLibraryStrategy:
    """The artifact is a library of SKILL.md files; a proposal is `name :: body`."""

    def initial(self):
        return {}

    def render(self, state):
        return render_skills(state)

    def to_diff(self, state, proposal, author, base_version, target):
        name, _, body = proposal.partition(" :: ")
        name, body = name.strip(), body.strip()
        if not (name and body) or state.get(name) == body:
            return None
        return Diff(diff_id=f"{author}:{name}:{base_version}", target=target,
                    ops={name: body}, author=author)


def make_propose(ctx: EvoSkillContext, complete: Completion):
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
            ctx.recent_failures.append((task.prompt, output, task.meta["answer"]))
            if len(ctx.recent_failures) - ctx.last_propose_at < ctx.batch_size:
                return None                                # batch not full yet -> keep collecting
            ctx.last_propose_at = len(ctx.recent_failures)
            batch = list(ctx.recent_failures[-ctx.batch_size:])
        skills = _parse_rendered_skills(rendered)          # Proposer + Generator over the batch
        proposed = propose_and_generate(complete, skills, batch, ctx.feedback)
        if not proposed:
            return None
        name, body = proposed
        return f"{name} :: {body}"
    return propose


class TopKFrontierAggregator(AggregatorProtocol):
    """EvoSkill's optimizer: the bounded top-K aggregate frontier."""

    def __init__(self, ledger: Ledger, verifier, ctx: EvoSkillContext,
                 artifact_id: str = "skill_library"):
        self.ledger = ledger
        self.verifier = verifier
        self.ctx = ctx
        self.aid = artifact_id
        self.cards: List[EvidenceCard] = []
        self._seeded = False

    def ingest(self, card: EvidenceCard) -> None:
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

        cards, self.cards = self.cards, []
        for card in cards:
            candidate = head.apply(card.diff)
            score = self._eval(candidate)
            admitted = self.ctx.frontier.update(dict(candidate.state), score)
            self.ctx.best_score = max(self.ctx.best_score, score)
            self.ctx.feedback.append(
                f"{list(card.diff.ops)[0]}: {'admitted' if admitted else 'discarded'} "
                f"(val {score:.3f})")

        parent_state, _ = self.ctx.frontier.select_parent()   # strategy="best"
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


class SgdSkillAggregator(AggregatorProtocol):
    """Async SGD-style skill descent -- the *reasonable* async acceleration.

    The merger's job is only to **apply** each proposed skill as an update step
    (cheap, no val), so the head keeps moving and the workers build on it -- like
    taking gradient steps.  Held-out validation is **amortised**: only every
    ``val_every`` applied updates does it score the accumulated head; it keeps the
    mini-batch of updates iff val improved, otherwise it **rolls back** to the
    last validated checkpoint.  This costs ~``val_every``x fewer held-out evals
    than validating every candidate (the frontier), while keeping the faithful
    failure-driven skill-induction flow.  Used on the async path; the sync path
    keeps the strict per-candidate :class:`TopKFrontierAggregator` (the repo)."""

    def __init__(self, ledger: Ledger, verifier, ctx: EvoSkillContext,
                 artifact_id: str = "skill_library"):
        self.ledger = ledger
        self.verifier = verifier
        self.ctx = ctx
        self.aid = artifact_id
        self.cards: List[EvidenceCard] = []
        self._seeded = False
        self.checkpoint: Dict[str, str] = {}   # last val-confirmed skill library
        self.ckpt_score = 0.0
        self.steps = 0                          # applied updates since last validation

    def ingest(self, card: EvidenceCard) -> None:
        self.cards.append(card)

    def _eval(self, artifact) -> float:
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

    def _commit_state(self, state: Dict[str, str], msg: str) -> Optional[int]:
        """Set the DEV head to *exactly* ``state`` (apply() only merges, so a
        rollback that drops skills needs a full replacement artifact)."""
        snap = self.ledger.snapshot(Ledger.DEV)
        head = snap.get(self.aid)
        base_vv = {self.aid: snap.version.get(self.aid, 0)}
        art = EvolvingArtifact(self.aid, dict(state), head.version + 1,
                               head.blast_radius, head._rt, head._strategy)
        try:
            _, v = self.ledger.commit(art, base_vv, branch=Ledger.DEV, message=msg)
            return v
        except CASConflict:
            return None

    def step(self) -> List[MergeReport]:
        head = self.ledger.snapshot(Ledger.DEV).get(self.aid)
        if not self._seeded:
            self.ckpt_score = self._eval(head)              # base agent, no skills
            self.ctx.seed_score = self.ckpt_score
            self.ctx.best_score = self.ckpt_score
            self.checkpoint = dict(head.state)
            self._seeded = True

        cards, self.cards = self.cards, []
        if not cards:
            # Always report ckpt_score so the async runtime never falls back to a
            # full held-out re-eval (which would defeat eval-at-end / val_every).
            return [MergeReport(self.aid, None, False, 0, 0, 0, 0, self.ckpt_score, None, "idle")]

        # 1) APPLY the skill updates as SGD steps -- cheap, NO val. Commit so the
        #    workers immediately build on the moved head.
        working = head
        for card in cards:
            working = working.apply(card.diff)
            self.steps += 1
        committed = self._commit_state(dict(working.state), f"sgd: apply {len(cards)} skill update(s)")

        # 2) Validate only every ``val_every`` steps -> keep the batch or ROLL BACK.
        did_val = False
        score = None
        if self.steps >= self.ctx.val_every:
            did_val = True
            n_updates = self.steps
            score = self._eval(working)
            kept = score > self.ckpt_score
            if kept:                                        # improved -> new checkpoint
                self.checkpoint = dict(working.state)
                self.ckpt_score = score
            else:                                           # no gain -> roll back to checkpoint
                committed = self._commit_state(dict(self.checkpoint), "sgd: rollback (no val gain)")
            self.ctx.best_score = self.ckpt_score
            self.ctx.feedback.append(
                f"+{n_updates} update(s) {'kept' if kept else 'rolled back'} "
                f"(val {score:.3f} vs ckpt {self.ckpt_score:.3f})")
            self.steps = 0

        reason = (f"val {score:.3f} ckpt={self.ckpt_score:.3f}" if did_val
                  else f"applied {len(cards)} (val in {max(0, self.ctx.val_every - self.steps)} steps)")
        return [MergeReport(self.aid, None, False, len(cards), len(cards), 0, 0,
                            self.ckpt_score, committed, reason)]


@dataclass
class EvoResult:
    skills: Dict[str, str]
    seed_score: float
    best_score: float
    iterations: int


def run_evoskill(complete: Completion, docs: Dict[str, str],
                 train: List[dict], val: List[dict], iterations: int = 6,
                 max_frontier: int = 3, seed: int = 0, asynchronous: bool = False,
                 async_ratio: int = 3, max_seconds: float = 30.0, backend=None,
                 eval_concurrency: int = 8, batch_size: int = 4, val_every: int = 3,
                 eval_at_end: bool = False, verbose: bool = False) -> EvoResult:
    """Drive EvoSkill through `evolve()` (`val` is the held-out frontier metric).

    ``backend`` (an :class:`~agentdescent.backends.AgentBackend`) replaces the passive
    keyword-retriever base agent with a tool-using one (OpenHands / grep-loop) so
    the agent can actually navigate the source documents; ``None`` keeps the
    dependency-free retriever."""
    def to_task(i, it):
        return Task(id=f"oqa{i}", prompt=it["question"],
                    meta={"answer": it["answer"], "source_files": it["source_files"]})

    tasks = [to_task(i, it) for i, it in enumerate(list(train) + list(val))]
    # eval-at-end: apply every skill update during the run (no mid-run val or
    # roll back) and score the accumulated library on held-out once, at the end.
    if eval_at_end:
        val_every = 10 ** 9        # SGD merger never validates mid-run -> pure apply
    ctx = EvoSkillContext(docs=docs, frontier=Frontier(max_size=max_frontier),
                          eval_concurrency=eval_concurrency, batch_size=batch_size,
                          val_every=val_every)

    def run(rendered, task):
        if backend is not None:                    # tool-using base agent (OpenHands / grep-loop)
            return backend.answer(task.prompt, ctx.docs.get(task.meta["source_files"], ""),
                                  skills=rendered)
        return agent_answer(complete, ctx.docs, rendered, task)

    def reward(task, output):
        return score_multi_tolerance(output, task.meta["answer"])

    def factory(ledger, verifier, audit, config, policy):
        # Async path -> SGD-style skill descent (apply updates, validate every
        # val_every steps, roll back on no gain).  Sync path -> the strict
        # per-candidate top-K frontier (faithful to registry/manager.py).
        cls = SgdSkillAggregator if asynchronous else TopKFrontierAggregator
        return cls(ledger, verifier, ctx, artifact_id="skill_library")

    workers = min(3, len(train))
    result = evolve(tasks, reward, run=run, propose=make_propose(ctx, complete),
                    strategy=SkillLibraryStrategy(), blast_radius=0.2,
                    artifact_id="skill_library", rounds=iterations,
                    n_workers=workers, max_concurrency=workers,
                    asynchronous=asynchronous, async_ratio=async_ratio,
                    max_seconds=max_seconds if asynchronous else None,
                    held_out_frac=len(val) / max(1, len(tasks)),
                    self_verify=False,   # repo evaluates the child on val only -- no per-trajectory re-run
                    aggregator_factory=factory, verbose=verbose)

    best = ctx.best_score
    if eval_at_end and val:
        # single held-out eval of the FINAL accumulated skill library (concurrent).
        val_tasks = tasks[len(train):]
        rendered = SkillLibraryStrategy().render(dict(result.state))
        from concurrent.futures import ThreadPoolExecutor
        n = max(1, min(eval_concurrency, len(val_tasks)))
        with ThreadPoolExecutor(max_workers=n) as pool:
            scores = list(pool.map(lambda t: reward(t, run(rendered, t)), val_tasks))
        best = sum(scores) / len(scores)
        if verbose:
            print(f"\n[eval-at-end] final held-out score over {len(val_tasks)} items: {best:.3f}")
    return EvoResult(dict(result.state), ctx.seed_score, best, iterations)


# ===========================================================================
# Dataset + test evaluation
# ===========================================================================


def load_dataset(seed: int = 0, ratios=(0.5, 0.25, 0.25)) -> Dataset:
    """OfficeQA items split train/val/test, stratified by difficulty."""
    return split_dataset(load_officeqa(), ratios=ratios, seed=seed,
                         stratify_key=lambda it: it["difficulty"], name="OfficeQA")


def evaluate(complete: Completion, docs: Dict[str, str], skills: Dict[str, str],
             items: List[dict], backend=None) -> float:
    """Mean multi-tolerance score of a skill library on a held-out split."""
    if not items:
        return 0.0
    rendered = render_skills(skills)
    total = 0.0
    for it in items:
        task = Task(id=it["uid"], prompt=it["question"],
                    meta={"answer": it["answer"], "source_files": it["source_files"]})
        pred = (backend.answer(it["question"], docs.get(it["source_files"], ""), skills=rendered)
                if backend is not None else agent_answer(complete, docs, rendered, task))
        total += score_multi_tolerance(pred, it["answer"])
    return total / len(items)


# ===========================================================================
# main
# ===========================================================================


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", default="claude",
                   choices=["claude", "openai", "glm"], help="claude, or any OpenAI-compatible endpoint (DeepSeek, GLM, vLLM, ...) via OPENAI_BASE_URL + OPENAI_API_KEY; 'glm' is a legacy alias")
    p.add_argument("--model", default="claude-haiku-4-5")
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--frontier", type=int, default=3)
    p.add_argument("--backend", default="retrieval",
                   choices=["retrieval", "toolloop", "openhands"],
                   help="base agent: passive keyword retriever (default), a local "
                        "grep/read ReAct loop, or a real OpenHands tool-using agent")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--async", dest="asynchronous", action="store_true",
                   help="run barrier-free (async_evolve)")
    p.add_argument("--async-ratio", type=int, default=3, help="staleness lag budget")
    p.add_argument("--max-seconds", type=float, default=40.0, help="async wall-clock budget")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args()

    print("Algorithm: EvoSkill -- failure-driven skill discovery (top-K frontier)")
    print("Dataset  : OfficeQA (U.S. Treasury Bulletins, deterministic numeric scorer)")
    ds = load_dataset(seed=args.seed)
    docs = fetch_docs(ds.train + ds.val + ds.test)
    art = EvolvingArtifact("skill_library", blast_radius=0.2)
    ntr, nva, nte = ds.sizes()
    print(f"Governance: skill library blast_radius={art.blast_radius} -> {classify(art).name}")
    src = "full HF OfficeQA" if len(ds) > 12 else "bundled 12-row sample (HF gated)"
    print(f"Loaded   : {len(ds)} items ({src}); {ntr} train / {nva} val / {nte} test; "
          f"{len(docs)} docs")
    print("\nExample problem:")
    print("  Q:", ds.train[0]["question"][:150])
    print("  A:", ds.train[0]["answer"], f"(difficulty={ds.train[0]['difficulty']})")

    calls = args.iterations * (3 + 2 + nva) + nte
    print(f"\nPlan     : model={args.model}, iterations={args.iterations}, frontier={args.frontier}")
    if args.asynchronous:
        print(f"Async    : {min(3, ntr)} workers, barrier-free (async_ratio={args.async_ratio}, "
              f"max {args.max_seconds:.0f}s); the frontier rebases/discards stale skills")
    else:
        print(f"Parallel : {min(3, ntr)} workers run concurrently each round "
              f"(synchronous DP; the frontier merge is the barrier)")
    print(f"Budget   : up to ~{calls} model calls")

    if args.dry_run:
        print("\n[dry-run] not calling the API. Drop --dry-run to discover skills.")
        return

    if not args.yes and sys.stdin.isatty():
        if input("\nProceed with real API calls? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted.")
            return

    completion = (openai_compatible(model=args.model) if args.provider in ("openai", "glm")
                  else claude(model=args.model))
    try:
        completion("Reply with the single word: ok")
    except Exception as e:  # noqa: BLE001
        print(f"\nCould not reach the model ({type(e).__name__}: {e}).")
        return

    # base agent: passive retriever (default), a local grep/read loop, or OpenHands.
    backend = None
    if args.backend == "openhands":
        from agentdescent.backends import openhands_backend
        oh_model = args.model if args.model.startswith("openai/") else f"openai/{args.model}"
        base = ("https://api.deepseek.com" if args.provider in ("openai", "glm")
                else os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        backend = openhands_backend(model=oh_model, base_url=base)
        print(f"Backend  : real OpenHands agent (terminal + file_editor) on {oh_model}")
    elif args.backend == "toolloop":
        from agentdescent.backends import tool_loop_backend
        backend = tool_loop_backend(completion)
        print("Backend  : local grep/read ReAct loop")

    print("\nDiscovering skills (failure analysis + top-K frontier, L2)...\n")
    result = run_evoskill(completion, docs, ds.train, ds.val, iterations=args.iterations,
                          max_frontier=args.frontier, seed=args.seed,
                          asynchronous=args.asynchronous, async_ratio=args.async_ratio,
                          max_seconds=args.max_seconds, backend=backend, verbose=True)

    test_score = evaluate(completion, docs, result.skills, ds.test, backend=backend)
    print("\n=== discovered skill library ===")
    print(render_skills(result.skills))
    print(f"\nval score : {result.seed_score:.3f} -> {result.best_score:.3f}")
    print(f"test score: {test_score:.3f}  (held out, never seen by the frontier)")
    print(f"skills discovered: {len(result.skills)}")


if __name__ == "__main__":
    main()
