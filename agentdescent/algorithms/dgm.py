"""DGM's keep-all archive and parent selection, shipped as a named algorithm (#384).

The mechanism is the Darwin Gödel Machine's open-ended search: every
self-modification is archived (stepping stones stay), each child is scored on
the held-out set, and the next parent is sampled by ``sigmoid(score) *
1/(1+children)`` rather than always extending the best average. The example
keeps its coding-agent domain and Docker harness; this registry entry runs the
same archive on a spec's own artifact and data.

Use it from a spec as ``"algorithm": "dgm"`` (``kind: text``). The artifact is
the capability list the spec's ``run`` sees (one comma-separated line); the
spec's ``reflect`` model proposes one new capability from each failure.
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..agents import Completion
from ..aggregator import AggregatorProtocol, MergeReport
from ..evolvable import Diff, EvidenceCard
from ..ledger import CASConflict, Ledger
from ..selection import Archive, Candidate, SelectionContext
from . import Algorithm, BuildContext, Overrides, Param, register

__all__ = ["DGMArchiveAggregator", "DGMContext", "HarnessStrategy",
           "dgm_aggregator_factory", "make_propose"]

_PROPOSE_TMPL = """You are improving a task-solving agent by adding ONE capability.

Current capabilities: {capabilities}

A task failed (reward {reward:.2f}):
{prompt}
The agent answered:
{output}

Propose ONE short, general capability that would help with this class of task.
Output only the capability name/phrase (no quotes, no explanation)."""


def _caps_of(rendered: str) -> set:
    return {c.strip() for c in rendered.split(",") if c.strip()}


class HarnessStrategy:
    """The artifact is the harness's capability set; a proposal adds one capability."""

    def __init__(self, seed: str = "") -> None:
        self.seed = seed

    def initial(self) -> Dict[str, str]:
        return {"capabilities": self.seed}

    def render(self, state: Dict[str, str]) -> str:
        return state.get("capabilities", self.seed)

    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]:
        cap = ((proposal or "").strip().splitlines()[0]
               .replace(",", " ").strip(" -`\"'")[:120] if proposal else "")
        current = _caps_of(state.get("capabilities", self.seed))
        if not cap or cap in current:
            return None
        new = ",".join(sorted(current | {cap}))
        return Diff(diff_id=f"{author}:{cap}:{base_version}", target=target,
                    ops={"capabilities": new}, author=author)


@dataclass
class _Agent:
    capabilities: Tuple[str, ...]
    score: float = 0.0
    parent: Optional[int] = None
    children: int = 0
    generation: int = 0

    def key(self) -> str:
        return ",".join(sorted(self.capabilities))


@dataclass
class DGMContext:
    rng: random.Random
    archive_mode: str = "keep_all"
    archive: List[_Agent] = field(default_factory=list)
    seen: set = field(default_factory=set)
    seed_score: float = 0.0
    best_score: float = 0.0
    head_index: int = 0


def make_propose(complete: Completion, *, threshold: float = 0.8):
    """One capability per genuine failure, or ``None`` when the agent succeeded."""
    def propose(rendered, task, output, reward):
        if reward >= threshold:
            return None
        return complete(_PROPOSE_TMPL.format(
            capabilities=rendered or "(none)",
            reward=reward, prompt=task.prompt, output=output)).strip()
    return propose


class DGMArchiveAggregator(AggregatorProtocol):
    """DGM's optimizer: keep-all archive + sigmoid×novelty parent selection."""

    def __init__(self, ledger: Ledger, verifier, ctx: DGMContext,
                 artifact_id: str = "artifact",
                 selection: Optional[Archive] = None):
        self.ledger = ledger
        self.verifier = verifier
        self.ctx = ctx
        self.selection = selection or Archive(
            sampling="sigmoid_novelty", rng=ctx.rng)
        self.aid = artifact_id
        self.cards: List[EvidenceCard] = []
        self._lock = threading.Lock()
        self._seeded = False

    def ingest(self, card: EvidenceCard) -> None:
        with self._lock:
            self.cards.append(card)

    def _score(self, artifact) -> float:
        tasks = self.verifier.held_out
        return artifact.score(tasks) if tasks else 0.0

    def step(self) -> List[MergeReport]:
        snap = self.ledger.snapshot(Ledger.DEV)
        head = snap.get(self.aid)
        base_vv = {self.aid: snap.version.get(self.aid, 0)}
        if not self._seeded:
            seed_caps = _caps_of(head.state.get("capabilities", ""))
            seed_agent = _Agent(tuple(sorted(seed_caps)), parent=None,
                                generation=0)
            seed_agent.score = self._score(head)
            self.ctx.archive.append(seed_agent)
            self.ctx.seen.add(seed_agent.key())
            self.ctx.seed_score = self.ctx.best_score = seed_agent.score
            self._seeded = True

        parent_idx = self.ctx.head_index
        parent = self.ctx.archive[parent_idx]
        with self._lock:
            cards, self.cards = self.cards, []
        for card in cards:
            child_caps = tuple(sorted(_caps_of(
                card.diff.ops.get("capabilities", ""))))
            child = _Agent(child_caps, parent=parent_idx,
                           generation=parent.generation + 1)
            if not child_caps or child.key() in self.ctx.seen:
                continue
            candidate = head.apply(card.diff)
            child.score = self._score(candidate)
            self.ctx.seen.add(child.key())
            if self.ctx.archive_mode == "keep_better" and child.score <= parent.score:
                continue
            self.ctx.archive.append(child)
            self.ctx.best_score = max(self.ctx.best_score, child.score)
        parent.children += 1

        rows = [Candidate(artifact_id=self.aid, version=i,
                          score=a.score, selected=a.children)
                for i, a in enumerate(self.ctx.archive)]
        ctx = SelectionContext(head=rows[self.ctx.head_index],
                               candidates=tuple(rows), n_workers=1)
        self.ctx.head_index = self.selection.select(ctx, 1)[0].version
        target = ",".join(sorted(self.ctx.archive[self.ctx.head_index].capabilities))
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
                            f"archive={len(self.ctx.archive)} "
                            f"best={self.ctx.best_score:.3f}",
                            category="dgm-select")]


    def finalize(self) -> None:
        """Leave the archive's best agent on dev, not the last sampled parent.

        ``step()`` samples a parent every round; the engine returns the dev head,
        so without this a spec/CLI/MCP caller would get a random draw while the
        archive held the better agent. Same hook GEPA uses.
        """
        if not self.ctx.archive:
            return
        best = max(self.ctx.archive, key=lambda a: a.score)
        target = ",".join(sorted(best.capabilities))
        try:
            snap = self.ledger.snapshot(Ledger.DEV)
        except Exception:  # noqa: BLE001 - finalize is a courtesy, never fatal
            return
        head = snap.get(self.aid)
        if head is None or head.state.get("capabilities") == target:
            return
        base_vv = {self.aid: snap.version.get(self.aid, 0)}
        try:
            self.ledger.commit(
                head.apply(Diff(diff_id="dgm-final", target=self.aid,
                                ops={"capabilities": target}, author="dgm")),
                base_vv, branch=Ledger.DEV,
                message=f"dgm: final best (score {best.score:.3f})")
        except CASConflict:
            pass


def dgm_aggregator_factory(ctx: DGMContext, *, artifact_id: str = "artifact"):
    holder = {}

    def factory(ledger, verifier, audit, config, policy):
        agg = DGMArchiveAggregator(ledger, verifier, ctx,
                                   artifact_id=artifact_id)
        holder["agg"] = agg
        return agg
    factory.holder = holder  # type: ignore[attr-defined]
    return factory


def _build(ctx: BuildContext, p: Dict[str, object]) -> Overrides:
    from ..evolvespec import _read_text_target
    seed = _read_text_target(str(ctx.spec.target))
    context = DGMContext(rng=random.Random(int(p["seed"])),
                          archive_mode=str(p["archive_mode"]))
    knobs: Dict[str, object] = {
        "strategy": HarnessStrategy(seed=seed),
        "initial_state": {"capabilities": seed},
        "aggregator_factory": dgm_aggregator_factory(
            context, artifact_id=ctx.spec.artifact_id()),
    }
    if callable(ctx.reflect):
        knobs["propose"] = make_propose(
            ctx.reflect, threshold=float(p["failure_threshold"]))
    return Overrides(
        knobs=knobs,
        notes=["algorithm dgm: a keep-all capability archive with DGM's "
               "sigmoid×novelty parent draw; the spec's run/score judge each "
               "candidate on the held-out set. The example's staged evaluator "
               "and Docker harness stay example-only."])


register(Algorithm(
    name="dgm",
    summary="Darwin Gödel Machine's open-ended archive: keep every capability "
            "self-modification and sample the next parent by sigmoid(score) × "
            "1/(1+times chosen).",
    kinds=("text",),
    fidelity="mechanism_microport",
    params=(
        Param("seed", "int", 0, "seed for the parent draw"),
        Param("archive_mode", "str", "keep_all",
              "keep_all keeps stepping stones; keep_better drops a child that "
              "does not beat its parent",
              choices=("keep_all", "keep_better")),
        Param("failure_threshold", "float", 0.8,
              "only rollouts below this reward induce a capability",
              minimum=0.0, maximum=1.0),
    ),
    build=_build,
    source="Darwin Gödel Machine (Zhang et al., 2025); see docs/algo-dgm.md",
    extra_calls="one reflection call per failed rollout; each candidate is "
                "scored on the held-out set",
    doc="algo-dgm.md"))
