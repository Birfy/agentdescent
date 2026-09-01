"""Flat-PUCT tree search over molecules, for porous molecular crystals.

Start from one molecule. Every expansion picks a node out of the whole tree by
PUCT, asks for **one deliberate modification** of it, validates the SMILES that
comes back, scores it against a five-criterion rubric, and appends it to the
tree -- valid or not, exactly as ERA's FUTS search appends a program that failed
to run.

    seed molecule
        |-- PUCT picks a node (rank + c_puct * P(s,a) * sqrt(N) / (1 + n))
        |-- a modification is proposed, and must come back as a SMILES
        |-- the SMILES is validated: parseable, sensible valences, neutral,
        |   closed shell, one molecule, under the atom cap
        |-- the molecule is scored on held-out weight profiles
        `-- the node is appended; visits backpropagate up the parent chain

What is shared with the rest of the repository, rather than written again here:

* :class:`agentdescent.selection.FlatPuct` is the selection rule, including the
  ``prior_exponent`` that turns ``P(s,a)`` from ERA's uniform ``1/N`` into a
  real prior;
* :func:`agentdescent.evolution.evolve` (or ``async_evolve``) supplies the
  workers, the ledger, evidence cards, staleness handling and the merge loop;
* the aggregator is replaced through ``aggregator_factory=``, which is the seam
  that lets the tree be the optimiser.

What is this example's own:

* :mod:`examples.porous._smiles` -- a dependency-free SMILES parser, kekuliser
  and validity gate, because every expansion has to decide whether what a model
  wrote is a molecule at all;
* :mod:`examples.porous._descriptors` and :mod:`examples.porous._score` -- the
  rubric: rigidity, symmetry, directional interaction sites, an open packing
  that is still competitive on lattice energy, and synthetic accessibility;
* :mod:`examples.porous._prior` -- the prior: structural headroom blended with
  the model's own rating of the direction, read out of the same reply;
* :mod:`examples.porous._mutations` -- rule-based edits, so ``--offline`` runs
  the whole search with no API key and gives the model-driven run a control arm.

The scoring rubric is a topological proxy, not a lattice energy. Every number
this prints is a proposal for a crystal structure prediction run, not a
substitute for one; ``docs/porous-molecules.md`` says where the proxy is weakest.

Run::

    python -m examples.porous.porous_tree_search --dry-run
    python -m examples.porous.porous_tree_search --offline --iterations 40 --workers 4
    python -m examples.porous.porous_tree_search --provider openai \\
        --model deepseek-v4-pro --iterations 24 --workers 4 --yes
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import tempfile
import threading
import time
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
from agentdescent.evolution import EvolutionResult, Task, evolve
from agentdescent.evolvable import Diff, EvidenceCard, vv_staleness
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
from examples.porous._mutations import propose_offline
from examples.porous._prior import Headroom, expansion_prior, structural_headroom
from examples.porous._prompts import extract_molecule, mutation_prompt, repair_prompt
from examples.porous._score import (
    DEFAULT_WEIGHTS,
    TERMS,
    ScoreReport,
    Weights,
    evaluate_smiles,
    parse_weights,
    weight_profiles,
)
from examples.porous._smiles import DEFAULT_ELEMENTS, similarity

ARTIFACT_ID = "porous_molecule"
TREE_UPDATED = "tree-updated"
NO_VALID_CANDIDATES = "no-valid-candidates"
DEFAULT_OUTPUT = Path("porous-tree-search-result.json")

#: Benzene. Small, rigid, symmetric, and useless as a porous crystal -- its
#: packing term is exactly 0.0, because a flat disc is the densest thing there
#: is. That is the point of the default: the search has to *find* the third
#: dimension rather than start with it.
DEFAULT_SEED = "c1ccccc1"


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
    """`-inf` is the failure sentinel and is not valid strict JSON."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


# ---------------------------------------------------------------------------
# The tree
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """One molecule in the search tree, valid or not."""

    index: int
    parent_index: Optional[int]
    smiles: str
    summary: str
    #: Mean score over the held-out weight profiles. `-inf` for a candidate the
    #: gate refused, which is what makes an invalid molecule a permanent dead
    #: end rather than a competitor.
    score: float
    report: Optional[ScoreReport] = None
    num_visits: int = 0
    iteration: int = 0
    #: `P(s,a)`. Headroom and the model's own rating, blended -- see
    #: `examples.porous._prior`.
    prior: Optional[float] = None
    headroom: Optional[Headroom] = None
    promise: Optional[float] = None
    duplicate_of: Optional[int] = None
    #: Multiset Tanimoto against the parent this was expanded from, radius 1.
    #: Reported, never gated on -- see `PorousTreeAggregator.step`.
    parent_similarity: Optional[float] = None
    profile_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.report is not None and self.report.ok

    def as_dict(self) -> Dict[str, Any]:
        report = self.report
        return {
            "index": self.index,
            "parent_index": self.parent_index,
            "iteration": self.iteration,
            "smiles": self.smiles,
            "change": self.summary,
            "score": _finite(self.score),
            "valid": self.valid,
            "reason": "" if self.valid else (report.reason if report else "no molecule"),
            "formula": report.formula if report and report.ok else "",
            "atom_count": (report.details.get("atom_count") if report and report.ok
                           else None),
            "terms": dict(report.terms) if report and report.ok else {},
            "num_visits": self.num_visits,
            "prior": self.prior,
            "promise": self.promise,
            "headroom": self.headroom.as_dict() if self.headroom else {},
            "duplicate_of": self.duplicate_of,
            "parent_similarity": self.parent_similarity,
            "profile_scores": dict(self.profile_scores),
        }


@dataclass
class MoleculeTree:
    """The node list, the PUCT rule over it, and the locking N workers need.

    Identical in shape to the ERA port's tree, and for the same reason: upstream
    FUTS backpropagates a visit *after* an expansion finishes, and with several
    proposals in flight that would let every worker pick the same node. A visit
    is reserved at selection instead, which is what the serial loop would have
    seen had those expansions already been dispatched, and is the same rule at
    one worker.
    """

    c_puct: float = 1.0
    #: ``0.0`` reproduces ERA exactly -- a uniform ``1/N`` prior. This example
    #: defaults to 1.0 because it *has* a prior worth using; see
    #: `examples.porous._prior` for what it is made of.
    prior_exponent: float = 1.0
    candidate_limit: Optional[int] = None
    nodes: List[Node] = field(default_factory=list)
    _next_iteration: int = 1
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _policy: FlatPuct = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._policy = FlatPuct(self.c_puct, self.prior_exponent)

    def seed(self, node: Node) -> Node:
        with self._lock:
            if self.nodes:
                return self.nodes[0]
            node.index = 0
            self.nodes.append(node)
            return node

    def _backpropagate_locked(self, node: Node) -> None:
        node.num_visits += 1
        if node.parent_index is not None:
            self._backpropagate_locked(self.nodes[node.parent_index])

    def select_parent(self) -> Optional[Tuple[int, Node]]:
        """The PUCT pick, with its visit reserved. ``None`` when the budget is spent."""
        with self._lock:
            if not self.nodes:
                raise RuntimeError("the molecule tree has not been seeded")
            iteration = self._next_iteration
            if self.candidate_limit is not None and iteration > self.candidate_limit:
                return None
            self._next_iteration += 1
            rows = tuple(
                Candidate(
                    artifact_id=ARTIFACT_ID,
                    version=node.index,
                    score=node.score if math.isfinite(node.score) else None,
                    selected=node.num_visits,
                    parent=node.parent_index,
                    prior=node.prior,
                )
                for node in self.nodes
            )
            ctx = SelectionContext(head=rows[0], candidates=rows, n_workers=1)
            chosen = self.nodes[self._policy.select(ctx, 1)[0].version]
            self._backpropagate_locked(chosen)
            return iteration, chosen

    def add_node(self, node: Node) -> Node:
        with self._lock:
            parent = node.parent_index
            if parent is None or not 0 <= parent < len(self.nodes):
                node.parent_index = 0
            node.index = len(self.nodes)
            node.num_visits = 1
            seen = {n.report.canonical: n.index for n in self.nodes
                    if n.report is not None and n.report.ok}
            if node.report is not None and node.report.ok:
                node.duplicate_of = seen.get(node.report.canonical)
            self.nodes.append(node)
            return node

    def node_at(self, index: int) -> Optional[Node]:
        with self._lock:
            return self.nodes[index] if 0 <= index < len(self.nodes) else None

    def children_of(self, index: int) -> List[Node]:
        with self._lock:
            return [n for n in self.nodes if n.parent_index == index]

    def best(self) -> Node:
        with self._lock:
            if not self.nodes:
                raise RuntimeError("the molecule tree is empty")
            return max(self.nodes, key=lambda node: node.score)

    def root(self) -> Node:
        with self._lock:
            return self.nodes[0]

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            depths = []
            for node in self.nodes:
                depth, cursor = 0, node
                while cursor.parent_index is not None:
                    cursor = self.nodes[cursor.parent_index]
                    depth += 1
                depths.append(depth)
            valid = [n for n in self.nodes if n.valid]
            return {
                "nodes": len(self.nodes),
                "valid_nodes": len(valid),
                "invalid_nodes": len(self.nodes) - len(valid),
                "duplicates": sum(1 for n in self.nodes if n.duplicate_of is not None),
                "max_depth": max(depths) if depths else 0,
                "root_visits": self.nodes[0].num_visits if self.nodes else 0,
                "c_puct": self.c_puct,
                "prior_exponent": self.prior_exponent,
                "tree": [node.as_dict() for node in self.nodes],
            }


# ---------------------------------------------------------------------------
# The strategy: the artifact is one molecule
# ---------------------------------------------------------------------------


class MoleculeStrategy:
    """One SMILES string, plus the bookkeeping a tree expansion carries."""

    def __init__(self, seed_smiles: str = DEFAULT_SEED) -> None:
        self.seed_smiles = seed_smiles

    def initial(self) -> Dict[str, str]:
        return {"smiles": self.seed_smiles, "change_summary": "seed molecule",
                "parent_index": "0", "promise": "", "iteration": "0"}

    def render(self, state: Dict[str, str]) -> str:
        return state.get("smiles", self.seed_smiles)

    def keys(self) -> Sequence[str]:
        return ("smiles", "change_summary", "parent_index", "promise", "iteration")

    def to_diff(self, state: Dict[str, str], proposal: str, author: str,
                base_version: int, target: str) -> Optional[Diff]:
        try:
            payload = json.loads(proposal)
            smiles = str(payload["smiles"]).strip()
            iteration = int(payload["iteration"])
            parent_index = int(payload["parent_index"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        # An unreadable reply still becomes a diff, and so still becomes a node
        # the gate refuses. Dropping it here would shrink the rank denominator
        # and quietly raise every later node's exploration term -- a different
        # search. It can never reach the ledger: `-inf` is never the best node.
        return Diff(
            diff_id=f"{author}:{abs(hash(smiles)) & 0xFFFFFFFF:08x}:{iteration}:{base_version}",
            target=target,
            ops={
                "smiles": smiles,
                "change_summary": str(payload.get("change_summary") or ""),
                "parent_index": str(parent_index),
                "promise": str(payload.get("promise") or ""),
                "iteration": str(iteration),
            },
            author=author,
        )


# ---------------------------------------------------------------------------
# Proposing: one deliberate modification of the node PUCT chose
# ---------------------------------------------------------------------------


def make_propose(
    tree: MoleculeTree,
    complete: Optional[Completion],
    *,
    max_atoms: int = 100,
    repair_attempts: int = 2,
    offline_seed: int = 0,
    counters: Optional[Dict[str, int]] = None,
) -> Callable[[str, Task, str, float], Optional[str]]:
    """The expansion step: PUCT picks the parent, the model writes the child.

    ``complete=None`` runs the rule-based operators instead, which is what
    ``--offline`` does.

    The **repair loop** is on by default here, unlike the ERA port, and the
    reason is specific to this domain: an invalid SMILES is not a failed idea,
    it is a typo -- an unclosed ring digit, a five-bonded carbon, an aromatic
    ring that will not kekulise. Turning that into a permanent `-inf` dead end
    throws away the chemistry along with the punctuation, so the gate's own
    reason goes straight back to the model and it gets another try.
    """
    tally = counters if counters is not None else {}

    def bump(key: str) -> None:
        tally[key] = tally.get(key, 0) + 1

    def propose(rendered: str, task: Task, output: str, reward: float) -> Optional[str]:
        selection = tree.select_parent()
        if selection is None:
            return None
        iteration, parent = selection
        best = tree.best()
        smiles, summary, promise = "", "", None

        if complete is None:
            rng = random.Random((offline_seed * 1_000_003) ^ (iteration * 7919))
            mutation = propose_offline(parent.smiles, rng, max_atoms=max_atoms)
            if mutation is not None:
                smiles, summary = mutation.smiles, mutation.summary
                bump(f"operator:{mutation.operator}")
            else:
                bump("offline:no-moves")
        else:
            siblings = [f"{child.smiles} -- {child.summary or 'no summary'}"
                        for child in tree.children_of(parent.index)]
            explain = parent.report.explain() if parent.report else "not scored"
            best_line = (f"{best.smiles} (score {best.score:.3f})"
                         if best.valid else "")
            prompt = mutation_prompt(
                parent.smiles, explain, max_atoms=max_atoms,
                elements=", ".join(sorted(DEFAULT_ELEMENTS)),
                siblings=siblings, best=best_line)
            for attempt in range(max(1, repair_attempts)):
                reply = complete(prompt)
                smiles, summary, found = extract_molecule(reply or "")
                promise = found if promise is None else promise
                if not smiles:
                    bump("reply:unreadable")
                check = evaluate_smiles(smiles, max_atoms=max_atoms) if smiles else None
                if check is not None and check.ok:
                    bump("repair:succeeded" if attempt else "reply:valid")
                    break
                if attempt == max(1, repair_attempts) - 1:
                    # Out of retries: the last draw goes forward and becomes the
                    # dead-end node it would have been on the first attempt. The
                    # loop buys attempts, it does not hide a failure.
                    bump("repair:gave-up")
                    break
                bump("repair:drawn")
                prompt = repair_prompt(
                    parent.smiles, smiles,
                    check.reason if check is not None else "no SMILES in the reply",
                    max_atoms=max_atoms)

        return json.dumps(
            {
                "smiles": smiles,
                "change_summary": summary,
                "iteration": iteration,
                "parent_index": parent.index,
                "promise": "" if promise is None else repr(promise),
            },
            separators=(",", ":"),
        )

    return propose


# ---------------------------------------------------------------------------
# Rollouts: one weight profile is one task
# ---------------------------------------------------------------------------


def build_tasks(count: int, seed: int = 0, *, jitter: float = 0.45,
                base: Weights = DEFAULT_WEIGHTS) -> List[Task]:
    """One task per weight profile -- the shards of a problem with no data in it.

    The rubric is deterministic, so there is nothing to sample and no train/test
    split in the usual sense. What there *is* to hold out is the **weighting**:
    a molecule that only wins under one exact set of five weights has not been
    shown to be good. Each task carries a perturbed weighting, ``evolve`` splits
    them into a train half and a held-out half, and profiles outside this list
    entirely are what the final number is reported on.
    """
    profiles = weight_profiles(count, seed=seed, jitter=jitter, base=base)
    return [
        Task(id=f"profile-{index}",
             prompt=("Score the candidate molecule under weighting "
                     + ", ".join(f"{term}={getattr(profile, term):.2f}"
                                 for term in TERMS)),
             meta={"profile": index, "weights": profile.as_dict()})
        for index, profile in enumerate(profiles)
    ]


def make_run(profiles: Sequence[Weights], *, max_atoms: int = 100
             ) -> Callable[[str, Task], str]:
    """A rollout is scoring the current molecule under one weight profile."""

    def run(rendered: str, task: Task) -> str:
        index = int(task.meta["profile"])
        report = evaluate_smiles(rendered, weights=profiles[index],
                                 max_atoms=max_atoms)
        return json.dumps(
            {"ok": report.ok, "score": report.total, "reason": report.reason,
             "terms": report.terms},
            separators=(",", ":"), default=str)

    return run


def reward_molecule(task: Task, output: str) -> float:
    """The engine's reward: the rubric total, already in [0, 1]."""
    try:
        payload = json.loads(output)
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0
    if not payload.get("ok"):
        return 0.0
    return max(0.0, min(1.0, float(payload.get("score", 0.0))))


# ---------------------------------------------------------------------------
# The aggregator: expand, score, append, commit
# ---------------------------------------------------------------------------


class PorousTreeAggregator(AggregatorProtocol):
    """The tree's expand-score-append step, as AgentDescent's merge optimizer."""

    def __init__(
        self,
        ledger: Ledger,
        verifier: Any,
        tree: MoleculeTree,
        config: AggregatorConfig,
        staleness_policy: Any,
        *,
        profiles: Sequence[Weights],
        seed_smiles: str = DEFAULT_SEED,
        max_atoms: int = 100,
        artifact_id: str = ARTIFACT_ID,
    ) -> None:
        self.ledger = ledger
        self.verifier = verifier
        self.tree = tree
        self.config = config
        self.staleness_policy = staleness_policy or get_policy("guarded")
        self.profiles = list(profiles)
        self.seed_smiles = seed_smiles
        self.max_atoms = max_atoms
        self.artifact_id = artifact_id
        self.cards: List[EvidenceCard] = []
        self._cards_lock = threading.Lock()
        self._seeded = False
        self.meter = None

    # -- scoring ------------------------------------------------------------

    def _held_out_profiles(self) -> Tuple[int, ...]:
        indices = tuple(sorted(int(task.meta["profile"])
                               for task in self.verifier.held_out))
        if not indices:
            raise RuntimeError("the porous search needs held-out weight profiles")
        return indices

    def evaluate(self, smiles: str) -> Tuple[ScoreReport, float, Dict[str, float]]:
        """Score on every held-out weighting; the node's score is their mean.

        A candidate the gate refuses scores ``-inf`` -- the sentinel FUTS uses
        for a program that would not run, and the reason a dead end is never
        selected again while still counting towards the rank denominator.
        """
        report = evaluate_smiles(smiles, max_atoms=self.max_atoms)
        if not report.ok:
            return report, -math.inf, {}
        per_profile = {
            f"profile-{index}": report.score_with(self.profiles[index])
            for index in self._held_out_profiles()
        }
        mean = sum(per_profile.values()) / len(per_profile)
        return report, round(mean, 6), per_profile

    def _lineage(self, parent_index: int, report: ScoreReport) -> Optional[float]:
        """How much of the parent survives in the child, as a diagnostic.

        Recorded, and deliberately **not** enforced. A floor on this would be
        the obvious way to hold the search to "modify the molecule you were
        given", and measurement says it cannot be: benzene ->
        hexakis(4-bromophenyl)benzene, the best molecule the live run found,
        scores 0.06 here, while benzene -> hexane, which shares nothing at all,
        scores 0.00. There is no threshold between "bold but legitimate" and
        "unrelated", because substituting every symmetry-equivalent position at
        once -- the move this rubric most wants -- rewrites every atom
        environment in the parent. The prompt asks for a modification; this
        number is how a reader of the result file checks that it got one.
        """
        parent = self.tree.node_at(parent_index)
        if not report.ok or report.validation is None or report.validation.molecule is None:
            return None
        if (parent is None or parent.report is None
                or parent.report.validation is None
                or parent.report.validation.molecule is None):
            return None
        return similarity(parent.report.validation.molecule,
                          report.validation.molecule, radius=1)

    def seed(self) -> None:
        if self._seeded:
            return
        self._seeded = True
        head = self.ledger.snapshot(Ledger.DEV).get(self.artifact_id)
        smiles = head.state.get("smiles", self.seed_smiles)
        report, score, per_profile = self.evaluate(smiles)
        if not report.ok:
            # Refused rather than seeded with a dead end: every child would be a
            # modification of something that is not a molecule.
            raise RuntimeError(
                f"the seed molecule {smiles!r} is not usable: {report.reason}")
        headroom = structural_headroom(report.descriptors, max_atoms=self.max_atoms)
        self.tree.seed(Node(
            index=0, parent_index=None, smiles=smiles, summary="seed molecule",
            score=score, report=report, headroom=headroom,
            prior=expansion_prior(headroom), profile_scores=per_profile))

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
                # Rebasing is safe here: a survivor is re-scored from its SMILES
                # before it can become a node, and its place in the tree is its
                # parent index, which a moving head cannot invalidate.
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
            smiles = ops.get("smiles", "")
            report, score, per_profile = self.evaluate(smiles)
            raw_promise = ops.get("promise")
            try:
                promise = float(raw_promise) if raw_promise not in (None, "") else None
            except (TypeError, ValueError):
                promise = None
            headroom = (structural_headroom(report.descriptors, max_atoms=self.max_atoms)
                        if report.ok and report.descriptors else None)
            parent_index = int(ops.get("parent_index", "0") or 0)
            self.tree.add_node(Node(
                index=-1,
                parent_index=parent_index,
                smiles=smiles,
                summary=ops.get("change_summary", ""),
                score=score,
                report=report,
                iteration=int(ops.get("iteration", "0") or 0),
                prior=expansion_prior(headroom, promise),
                headroom=headroom,
                promise=promise,
                parent_similarity=self._lineage(parent_index, report),
                profile_scores=per_profile,
            ))
            valid_candidates += int(report.ok)

        best = self.tree.best()
        accepted: Optional[Diff] = None
        committed_version: Optional[int] = None
        category = TREE_UPDATED
        if not survivors:
            category = MergeOutcome.ALL_STALE.value
        elif not valid_candidates:
            category = NO_VALID_CANDIDATES
        if best.valid and best.smiles != head.state.get("smiles"):
            accepted = Diff(
                diff_id=f"tree-best:{best.index}:{head.version}",
                target=self.artifact_id,
                ops={
                    "smiles": best.smiles,
                    "change_summary": best.summary,
                    "parent_index": str(best.parent_index or 0),
                    "promise": "" if best.promise is None else repr(best.promise),
                    "iteration": str(best.iteration),
                },
                author="porous-tree",
            )
            try:
                _, committed_version = self.ledger.commit(
                    head.apply(accepted), base_vv, branch=Ledger.DEV,
                    message="porous: commit the best node in the tree")
                category = MergeOutcome.COMMITTED.value
            except CASConflict:
                accepted = None
                category = MergeOutcome.CAS_CONFLICT.value

        return [
            MergeReport(
                self.artifact_id, accepted, False, len(cards), len(survivors),
                len(discarded), 0,
                max(0.0, best.score) if math.isfinite(best.score) else 0.0,
                committed_version,
                (f"valid={valid_candidates}/{len(survivors)} "
                 f"nodes={len(self.tree.nodes)} "
                 f"best={best.smiles} ({best.score:.3f})"),
                category,
            )
        ]


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass
class PorousRun:
    mode: str
    result: EvolutionResult
    tree: MoleculeTree
    profiles: List[Weights]
    test_profiles: List[Weights]
    wall_seconds: float
    counters: Dict[str, int] = field(default_factory=dict)

    def _test_scores(self, node: Node) -> Dict[str, Any]:
        """The held-back weightings -- never in a task, never seen by the search."""
        if not node.valid or node.report is None:
            return {"mean": None, "profiles": {}}
        scores = {f"test-{i}": node.report.score_with(profile)
                  for i, profile in enumerate(self.test_profiles)}
        return {"mean": round(sum(scores.values()) / len(scores), 6),
                "profiles": scores}

    def payload(self, config: Dict[str, Any], usage: Optional[Usage]) -> Dict[str, Any]:
        root, best = self.tree.root(), self.tree.best()
        return {
            "run": {
                "finished_at": _utc_now(),
                "mode": self.mode,
                "wall_seconds": round(self.wall_seconds, 3),
                "config": config,
            },
            "seed_molecule": root.as_dict(),
            "best_molecule": best.as_dict(),
            "gain": (round(best.score - root.score, 6)
                     if math.isfinite(best.score) and math.isfinite(root.score)
                     else None),
            "held_back_weightings": {
                "seed": self._test_scores(root),
                "best": self._test_scores(best),
            },
            "breakdown": best.report.as_dict() if best.report else {},
            "proposal_counters": dict(self.counters),
            "tree": self.tree.summary(),
            "engine": {
                "final_reward": self.result.final_reward,
                "outcomes": {str(k): v for k, v in self.result.outcomes().items()},
                "error": str(self.result.error) if self.result.error else None,
            },
            "usage": ({
                "calls": usage.calls, "failures": usage.failures,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "model_seconds": round(usage.seconds, 3),
            } if usage else {}),
        }


def run_search(
    complete: Optional[Completion],
    *,
    mode: str = "sync",
    seed_smiles: str = DEFAULT_SEED,
    iterations: int = 24,
    workers: int = 4,
    profiles: int = 8,
    test_profiles: int = 4,
    c_puct: float = 1.0,
    prior_exponent: float = 1.0,
    max_atoms: int = 100,
    weights: Weights = DEFAULT_WEIGHTS,
    jitter: float = 0.45,
    repair_attempts: int = 2,
    held_out_frac: float = 0.5,
    async_ratio: int = 1,
    max_seconds: float = 600.0,
    #: How long `--async` waits for expansions already in flight when the
    #: budget runs out. The engine's own default is two seconds, which is right
    #: when a rollout is a fast API call and wrong here: one expansion on a
    #: reasoning model takes minutes, so a two-second grace throws away work
    #: that was seconds from landing.
    shutdown_grace: float = 300.0,
    staleness: str = "guarded",
    eval_concurrency: Optional[int] = None,
    seed: int = 0,
    usage: Optional[Usage] = None,
    verbose: bool = False,
) -> PorousRun:
    """One search: ``iterations`` expansions of the tree, over ``workers`` workers."""
    if mode not in ("serial", "sync", "async"):
        raise ValueError("mode must be serial, sync, or async")
    if iterations < 1 or workers < 1 or iterations % workers:
        raise ValueError("iterations must be positive and divisible by workers")

    every = weight_profiles(profiles + test_profiles, seed=seed, jitter=jitter,
                            base=weights)
    search_profiles, held_back = every[:profiles], every[profiles:]
    tasks = build_tasks(profiles, seed, jitter=jitter, base=weights)
    tree = MoleculeTree(c_puct=c_puct, prior_exponent=prior_exponent,
                        candidate_limit=iterations)
    counters: Dict[str, int] = {}
    propose = make_propose(tree, complete, max_atoms=max_atoms,
                           repair_attempts=repair_attempts, offline_seed=seed,
                           counters=counters)

    def factory(ledger, verifier, audit, config, policy):
        aggregator = PorousTreeAggregator(
            ledger, verifier, tree, config, policy, profiles=search_profiles,
            seed_smiles=seed_smiles, max_atoms=max_atoms)
        aggregator.seed()
        return aggregator

    started = time.monotonic()
    common: Dict[str, Any] = {
        "run": make_run(search_profiles, max_atoms=max_atoms),
        "propose": propose,
        "strategy": MoleculeStrategy(seed_smiles),
        # A molecule is an L2 artifact: it is evaluated in-process by a pure
        # function, it executes nothing, and it cannot break the harness. The
        # ERA port is L1 because it runs generated code in a sandbox.
        "blast_radius": 0.2,
        "artifact_id": ARTIFACT_ID,
        "n_workers": workers,
        "self_verify": False,
        # Scoring is a pure function of the SMILES and takes microseconds, so
        # the held-out sweep is not the bottleneck any port-shaped default was
        # written for.
        "eval_concurrency": eval_concurrency if eval_concurrency else workers,
        "held_out_frac": held_out_frac,
        "solved_threshold": 1.0,
        "usage": usage,
        "aggregator_factory": factory,
        "verbose": verbose,
        "seed": seed,
        "staleness_policy": get_policy(staleness),
    }
    if mode == "async":
        result = async_evolve(tasks, reward_molecule, async_ratio=async_ratio,
                              max_seconds=max_seconds, max_iters=iterations,
                              shutdown_grace=shutdown_grace, **common)
    else:
        result = evolve(tasks, reward_molecule, rounds=iterations // workers,
                        max_concurrency=1 if mode == "serial" else workers,
                        **common)
    if verbose:
        report_engine(result)
    return PorousRun(mode, result, tree, search_profiles, held_back,
                     time.monotonic() - started, counters)


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_standard_args(parser, model_default=None,
                      model_help="model id; omit with --offline",
                      max_seconds_default=600.0, async_ratio_default=1,
                      eval_concurrency_default=None, include_val_cap=False)
    parser.add_argument("--iterations", type=int, default=24,
                        help="tree expansions in total (rounds = iterations // workers)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-smiles", default=DEFAULT_SEED,
                        help=f"the molecule the search starts from (default {DEFAULT_SEED})")
    parser.add_argument("--max-atoms", type=int, default=100,
                        help="hard cap on atoms including hydrogens (default 100)")
    parser.add_argument("--c-puct", type=float, default=1.0,
                        help="PUCT exploration constant (ERA's value is 1.0)")
    parser.add_argument("--prior-exponent", type=float, default=1.0,
                        help=("how sharply P(s,a) bends selection: 0 is ERA's "
                              "uniform 1/N prior, 1 uses the headroom-and-promise "
                              "prior, 2 squares it"))
    parser.add_argument("--weights", default="",
                        help=("override the rubric, e.g. "
                              "'rigidity=0.3,packing=0.3'; the rest keep their "
                              "defaults and all five are renormalised"))
    parser.add_argument("--profiles", type=int, default=8,
                        help="weight profiles the search is scored on (default 8)")
    parser.add_argument("--test-profiles", type=int, default=4,
                        help="weight profiles held back entirely (default 4)")
    parser.add_argument("--jitter", type=float, default=0.45,
                        help="log-normal spread of the perturbed weight profiles")
    parser.add_argument("--repair-attempts", type=int, default=2,
                        help=("draws per expansion: an invalid SMILES goes back "
                              "to the model with the gate's reason attached"))
    parser.add_argument("--shutdown-grace", type=float, default=300.0,
                        help=("--async only: how long to wait for expansions "
                              "already in flight when the budget runs out"))
    parser.add_argument("--offline", action="store_true",
                        help=("propose with the rule-based edit operators "
                              "instead of a model -- no API key, no network"))
    # Both defaults are set by what a reasoning model actually does here, not
    # by the size of the reply. One measured `deepseek-v4-pro` expansion took
    # 205 s and 14 000 tokens to produce three lines, because the thinking is
    # billed and timed like output: at 180 s three of four workers timed out in
    # the first round, and at a low `--max-tokens` the budget goes to hidden
    # reasoning and the reply arrives empty -- which this search would record as
    # a node the gate refused, indistinguishable from bad chemistry.
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--api-timeout", type=float, default=300.0)
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="a tree search wants varied children, not the mode")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quiet", action="store_true",
                        help="do not print the tree at the end")
    return parser


def _print_leaderboard(run: PorousRun, top: int = 8) -> None:
    ranked = sorted((n for n in run.tree.nodes if n.valid),
                    key=lambda n: n.score, reverse=True)[:top]
    print(f"\nTop {len(ranked)} of {len(run.tree.nodes)} nodes "
          f"({run.tree.summary()['invalid_nodes']} refused by the gate):")
    header = "  score  " + "".join(f"{term[:5]:>7s}" for term in TERMS) + "  molecule"
    print(header)
    for node in ranked:
        terms = "".join(f"{node.report.terms[t]:7.2f}" for t in TERMS)
        print(f"  {node.score:.3f}{terms}  {node.smiles}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.budget_rollouts:
        # This port's iteration flag already *is* the rollout budget:
        # rounds = iterations // workers, so the total is fixed as workers vary.
        args.iterations = args.budget_rollouts
    args.workers = worker_count(args, args.workers)
    weights = parse_weights(args.weights) if args.weights else DEFAULT_WEIGHTS
    mode = "async" if args.asynchronous else ("serial" if args.workers == 1 else "sync")

    print("Search   : porous molecular crystals -- flat-PUCT tree over molecules")
    print(f"Seed     : {args.seed_smiles}")
    print(f"Rubric   : " + ", ".join(f"{term}={getattr(weights.normalized(), term):.2f}"
                                     for term in TERMS))
    print(f"Proposer : " + ("rule-based edit operators (offline)" if args.offline
                            else f"{args.provider}/{args.model}"))
    print(f"Plan     : mode={mode}, iterations={args.iterations}, "
          f"workers={args.workers}, c_puct={args.c_puct}, "
          f"prior_exponent={args.prior_exponent}, max_atoms={args.max_atoms}")
    print(f"Profiles : {args.profiles} scored ({args.test_profiles} held back "
          f"entirely), jitter={args.jitter}")

    if args.dry_run:
        print("Data     : none to load -- the rubric is a pure function of the SMILES")
        print("Dry-run  : plan only; no model API was accessed.")
        return 0

    check = evaluate_smiles(args.seed_smiles, max_atoms=args.max_atoms)
    if not check.ok:
        print(f"The seed molecule is not usable: {check.reason}", file=sys.stderr)
        return 2
    print(f"Seed     : scores {check.total:.3f} -- " + ", ".join(
        f"{term}={check.terms[term]:.2f}" for term in TERMS))

    usage = Usage()
    complete: Optional[Completion] = None
    if not args.offline:
        if not args.model:
            print("--model is required unless --offline is given", file=sys.stderr)
            return 2
        if not confirm(args):
            return 1
        complete = completion_for(args, usage=usage, max_tokens=args.max_tokens,
                                  timeout=args.api_timeout,
                                  temperature=args.temperature)

    run = run_search(
        complete, mode=mode, seed_smiles=args.seed_smiles,
        iterations=args.iterations, workers=args.workers,
        profiles=args.profiles, test_profiles=args.test_profiles,
        c_puct=args.c_puct, prior_exponent=args.prior_exponent,
        max_atoms=args.max_atoms, weights=weights, jitter=args.jitter,
        repair_attempts=args.repair_attempts, async_ratio=args.async_ratio,
        max_seconds=args.max_seconds, shutdown_grace=args.shutdown_grace,
        eval_concurrency=args.eval_concurrency,
        seed=args.seed, usage=usage if not args.offline else None,
        verbose=not args.quiet,
    )

    root, best = run.tree.root(), run.tree.best()
    config = {
        "mode": run.mode, "iterations": args.iterations, "workers": args.workers,
        "seed_smiles": args.seed_smiles, "max_atoms": args.max_atoms,
        "c_puct": args.c_puct, "prior_exponent": args.prior_exponent,
        "weights": weights.normalized().as_dict(), "profiles": args.profiles,
        "test_profiles": args.test_profiles, "jitter": args.jitter,
        "repair_attempts": args.repair_attempts, "seed": args.seed,
        "shutdown_grace": args.shutdown_grace if args.asynchronous else None,
        "proposer": "offline-operators" if args.offline else f"{args.provider}/{args.model}",
    }
    payload = run.payload(config, None if args.offline else usage)
    _write_json(args.output, payload)

    print(f"\nSeed     : {root.smiles}  score {root.score:.3f}")
    print(f"Best     : {best.smiles}")
    if best.report is not None and best.report.ok:
        print(best.report.explain())
    held = payload["held_back_weightings"]
    if held["best"]["mean"] is not None and held["seed"]["mean"] is not None:
        print(f"Held-back weightings: seed {held['seed']['mean']:.3f} -> "
              f"best {held['best']['mean']:.3f}")
    if not args.quiet:
        _print_leaderboard(run)
    if run.counters:
        print(f"\nProposals: {dict(sorted(run.counters.items()))}")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
