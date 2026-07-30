"""The Aggregator: an optimizer over a discrete artifact space.

This is where the central analogy deliberately breaks (design doc, section 1):
gradients add, diffs do not.  Aggregation is therefore not averaging but
**conflict resolution + statistical acceptance + transactional commit**.

The per-bucket pipeline (design doc, section 4):

    1. trigger + bucket        (section 4.1)
    2. staleness filter        (section 4.2)  per-diff eta, three-way split
       + rebase & cheap re-verify
    3. conflict resolution     (section 4.3)  semantic contradiction / PCGrad-drop
       + candidate fusion tournament          (model-soup style)
    4. Beta-posterior accept   (section 4.4)  P(delta > 0) > 1 - delta
    5. CAS / 2PC commit        (section 4.1)
    6. dual-branch promotion   (section 4.5)  EMA-style dev -> stable
    7. audit submission        (section 5.3)  the optimizer audits itself
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import (Callable, Deque, Dict, List, Optional, Protocol, Tuple,
                    runtime_checkable)

from .evolvable import (
    ContractError, Diff, EvidenceCard, Evolvable, VersionVector, stable_hash,
    vv_staleness,
)
from .governance import Layer, classify, assert_mutable
from .ledger import CASConflict, Ledger
from .scheduler import AuditScheduler
from .staleness import StaleAction, StalenessPolicy, get_policy
from .stats import BetaPosterior, annealed_delta, prob_improvement
from .verifier import ThreeLayerVerifier


@runtime_checkable
class AggregatorProtocol(Protocol):
    """The contract a custom aggregator must satisfy to plug into ``evolve``.

    An aggregator is the framework's *optimizer*: it receives evidence cards
    (diffs + evidence) and, on ``step()``, decides what to merge into the shared
    ledger. Implement these two methods (see :class:`Aggregator` for the
    reference 7-stage pipeline) to swap in your own merge/acceptance logic."""

    def ingest(self, card: EvidenceCard) -> None: ...

    def step(self) -> List["MergeReport"]: ...


# Builds an aggregator from the runtime deps ``evolve`` owns. The default is the
# reference :class:`Aggregator`; pass your own to ``evolve(aggregator_factory=)``.
AggregatorFactory = Callable[
    [Ledger, ThreeLayerVerifier, AuditScheduler, "AggregatorConfig", StalenessPolicy],
    AggregatorProtocol,
]


@dataclass
class AggregatorConfig:
    batch_trigger: int = 4          # B: hot artifacts trigger by batch size
    max_wait_rounds: int = 3        # T_max: cold artifacts trigger by time
    base_delta: float = 0.5         # acceptance risk (annealed by version)
    alpha_head: int = 5             # staleness tolerance for hot artifacts
    alpha_tail: int = 1             # staleness tolerance for cold artifacts
    trust_region_ops: int = 6       # max ops per diff (trust region)
    #: Max characters in any single op's value. The op-count trust region did not
    #: bound *size*, so one runaway proposal (a reflector that echoes its input,
    #: say) could commit a 500 KB value that then renders into every later prompt.
    #: Real ops in the shipped ports are ~2.5k chars, so this is ~12x headroom.
    trust_region_chars: int = 32_000
    promote_after_k: int = 3        # dev->stable survival rounds (EMA)


class AggregatorContractError(ContractError, TypeError):
    """A custom aggregator returned something ``step()`` may not return."""


def check_reports(reports, aggregator) -> List["MergeReport"]:
    """Validate what a custom ``step()`` handed back.

    Returning ``None`` or a list of the wrong thing otherwise surfaces as
    ``'NoneType' object is not iterable`` or ``'str' object has no attribute
    'committed_version'`` from inside the driver, with nothing naming the
    aggregator that caused it."""
    name = type(aggregator).__name__
    if reports is None:
        raise AggregatorContractError(
            f"{name}.step() returned None; it must return a list of MergeReport "
            "(return [] when nothing merged)")
    try:
        reports = list(reports)
    except TypeError:
        raise AggregatorContractError(
            f"{name}.step() returned {type(reports).__name__}, which is not "
            "iterable; it must return a list of MergeReport") from None
    for r in reports:
        if not hasattr(r, "committed_version"):
            raise AggregatorContractError(
                f"{name}.step() returned a {type(r).__name__} where a MergeReport "
                "was expected")
    return reports


@dataclass
class MergeReport:
    artifact_id: str
    accepted: Optional[Diff]
    fused: bool
    considered: int
    survived_staleness: int
    discarded_stale: int
    conflicts_dropped: int
    prob_improve: float
    committed_version: Optional[int]
    reason: str = ""


class EvidenceBuffer:
    """Cards bucketed by target artifact (design doc, section 4.1).

    Thread-safe: in the asynchronous runtime many worker threads call
    :meth:`add` concurrently while the aggregator thread calls :meth:`ready` /
    :meth:`drain`.  All bucket mutations are guarded by an internal lock."""

    #: Cap on the settled pool: at most this many cards, and at most this many
    #: characters of diff payload across them. It is a *diagnostic* ring, so old
    #: entries are evicted rather than allowed to accumulate -- see :meth:`settle`.
    SETTLED_MAX_CARDS = 256
    SETTLED_MAX_CHARS = 2_000_000

    def __init__(self) -> None:
        self._buckets: Dict[str, List[EvidenceCard]] = defaultdict(list)
        self._waited: Dict[str, int] = defaultdict(int)
        # (payload chars, card), oldest first -- see :meth:`settle` for the bound.
        self._settled: Deque[Tuple[int, EvidenceCard]] = deque()
        self._settled_chars = 0
        self._lock = threading.Lock()

    def add(self, card: EvidenceCard) -> None:
        with self._lock:
            self._buckets[card.diff.target].append(card)

    def tick(self) -> None:
        with self._lock:
            for aid in list(self._buckets):
                self._waited[aid] += 1

    def ready(self, config: AggregatorConfig) -> List[str]:
        with self._lock:
            out = []
            for aid, cards in self._buckets.items():
                if not cards:
                    continue
                if len(cards) >= config.batch_trigger or self._waited[aid] >= config.max_wait_rounds:
                    out.append(aid)
            return out

    def drain(self, artifact_id: str) -> List[EvidenceCard]:
        with self._lock:
            cards = self._buckets.pop(artifact_id, [])
            self._waited.pop(artifact_id, None)
            return cards

    def pending(self) -> int:
        with self._lock:
            return sum(len(c) for c in self._buckets.values())

    def settle(self, cards: List[EvidenceCard]) -> None:
        """Keep discarded-diff evidence addressable, under a hard bound.

        This is the structural advantage of artifacts over gradients (design
        doc, section 3.3): a stale gradient is simply lost, but a stale diff's
        evidence survives. Nothing in the library consumes the pool yet -- a
        fuller system would re-file the cards into the trajectory pool -- so
        treat it as a **diagnostic ring**, not a queue.

        It is bounded because it was not, and the unbounded version leaked
        precisely the payloads the trust region exists to reject: a runaway
        reflector's oversized diffs are settled here, and 500 of them retained
        250 MB that no code path could ever read. Oldest entries are evicted
        once either :attr:`SETTLED_MAX_CARDS` or :attr:`SETTLED_MAX_CHARS` is
        exceeded.
        """
        with self._lock:
            for card in cards:
                cost = sum(len(str(v)) for v in card.diff.ops.values())
                self._settled.append((cost, card))
                self._settled_chars += cost
            while self._settled and (
                len(self._settled) > self.SETTLED_MAX_CARDS
                or self._settled_chars > self.SETTLED_MAX_CHARS
            ):
                cost, _ = self._settled.popleft()
                self._settled_chars -= cost

    @property
    def settled(self) -> List[EvidenceCard]:
        """The retained tail of discarded evidence (bounded; see :meth:`settle`)."""
        with self._lock:
            return [c for _, c in self._settled]

    @property
    def settled_chars(self) -> int:
        """Diff payload currently retained by the settled pool."""
        with self._lock:
            return self._settled_chars


def diffs_conflict(a: Diff, b: Diff) -> bool:
    """Syntactic overlap: do two diffs edit an overlapping set of keys?

    A primitive for custom aggregators; the reference pipeline deliberately does
    **not** gate on this. Overlap alone is not a conflict -- two workers proposing
    the *same* value for a key are duplicates, and collapsing them is the point of
    content-addressing. Resolution keys on :func:`diffs_contradict` instead.
    """
    return bool(set(a.ops) & set(b.ops))


def diffs_contradict(a: Diff, b: Diff) -> bool:
    """Semantic contradiction: same key, different proposed value."""
    for k in set(a.ops) & set(b.ops):
        if a.ops[k] != b.ops[k]:
            return True
    return False


def fuse_diffs(diffs: List[Diff]) -> Diff:
    """Merge complementary (non-contradicting) diffs into one candidate.

    The model-soup analogy (design doc, section 4.3): the fusion of several
    local improvements often beats any single one -- but only when they share a
    base and do not contradict, which the caller guarantees."""
    ops: Dict = {}
    for d in diffs:
        ops.update(d.ops)
    ids = "+".join(sorted(d.diff_id for d in diffs))
    breaking = any(d.contract_breaking for d in diffs)
    return Diff(diff_id=f"fused({ids})", target=diffs[0].target, ops=ops,
                contract_breaking=breaking, author="aggregator")


class Aggregator:
    """Per-artifact optimizer step over the ledger."""

    def __init__(
        self,
        ledger: Ledger,
        verifier: ThreeLayerVerifier,
        audit: AuditScheduler,
        config: Optional[AggregatorConfig] = None,
        staleness_policy: Optional[StalenessPolicy] = None,
    ) -> None:
        self.ledger = ledger
        self.verifier = verifier
        self.audit = audit
        # swap Full / Guarded / Reflective without touching the merge pipeline.
        self.staleness_policy = staleness_policy or get_policy("guarded")
        self.config = config or AggregatorConfig()
        self.buffer = EvidenceBuffer()
        self._posteriors: Dict[str, BetaPosterior] = defaultdict(BetaPosterior)
        # dev-branch survival counter for EMA-style promotion.
        self._survival: Dict[str, int] = defaultdict(int)

    # -- staleness (section 4.2) ---------------------------------------------

    def _alpha_for(self, artifact: Evolvable, card: EvidenceCard) -> int:
        if card.diff.contract_breaking:
            return 0  # contract-breaking diffs must be re-proposed, not rebased.
        # hotter artifacts (bigger blast radius) tolerate more staleness.
        return self.config.alpha_head if artifact.blast_radius > 0.5 else self.config.alpha_tail

    def _staleness_filter(
        self, artifact: Evolvable, head: VersionVector, cards: List[EvidenceCard]
    ) -> Tuple[List[EvidenceCard], List[EvidenceCard]]:
        """Split cards into (survivors, discarded), delegating to the policy.

        The active :class:`~agentdescent.staleness.StalenessPolicy` (Full / Guarded
        / Reflective) decides ACCEPT / REBASE / DISCARD from ``eta`` and
        ``alpha``; the aggregator only executes the mechanical rebase-and-
        re-verify for the REBASE case."""
        survivors: List[EvidenceCard] = []
        discarded: List[EvidenceCard] = []
        for card in cards:
            eta = vv_staleness(head, card.base_version)
            alpha = self._alpha_for(artifact, card)
            action = self.staleness_policy.decide(eta, alpha, card.diff.contract_breaking)
            if action is StaleAction.ACCEPT:
                survivors.append(card if eta == 0 else card.rebased_onto(head))
            elif action is StaleAction.REBASE:
                # cheaply re-verify the delta still holds on the current head.
                candidate = artifact.apply(card.diff)
                if artifact.cheap_eval(card) <= candidate.cheap_eval(card):
                    survivors.append(card.rebased_onto(head))
                else:
                    discarded.append(card)
            else:  # DISCARD
                discarded.append(card)
        return survivors, discarded

    # -- conflict resolution (section 4.3) -----------------------------------

    def _resolve_conflicts(
        self, artifact: Evolvable, cards: List[EvidenceCard]
    ) -> Tuple[List[EvidenceCard], int]:
        """Drop contradicting diffs (PCGrad-style) and return surviving cards."""
        dropped = 0
        kept: List[EvidenceCard] = []
        for card in cards:
            # Resolve against everything already kept, and keep going after a win:
            # a card that displaces one survivor may contradict another (it can
            # touch several keys). Stopping at the first conflict left mutually
            # contradicting cards in `kept`, which then made the tournament's
            # "no contradictions" guard false and silently skipped the fusion.
            survivor: Optional[EvidenceCard] = card
            while survivor is not None:
                idx = next((i for i, k in enumerate(kept)
                            if diffs_contradict(survivor.diff, k.diff)), None)
                if idx is None:
                    break
                # project out the worse of the two on the held-out subset.
                d_score = self.verifier.cheap_eval(artifact.apply(survivor.diff))
                k_score = self.verifier.cheap_eval(artifact.apply(kept[idx].diff))
                dropped += 1
                if d_score > k_score:
                    kept.pop(idx)          # the newcomer wins; re-check the rest
                else:
                    survivor = None        # the newcomer loses; drop it
            if survivor is not None:
                kept.append(survivor)
        return kept, dropped

    # -- fusion tournament (section 4.3) -------------------------------------

    def _tournament(self, artifact: Evolvable, diffs: List[Diff]) -> Tuple[Diff, Evolvable, bool]:
        """Run candidates (plus their fusion) in a held-out tournament."""
        candidates: List[Tuple[Diff, Evolvable, bool]] = []
        for d in diffs:
            candidates.append((d, artifact.apply(d), False))
        # add a fused candidate if the survivors are mutually complementary.
        if len(diffs) > 1 and not any(
            diffs_contradict(a, b) for i, a in enumerate(diffs) for b in diffs[i + 1:]
        ):
            fused = fuse_diffs(diffs)
            candidates.append((fused, artifact.apply(fused), True))

        best = max(candidates, key=lambda c: self.verifier.cheap_eval(c[1]))
        return best

    # -- public entry points -------------------------------------------------

    def ingest(self, card: EvidenceCard) -> None:
        self.buffer.add(card)

    def step(self) -> List[MergeReport]:
        """Fire every artifact bucket that is ready and return per-artifact reports."""
        self.buffer.tick()
        reports: List[MergeReport] = []
        for aid in self.buffer.ready(self.config):
            reports.append(self._process(aid))
        return reports

    def _process(self, artifact_id: str) -> MergeReport:
        cards = self.buffer.drain(artifact_id)
        snap = self.ledger.snapshot(Ledger.DEV)
        artifact = snap.get(artifact_id)
        if artifact is None:
            return MergeReport(artifact_id, None, False, len(cards), 0, 0, 0, 0.0, None,
                               "unknown artifact")
        assert_mutable(artifact)  # L0 guard

        # trust-region: reject over-large diffs up front. They were previously
        # dropped *before* `considered` was computed and never settled, so they
        # vanished from both the report and the evidence pool -- silently.
        n_considered = len(cards)
        def _within_trust_region(card) -> bool:
            if card.diff.size() > self.config.trust_region_ops:
                return False
            return all(len(str(v)) <= self.config.trust_region_chars
                       for v in card.diff.ops.values())

        oversized = [c for c in cards if not _within_trust_region(c)]
        cards = [c for c in cards if _within_trust_region(c)]
        if oversized:
            self.buffer.settle(oversized)

        head = snap.version
        survivors, discarded = self._staleness_filter(artifact, head, cards)
        self.buffer.settle(discarded)
        if not survivors:
            return MergeReport(artifact_id, None, False, n_considered, 0,
                               len(discarded) + len(oversized), 0, 0.0, None,
                               "all stale / rejected")

        kept_cards, conflicts = self._resolve_conflicts(artifact, survivors)
        kept_diffs = [c.diff for c in kept_cards]
        best_diff, best_state, fused = self._tournament(artifact, kept_diffs)

        # -- Beta-posterior acceptance (section 4.4) -------------------------
        # Compare candidate vs base as two Beta posteriors over their held-out
        # success rate, with the running per-artifact posterior as a shared
        # prior (so evidence-starved tail artifacts get a wider, more
        # conservative test).  Accept iff P(candidate_rate > base_rate) clears
        # the annealed threshold -- not a point estimate crossing a line.
        prior = self._posteriors[artifact_id]
        base_s, base_f = self.verifier.eval_counts(artifact)
        cand_s, cand_f = self.verifier.eval_counts(best_state)
        base_score = self.verifier.cheap_eval(artifact)
        cand_score = self.verifier.cheap_eval(best_state)

        baseline_post = BetaPosterior(prior.successes + base_s, prior.failures + base_f)
        candidate_post = BetaPosterior(prior.successes + cand_s, prior.failures + cand_f)
        # fold each contributing worker's local before/after delta as extra
        # evidence for the candidate (the "gradient magnitude").
        for card in kept_cards:
            if set(card.diff.ops) & set(best_diff.ops):
                candidate_post.observe_delta(card.before_after_delta)

        delta = annealed_delta(self.config.base_delta, artifact.version)
        # Seed from (version, candidate) rather than the version alone. A shared
        # seed made the ~0.003 Monte-Carlo error identical for every candidate in
        # the round, so on knife-edge cases the draw decided them as a block --
        # one stream accepted every marginal diff, another rejected all of them.
        # stable_hash keeps this reproducible across processes.
        p_improve = prob_improvement(
            candidate_post, baseline_post,
            seed=stable_hash((artifact.version, best_diff.diff_id)) & 0x7FFFFFFF)

        # -- audit (section 5.3): the optimizer audits its own decision ------
        _, uncertainty = self.verifier.learned_eval(best_state)
        self.audit.submit(best_diff.diff_id, artifact_id, artifact.blast_radius,
                          uncertainty, payload=best_diff)
        if self.audit.force_oracle(artifact.blast_radius, artifact_id):
            oracle_base = self.verifier.oracle_eval(artifact)
            oracle_cand = self.verifier.oracle_eval(best_state)
            agreed = (oracle_cand > oracle_base) == (cand_score > base_score)
            self.audit.update_trust(artifact_id, agreed)
            if oracle_cand <= oracle_base:
                prior.observe_delta(oracle_cand - oracle_base)
                return MergeReport(artifact_id, None, fused, n_considered, len(survivors),
                                   len(discarded), conflicts, p_improve, None,
                                   "oracle rejected")

        if p_improve <= 1.0 - delta or cand_score < base_score:
            prior.observe_delta(cand_score - base_score)
            return MergeReport(artifact_id, None, fused, n_considered, len(survivors),
                               len(discarded), conflicts, p_improve, None,
                               f"P(delta>0)={p_improve:.2f} <= {1-delta:.2f}")

        # -- commit (section 4.1): CAS on dev --------------------------------
        base_vv = {artifact_id: head.get(artifact_id, 0)}
        try:
            _, new_version = self.ledger.commit(
                best_state, base_vv, branch=Ledger.DEV,
                message=f"merge {best_diff.diff_id} -> {artifact_id}",
            )
        except CASConflict:
            self.buffer.settle(survivors)
            return MergeReport(artifact_id, None, fused, n_considered, len(survivors),
                               len(discarded), conflicts, p_improve, None, "CAS conflict")

        prior.update(True, weight=2.0)  # a committed improvement is strong evidence.

        # -- dual-branch EMA promotion (section 4.5) -------------------------
        self._survival[artifact_id] += 1
        if self._survival[artifact_id] >= self.config.promote_after_k:
            self.ledger.promote_to_stable(artifact_id)
            self._survival[artifact_id] = 0

        return MergeReport(artifact_id, best_diff, fused, n_considered, len(survivors),
                           len(discarded), conflicts, p_improve, new_version,
                           "committed")
