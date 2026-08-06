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
    4. audit gate              (section 5.3)  high blast radius / low trust -> the
       oracle decides, and can veto here      optimizer audits itself
    5. Beta-posterior accept   (section 4.4)  P(delta > 0) > 1 - delta
    6. CAS / 2PC commit        (section 4.1)
    7. dual-branch promotion   (section 4.5)  EMA-style dev -> stable, after K
                                              regression-free rounds

The audit is stage **4**, not a post-commit spot-check. It runs before the
acceptance test and returns ``oracle-rejected`` outright, so it is a blocking gate
on the accept path -- the diagrams used to draw it after the commit with a dotted
"spot-check" arrow, which reads as advisory when it holds a veto.
"""

from __future__ import annotations

import random
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import (Callable, Deque, Dict, List, Optional, Protocol, Tuple,
                    runtime_checkable)

from .evolvable import (
    ContractError, Diff, EvidenceCard, Evolvable, VersionVector, vv_staleness,
)
from .defaults import (
    DefaultAcceptance, DefaultConflict, DefaultFusion, DefaultPromotion,
)
from .governance import Layer, classify, assert_mutable
from .ledger import CASConflict, Ledger
from .metrics import Meter
from .policies import (
    AcceptancePolicy, ConflictPolicy, FusionPolicy, MergeContext,
    PromotionPolicy,
)
from .scheduler import AuditScheduler
from .staleness import StaleAction, StalenessPolicy, get_policy
from .stats import BetaPosterior
from .verifier import ThreeLayerVerifier


@runtime_checkable
class AggregatorProtocol(Protocol):
    """The contract a custom aggregator must satisfy to plug into ``evolve``.

    An aggregator is the framework's *optimizer*: it receives evidence cards
    (diffs + evidence) and, on ``step()``, decides what to merge into the shared
    ledger. Implement these two methods (see :class:`Aggregator` for the
    reference 7-stage pipeline) to swap in your own merge/acceptance logic.

    **Threading contract.** ``ingest`` may be called concurrently from many worker
    threads; ``step`` is called from one. Guard anything both touch. This was
    unwritten, and the shipped examples all relied on ``list.append`` happening to
    be atomic under the GIL -- which stops being true the moment ``ingest`` grows a
    counter, a dict update or a dedup check, and is already not enough when
    ``evolve(round_timeout=)`` abandons a straggler that keeps running and can
    ``ingest`` in the middle of a ``step``'s drain. :class:`EvidenceBuffer` is the
    worked example."""

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
    #: Optional :class:`~agentdescent.advantage.AdaptiveTrustRegion`. `None` keeps
    #: the two constants above, which is what every measurement in this repository
    #: was taken with. Both of them are guesses -- the character cap exists because
    #: a comment records that the op count alone was not enough -- and a guess that
    #: fits a rule table fits a file tree in neither direction. Opt-in until an A/B
    #: says letting it move is better than leaving it still.
    trust_region_policy: Optional[Any] = None
    promote_after_k: int = 3        # dev->stable: regression-free ROUNDS (EMA)
    #: Version half-life of the acceptance risk. `base_delta` is exposed but the
    #: half-life that turns it into the actual threshold was a default argument
    #: buried in `stats.annealed_delta`, unreachable from here or from `evolve()` --
    #: and it sets the shape of the whole run: the threshold `P(delta>0)` must clear
    #: goes 0.505 at v1, 0.750 at v64, 0.875 at v128, and floors at 0.99. A caller
    #: wanting a flatter or steeper schedule had to monkeypatch the module.
    anneal_half_life: int = 64
    #: Monte-Carlo draws behind each acceptance decision. Also unreachable before.
    accept_samples: int = 4000
    #: How many times a losing commit rebases and tries again. 1 restores the old
    #: behaviour: settle the evidence back into the pool and wait a round, which
    #: is the wrong answer to "somebody committed first" as soon as there is more
    #: than one writer -- and with one writer this never fires at all.
    cas_attempts: int = 3
    #: Upper bound of the jittered backoff, doubling per attempt. Jittered because
    #: two writers backing off by the same amount collide again on the same
    #: schedule.
    cas_backoff: float = 0.05
    #: Rank candidates against their fusion on the cheap layer before putting one
    #: forward. **Off**, because the ranking is paid every round while the only
    #: decision it changes from the gate's is recoverable -- see
    #: :class:`~agentdescent.defaults.DefaultFusion` for the case analysis.
    #:
    #: Turn it on to *measure*: `best_single_score`, and therefore
    #: :attr:`~agentdescent.evolution.FusionStats.win_rate`, exist only when a
    #: single was actually scored. Ignored by a fusion policy that does its own
    #: thing, which :class:`~agentdescent.fusion.ReflectiveFusion` does.
    fusion_tournament: bool = False


class MergeOutcome(str, Enum):
    """The vocabulary of :attr:`MergeReport.category`.

    ``result.outcomes()`` is the framework's primary diagnostic -- the README, the
    quickstart and ``docs/evolution.md`` all point at it to answer "why did nothing
    commit?" -- and its keys were bare string literals with no declared vocabulary
    anywhere. To learn them you had to read this file and collect the sixth
    argument of six different ``MergeReport(...)`` constructions.

    Subclasses ``str``, so every existing dict key, comparison and format string
    keeps working: ``outcomes()["below-threshold"]`` is unchanged.
    """

    #: The candidate was merged into dev.
    COMMITTED = "committed"
    #: Proposals reached the gate and failed to beat the baseline. The reflector is
    #: the thing to look at.
    BELOW_THRESHOLD = "below-threshold"
    #: Nothing survived the staleness filter -- they never reached the gate. The
    #: lag budget is the thing to look at.
    ALL_STALE = "all-stale"
    #: The diff exceeded the trust region (too many ops, or one oversized value).
    #: Used to be folded into ``all-stale``, which pointed at the opposite fix.
    OVERSIZED = "oversized"
    #: The oracle overruled the cheap layer's verdict (see the audit gate).
    ORACLE_REJECTED = "oracle-rejected"
    #: Another writer committed first; the diff was re-filed for a later round.
    CAS_CONFLICT = "cas-conflict"
    #: The bucket named an artifact the ledger does not have.
    UNKNOWN_ARTIFACT = "unknown-artifact"

    def __str__(self) -> str:            # so f-strings render the value, not the name
        return self.value

    def __repr__(self) -> str:
        """Render as the plain string, which is what every reader is shown.

        ``outcomes()`` is a *dict*, and printing a dict uses ``repr`` on its keys
        -- so the diagnostic the README, both quickstarts, ``evolution.md`` and
        this class's own docstring all print as ``{'committed': 1}`` actually came
        out as ``{<MergeOutcome.COMMITTED: 'committed'>: 1}``. The value is the
        identity here (the enum subclasses ``str`` precisely so callers can use
        the bare string), so the enum repr showed the reader nothing they needed
        and contradicted every example of it. ``enum.StrEnum`` does this for free
        on 3.11+; this package supports 3.9."""
        return repr(self.value)


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
    #: Human-readable, may embed measured numbers -- good for a log line.
    reason: str = ""
    #: Stable bucket for the same outcome, safe to count across rounds. ``reason``
    #: interpolates values ("P(delta>0)=0.42 <= 0.75"), so it makes a useless key.
    category: str = ""



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
        meter: Optional["Meter"] = None,
        conflict: Optional["ConflictPolicy"] = None,
        fusion: Optional["FusionPolicy"] = None,
        acceptance: Optional["AcceptancePolicy"] = None,
        promotion: Optional["PromotionPolicy"] = None,
    ) -> None:
        self.ledger = ledger
        self.verifier = verifier
        self.audit = audit
        # swap Full / Guarded / Reflective without touching the merge pipeline.
        self.staleness_policy = staleness_policy or get_policy("guarded")
        #: Where the staleness ratio is recorded. `None` for a hand-built
        #: aggregator, which then simply reports nothing -- a missing counter is
        #: better than one that counts a fraction of the run.
        self.meter = meter
        self.config = config or AggregatorConfig()
        # The decisions, as objects. Swapping one is the point; the defaults are
        # the code that used to be inline here, moved rather than rewritten.
        self.conflict_policy = conflict or DefaultConflict(verifier)
        self.fusion_policy = fusion or DefaultFusion(
            verifier, tournament=self.config.fusion_tournament)
        # A fusion policy ranks candidates, so it needs the verifier -- and a
        # caller building one cannot supply it, because the verifier is built
        # inside the engine. Hand it over to any policy that asked by exposing
        # `bind`; the shipped one takes it in its constructor and has none.
        _bind = getattr(self.fusion_policy, "bind", None)
        if callable(_bind):
            _bind(verifier)
        cfg = self.config
        self.acceptance_policy = acceptance or DefaultAcceptance(
            cfg.base_delta, cfg.anneal_half_life, cfg.accept_samples)
        self.promotion_policy = promotion or DefaultPromotion(cfg.promote_after_k)
        self.buffer = EvidenceBuffer()
        self._posteriors: Dict[str, BetaPosterior] = defaultdict(BetaPosterior)
        # dev-branch survival counter for EMA-style promotion.
        # Artifacts this aggregator has seen, so `finalize` knows what to publish.
        # The survival *counter* belongs to the promotion policy; this is only the
        # set of ids, which the aggregator needs whether or not that policy keeps
        # one.
        self._seen: "set" = set()
        # dev version last copied onto stable, so an unchanged head is not re-promoted.
        self._promoted_at: Dict[str, int] = {}

    # -- staleness (section 4.2) ---------------------------------------------

    def _alpha_for(self, artifact: Evolvable, card: EvidenceCard) -> int:
        if card.diff.contract_breaking:
            return 0  # contract-breaking diffs must be re-proposed, not rebased.
        # Hotter artifacts (bigger blast radius) tolerate more staleness. The
        # boundary is `governance.classify`, not a second hand-written threshold:
        # this used to test `blast_radius > 0.5` while governance drew the line at
        # 0.30, so an artifact at 0.4 was L1 by governance and got the *cold*
        # tolerance meant for an L2 skill.
        return (self.config.alpha_head if classify(artifact) is Layer.L1_SLOW
                else self.config.alpha_tail)

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
                if artifact.evidence_eval(card) <= candidate.evidence_eval(card):
                    survivors.append(card.rebased_onto(head))
                else:
                    discarded.append(card)
            else:  # DISCARD
                discarded.append(card)
        if self.meter is not None:
            # The denominator has to come from here: `MergeReport` records what
            # was discarded, and without "out of how many" a stale ratio cannot
            # be computed at all -- 3 discards out of 4 and out of 400 need
            # opposite fixes.
            self.meter.add("stale_considered", len(cards))
            self.meter.add("stale_discarded", len(discarded))
        return survivors, discarded

    # -- conflict resolution and fusion (section 4.3) ------------------------
    #
    # Both live in `agentdescent.defaults` now. These stay as one-line
    # delegations rather than being inlined at the call site because the call
    # site is `_process`, which is already the longest method here, and because
    # a named seam is easier to find than a policy attribute used once.

    def _resolve_conflicts(
        self, artifact: Evolvable, cards: List[EvidenceCard]
    ) -> Tuple[List[EvidenceCard], int]:
        """Drop contradicting diffs (PCGrad-style) and return surviving cards."""
        return self.conflict_policy.resolve(artifact, cards)

    def _tournament(self, artifact: Evolvable,
                    diffs: List[Diff]) -> Tuple[Diff, Evolvable, bool]:
        """Run candidates (plus their fusion) in a held-out tournament."""
        return self.fusion_policy.select(artifact, diffs)

    # -- public entry points -------------------------------------------------

    def ingest(self, card: EvidenceCard) -> None:
        self.buffer.add(card)

    def step(self) -> List[MergeReport]:
        """Fire every artifact bucket that is ready and return per-artifact reports."""
        self.buffer.tick()
        reports: List[MergeReport] = []
        for aid in self.buffer.ready(self.config):
            report = self._process(aid)
            # Told after the fact, from the category the merge actually reported
            # -- not from a second reading of the same decision. A trust region
            # that adapted to its own guess about what happened would be tuning
            # itself against a number nothing else in the run agrees with.
            self._observe_trust_region(report.category)
            reports.append(report)
        self._age_and_promote(reports)
        return reports

    # -- dual-branch EMA promotion (section 4.5) -----------------------------

    def _age_and_promote(self, reports: List[MergeReport]) -> None:
        """Ask the promotion policy what has proved itself, and publish it.

        The rule lives in `agentdescent.defaults.DefaultPromotion`; the git work
        stays here, because "which artifact deserves stable" is an algorithm
        decision and "copy a branch without doing redundant git work" is not."""
        for promotion in self.promotion_policy.observe(reports):
            self._promote(promotion.artifact_id)

    def _promote(self, artifact_id: str) -> None:
        """Copy dev onto stable, skipping the git work when they already agree.

        The synchronous driver calls ``step()`` once per round, but the async
        merger polls it every couple of milliseconds -- so without this a converged
        async run re-promoted an unchanged artifact every K sweeps (52 times in one
        6-second run), each a handful of git operations under the ledger lock that
        every worker is queued behind."""
        dev_v = self.ledger.head_version(Ledger.DEV).get(artifact_id, 0)
        if self._promoted_at.get(artifact_id) == dev_v:
            return
        if self.ledger.promote_to_stable(artifact_id) is not None:
            self._promoted_at[artifact_id] = dev_v

    def finalize(self) -> None:
        """Publish the current dev head to stable at the end of a clean run.

        Confirmation takes ``promote_after_k`` rounds, and a run can legitimately
        stop before that many have elapsed -- ``target_reward`` fires on the very
        commit that reaches it, so the artifact the run was *for* would otherwise
        never reach the branch production reads. Only ever called when a run
        finishes without an error, and never in place of the round-based rule."""
        for aid in list(self._known_artifacts()):
            self._promote(aid)

    @property
    def _survival(self) -> Dict[str, int]:
        """Rounds survived per artifact -- now owned by the promotion policy.

        Kept under the old name because it is what the promotion tests and any
        diagnostic reach for, and because moving where a number lives is not a
        reason to make callers hunt for it. Read-only: the policy decides when it
        moves."""
        return getattr(self.promotion_policy, "survival", {})

    def _apply_trust_region(self, cards):
        """Split cards into (within budget, oversized), settling the rejects.

        Oversized diffs used to be dropped *before* `considered` was computed and
        never settled, so a runaway reflector vanished from the report and from
        the evidence pool at once -- the most expensive failure to diagnose is
        the one that leaves no trace."""
        region = self._trust_region()

        def within(card) -> bool:
            if card.diff.size() > region.ops:
                return False
            return all(len(str(v)) <= region.chars
                       for v in card.diff.ops.values())

        kept = [c for c in cards if within(c)]
        oversized = [c for c in cards if not within(c)]
        if oversized:
            self.buffer.settle(oversized)
        return kept, oversized

    def _trust_region(self):
        """The size cap for this merge: the constants, or whatever adapted them.

        Read through the policy rather than from `config` directly so the two
        cannot disagree -- an adaptive region that the size check did not consult
        would move, be recorded as moving, and change nothing."""
        from .advantage import TrustRegion

        policy = getattr(self.config, "trust_region_policy", None)
        if policy is None:
            return TrustRegion(ops=self.config.trust_region_ops,
                               chars=self.config.trust_region_chars)
        return policy.current

    def _observe_trust_region(self, outcome: str) -> None:
        """Tell an adaptive region how the merge it sized turned out."""
        policy = getattr(self.config, "trust_region_policy", None)
        if policy is not None:
            policy.observe(outcome)

    def _stable_distance(self, artifact_id: str, candidate) -> float:
        """How far a candidate sits from the confirmed branch, in ``[0, 1]``.

        Read here rather than by the policy, because a policy is not allowed to
        know about the ledger -- that is the boundary `policies.py` exists to
        keep. Any failure to read `stable` yields ``0.0``: a branch that does not
        exist yet, or cannot be read, is not a reference to be far from, and
        inventing a distance would be worse than reporting none.
        """
        from .advantage import state_distance

        try:
            stable = self.ledger.snapshot(Ledger.STABLE).get(artifact_id)
        except Exception:  # noqa: BLE001 - a missing branch is the normal case
            return 0.0
        if stable is None:
            return 0.0
        return state_distance(getattr(stable, "state", {}) or {},
                              getattr(candidate, "state", {}) or {})

    def _audit(self, artifact, artifact_id, best_state, best_diff,
               base_full, cand_full, base_score, cand_score, prior) -> bool:
        """The optimizer audits its own decision. True means the oracle refused.

        Not an `AcceptancePolicy`: acceptance asks "is this candidate better",
        and this asks "is the cheap layer still trustworthy" -- a question about
        the *measuring instrument*, which is the infrastructure's to answer and
        not the algorithm's to replace.
"""
        _, uncertainty = self.verifier.learned_eval(best_state)
        self.audit.submit(best_diff.diff_id, artifact_id, artifact.blast_radius,
                          uncertainty, payload=best_diff)
        # Trust is "how often does the cheap layer agree with the full held-out
        # set", and it has to be measurable WITHOUT spending oracle budget --
        # otherwise it is circular. It was: `force_oracle` fires on
        # `blast_radius >= 0.5 or trust < 0.75`, and the only writer of trust sat
        # inside that branch, so for any artifact below 0.5 the condition could
        # never become true and the audit never ran at all. Measured on the default
        # blast_radius=0.2: oracle_calls_used == 0 for the whole run, trust pinned
        # at its initial 1.0.
        #
        # The signal is free here: `eval_counts` already scored base and candidate
        # on the full held-out set for the Beta test above, so comparing its verdict
        # with the cheap layer's costs nothing and happens on every merge.
        if cand_full != base_full or cand_score != base_score:
            self.audit.update_trust(
                artifact_id, (cand_full > base_full) == (cand_score > base_score))
        if self.audit.force_oracle(artifact.blast_radius, artifact_id):
            # A verifier whose oracle scores the same set `eval_counts` does has
            # already been paid for: `base_full` / `cand_full` ARE that
            # measurement. Re-buying it was not merely wasteful -- `oracle_eval`
            # degrades to `rule_eval` once the budget is gone, so past that point
            # the gate vetoed on the cheap SUB-SAMPLE. Measured: a candidate that
            # took the full-set rate from 0.5 to 1.0 came back `oracle-rejected`
            # because the two-task sample scored both at 0.5. The verifier page
            # promises in two places that sub-sampling can never decide a commit;
            # this was the path that made it false.
            if getattr(self.verifier, "oracle_shares_full_set", False):
                oracle_base, oracle_cand = base_full, cand_full
            else:
                oracle_base = self.verifier.oracle_eval(artifact)
                oracle_cand = self.verifier.oracle_eval(best_state)
            agreed = (oracle_cand > oracle_base) == (cand_score > base_score)
            self.audit.update_trust(artifact_id, agreed)
            if oracle_cand <= oracle_base:
                prior.observe_delta(oracle_cand - oracle_base)
                # Not settled, unlike the CAS-conflict path below -- and the
                # asymmetry is deliberate. A CAS-conflicted diff lost a race and
                # was never judged against the new head, so it deserves another
                # look; one rejected here was judged and lost, and its tournament
                # rivals scored below it on the same held-out set. Re-filing them
                # would just buy the same rejection again.
                return True
        return False

    def _commit_with_retry(self, artifact_id, candidate, diff, head):
        """CAS, rebasing onto whatever landed first. ``None`` when it kept losing.

        With one writer a conflict cannot happen: every commit goes through a
        single merger. With several -- separate processes sharing a ledger --
        losing the race is ordinary, and settling the evidence back into the pool
        to wait a round is the wrong answer to "somebody committed first". The
        right one is to rebase and try again.

        **Rebasing means re-applying the diff to the new head**, not re-sending
        the old candidate. The candidate was computed against a head that no
        longer exists; committing it would discard whatever won the race, which is
        the lost update CAS is there to prevent, arriving through the retry that
        was supposed to preserve it.

        Bounded, and jittered: two writers backing off by the same amount collide
        again on the same schedule. The acceptance decision is **not** re-run --
        it was made against a measurement of this diff's effect, and that
        measurement is what the retry is trying to preserve. A diff whose value
        depends on which of two commits landed first is a diff the staleness
        policy should have caught.
        """
        attempts = max(1, self.config.cas_attempts)
        for attempt in range(attempts):
            base_vv = {artifact_id: head.get(artifact_id, 0)}
            try:
                _, new_version = self.ledger.commit(
                    candidate, base_vv, branch=Ledger.DEV,
                    message=f"merge {diff.diff_id} -> {artifact_id}")
                return new_version
            except CASConflict:
                if self.meter is not None:
                    self.meter.add("cas_conflicts")
                if attempt + 1 >= attempts:
                    return None
                time.sleep(random.uniform(0.0, self.config.cas_backoff)
                           * (2 ** attempt))
                snap = self.ledger.snapshot(Ledger.DEV)
                fresh = snap.get(artifact_id)
                if fresh is None:
                    return None
                head, candidate = snap.version, fresh.apply(diff)
                if candidate.state == fresh.state:
                    # Whoever won applied the same change. Nothing left to commit,
                    # and reporting a conflict would be truer than reporting a
                    # commit that did not happen.
                    return None
        return None

    def _known_artifacts(self):
        """Every artifact id this aggregator has merged for.

        Taken from the promotion policy when it tracks them (the default does),
        and from the aggregator's own record otherwise -- a replacement policy is
        not obliged to keep a survival table, and `finalize` still has to know
        what to publish."""
        tracked = getattr(self.promotion_policy, "survival", None)
        return set(tracked) | self._seen if tracked is not None else set(self._seen)

    def _process(self, artifact_id: str) -> MergeReport:
        self._seen.add(artifact_id)
        cards = self.buffer.drain(artifact_id)
        snap = self.ledger.snapshot(Ledger.DEV)
        artifact = snap.get(artifact_id)
        if artifact is None:
            return MergeReport(artifact_id, None, False, len(cards), 0, 0, 0, 0.0, None,
                               "unknown artifact", MergeOutcome.UNKNOWN_ARTIFACT)
        assert_mutable(artifact)  # L0 guard

        # trust-region: reject over-large diffs up front. They were previously
        # dropped *before* `considered` was computed and never settled, so they
        # vanished from both the report and the evidence pool -- silently.
        n_considered = len(cards)
        cards, oversized = self._apply_trust_region(cards)

        head = snap.version
        survivors, discarded = self._staleness_filter(artifact, head, cards)
        self.buffer.settle(discarded)
        if not survivors:
            # Report the reason that dominated, rather than folding oversized
            # diffs into the staleness count. "my reflector emits 500 KB values and
            # every one is rejected" and "my lag budget is too tight" are opposite
            # fixes, and the trust-region case used to be invisible in both
            # `outcomes()` and the report.
            if oversized and not discarded:
                return MergeReport(artifact_id, None, False, n_considered, 0,
                                   len(oversized), 0, 0.0, None,
                                   f"{len(oversized)} diff(s) outside the trust "
                                   f"region ({self.config.trust_region_ops} ops / "
                                   f"{self.config.trust_region_chars} chars)",
                                   MergeOutcome.OVERSIZED)
            return MergeReport(artifact_id, None, False, n_considered, 0,
                               len(discarded) + len(oversized), 0, 0.0, None,
                               "all stale / rejected", MergeOutcome.ALL_STALE)

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
        decision = self.acceptance_policy.accept(MergeContext(
            artifact=artifact, candidate=best_state, cards=kept_cards,
            base_counts=(base_s, base_f), cand_counts=(cand_s, cand_f),
            diff=best_diff, base_cheap=base_score, cand_cheap=cand_score,
            prior=prior, trust_radius=artifact.blast_radius,
            stable_distance=self._stable_distance(artifact_id, best_state)))
        p_improve = decision.p_improve

        # The full held-out rates. The regression guard below must use *these*
        # rather than `cand_score`/`base_score`: those come from the cheap layer,
        # which `cheap_eval_tasks` sub-samples -- so a 4-task sample could veto a
        # commit the full-set Beta test had just approved, while three comments and
        # two doc pages promised sub-sampling "never affects commit safety". It was
        # the promise that was wrong, not the intent; the guard is meant to stop a
        # measured regression, and ground truth is what measures one.
        base_full = MergeContext.rate((base_s, base_f))
        cand_full = MergeContext.rate((cand_s, cand_f))

        rejected = self._audit(artifact, artifact_id, best_state, best_diff,
                               base_full, cand_full, base_score, cand_score, prior)
        if rejected:
            return MergeReport(artifact_id, None, fused, n_considered, len(survivors),
                               len(discarded), conflicts, p_improve, None,
                               "oracle rejected", MergeOutcome.ORACLE_REJECTED)

        if not decision.accept:
            prior.observe_delta(decision.observed_delta)
            return MergeReport(artifact_id, None, fused, n_considered, len(survivors),
                               len(discarded), conflicts, p_improve, None,
                               decision.detail, MergeOutcome(decision.category))

        # -- commit (section 4.1): CAS on dev --------------------------------
        new_version = self._commit_with_retry(artifact_id, best_state, best_diff, head)
        if new_version is None:
            self.buffer.settle(survivors)
            return MergeReport(artifact_id, None, fused, n_considered, len(survivors),
                               len(discarded), conflicts, p_improve, None,
                               "CAS conflict", MergeOutcome.CAS_CONFLICT)

        prior.update(True, weight=2.0)  # a committed improvement is strong evidence.

        return MergeReport(artifact_id, best_diff, fused, n_considered, len(survivors),
                           len(discarded), conflicts, p_improve, new_version,
                           "committed", MergeOutcome.COMMITTED)
