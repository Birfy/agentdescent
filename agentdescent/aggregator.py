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

import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
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
        self._age_and_promote(reports)
        return reports

    # -- dual-branch EMA promotion (section 4.5) -----------------------------

    def _age_and_promote(self, reports: List[MergeReport]) -> None:
        """Promote dev -> stable after K **regression-free rounds** on dev.

        The counter used to be bumped on the *commit* path, so it measured how many
        times an artifact had **changed**, not how long it had held up -- the exact
        opposite of what every description of it says ("survival rounds",
        "regression-free rounds", "EMA confirmation rounds"). The incentive was
        inverted: an artifact that converged, which is the success state, stopped
        committing and could therefore never be promoted, while one that thrashed
        promoted every K commits. In the shipped demo the artifact reached 1.000
        held-out accuracy after two commits and then sat there for 36 rounds with
        `stable` stuck at 0.000 for the entire run.

        So: one round is one ``step()``. A commit restarts the clock (the new
        version has survived nothing yet) and so does an oracle rejection (a
        measured regression). Everything else is a round survived.
        """
        by_id = {r.artifact_id: r for r in reports}
        for aid in set(self._survival) | set(by_id):
            rep = by_id.get(aid)
            if rep is not None and (rep.committed_version is not None
                                    or rep.category == MergeOutcome.ORACLE_REJECTED):
                self._survival[aid] = 0        # changed, or measurably worse
                continue
            self._survival[aid] += 1
            if self._survival[aid] >= self.config.promote_after_k:
                self._promote(aid)
                self._survival[aid] = 0

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
        for aid in list(self._survival):
            self._promote(aid)

    def _process(self, artifact_id: str) -> MergeReport:
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

        baseline_post = BetaPosterior(prior.successes + base_s, prior.failures + base_f)
        candidate_post = BetaPosterior(prior.successes + cand_s, prior.failures + cand_f)
        # fold each contributing worker's local before/after delta as extra
        # evidence for the candidate (the "gradient magnitude").
        for card in kept_cards:
            if set(card.diff.ops) & set(best_diff.ops):
                candidate_post.observe_delta(card.before_after_delta)

        delta = annealed_delta(self.config.base_delta, artifact.version,
                               half_life=self.config.anneal_half_life)
        # Seed from (version, candidate) rather than the version alone. A shared
        # seed made the ~0.003 Monte-Carlo error identical for every candidate in
        # the round, so on knife-edge cases the draw decided them as a block --
        # one stream accepted every marginal diff, another rejected all of them.
        # stable_hash keeps this reproducible across processes.
        p_improve = prob_improvement(
            candidate_post, baseline_post, samples=self.config.accept_samples,
            seed=stable_hash((artifact.version, best_diff.diff_id)) & 0x7FFFFFFF)

        # -- audit (section 5.3): the optimizer audits its own decision ------
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
        base_full = base_s / max(1e-9, base_s + base_f)
        cand_full = cand_s / max(1e-9, cand_s + cand_f)
        if cand_full != base_full or cand_score != base_score:
            self.audit.update_trust(
                artifact_id, (cand_full > base_full) == (cand_score > base_score))
        if self.audit.force_oracle(artifact.blast_radius, artifact_id):
            oracle_base = self.verifier.oracle_eval(artifact)
            oracle_cand = self.verifier.oracle_eval(best_state)
            agreed = (oracle_cand > oracle_base) == (cand_score > base_score)
            self.audit.update_trust(artifact_id, agreed)
            if oracle_cand <= oracle_base:
                prior.observe_delta(oracle_cand - oracle_base)
                return MergeReport(artifact_id, None, fused, n_considered, len(survivors),
                                   len(discarded), conflicts, p_improve, None,
                                   "oracle rejected", MergeOutcome.ORACLE_REJECTED)

        # Not settled, unlike the CAS-conflict path below -- and the asymmetry is
        # deliberate. A CAS-conflicted diff lost a race and was never judged against
        # the new head, so it deserves another look; one rejected here was judged and
        # lost, and its tournament rivals scored below it on the same held-out set.
        # Re-filing them would just buy the same rejection again.
        if p_improve <= 1.0 - delta or cand_score < base_score:
            prior.observe_delta(cand_score - base_score)
            return MergeReport(artifact_id, None, fused, n_considered, len(survivors),
                               len(discarded), conflicts, p_improve, None,
                               f"P(delta>0)={p_improve:.2f} <= {1-delta:.2f}",
                               MergeOutcome.BELOW_THRESHOLD)

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
                               len(discarded), conflicts, p_improve, None,
                               "CAS conflict", MergeOutcome.CAS_CONFLICT)

        prior.update(True, weight=2.0)  # a committed improvement is strong evidence.

        # Promotion is decided in `_age_and_promote`, per round, not per commit:
        # a commit restarts the survival clock rather than advancing it.
        self._survival[artifact_id] = 0

        return MergeReport(artifact_id, best_diff, fused, n_considered, len(survivors),
                           len(discarded), conflicts, p_improve, new_version,
                           "committed", MergeOutcome.COMMITTED)
