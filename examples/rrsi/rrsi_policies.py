"""RRSI policies wired into ``evolve()`` through the engine's seams.

This is the integration the upstream-faithful fixtures were waiting for. The
selection core (``selection.py``, pinned to Algorithm 2) and the cosine-annealed
edit budget (``schedule.py``) are **reused here, not rewritten**; this module
only adapts them to the engine's ``Policies`` slots:

* :class:`RRsiCritic` -- the pre-evaluation screen (``Policies(screen=...)``):
  leakage screening against the evolve set plus the annealed edit-budget gate.
  It runs before any measurement (issue #221: "screening before evaluation is
  important"), so a leaking or over-budget card is settled without ever buying
  a held-out sweep. The pinned selection core already has a home for its
  verdict: ``Candidate.gate_failure``.
* :class:`RRsiProposalPolicy` -- the regularized proposer
  (``Policies(proposal=...)``): injects the annealed budget, the measured
  per-edit history, pruning targets and stall-exploration guidance into the
  reflection prompt, so the proposal distribution is constrained without
  restricting which harness components may be edited.
* :class:`RRsiAcceptance` -- the regularized selector
  (``Policies(acceptance=...)``): the pinned ``judge`` decision (historical-
  best floor, both cost-rule branches, novelty) with ``S*`` maintained through
  the pinned ``update_best_score`` and a checkpointable state, so a resumed
  run does not lose the floor.

The three share one :class:`RRsiState` -- ``S*``, the measured per-edit
ledger, the component bookkeeping, the annealed round counter -- which is what
makes them one algorithm rather than three knobs. Build them with
:func:`build_rrsi_bundle`.

**Fidelity and deviations** (recorded, per the port conventions):

* The selection decision is the pinned ``judge`` verbatim. The engine merges
  to **one candidate per round**, so the multi-candidate ``select_round``
  "choose highest admissible" step is not exercised here; each merge's
  candidate is judged against the incumbent and ``S*``. This is the "one
  winner per round" semantics issue #221 asks to establish first.
* The annealed round counter advances once per **merge** (per round barrier on
  the synchronous path). On the barrier-free path there is no round barrier,
  so the schedule anneals per merge instead of per RRSI round -- a declared
  semantic deviation, not a silent one.
* ``delta`` (the noise band) is a configured :class:`RRsiConfig.noise_band`,
  default ``0.0`` (the conservative floor ``score >= S*``). The paper estimates
  it by repeatedly evaluating the unchanged base harness; this module treats it
  as a parameter and documents how to set it.
* Token cost is read from ``EvidenceCard.cost_tokens`` (the proposing rollout's
  usage, when the caller shares a ``Usage`` with its adapters). Missing or zero
  telemetry yields ``dC = 0``, matching the pinned upstream behaviour.

Privacy and data isolation follow the port conventions: the critic's
``protected`` strings must be derived from the **evolve set only** -- never the
final holdout -- and the integration tests enforce that discipline by
construction.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from agentdescent.aggregator import MergeOutcome
from agentdescent.policies import (
    AcceptDecision,
    AcceptancePolicy,
    MergeContext,
    Policies,
    ProposalContext,
    ProposalPolicy,
    ScreenPolicy,
)

from .schedule import edit_budget
from .selection import (
    Candidate,
    Measurement,
    SelectionConfig,
    judge,
    update_best_score,
)

__all__ = [
    "RRsiAcceptance",
    "RRsiConfig",
    "RRsiCritic",
    "RRsiProposalPolicy",
    "RRsiState",
    "build_rrsi_bundle",
]


# -- configuration and shared state -------------------------------------------


@dataclass(frozen=True)
class RRsiConfig:
    """The RRSI knobs: Algorithm 2's selection rule plus the annealed budget.

    ``beta0`` / ``beta1`` are the large-gain cost rule's tolerance and slope;
    ``w_s`` / ``w_c`` / ``w_n`` weight the within-band score/cost/novelty rule;
    ``noise_band`` is the historical-best floor tolerance (the paper's
    empirically estimated ``delta``). ``b_min`` / ``b_max`` / ``T`` are the
    annealed edit budget's schedule.
    """

    # -- selection (Algorithm 2) ---------------------------------------------
    beta0: float = 0.0
    beta1: float = 0.0
    w_s: float = 1.0
    w_c: float = 1.0
    w_n: float = 0.0
    #: The historical-best noise floor: ``score >= S* - noise_band``.
    noise_band: float = 0.0

    # -- proposal (annealed edit budget) -------------------------------------
    b_min: int = 1
    b_max: int = 5
    #: Number of RRSI rounds the budget anneals over.
    T: int = 20
    #: Rounds without S* movement (beyond ``noise_band``) that count as a stall.
    stall_window: int = 5
    #: Rounds of component yield a component must show positive gain over to
    #: keep its place (the L1-style pruner's window).
    prune_window: int = 5
    #: How much of the measured per-edit history fits in the reflection prompt.
    max_history_chars: int = 1500


@dataclass
class EditRecord:
    """One measured candidate, as the proposer conditions on in later rounds."""

    round: int
    components: Tuple[str, ...]
    diff_size: int
    delta_score: float
    delta_cost: float
    accepted: bool


#: A diff -> component-set extractor. The default takes the first path segment
#: of each op key, so a FileTree harness ("skills/x/SKILL.md") attributes an
#: edit to its top-level component ("skills"), which is what RRSI's novelty and
#: pruning rules count.
def default_components(diff: Any) -> Tuple[str, ...]:
    out: Set[str] = set()
    for key in diff.ops:
        head = str(key).split("/", 1)[0]
        if head:
            out.add(head)
    return tuple(sorted(out))


class RRsiState:
    """S*, the per-edit ledger, component bookkeeping and the annealed round.

    One instance is shared by the critic, the proposer and the selector -- that
    sharing is what makes the three slots one regularized algorithm. It is
    checkpointable (see :meth:`checkpoint`) so a resumed run does not lose the
    historical-best floor or the measured history.
    """

    def __init__(self, config: Optional[RRsiConfig] = None, *,
                 base_score: float = 0.0,
                 components: Optional[Callable[[Any], Tuple[str, ...]]] = None):
        self.config = config or RRsiConfig()
        self.components = components or default_components
        #: The annealed round counter. Advanced once per merge (see the module
        #: docstring for the barrier-free deviation). Never reset on resume.
        self.round = 0
        #: The historical-best evolve-set score ``S*``, monotone via the pinned
        #: ``update_best_score``.
        self.best_score = float(base_score)
        #: The incumbent's measured policy-token cost, ``None`` until a first
        #: accepted candidate carried cost telemetry. ``None`` is "unknown",
        #: which the pinned cost rule reads as ``dC = 0``.
        self.incumbent_cost: Optional[float] = None
        #: component -> how often it appeared in a proposed diff (novelty's
        #: denominator, and the pruner's "has it been exercised" test).
        self.attempts: Counter = Counter()
        #: component -> rolling measured gain per round, for the pruner.
        self._round_gains: List[Dict[str, float]] = []
        #: The measured per-edit history the proposer conditions on.
        self.ledger: List[EditRecord] = []
        #: ``S*`` after each merge, for stall detection.
        self.progress: List[float] = []

    def record_merge(self) -> None:
        """Advance the schedule by one merge, whatever its outcome.

        ``round`` and ``progress`` are the anneal and the stall detector, and
        both must move on **every** merge decision -- a rejection is a round the
        search spent stuck, and the sparsity anneal is exactly what makes a
        stuck search try smaller bundles, while a flat ``progress`` is what
        makes :meth:`stalled` trip into structured exploration. Advancing only
        on accepts left the budget pinned at ``b_max`` and ``stalled()`` False
        through a streak of rejections (review)."""
        self.round += 1
        self.progress.append(self.best_score)

    def budget(self) -> int:
        """The annealed edit budget for the current round (``b_t``)."""
        return edit_budget(self.round, self.config.T,
                           self.config.b_min, self.config.b_max)

    def stalled(self) -> bool:
        """True when ``S*`` has not moved beyond the noise band recently.

        Structured exploration fires here: the proposer reserves part of its
        budget for components never yet exercised."""
        window = (self.progress[-self.config.stall_window:]
                  if self.config.stall_window > 0 else [])
        if len(window) < 2:
            return False
        return (window[-1] - window[0]) <= self.config.noise_band

    def pruning_targets(self) -> List[str]:
        """Components whose recent measured gain is non-positive (L1-style).

        A component that has been exercised in the last ``prune_window`` rounds
        but whose net gain over that window is not strictly positive has stopped
        earning its place; the proposer is told it is a deletion candidate. A
        component with no gain record is not a target (it may simply be new)."""
        window = self._round_gains[-self.config.prune_window:]
        if not window:
            return []
        totals: Dict[str, float] = defaultdict(float)
        exercised: Set[str] = set()
        for gains in window:
            for comp, gain in gains.items():
                totals[comp] += gain
                exercised.add(comp)
        return sorted(comp for comp in exercised
                      if totals.get(comp, 0.0) <= 0.0
                      and self.attempts.get(comp, 0) > 0)

    def components_of(self, diff: Any) -> Tuple[str, ...]:
        return self.components(diff)

    def record_measurement(self, diff: Any, *, delta_score: float,
                           delta_cost: float, accepted: bool) -> None:
        """Record one measured candidate into the ledger and the pruner's books.

        Called for accepted *and* rejected candidates: rejected mechanisms are
        negative evidence the proposer conditions on (the paper's evidence-aware
        credit assignment), and a rejection is exactly what the pruner should
        count against a component."""
        comps = self.components_of(diff)
        for c in comps:
            self.attempts[c] += 1
        gains = {c: delta_score for c in comps}
        self._round_gains.append(gains)
        self.ledger.append(EditRecord(
            round=self.round, components=comps, diff_size=len(diff.ops),
            delta_score=delta_score, delta_cost=delta_cost, accepted=accepted))

    def checkpoint(self) -> dict:
        """Serialise the state the ledger does not persist."""
        return {
            "round": self.round,
            "best_score": self.best_score,
            "incumbent_cost": self.incumbent_cost,
            "attempts": dict(self.attempts),
            "round_gains": self._round_gains,
            "ledger": [{"round": r.round, "components": list(r.components),
                        "diff_size": r.diff_size, "delta_score": r.delta_score,
                        "delta_cost": r.delta_cost, "accepted": r.accepted}
                       for r in self.ledger],
            "progress": self.progress,
        }

    def restore(self, data: dict) -> None:
        """Restore :meth:`checkpoint`'s form. Unknown keys are ignored."""
        try:
            if isinstance(data.get("round"), int):
                self.round = data["round"]
            if isinstance(data.get("best_score"), (int, float)):
                self.best_score = float(data["best_score"])
            if data.get("incumbent_cost") is not None:
                self.incumbent_cost = float(data["incumbent_cost"])
            if isinstance(data.get("attempts"), dict):
                self.attempts = Counter(
                    {k: int(v) for k, v in data["attempts"].items()
                     if isinstance(v, (int, float))})
            if isinstance(data.get("round_gains"), list):
                self._round_gains = [
                    {k: float(v) for k, v in gains.items()
                     if isinstance(v, (int, float))}
                    for gains in data["round_gains"] if isinstance(gains, dict)]
            if isinstance(data.get("ledger"), list):
                self.ledger = []
                for r in data["ledger"]:
                    if not isinstance(r, dict):
                        continue
                    try:
                        self.ledger.append(EditRecord(
                            round=int(r["round"]),
                            components=tuple(r.get("components", ())),
                            diff_size=int(r.get("diff_size", 0)),
                            delta_score=float(r.get("delta_score", 0.0)),
                            delta_cost=float(r.get("delta_cost", 0.0)),
                            accepted=bool(r.get("accepted", False))))
                    except (TypeError, ValueError, KeyError):
                        continue
            if isinstance(data.get("progress"), list):
                self.progress = [float(v) for v in data["progress"]
                                 if isinstance(v, (int, float))]
        except Exception:  # noqa: BLE001 - a bad payload must not take the run down
            return


# -- the critic: pre-evaluation screen ------------------------------------------


def _diff_text(diff: Any) -> str:
    """The card's editable content, for the deterministic leakage scan."""
    return "\n".join([str(k) for k in diff.ops]
                     + [str(v) for v in diff.ops.values()])


class RRsiCritic:
    """The pre-evaluation screen: leakage + annealed edit budget.

    Wired as ``Policies(screen=...)`` and run by the aggregator **before**
    measurement (issue #221: "screening before evaluation is important"). Each
    card is checked deterministically for

    * **edit budget** -- ``len(diff.ops) > b_t`` is rejected: the proposer was
      told to bundle at most ``b_t`` independent edits, and an over-budget
      bundle is exactly the "many unrelated modifications in one candidate"
      the annealed sparsity constraint exists to stop;
    * **leakage** -- any ``protected`` string appearing in the diff's keys or
      values is rejected: task names, entity names, task-specific values or
      answers encoded into the harness are benchmark-specific content, and a
      leaking candidate must never receive the inflated evolve-set score that
      would make it attractive to later rounds.

    An optional model ``reviewer`` (a ``Completion``) performs the paper's
    model critic: it reads the candidate and returns ``"leak"``, ``"inert"``
    or ``"ok"``. The deterministic checks run first; the model is only asked
    when they pass, so a leaking card never pays a model call either.

    ``protected`` must come from the **evolve set only** -- never the final
    holdout. The integration tests enforce that discipline by construction.
    """

    def __init__(self, state: RRsiState, *, protected: Sequence[str] = (),
                 reviewer: Optional[Any] = None):
        self.state = state
        self._protected = tuple(s for s in protected if s)
        self._reviewer = reviewer

    def screen(self, card: Any) -> Optional[str]:
        """Return ``None`` to let the card through, or a reason to reject it."""
        diff = getattr(card, "diff", None)
        if diff is None:
            return "card carries no diff"
        budget = self.state.budget()
        if len(diff.ops) > budget:
            return (f"edit bundle has {len(diff.ops)} edit(s); annealed budget "
                    f"at round {self.state.round} is {budget}")
        text = _diff_text(diff)
        for token in self._protected:
            if token in text:
                return f"leaks evolve-set content ({token!r})"
        if self._reviewer is not None:
            verdict = self._reviewer_verdict(diff, text)
            if verdict != "ok":
                return f"critic: {verdict}"
        return None

    def _reviewer_verdict(self, diff: Any, text: str) -> str:
        """Ask the model critic for its verdict. Best-effort: an unusable reply
        lets the card through (the deterministic checks remain the gate)."""
        prompt = (
            "You are screening an edit to an agent harness before it is "
            "evaluated.\n\nEdit:\n{text}\n\nReply exactly one word: 'leak' if "
            "the edit encodes benchmark-specific content (task names, entity "
            "names, values, answers), 'inert' if it adds machinery that cannot "
            "change behavior, or 'ok'.")
        try:
            reply = (self._reviewer(prompt.format(text=text[:4000])).strip().lower())
        except Exception:  # noqa: BLE001 - a broken reviewer must not block a run
            return "ok"
        return "ok" if reply not in ("leak", "inert") else reply


# -- the regularized proposer ---------------------------------------------------


def _expected_block(task: Any, show_meta: bool, limit: int) -> str:
    meta = getattr(task, "meta", None)
    if not show_meta or not meta:
        return ""
    text = ", ".join(f"{k}={v!r}" for k, v in meta.items())
    if len(text) > limit:
        text = text[:limit] + " ..."
    return f"\nWhat the scorer expected (task metadata):\n{text}\n"


def _history_summary(ledger: Sequence[EditRecord], max_chars: int) -> str:
    """A compact recent-history summary for the reflection prompt."""
    if not ledger:
        return ""
    recent = ledger[-8:]
    lines = []
    for r in recent:
        tag = "accepted" if r.accepted else "rejected"
        comps = ",".join(r.components) or "?"
        lines.append(
            f"- r{r.round} [{tag}] {comps}: dS={r.delta_score:+.2f} "
            f"dC={r.delta_cost:+.2f} ({r.diff_size} edit(s))")
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[:max_chars] + " ..."
    return out


class RRsiProposalPolicy:
    """The regularized proposer: ``Policies(proposal=...)``.

    Wraps a ``Completion`` and builds the reflection prompt itself, so the
    RRSI guidance -- the annealed budget, the measured per-edit history, the
    pruning targets and the stall-exploration note -- reaches the reflector
    exactly like the ordinary failure prompt does. The strategy still turns the
    reply into a diff; the critic then *enforces* the budget deterministically,
    so guidance and enforcement agree.

    ``complete`` has the repo's ``Completion`` shape (``prompt -> str``); the
    same model can drive solve and reflection, or a cheap model can reflect on
    an expensive agent.
    """

    def __init__(self, complete: Callable[[str], str], state: RRsiState, *,
                 show_meta: bool = True, meta_chars: int = 600):
        self.complete = complete
        self.state = state
        self.show_meta = show_meta
        self.meta_chars = meta_chars

    def propose(self, ctx: ProposalContext) -> Sequence[str]:
        config = self.state.config
        budget = self.state.budget()
        lines = [
            f"The artifact just failed a task (score {ctx.reward:.2f} out of 1.0).",
            "",
            "Artifact so far:",
            ctx.rendered,
            "",
            "Task input:",
            str(ctx.task.prompt),
            "",
            "It produced:",
            ctx.output,
            _expected_block(ctx.task, self.show_meta, self.meta_chars),
            "",
            f"[RRSI budget] Round {self.state.round} of {config.T} allows at most "
            f"{budget} independent edit(s) in one candidate; bundle fewer, more "
            "attributable edits later in the run.",
        ]
        history = _history_summary(self.state.ledger, config.max_history_chars)
        if history:
            lines.append("")
            lines.append("[RRSI history] Measured so far:")
            lines.append(history)
        targets = self.state.pruning_targets()
        if targets:
            lines.append("")
            lines.append("[RRSI pruning] These components have not earned their "
                         f"place recently; consider deleting them: {', '.join(targets)}")
        if self.state.stalled():
            lines.append("")
            lines.append("[RRSI exploration] The search has stalled within the "
                         "noise band; spend part of the budget on components never "
                         "yet exercised.")
        lines.extend([
            "",
            "Propose exactly ONE concise candidate (a single imperative rule, or "
            "an edit script), or NONE if no change would help.",
        ])
        prompt = "\n".join(lines)
        reply = (self.complete(prompt) or "").strip()
        if not reply or reply.upper().startswith("NONE"):
            return []
        return [reply]


# -- the regularized selector ---------------------------------------------------


def _card_cost(cards: Sequence[Any]) -> Optional[float]:
    """The candidate's measured policy-token cost, or ``None`` when unknown.

    A **per-rollout mean**, not a batch sum: `EvidenceCard.cost_tokens` is one
    proposing rollout's usage, so summing across a merge's cards would scale
    with however many cards happened to land in the bucket and reject a
    candidate purely for batch size (review). The mean is still a proxy --
    Algorithm 2's ``dC`` is the *candidate* harness's cost relative to the
    incumbent's on the same evaluation, and the proposing rollout runs the
    incumbent -- so this is a documented deviation; callers who share a
    ``Usage`` get an approximate, batch-size-independent cost signal, and
    missing/zero telemetry yields ``dC = 0`` (the pinned upstream behaviour).
    """
    costs = [int(getattr(card, "cost_tokens", 0) or 0) for card in cards]
    if not costs:
        return None
    return sum(costs) / len(costs) if any(costs) else None


class RRsiAcceptance:
    """Algorithm 2's selector, adapted to the engine's per-merge decision.

    Wired as ``Policies(acceptance=...)``. Each merge's candidate is judged by
    the **pinned** ``selection.judge`` -- historical-best floor, both cost-rule
    branches, novelty -- against the incumbent and ``S*``; ``S*`` is advanced
    through the pinned ``update_best_score``. The state (floor, ledger, round)
    checkpoint/restore through the aggregator's optional acceptance hook, so a
    resumed run does not lose the floor.
    """

    def __init__(self, state: RRsiState):
        self.state = state

    def accept(self, ctx: MergeContext) -> AcceptDecision:
        config = self.state.config
        cand_rate = MergeContext.rate(ctx.cand_counts)
        base_rate = MergeContext.rate(ctx.base_counts)
        cand_cost = _card_cost(ctx.cards)
        base_cost = self.state.incumbent_cost
        observed = cand_rate - base_rate

        candidate = Candidate(
            variant=ctx.diff.diff_id if ctx.diff is not None else "?",
            components=self.state.components_of(ctx.diff),
            measurement=Measurement(score=cand_rate, cost=cand_cost))
        incumbent = Measurement(score=base_rate, cost=base_cost)
        selection_config = SelectionConfig(
            beta0=config.beta0, beta1=config.beta1,
            w_s=config.w_s, w_c=config.w_c, w_n=config.w_n)
        decision = judge(
            candidate, incumbent, self.state.best_score, config.noise_band,
            selection_config, dict(self.state.attempts))

        # The schedule advances on *every* decision this policy makes -- a
        # rejection is a round the search spent stuck, not a free one (see
        # RRsiState.record_merge). best_score is advanced only on an accept.
        self.state.record_merge()

        if not decision.admissible:
            self.state.record_measurement(ctx.diff, delta_score=observed,
                                          delta_cost=decision.delta_cost or 0.0,
                                          accepted=False)
            category = (MergeOutcome.NOISE_FLOOR.value
                        if "floor" in decision.reason
                        else MergeOutcome.COST_RULE.value)
            return AcceptDecision(False, category, decision.reason,
                                  p_improve=0.0, observed_delta=observed)

        self.state.best_score = update_best_score(self.state.best_score, candidate)
        self.state.incumbent_cost = cand_cost
        self.state.record_measurement(ctx.diff, delta_score=observed,
                                      delta_cost=decision.delta_cost or 0.0,
                                      accepted=True)
        return AcceptDecision(True, "", decision.reason,
                              p_improve=max(0.0, observed), observed_delta=observed)

    # -- checkpoint hooks (aggregator optional acceptance state) ---------------
    def checkpoint_state(self) -> dict:
        return self.state.checkpoint()

    def restore_state(self, data: dict) -> None:
        self.state.restore(data)

    def on_screen_reject(self) -> None:
        """Advance the schedule when the aggregator's screen rejected a batch.

        The screen (critic) runs in ``Aggregator._prepare``, before this policy's
        ``accept`` is ever reached, so a batch the screen rejects would otherwise
        leave the anneal and the stall detector frozen even though a merge was
        decided. The aggregator calls this hook once per screen-rejected batch;
        ``record_merge`` advances ``round``/``progress`` exactly as an acceptance
        decision does (review: the round must advance on every merge)."""
        self.state.record_merge()


# -- the bundle -----------------------------------------------------------------


def build_rrsi_bundle(
    complete: Callable[[str], str],
    *,
    protected: Sequence[str] = (),
    reviewer: Optional[Any] = None,
    config: Optional[RRsiConfig] = None,
    base_score: float = 0.0,
) -> Tuple[Policies, RRsiState]:
    """Assemble the three RRSI policies into one ``Policies`` bundle.

    Returns ``(bundle, state)``; keep ``state`` to read (or checkpoint) the
    historical-best floor and the measured history. ``protected`` is the evolve
    set's distinctive strings (task ids, gold answers, names) -- never the
    final holdout. ``complete`` drives the proposer; an optional ``reviewer``
    drives the model critic. ``base_score`` seeds ``S*`` (the base harness's
    evolve-set score when you have it)."""
    state = RRsiState(config, base_score=base_score)
    bundle = Policies(
        screen=RRsiCritic(state, protected=protected, reviewer=reviewer),
        proposal=RRsiProposalPolicy(complete, state),
        acceptance=RRsiAcceptance(state),
    )
    return bundle, state
