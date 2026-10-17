"""RRSI policies wired into ``evolve()`` through the engine's seams.

This is the integration the upstream-faithful fixtures were waiting for. The
selection core (``selection.py``, pinned to Algorithm 2), the cosine-annealed
edit budget (``schedule.py``), the component vocabulary (``components.py``, the
pinned ``K`` / ``K_STR`` / ``normalize`` evidence rules) and the per-edit JSONL
ledger (``history.py``, ``History``) are **reused here, not rewritten**; this
module only adapts them to the engine's ``Policies`` slots:

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

import math
import re
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
class _InMemoryHistory:
    """A memory-only twin of the pinned ``history.History`` (JSONL) backend.

    The durable product case passes ``history_path`` to
    :func:`build_rrsi_bundle` and uses :class:`~examples.rrsi.history.History`
    -- the pinned per-edit JSONL ledger, readable across processes and resume.
    A run that does not ask for durability (a scratch run, a test) gets this
    in-memory twin, which mirrors the same summaries (:meth:`prune_set`,
    :meth:`yield_g`, :meth:`incumbent_component_counts`, :meth:`render`).
    A parity test pins the two backends to the same answers, so the twin cannot
    silently drift from the pinned file backend."""

    def __init__(self) -> None:
        self._records: List[dict] = []

    def records(self) -> List[dict]:
        return list(self._records)

    def load_records(self, records: List[dict]) -> None:
        """Replace the ledger with ``records`` (a :meth:`records` snapshot), so a
        resumed run starts from the evidence the checkpointed one had."""
        self._records = [dict(r) for r in records]

    def append_candidate(self, t: int, variant: str, edits: Optional[List[dict]],
                         outcome: str, delta_S: Optional[float],
                         delta_C: Optional[float], accepted: bool,
                         S: Optional[float], C: Optional[float],
                         diff: Optional[str], detail: str = "") -> None:
        for e in edits or [{"id": "C1", "component": None, "hypothesis": None}]:
            self._records.append({
                "t": t, "variant": variant, "edit_id": e.get("id"),
                "component": e.get("component"),
                "hypothesis": e.get("hypothesis") or e.get("mechanism"),
                "diff": diff, "delta_S": delta_S, "delta_C": delta_C,
                "accepted": bool(accepted), "outcome": outcome,
                "S": S, "C": C, "bundle": len(edits or []), "detail": detail,
            })

    def _measured(self) -> List[dict]:
        from .components import K
        return [r for r in self._records if r.get("delta_S") is not None
                and r.get("component") in K]

    def tried(self) -> Set[str]:
        return {r["component"] for r in self._measured()}

    def attempted(self) -> Dict[str, int]:
        counts: Dict[str, int] = {c: 0 for c in _K_ALL}
        for r in self._records:
            if r.get("component") in counts and r.get("edit_id"):
                counts[r["component"]] += 1
        return counts

    def accepted_edits(self) -> Dict[str, List[dict]]:
        out: Dict[str, List[dict]] = {c: [] for c in _K_ALL}
        for r in self._records:
            if r.get("accepted") and r.get("component") in out:
                out[r["component"]].append(
                    {"t": r["t"], "edit_id": r.get("edit_id"),
                     "hypothesis": r.get("hypothesis"),
                     "delta_S": r.get("delta_S")})
        return out

    def incumbent_component_counts(self) -> Dict[str, int]:
        return {c: len(v) for c, v in self.accepted_edits().items()}

    def yield_g(self, t: int, n_prune: int) -> Dict[str, float]:
        g = {c: -math.inf for c in self.tried()}
        for r in self._measured():
            if t - int(r["t"]) <= n_prune:
                c = r["component"]
                g[c] = max(g.get(c, -math.inf), float(r["delta_S"]))
        return g

    def prune_set(self, t: int, n_prune: int) -> List[dict]:
        g = self.yield_g(t, n_prune)
        acc = self.accepted_edits()
        out = []
        for c in sorted(g):
            if g[c] <= 0:
                out.append({"component": c,
                            "recent_best_gain": (None if g[c] == -math.inf
                                                 else round(g[c], 5)),
                            "accepted_edits_in_incumbent": acc.get(c, [])})
        return out

    def render(self, n: int = 40) -> List[dict]:
        recs = list(reversed(self._records))
        kept, unmeasured = [], 0
        for r in recs:
            if r.get("delta_S") is None:
                unmeasured += 1
                if unmeasured > 4:
                    continue
            kept.append(r)
            if len(kept) >= n:
                break
        keys = ("t", "variant", "edit_id", "component", "hypothesis",
                "targets_mode", "delta_S", "delta_C", "accepted", "outcome",
                "bundle", "detail")
        return [{k: r.get(k) for k in keys if r.get(k) is not None}
                for r in reversed(kept)]


#: The full component vocabulary (structural + non-structural), for the
#: in-memory twin's ``attempted()`` default keys -- the same ``K`` the pinned
#: file backend initialises its counts from.
from .components import K as _K_ALL  # noqa: E402
_K_ALL = tuple(_K_ALL)


_PATHLIKE = re.compile(r"[/\\]|\.\w{1,8}$")


def _op_diff_text(key: Any, value: Any) -> str:
    """One op as the unified-diff text ``components.normalize`` expects.

    ``text_only`` reads ``+``/``-`` changed-line markers; feeding it the raw
    ``key\nvalue`` made every line look like code, so prose that merely
    *mentions* ``ToolRegistry`` was classified ``client_tool`` and won a
    structural-novelty tie-break (#337). The key becomes the file header (its
    path still carries ``skills/``-style signals) and every value line is an
    added line, so a comment- or string-only edit is recognised as ``prompt``.
    """
    lines = [f"--- a/{key}", f"+++ b/{key}"]
    if value is not None:
        lines += ["+ " + ln for ln in str(value).splitlines()]
    return "\n".join(lines)


def upstream_components(diff: Any) -> Tuple[str, ...]:
    """Classify a diff's ops through the **pinned upstream vocabulary**.

    Each op is classified by :func:`~examples.rrsi.components.normalize`
    (declared=None) over a real changed-line diff of that op, which applies
    ``text_only`` (a pure prose/comment/string edit is a ``prompt``) and then
    the shared structural signals (``skills/`` -> ``skill``, ``Memory(`` ->
    ``memory``, tool registries -> ``client_tool``, subagent calls ->
    ``subagent``).

    An op whose key is a *slot* rather than a file path (``value``, a rule id --
    what ``SingleSlot`` / ``KeyedRules`` / ``AppendRules`` write) holds model
    -facing prose: it is a ``prompt`` edit however it is worded, since no
    machinery is added by words. A FileTree harness's ``skills/x/SKILL.md``
    still attributes its edit to the upstream ``skill`` component.
    """
    from .components import normalize

    out: Set[str] = set()
    for key, value in (diff.ops or {}).items():
        if not _PATHLIKE.search(str(key)):
            out.add("prompt")
            continue
        out.add(normalize(None, _op_diff_text(key, value)))
    return tuple(sorted(out))


class RRsiState:
    """S*, the per-edit history, component bookkeeping and the annealed round.

    One instance is shared by the critic, the proposer and the selector -- that
    sharing is what makes the three slots one regularized algorithm.

    **Fidelity.** The per-edit ledger and the pruning/stall summaries live in a
    :class:`~examples.rrsi.history.History` backend (the pinned upstream JSONL
    ledger) when ``history_path`` is given, or an :class:`_InMemoryHistory`
    twin otherwise. Component attribution goes through the pinned
    :mod:`~examples.rrsi.components` vocabulary (``upstream_components``), not
    a path-segment heuristic. ``checkpoint``/``restore`` carry the *search*
    state (``round``, ``S*``, ``incumbent_cost``, ``progress``); the per-edit
    ledger is the History's own durable file (the in-memory twin's records ride
    the checkpoint instead).
    """

    def __init__(self, config: Optional[RRsiConfig] = None, *,
                 base_score: float = 0.0,
                 components: Optional[Callable[[Any], Tuple[str, ...]]] = None,
                 history_path: Optional[str] = None):
        self.config = config or RRsiConfig()
        self.components = components or upstream_components
        if history_path is not None:
            from .history import History
            self.history = History(history_path)
        else:
            self.history = _InMemoryHistory()
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

        Uses the pinned ``history.stall_flag`` over the ``progress``
        trajectory: ``progress[i]`` is ``S*`` after merge ``i+1`` (the anneal
        advances on every merge), so the latest point is the last element and
        the pinned guard "0 while fewer than ``w`` rounds exist" applies
        through its ``t < w`` check. Structured exploration fires here: the
        proposer reserves part of its budget for components never yet
        exercised."""
        if self.config.stall_window <= 0 or not self.progress:
            return False
        from .history import stall_flag
        return bool(stall_flag(self.progress, len(self.progress) - 1,
                               self.config.stall_window, self.config.noise_band))

    def pruning_targets(self) -> List[str]:
        """Components whose recent measured gain is non-positive (L1-style),
        from the pinned ``history.prune_set``."""
        return sorted({t["component"] for t in
                       self.history.prune_set(self.round, self.config.prune_window)})

    def pruning_edits(self) -> List[dict]:
        """The actionable deletion list from ``history.prune_set``: component +
        the accepted machinery to remove."""
        return self.history.prune_set(self.round, self.config.prune_window)

    def components_of(self, diff: Any) -> Tuple[str, ...]:
        return self.components(diff)

    def record_measurement(self, diff: Any, *, delta_score: float,
                           delta_cost: float, accepted: bool) -> None:
        """Record one measured candidate into the per-edit History.

        Called for accepted *and* rejected candidates: rejected mechanisms are
        negative evidence the proposer conditions on (the paper's evidence-aware
        credit assignment), and a rejection is exactly what the pruner should
        count against a component. Each op is its own edit record, tagged with
        its upstream component."""
        comps = self.components_of(diff)
        self.history.append_candidate(
            t=self.round, variant=getattr(diff, "diff_id", "?"),
            edits=[{"id": f"e{i}", "component": c}
                   for i, c in enumerate(comps)],
            outcome="accepted" if accepted else "rejected",
            delta_S=delta_score, delta_C=delta_cost, accepted=accepted,
            S=self.best_score, C=self.incumbent_cost, diff=_diff_text(diff))

    def checkpoint(self) -> dict:
        """Serialise the search state.

        With a durable ``history_path`` the per-edit ledger is the History's
        own file and is not part of this payload. The in-memory backend has no
        file, so its records ride the checkpoint (``history``): without them a
        resumed bundle kept ``S*``/round/progress but lost accepted component
        counts, pruning and proposer evidence, and an already-accepted skill
        became "novel" again (#337)."""
        out = {
            "round": self.round,
            "best_score": self.best_score,
            "incumbent_cost": self.incumbent_cost,
            "progress": list(self.progress),
        }
        if isinstance(self.history, _InMemoryHistory):
            out["history"] = self.history.records()
        return out

    def restore(self, data: dict) -> None:
        """Restore :meth:`checkpoint`'s form. Unknown keys are ignored."""
        try:
            if isinstance(data.get("round"), int):
                self.round = data["round"]
            if isinstance(data.get("best_score"), (int, float)):
                self.best_score = float(data["best_score"])
            if data.get("incumbent_cost") is not None:
                self.incumbent_cost = float(data["incumbent_cost"])
            if isinstance(data.get("progress"), list):
                self.progress = [float(v) for v in data["progress"]
                                 if isinstance(v, (int, float))]
            # Only the in-memory twin restores records from the payload; a
            # durable History already reads its own file, which stays the source
            # of truth on an explicit resume.
            if (isinstance(self.history, _InMemoryHistory)
                    and isinstance(data.get("history"), list)):
                self.history.load_records(
                    [r for r in data["history"] if isinstance(r, dict)])
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


def _history_summary(state: "RRsiState", max_chars: int) -> str:
    """A compact recent-history summary for the reflection prompt, from the
    pinned ``History.render`` (measured outcomes dominate; a wall of
    unmeasured aborts is not evidence)."""
    records = state.history.render(n=8)
    if not records:
        return ""
    lines = []
    for r in records:
        tag = "accepted" if r.get("accepted") else "rejected"
        comps = r.get("component") or "?"
        dS = r.get("delta_S")
        dC = r.get("delta_C")
        head = f"- r{r.get('t')} [{tag}] {comps}: "
        if isinstance(dS, (int, float)):
            line = head + f"dS={dS:+.2f}"
        else:
            line = head + "dS=?"
        if isinstance(dC, (int, float)):
            line += f" dC={dC:+.2f}"
        lines.append(line)
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
        history = _history_summary(self.state, config.max_history_chars)
        if history:
            lines.append("")
            lines.append("[RRSI history] Measured so far:")
            lines.append(history)
        targets = self.state.pruning_edits()
        if targets:
            parts = []
            for t in targets:
                edits = t["accepted_edits_in_incumbent"]
                ids = ", ".join(e["edit_id"] for e in edits) if edits else ""
                parts.append(f"{t['component']}" + (f" (remove {ids})" if ids else ""))
            lines.append("")
            lines.append("[RRSI pruning] These components have not earned their "
                         "place recently; consider deleting them: "
                         + "; ".join(parts))
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
            selection_config, self.state.history.incumbent_component_counts())

        # The schedule advances on *every* decision this policy makes -- a
        # rejection is a round the search spent stuck, not a free one (see
        # RRsiState.record_merge). best_score is advanced only on an accept,
        # and the accept records the **post-decision** S*: appending before the
        # update would make an improve-after-rejection-streak look stalled even
        # though S* just moved (review).
        if not decision.admissible:
            self.state.record_merge()
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
        self.state.record_merge()
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
        """Advance the schedule on a pre-measurement merge refusal.

        The screen (critic) and the governance invariant gate both run in
        ``Aggregator._prepare``, before this policy's ``accept`` is ever reached.
        Without this hook a streak of screen-rejected or violates-invariant
        batches would leave the anneal and the stall detector frozen even though
        a merge was decided. The aggregator calls this once per such refusal;
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
    history_path: Optional[str] = None,
) -> Tuple[Policies, RRsiState]:
    """Assemble the three RRSI policies into one ``Policies`` bundle.

    Returns ``(bundle, state)``; keep ``state`` to read (or checkpoint) the
    historical-best floor and the measured history. ``protected`` is the evolve
    set's distinctive strings (task ids, gold answers, names) -- never the
    final holdout. ``complete`` drives the proposer; an optional ``reviewer``
    drives the model critic. ``base_score`` seeds ``S*`` (the base harness's
    evolve-set score when you have it). ``history_path``, when given, is the
    durable per-edit ledger (the pinned ``history.History`` JSONL) -- pass the
    run's ``repo_path``-adjacent file for resume that reads the same evidence
    the loop conditioned on; ``None`` keeps an in-memory ledger (scratch runs,
    tests)."""
    state = RRsiState(config, base_score=base_score, history_path=history_path)
    bundle = Policies(
        screen=RRsiCritic(state, protected=protected, reviewer=reviewer),
        proposal=RRsiProposalPolicy(complete, state),
        acceptance=RRsiAcceptance(state),
    )
    return bundle, state
