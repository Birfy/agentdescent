"""The seam between *the algorithm* and *the machinery that runs it*.

`evolve()` already had nine replaceable pieces -- the actors, the strategy, the
parallel plan, the task sampler, the staleness policy, the aggregator factory.
What it did not have was any way to replace one *decision* without replacing the
whole optimizer: acceptance, conflict resolution, fusion and promotion all live
inside a single 150-line method, so "I want a different accept rule" meant
reimplementing `Aggregator`.

This module is the contract for those decisions, and nothing else. It holds
`Protocol` definitions and the small frozen records they exchange; the default
implementations stay exactly where they are today. Two consequences worth being
explicit about:

**Protocols, not base classes.** `typing.Protocol` is structural, so
`ThreeLayerVerifier`, `Ledger` and `Aggregator` satisfy these contracts without
inheriting anything and without a line changing. An ABC would have required
edits to all three, which is a behaviour change dressed as a definition.

**Written from the call sites, not from the docs.** `docs/verifier.md` once
described `ThreeLayerVerifier` as three methods, having missed `learned_eval`;
a verifier written against that page raises `AttributeError` half an hour into a
run, not at startup. So every method here comes from grepping what the engine and
the aggregator actually call -- and `tests/test_policy_contract.py` re-derives
those sets and fails if this file drifts from them.

That rule already earned its keep: writing `LedgerProtocol` from the aggregator's
four calls would have shipped a contract that omits `register`, `close` and
`log`, which the *engine* calls -- so the first injected ledger would have died
in `_build_engine`.

What is *not* here: sandbox provisioning, executors, worker pools. Those are the
execution and resource planes; the algorithm is not allowed to know about them,
which is what keeps a run reproducible when it moves to another machine.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import (
    TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Sequence, Tuple,
    runtime_checkable,
)

if TYPE_CHECKING:                                   # pragma: no cover
    from .aggregator import AggregatorFactory, MergeReport
    from .evalcache import CacheProtocol
    from .executor import Executor
    from .evaluator import EvaluatorGroup
    from .evolvable import Diff, EvidenceCard, Evolvable
    from .sampling import TaskSampler
    from .selection import SelectionPolicy
    from .staleness import StalenessPolicy
    from .stats import BetaPosterior

__all__ = [
    "AcceptDecision",
    "AcceptancePolicy",
    "ConflictPolicy",
    "FusionPolicy",
    "FusionTrial",
    "LedgerProtocol",
    "MergeContext",
    "Policies",
    "Promotion",
    "PromotionPolicy",
    "ProposalContext",
    "ProposalPolicy",
    "Sandbox",
    "SandboxProvider",
    "SandboxSpec",
    "VerifierProtocol",
]


# ---------------------------------------------------------------------------
# What a decision is given, and what it returns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeContext:
    """Everything an :class:`AcceptancePolicy` is allowed to look at.

    ``base_counts`` / ``cand_counts`` are the **full** held-out measurement
    (``verifier.eval_counts``); ``base_cheap`` / ``cand_cheap`` are the cheap
    layer's sub-sample (``verifier.cheap_eval``). They are separate fields
    because conflating them is a bug this repository has already shipped once:
    the regression guard read the cheap layer, so lowering ``cheap_eval_tasks``
    silently made the guard judge "quality dropped" from a sub-sample. Ranking
    may use the cheap numbers. **Deciding to commit may not.**
    """

    artifact: "Evolvable"
    candidate: "Evolvable"
    cards: Sequence["EvidenceCard"]
    #: ``(successes, failures)`` over the whole held-out set -- what
    #: ``verifier.eval_counts`` returns, and what the commit gates read. Not
    #: ``(successes, trials)``: this contract was written from the aggregator's
    #: call and the call's meaning, and a policy that divides by the second
    #: element gets a rate that is wrong and plausible.
    base_counts: Tuple[float, float]
    cand_counts: Tuple[float, float]
    #: The diff under consideration. Needed for a reproducible acceptance draw
    #: (the sampler is seeded per candidate) and to tell which cards contributed.
    diff: Optional["Diff"] = None
    #: Cheap-layer scores over a sub-sample -- ranking only.
    base_cheap: float = 0.0
    cand_cheap: float = 0.0
    prior: Optional["BetaPosterior"] = None
    #: Fraction of the artifact a single diff may touch.
    trust_radius: float = 0.2
    #: How far the candidate sits from the ``stable`` branch, as a fraction of
    #: keys that differ, in ``[0, 1]``. The KL-to-a-reference-policy analogy, and
    #: information today's default gate does not use: L1 safety rests entirely on
    #: the oracle's single veto, and "how far is this from the thing known to
    #: work" is exactly what a blast radius should be weighed against. ``0.0``
    #: when the run has no ``stable`` branch yet, which is the same as "no
    #: reference to be far from". See
    #: :class:`~agentdescent.advantage.StableDistanceAcceptance`.
    stable_distance: float = 0.0
    #: Fingerprint of the environment each side was measured in. Equal on a
    #: single-machine run, so the comparison is always valid there. Once
    #: measurements can come from different toolchains they stop being
    #: comparable, and a policy that ignores this reads an environment
    #: difference as a quality change -- in either direction.
    base_env: str = ""
    cand_env: str = ""

    @staticmethod
    def rate(counts: Tuple[float, float]) -> float:
        """Success rate from ``(successes, failures)``, guarding the empty case."""
        successes, failures = counts
        return successes / max(1e-9, successes + failures)

    def comparable(self) -> bool:
        """Were both sides measured in the same environment?"""
        return self.base_env == self.cand_env


@dataclass(frozen=True)
class FusionTrial:
    """One tournament: what the fused candidate scored against the best single.

    The sharpest objection to "diffs can be merged" is that two workers' local
    improvements can be worse together than either is alone. The tournament has
    always been the answer to it -- the fused candidate has to *win* on held-out
    before anything commits -- and the run recorded only how often a fusion was
    committed, never how often it won, by how much, or how badly it lost. That is
    a design, not evidence.

    ``baseline_score`` is why this is a record rather than a boolean. A fusion
    that loses to the best single diff while both beat the baseline is complementary
    changes ranking imperfectly; a fusion that loses to the *baseline* is two good
    changes making each other worse. Only the second is the failure mode the
    objection describes, and ``winner`` alone cannot tell them apart.

    All three scores come from the cheap layer, which is what the tournament
    ranks on. They are comparable with each other and not with a full held-out
    number.
    """

    artifact_id: str
    #: Surviving diffs entering the tournament -- the fused candidate is built
    #: from all of them, so this is what "of N" means in a merge-of-N claim.
    n_candidates: int
    best_single_score: float
    baseline_score: float
    #: ``None`` when no fused candidate was built at all; ``reason`` says why.
    #: Distinguishing that from "the fusion lost" is the difference between a
    #: mechanism that fails and a mechanism that never ran.
    fused_score: Optional[float] = None
    #: Score of a **model-synthesised** candidate, when a fusion policy built one
    #: -- :class:`~agentdescent.fusion.ReflectiveFusion` does, for keys the diffs
    #: disagree on. ``None`` when no model was involved, which is the default.
    #: Separate from ``fused_score`` so "the union of independent edits helped"
    #: and "a model rewrote two competing edits into one" are never one number.
    synthesized_score: Optional[float] = None
    #: ``"fused"`` / ``"synthesized"`` / ``"single"`` / ``"neither"``.
    #: ``"neither"`` means nothing in the tournament beat the artifact it
    #: started from.
    winner: str = "single"
    #: Why no fusion was built: ``"single-candidate"``, ``"contradiction"``, or
    #: ``"synthesis-failed"`` (a model was asked and could not be used -- a dead
    #: backend, an empty answer, an oversized one, or one that merely repeated an
    #: input). Empty when a fused candidate did reach the tournament.
    reason: str = ""

    @property
    def gain(self) -> Optional[float]:
        """Fused minus best single. Negative is the interesting direction."""
        if self.fused_score is None:
            return None
        return self.fused_score - self.best_single_score


@dataclass(frozen=True)
class AcceptDecision:
    """Commit or not, and -- when not -- which of the merge categories it was.

    ``category`` is not decoration: it is the difference between "the gate says
    these proposals do not help" and "they never reached the gate", which need
    opposite fixes and which a boolean cannot express.
    """

    accept: bool
    category: str = ""
    detail: str = ""
    #: The posterior the decision rested on, carried out so the caller can report
    #: it without recomputing -- the draw is sampled, so recomputing gives a
    #: different number than the one that decided.
    p_improve: float = 0.0
    #: Measured change in the full held-out rate. The aggregator folds this back
    #: into the artifact's running prior when a candidate is refused.
    observed_delta: float = 0.0


@dataclass(frozen=True)
class Promotion:
    """One artifact the :class:`PromotionPolicy` believes ``stable`` should hold."""

    artifact_id: str
    reason: str = ""


@dataclass(frozen=True)
class ProposalContext:
    """What a :class:`ProposalPolicy` is given for one rollout.

    A superset of today's ``propose(rendered, task, output, reward)``: that
    signature returns a single proposal and sees neither its own history nor the
    version it is proposing against, which rules out the whole family of
    algorithms that sample k variants and pick among them, and makes "do not
    re-propose what was just rejected" inexpressible.
    """

    rendered: str
    task: Any
    output: str
    reward: float
    #: The artifact version this proposal is based on -- what staleness is
    #: measured against once the proposal takes a while to come back.
    base_version: int = 0
    #: How many proposals the caller would like. The default policy ignores it
    #: and returns at most one; asking for more is a different algorithm.
    k: int = 1
    #: Proposals this worker already made, newest last. Empty unless the policy
    #: asked to be given them.
    history: Sequence[str] = ()


# ---------------------------------------------------------------------------
# The algorithm: what a run decides
# ---------------------------------------------------------------------------


@runtime_checkable
class ProposalPolicy(Protocol):
    """How a rollout becomes candidate changes."""

    def propose(self, ctx: ProposalContext) -> Sequence[str]: ...


@runtime_checkable
class ConflictPolicy(Protocol):
    """Which of a batch of mutually contradictory changes survive.

    Returns ``(survivors, dropped_count)``. Signature taken from
    ``Aggregator._resolve_conflicts``."""

    def resolve(self, artifact: "Evolvable",
                cards: Sequence["EvidenceCard"]) -> Tuple[List["EvidenceCard"], int]: ...


@runtime_checkable
class FusionPolicy(Protocol):
    """How complementary diffs become one candidate.

    Returns ``(chosen_diff, candidate_artifact, fused)``. Signature taken from
    ``Aggregator._tournament``.

    A policy may also expose ``trials`` -- a growing sequence of
    :class:`FusionTrial` -- and the engine will carry it onto
    :attr:`~agentdescent.evolution.EvolutionResult.fusion_trials`. Optional
    because it is instrumentation, not a decision: a replacement policy that
    keeps no trials still satisfies this protocol, and
    :meth:`EvolutionResult.fusion_stats` reports the trial count so that
    "not instrumented" cannot be misread as "fusion never won"."""

    def select(self, artifact: "Evolvable",
               diffs: List["Diff"]) -> Tuple["Diff", "Evolvable", bool]: ...


@runtime_checkable
class AcceptancePolicy(Protocol):
    """Whether a candidate is committed."""

    def accept(self, ctx: MergeContext) -> AcceptDecision: ...


@runtime_checkable
class PromotionPolicy(Protocol):
    """Which artifacts ``dev`` has proved well enough to copy onto ``stable``."""

    def observe(self, reports: Sequence["MergeReport"]) -> Sequence[Promotion]: ...


# ---------------------------------------------------------------------------
# The machinery: what a run is built on
# ---------------------------------------------------------------------------


@runtime_checkable
class VerifierProtocol(Protocol):
    """Four methods, from ``grep 'self\\.verifier\\.' agentdescent/aggregator.py``.

    Not three. `docs/verifier.md` said three for a while, and a verifier written
    from that page fails in the middle of a merge rather than at startup."""

    def cheap_eval(self, artifact: "Evolvable") -> float: ...
    def learned_eval(self, artifact: "Evolvable") -> Tuple[float, float]: ...
    def eval_counts(self, artifact: "Evolvable") -> Tuple[float, float]: ...
    def oracle_eval(self, artifact: "Evolvable") -> float: ...


@runtime_checkable
class LedgerProtocol(Protocol):
    """Seven methods: four the aggregator calls, three more the engine calls.

    Deriving this from the aggregator alone -- ``snapshot``, ``head_version``,
    ``commit``, ``promote_to_stable`` -- yields a contract that type-checks and
    then dies in ``_build_engine``, which calls ``register`` before any merge
    happens, in ``_safe_log`` at the end of every run, and in ``_Engine.cleanup``.
    """

    def register(self, artifact: "Evolvable", branch: str = ...) -> None: ...
    def snapshot(self, branch: str = ...) -> Any: ...
    def head_version(self, branch: str = ...) -> Dict[str, int]: ...
    def commit(self, new_state: "Evolvable", base_version: Dict[str, int],
               branch: str = ..., message: str = ...) -> Tuple[str, int]: ...
    def promote_to_stable(self, artifact_id: str) -> Optional[int]: ...
    def log(self, branch: str = ..., limit: int = ...) -> List[str]: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# The resource plane: declared here, implemented elsewhere
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SandboxSpec:
    """What environment one rollout needs. Must survive JSON: it crosses processes.

    Note what is *not* here: any secret. ``env_allowlist`` carries variable
    **names**, and the values are read inside the sandbox. A spec gets pickled,
    logged, cached and written to a journal, so a spec that carried an API key
    would leak it into all four.
    """

    #: Toolchain identity. ``None`` means "run here", which is today's behaviour.
    image: Optional[str] = None
    #: Names of environment variables to pass through. Names only.
    env_allowlist: Tuple[str, ...] = ()
    workspace_root: Optional[str] = None
    #: ``None`` leaves it to the provider, ``"none"`` forbids it, ``"inherit"``
    #: asks for the host's network.
    #:
    #: Three states rather than two, because "the default" means opposite things
    #: to the two providers. A local workspace cannot restrict the network at
    #: all, so for it the honest description is "inherited"; a container can, and
    #: for it the only safe default is off -- a candidate that can reach the
    #: network can send whatever it managed to read. With a two-state field the
    #: shared default had to be wrong for one of them.
    network: Optional[str] = None
    cpu: Optional[float] = None
    memory_mb: Optional[int] = None
    timeout_s: Optional[float] = None
    #: Reuse a warm sandbox instead of a fresh one. Off by default, and that is
    #: a correctness default rather than a conservative one: the frozen-file
    #: overlay assumes a clean workspace, and a dirty one produces scores that
    #: look entirely ordinary and are wrong.
    reuse: bool = False

    def fingerprint(self) -> str:
        """Stable identity of this environment.

        Everything that has to know "were these two numbers measured in the same
        place?" hangs off this: whether a candidate may be compared against a
        base, whether a cached evaluation may be reused, whether a row in a
        comparison table is trustworthy. So it is order-independent and stable
        across processes -- `hash()` is neither."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@runtime_checkable
class Sandbox(Protocol):
    """One acquired execution environment."""

    #: Unique across hosts: host + pid + counter. Attribution needs it.
    id: str
    #: Root of the workspace.
    root: str
    #: The fingerprint of the spec that produced it.
    fingerprint: str


@runtime_checkable
class SandboxProvider(Protocol):
    """Where sandboxes come from and go back to.

    ``acquire`` may block: a bounded pool that queues is the point, and one that
    fails when full would push the retry logic into every caller."""

    def acquire(self, spec: SandboxSpec) -> Sandbox: ...
    def release(self, sandbox: Sandbox) -> None: ...
    def reap(self, older_than: float) -> int: ...


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------


@dataclass
class Policies:
    """Every replaceable piece, in one argument.

    `evolve()` has thirty-five parameters. Each capability after this one --
    executors, sandbox providers, cache backends -- would otherwise add another
    handful, to two entry points, plus the generated API reference and the
    public-surface test. So they go here instead, and the existing keyword
    arguments stay as shortcuts onto these fields; nothing is deprecated.

    Every field is ``None`` by default and ``None`` means "the current
    behaviour", so `Policies()` and passing nothing are the same run.
    """

    # the algorithm
    task_sampler: Optional["TaskSampler"] = None
    #: Which candidate the next batch of workers starts from. `None` is
    #: :class:`~agentdescent.selection.SingleHead` -- the current head, for every
    #: worker, which is what the engine has always done.
    selection: Optional["SelectionPolicy"] = None
    proposal: Optional[ProposalPolicy] = None
    conflict: Optional[ConflictPolicy] = None
    fusion: Optional[FusionPolicy] = None
    acceptance: Optional[AcceptancePolicy] = None
    promotion: Optional[PromotionPolicy] = None
    staleness: Optional["StalenessPolicy"] = None

    # the machinery
    verifier: Optional[VerifierProtocol] = None
    ledger: Optional[LedgerProtocol] = None
    aggregator_factory: Optional["AggregatorFactory"] = None
    #: Where evaluations are memoised. The default is in-process; a shared one
    #: lets separate processes stop paying twice for the same gate.
    eval_cache: Optional["CacheProtocol"] = None
    #: Where rollouts run. `None` is in-process threads. Listed in #54's design
    #: and missed in its implementation, which nothing noticed until the round
    #: body tried to read it -- a bundle field that does not exist and a bundle
    #: field that is ignored fail in opposite ways, and only one of them is loud.
    executor: Optional["Executor"] = None
    #: The gate's own concurrency, separate from the rollouts'. Evaluation is
    #: batched, cacheable and decides commits; a rollout is long-tailed, often
    #: fails, and is one opinion among many. Sizing them together means sizing
    #: them for whichever matters less.
    evaluator: Optional["EvaluatorGroup"] = None

    # the resource plane (implementations land with the sandbox pool)
    sandbox_provider: Optional[SandboxProvider] = None
    sandbox_spec: Optional[SandboxSpec] = None

    def is_empty(self) -> bool:
        """True when this bundle asks for nothing, i.e. the default run."""
        return all(getattr(self, f) is None for f in self.__dataclass_fields__)

    def require_supported(self, supported: Sequence[str], where: str) -> None:
        """Raise if this bundle asks for something ``where`` cannot honour yet.

        The alternative -- accepting the field and ignoring it -- is the worst
        possible behaviour: a caller who passes a custom acceptance rule and sees
        a completed run has every reason to believe the rule ran. Failing names
        the field and the issue that implements it."""
        asked = [f for f in self.__dataclass_fields__
                 if getattr(self, f) is not None and f not in supported]
        if asked:
            raise NotImplementedError(
                f"{where} does not honour Policies({', '.join(sorted(asked))}) yet, "
                "so passing them would look like they took effect. Supported here: "
                f"{', '.join(sorted(supported))}.")

    def merged_with(self, **kwargs: Any) -> "Policies":
        """This bundle, with any non-``None`` keyword taking precedence.

        How the legacy keyword arguments fold in: they are shortcuts onto these
        fields, and an explicit argument should win over a bundle default rather
        than being quietly ignored."""
        merged = {f: getattr(self, f) for f in self.__dataclass_fields__}
        for key, value in kwargs.items():
            if key not in merged:
                raise TypeError(f"Policies has no field {key!r}")
            if value is not None:
                merged[key] = value
        return Policies(**merged)
