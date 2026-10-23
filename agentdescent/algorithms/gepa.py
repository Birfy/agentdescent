"""GEPA's Pareto illumination, shipped as a named algorithm (#384).

The mechanism is the paper's Algorithm 2: keep a pool of candidates each scored
on every ``D_pareto`` instance, admit a child that does not regress on its
feedback sample, then sample the next parent from the per-instance Pareto
frontier weighted by unique wins. The example keeps its HotpotQA domain,
prompts and CLI and imports this implementation, so there is one copy of the
algorithm rather than one per entry point.

Use it from a spec as ``"algorithm": "gepa"`` (``kind: text``). The engine's
rollouts and reflective mutations are unchanged; this replaces the merger and
acceptance path with the Pareto pool.
"""

from __future__ import annotations

import random
import threading
from typing import Dict, List, Optional, Set

from ..aggregator import AggregatorProtocol, MergeReport
from ..evolvable import Diff, EvidenceCard
from ..ledger import CASConflict, Ledger
from ..selection import ParetoFrontier, pareto_win_frequency
from . import Algorithm, BuildContext, Overrides, Param, register

__all__ = ["ParetoAggregator", "pareto_aggregator_factory", "pareto_frontier",
           "pareto_select"]

#: Algorithm 2 steps 1-4, now shipped as
#: :func:`agentdescent.selection.pareto_win_frequency`. Kept under this name
#: because it is what the paper's step numbering calls it and what the port's
#: tests reach for.
pareto_frontier = pareto_win_frequency


def pareto_select(scores: List[List[float]], rng: random.Random) -> int:
    """Sample a parent index ~ its Pareto win-frequency (Algorithm 2, step 5).

    The index form, for the tests and for anyone reading the algorithm rather
    than driving the engine. The engine drives
    :class:`~agentdescent.selection.ParetoFrontier` with ``mode='win_frequency'``,
    which is this function over `Candidate` rows and draws from the same rng in
    the same order -- `tests/test_port_selection_equivalence.py` steps the two
    in lockstep rather than comparing their distributions.
    """
    kept, freq = pareto_win_frequency(scores)
    cands = [c for c in sorted(kept) if freq[c] > 0]
    if not cands:  # degenerate: fall back to best average
        return max(range(len(scores)), key=lambda c: sum(scores[c]))
    weights = [freq[c] for c in cands]
    return rng.choices(cands, weights=weights, k=1)[0]


class ParetoAggregator(AggregatorProtocol):
    """A drop-in optimizer that realises GEPA's genetic/Pareto loop.

    Keeps a **pool** of candidate instructions (states), each with its per-
    instance score row over ``D_pareto`` (= the verifier's held-out set). Each
    round it admits the round's mutated candidates that did not regress on their
    feedback sample (GEPA Alg. 1: "if sigma' improves"), then samples the next
    parent from the Pareto frontier and makes it the dev head, so `evolve()`
    mutates the illumination-selected parent next round.

    **Documented deviation: the feedback minibatch has size 1.** GEPA evaluates a
    child on a minibatch of size *b* and compares means; `evolve()` gives one
    rollout per worker per round, so the comparison is a single Bernoulli draw.
    The admission test is therefore "did not regress" rather than "improved" --
    see the note at the gate. Everything downstream (Algorithm 2's per-instance
    frontier, domination pruning, win-frequency sampling) is scored on the full
    ``D_pareto`` row and is unaffected."""

    def __init__(self, ledger: Ledger, verifier, audit, config, policy,
                 artifact_id: str = "gepa_prompt", seed: int = 0,
                 eval_concurrency: int = 8, merge_round=None,
                 selection: Optional["ParetoFrontier"] = None):
        self.ledger = ledger
        self.eval_concurrency = eval_concurrency
        #: Optional :class:`~agentdescent.policies.FusionPolicy`. When set, the
        #: round's surviving diffs are combined into **one** candidate before
        #: admission, so the pool grows by one per round instead of one per
        #: worker.
        #:
        #: A policy, not a bare callable, because the shipped one needs the
        #: verifier: `ReflectiveFusion` builds its synthesis-failure fallback in
        #: `bind()`, which only the engine calls when the policy is installed
        #: through `Policies`. Used from here that never happens, and the
        #: fallback stays `None` until a merge fails and the run dies on
        #: `'NoneType' has no attribute 'select'` -- twenty minutes in, on the
        #: rare path. Binding here is what makes the policy reusable outside the
        #: engine's own wiring.
        self.merge_round = merge_round
        if merge_round is not None and hasattr(merge_round, "bind"):
            merge_round.bind(verifier)
        self.verifier = verifier
        self.artifact_id = artifact_id
        self.rng = random.Random(seed)
        # Algorithm 2 at the shipped seam. `rng=` rather than `seed=` because
        # this aggregator owns the stream and spends it elsewhere too: a
        # policy that re-seeded per call would agree on the distribution and
        # disagree with every number this port has published.
        self.selection = selection or ParetoFrontier(
            mode="win_frequency", rng=self.rng)
        self._cards: List[EvidenceCard] = []
        self._lock = threading.Lock()   # ingest: workers; step: one thread
        # pool: list of (state_dict, per_instance_scores, avg). Parallel lists so
        # score rows stay aligned for pareto_frontier.
        self.states: List[Dict[str, str]] = []
        self.scores: List[List[float]] = []
        self._seen: Set[str] = set()

    # -- per-instance evaluation over D_pareto -------------------------------

    def _score_rows(self, artifacts) -> List[List[float]]:
        """Per-instance rows over D_pareto for several candidates at once.

        The port's largest cost, and it was serial in **two** dimensions. Within
        a row it was a comprehension over D_pareto -- on 20 instances at ~34 s a
        call, eleven minutes per candidate, in silence. Across rows, `step()`
        admitted one card at a time, so a round from eight workers queued eight
        of those sweeps back to back. `evolve(eval_concurrency=)` reaches
        neither, because that knob bounds the *engine's* gate and this aggregator
        replaces the engine's.

        Both dimensions flatten into one pool rather than nesting, so in-flight
        work is exactly ``eval_concurrency`` instead of its square -- nesting two
        pools of 16 would put 256 requests on the endpoint and measure its rate
        limiter rather than the algorithm.

        Concurrency changes no result: each (candidate, instance) score is
        independent, ``eval_fn`` is pure, rows are reassembled in order, and the
        evaluation cache already memoises per (artifact, task).
        """
        artifacts = list(artifacts)
        tasks = list(self.verifier.held_out)
        if not artifacts or not tasks:
            return [[] for _ in artifacts]
        pairs = [(a, t) for a in artifacts for t in tasks]
        if len(pairs) < 2 or self.eval_concurrency < 2:
            flat = [self.verifier.eval_fn(a, [t]) for a, t in pairs]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(
                    max_workers=min(self.eval_concurrency, len(pairs))) as pool:
                flat = list(pool.map(
                    lambda pair: self.verifier.eval_fn(pair[0], [pair[1]]), pairs))
        width = len(tasks)
        return [flat[i * width:(i + 1) * width] for i in range(len(artifacts))]

    def _score_row(self, artifact) -> List[float]:
        """One candidate's row. For callers that admit a single artifact."""
        return self._score_rows([artifact])[0]

    def _admit_many(self, artifacts) -> int:
        """Admit several candidates, scoring all their rows in one pass.

        Dedup runs first and sequentially, so which candidates enter the pool --
        and in what order -- never depends on thread timing. Only the scoring is
        concurrent, and `states`/`scores` are appended in dedup order because
        `pareto_frontier` reads them as parallel lists.
        """
        fresh = []
        for artifact in artifacts:
            key = artifact.render()
            if key in self._seen:
                continue
            self._seen.add(key)
            fresh.append(artifact)
        if not fresh:
            return 0
        for artifact, row in zip(fresh, self._score_rows(fresh)):
            self.states.append(dict(artifact.state))
            self.scores.append(row)
        return len(fresh)

    def _admit(self, artifact) -> None:
        key = artifact.render()
        if key in self._seen:
            return
        self._seen.add(key)
        self.states.append(dict(artifact.state))
        self.scores.append(self._score_row(artifact))

    def best_index(self) -> int:
        return max(range(len(self.scores)),
                   key=lambda c: sum(self.scores[c])) if self.scores else -1

    @property
    def best_avg(self) -> float:
        i = self.best_index()
        return sum(self.scores[i]) / len(self.scores[i]) if i >= 0 else 0.0

    @property
    def best_state(self) -> Dict[str, str]:
        i = self.best_index()
        return dict(self.states[i]) if i >= 0 else {}

    def finalize(self) -> None:
        """Leave GEPA's best-average candidate on dev, not the last random draw.

        ``step()`` samples the next parent from the frontier; that is the right
        choice *during* the search, but the result `evolve()` returns is the dev
        head, so without this a spec run would hand back a random frontier
        member. The example read `factory.holder["agg"].best_state` instead, a
        workaround a spec/CLI/MCP caller cannot see. The engine calls
        ``finalize()`` on a clean run (the same hook the population aggregator
        uses for its best state).
        """
        if not self.states:
            return
        i = self.best_index()
        if i < 0:
            return
        target = self.states[i]
        try:
            snap = self.ledger.snapshot(Ledger.DEV)
        except Exception:  # noqa: BLE001 - finalize is a courtesy, never fatal
            return
        head = snap.get(self.artifact_id)
        if head is None or head.state == target:
            return
        base_vv = {self.artifact_id: snap.version.get(self.artifact_id, 0)}
        parent = head.apply(Diff(diff_id=f"gepa-final:{i}",
                                 target=self.artifact_id, ops=dict(target),
                                 author="gepa"))
        try:
            self.ledger.commit(parent, base_vv, branch=Ledger.DEV,
                               message=f"gepa: final best-average #{i}")
        except CASConflict:
            pass

    # -- AggregatorProtocol --------------------------------------------------

    def ingest(self, card: EvidenceCard) -> None:
        # ingest runs on worker threads, step on one: see AggregatorProtocol.
        with self._lock:
            self._cards.append(card)

    def step(self) -> List[MergeReport]:
        snap = self.ledger.snapshot(Ledger.DEV)
        head = snap.get(self.artifact_id)
        base_vv = {self.artifact_id: snap.version.get(self.artifact_id, 0)}
        if not self.states:                       # seed the pool with the base system
            self._admit(head)

        with self._lock:
            cards, self._cards = self._cards, []
        survivors: List[Diff] = []
        for card in cards:
            # GEPA Alg. 1 admits a child whose score on the feedback minibatch is
            # not worse than its parent's. Here that "minibatch" has size **1**:
            # `evolve()` rolls out one task per worker per round, so
            # `before_after_delta` is a single-instance measurement, and for a
            # binary reward like HotpotQA EM it is exactly {-1, 0, +1}.
            #
            # So this gate must not be `> 0`. That demanded the one sampled
            # instance flip wrong->right -- and it is the instance the mutation was
            # generated *from*, so a prompt that helps broadly but does not fix that
            # particular question was thrown away before it was ever scored.
            # Rejected candidates never enter `self.states`, never get a
            # `_score_row`, and so can never reach the Pareto frontier -- which is
            # precisely the "complementary specialist" the frontier exists to keep
            # alive. `>= 0` filters obvious regressions and leaves the selection to
            # domination pruning, which is what illumination is supposed to do.
            if card.before_after_delta < 0:
                continue
            survivors.append(card.diff)

        # **One candidate per round instead of one per worker.** Admission is what
        # this aggregator pays for: every candidate that enters the pool is scored
        # across the whole of D_pareto, so a round from N workers costs N sweeps.
        # Combining the round's survivors into a single diff first makes that one
        # sweep, which is the only way widening N makes the run *cheaper* rather
        # than merely wider.
        #
        # `fuse_diffs` is `ops.update()`, so on this port's one-key artifact it
        # would keep the last rewrite and silently drop the rest -- N-1 proposals
        # paid for and thrown away. `merge_round` is therefore a *model* merger
        # (`ReflectiveFusion`), which writes the union of the deltas.
        #
        # It changes what the Pareto pool contains: merged candidates rather than
        # individual mutations. Algorithm 2 itself -- the per-instance frontier,
        # domination pruning, win-frequency sampling -- is untouched and still
        # runs on whatever is in the pool. A row measured this way has to say so.
        if self.merge_round is not None and len(survivors) > 1:
            merged, _, _ = self.merge_round.select(head, survivors)
            if merged is not None:
                survivors = [merged]
        admitted = self._admit_many([head.apply(d) for d in survivors])

        # Algorithm 2 at the standard seam: candidates carry their per-instance
        # rows in Candidate.per_task; version carries the pool index.
        from agentdescent.selection import Candidate, SelectionContext
        task_ids = [t.id for t in self.verifier.held_out]
        rows = [Candidate(artifact_id=self.artifact_id, version=i,
                          state=dict(state),
                          score=sum(row) / len(row) if row else 0.0,
                          per_task=dict(zip(task_ids, row)))
                for i, (state, row) in enumerate(zip(self.states, self.scores))]
        ctx = SelectionContext(head=rows[0], candidates=tuple(rows), n_workers=1)
        k = self.selection.select(ctx, 1)[0].version
        target_state = self.states[k]
        report_diff = None
        committed_version = None
        if target_state != head.state:            # make it the dev head
            parent = head.apply(Diff(diff_id=f"pareto:{k}", target=self.artifact_id,
                                     ops=dict(target_state), author="gepa"))
            try:
                _, committed_version = self.ledger.commit(
                    parent, base_vv, branch=Ledger.DEV,
                    message=f"gepa: select pareto parent #{k}")
                report_diff = parent.diff(head)
            except CASConflict:
                committed_version = None
        return [MergeReport(self.artifact_id, report_diff, False, len(cards),
                            admitted, 0, 0, self.best_avg, committed_version,
                            f"pool={len(self.states)} parent=#{k} best_avg={self.best_avg:.3f}",
                            category="gepa-select")]


def pareto_aggregator_factory(artifact_id: str = "gepa_prompt", seed: int = 0,
                              eval_concurrency: int = 8, merge_round=None):
    """Build the `aggregator_factory=` callable evolve() expects, capturing the
    live aggregator in `.last` so the example can read GEPA's best candidate."""
    holder = {}

    def factory(ledger, verifier, audit, config, policy):
        agg = ParetoAggregator(ledger, verifier, audit, config, policy,
                               artifact_id=artifact_id, seed=seed,
                               eval_concurrency=eval_concurrency,
                               merge_round=merge_round)
        holder["agg"] = agg
        return agg

    factory.holder = holder  # type: ignore[attr-defined]
    return factory


def _build(ctx: BuildContext, p: Dict[str, object]) -> Overrides:
    return Overrides(
        knobs={"aggregator_factory": pareto_aggregator_factory(
            artifact_id=ctx.spec.artifact_id(), seed=int(p["seed"]),
            eval_concurrency=int(p["eval_concurrency"]))},
        notes=["algorithm gepa: the Pareto pool replaces the merge/acceptance path; "
               "the spec's agent/reflect still run the rollouts and mutations.",
               "algorithm gepa: the pool is in memory and re-seeds from dev on a "
               "resumed ledger; the example's ReflectiveFusion merge is example-only."])


register(Algorithm(
    name="gepa",
    summary="GEPA's genetic reflective loop: keep a Pareto pool of candidates scored "
            "per instance and mutate a win-frequency-selected parent instead of the "
            "single best average.",
    kinds=("text",),
    fidelity="mechanism_microport",
    params=(
        Param("seed", "int", 0, "seed for the Pareto parent draw"),
        Param("eval_concurrency", "int", 8,
              "per-instance scorings run in parallel when a candidate enters the pool",
              minimum=1),
    ),
    build=_build,
    source="GEPA: Reflective Prompt Evolution (Agrawal et al., 2025); arXiv:2507.19457",
    extra_calls="every admitted candidate is scored over the full held-out set once "
                "(the Pareto row); no extra reflection calls",
    doc="algo-gepa.md"))
