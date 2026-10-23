"""Co-evolving the verifier and the tasks it is judged on.

A verifier is a proxy. A curriculum that only samples tasks the verifier
already scores well measures the proxy against itself, so the two can agree
while both drift from the oracle (Goodhart). This module makes the curriculum
target the verifier's *disagreement* with oracle labels, and gives the verifier
a way to learn from those labels, so the task distribution and the proxy improve
against the same ground truth.

Two pieces, both independent of a model:

* :class:`VerifierCurriculum` -- probe a task pool with the verifier and the
  oracle, rank by the gap, and (optionally) feed the selected oracle labels back
  into the verifier via the caller's ``update`` callable.
* :class:`VerifierAgreement` -- an :class:`~agentdescent.policies.AcceptancePolicy`
  that audits a sample of commits against the oracle and refuses a
  verifier-only gain (the same failure the curriculum is looking for).

The verifier and oracle are ``(artifact, task) -> float`` callables, so this
works with the shipped verifier, a model judge, or a deterministic fixture.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .advantage import _ForwardsInstall
from .policies import AcceptDecision, MergeContext

__all__ = ["CoevolutionReport", "CoevolutionRound", "ProbeRow",
           "VerifierAgreement", "VerifierCurriculum"]

Scorer = Callable[[Any, Any], float]


@dataclass(frozen=True)
class ProbeRow:
    """One task's verifier and oracle scores, and how far apart they are."""

    task: Any
    verifier: float
    oracle: float

    @property
    def gap(self) -> float:
        return self.verifier - self.oracle

    @property
    def abs_gap(self) -> float:
        return abs(self.gap)


@dataclass(frozen=True)
class CoevolutionRound:
    """One curriculum step: the selected tasks, the bias before/after."""

    index: int
    tasks: Tuple[Any, ...]
    mean_gap: float
    abs_gap: float


@dataclass(frozen=True)
class CoevolutionReport:
    """The curriculum's effect on verifier/oracle agreement."""

    rounds: Tuple[CoevolutionRound, ...] = ()
    initial_abs_gap: float = 0.0
    final_abs_gap: float = 0.0
    improved: bool = False


class VerifierCurriculum:
    """Select the tasks where the verifier disagrees most with the oracle.

    ``probe`` scores every task with both callables. ``select`` takes the top
    ``n`` by absolute gap by default; ``mode='overestimate'`` targets the
    verifier being too generous and ``'underestimate'`` too harsh.
    ``coevolve`` repeats that selection and calls ``update(task, oracle_score)``
    for each selected task, so a caller can correct a contextual verifier. The
    ``update`` return value is ignored; the caller mutates its verifier.
    """

    def __init__(self, tasks: Sequence[Any], verifier: Scorer,
                 oracle: Scorer) -> None:
        self.tasks = tuple(tasks)
        self.verifier = verifier
        self.oracle = oracle

    def probe(self, artifact: Any) -> List[ProbeRow]:
        return [ProbeRow(task=task,
                         verifier=float(self.verifier(artifact, task)),
                         oracle=float(self.oracle(artifact, task)))
                for task in self.tasks]

    def _select_rows(self, rows: List[ProbeRow], n: int, *,
                     mode: str) -> List[ProbeRow]:
        if n <= 0 or not rows:
            return []
        if mode == "overestimate":
            rows.sort(key=lambda row: row.gap, reverse=True)
        elif mode == "underestimate":
            rows.sort(key=lambda row: row.gap)
        elif mode == "abs":
            rows.sort(key=lambda row: row.abs_gap, reverse=True)
        else:
            raise ValueError("mode must be 'abs', 'overestimate' or 'underestimate'")
        return rows[:n]

    def select(self, artifact: Any, n: int = 1, *,
               mode: str = "abs") -> List[Any]:
        return [row.task for row in self._select_rows(
            self.probe(artifact), n, mode=mode)]

    def coevolve(self, artifact: Any, *, rounds: int = 3, n: int = 4,
                 mode: str = "abs",
                 update: Optional[Callable[[Any, float], None]] = None,
                 ) -> CoevolutionReport:
        history: List[CoevolutionRound] = []
        initial = final = 0.0
        for index in range(max(0, rounds)):
            rows = self.probe(artifact)          # one oracle label per task this round
            if not rows:
                break
            if index == 0:
                initial = sum(row.abs_gap for row in rows) / len(rows)
            selected = self._select_rows(rows, n, mode=mode)
            selected_tasks = {id(row.task) for row in selected}
            if update is not None:
                for row in selected:
                    update(row.task, row.oracle)   # reuse the label already paid for
            # The verifier may have changed on the selected tasks; the oracle
            # labels are reused, so this costs verifier calls only.
            after = [ProbeRow(
                task=row.task,
                verifier=(float(self.verifier(artifact, row.task))
                          if id(row.task) in selected_tasks else row.verifier),
                oracle=row.oracle) for row in rows]
            mean_gap = sum(row.gap for row in after) / len(after)
            abs_gap = sum(row.abs_gap for row in after) / len(after)
            final = abs_gap
            history.append(CoevolutionRound(
                index=index, tasks=tuple(row.task for row in selected),
                mean_gap=mean_gap, abs_gap=abs_gap))
        return CoevolutionReport(
            rounds=tuple(history), initial_abs_gap=initial,
            final_abs_gap=final,
            improved=bool(history) and final < initial - 1e-12)


class VerifierAgreement(_ForwardsInstall):
    """Reject a commit whose gain the verifier claims and the oracle denies.

    ``audit_tasks`` is a fixed, held-out probe set (the same contract as an
    exam). A deterministic ``sample_rate`` fraction of commits is audited (the
    draw is seeded from the diff id, so it does not depend on merge order); the
    rest delegate straight to ``inner``. ``inner=None`` is the shipped
    ``DefaultAcceptance``, configured by the aggregator through the forwarded
    ``configure`` hook -- the wrapper can only make acceptance stricter. On an
    audit, an oracle regression is refused whatever the verifier says, and a
    verifier-only gain is refused with category ``oracle-disagrees``.
    """

    def __init__(self, audit_tasks: Sequence[Any], verifier: Scorer, oracle: Scorer,
                 *, inner: Optional[Any] = None, sample_rate: float = 0.2,
                 seed: int = 0) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be in [0, 1]")
        from .defaults import DefaultAcceptance

        self.audit_tasks = tuple(audit_tasks)
        self.verifier = verifier
        self.oracle = oracle
        self.inner = inner if inner is not None else DefaultAcceptance()
        self.sample_rate = float(sample_rate)
        self.seed = int(seed)

    def _mean(self, scorer: Scorer, artifact: Any) -> float:
        if not self.audit_tasks:
            return 0.0
        return sum(float(scorer(artifact, task)) for task in self.audit_tasks
                   ) / len(self.audit_tasks)

    def accept(self, ctx: MergeContext) -> AcceptDecision:
        decision = (self.inner.accept(ctx) if hasattr(self.inner, "accept")
                    else self.inner(ctx))
        if not decision.accept:
            return decision
        diff = getattr(ctx, "diff", None)
        key = f"{self.seed}:{getattr(diff, 'diff_id', '')}"
        if random.Random(key).random() > self.sample_rate or not self.audit_tasks:
            return decision
        base_v = self._mean(self.verifier, ctx.artifact)
        cand_v = self._mean(self.verifier, ctx.candidate)
        base_o = self._mean(self.oracle, ctx.artifact)
        cand_o = self._mean(self.oracle, ctx.candidate)
        verifier_gain = cand_v - base_v
        oracle_gain = cand_o - base_o
        if oracle_gain < 0.0:
            return AcceptDecision(
                False, "oracle-regression",
                f"oracle {base_o:.3f} -> {cand_o:.3f} on the audit set",
                observed_delta=0.0)
        if verifier_gain > 0.0 and oracle_gain <= 0.0:
            return AcceptDecision(
                False, "oracle-disagrees",
                f"verifier {base_v:.3f} -> {cand_v:.3f} but oracle "
                f"{base_o:.3f} -> {cand_o:.3f}",
                observed_delta=0.0)
        return decision
