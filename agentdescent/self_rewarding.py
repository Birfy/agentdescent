"""Self-rewarding meta-evolution: the judge improves as the run goes.

Self-Rewarding Language Models (Meta, 2024) train a model to reward its own
outputs, and improve *both* the outputs and the judge in alternation. The meta
level of this framework already pairs a fixed ``meta_reward`` (an ``auc``) with
an oracle that samples the truth (#426); the missing rung is that the reward
itself never improves. This module closes it: a :class:`LearnableMetaJudge` is
fit on oracle-labelled outcomes, and ``self_rewarding_meta_run`` alternates —
evolve the rule with the current judge, then fit a *better* judge on the newly
oracle-labelled outcomes, so each round the search moves against a closer
approximation of truth.

The judge is honest by construction: it is trained only on labels the oracle
actually produced (never on its own guesses), it is a small pure-Python linear
fit (no numpy, deterministic), and the run reports the judge's agreement with
the oracle on outcomes it never trained on — so "the judge got better" is
measured, not claimed.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .evolution import EvolutionResult, Task
from .meta import (
    MetaOutcome, MetaReward, Problem, SlotSpec, auc, meta_evolve,
    meta_validate,
)

__all__ = ["outcome_features", "LearnableMetaJudge", "SelfRewardingReport",
           "self_rewarding_meta_run"]


def outcome_features(outcome: MetaOutcome) -> List[float]:
    """The features a judge learns quality from: mean best-so-far, the final
    reward, and a constant intercept term. Sweep counts / rollouts are
    deliberately dropped: raw counts dwarf a ``[0, 1]`` quality in a linear fit
    and had the old judge saturating its clip at 1.0 for every outcome. The
    judge must learn quality from the curve's *shape*, not be handed the
    answer."""
    best = outcome.best_so_far()
    return [sum(best) / len(best) if best else 0.0,
            float(outcome.final),
            1.0]


class LearnableMetaJudge:
    """A small linear regressor from :func:`outcome_features` to oracle quality,
    fit by least squares (pure Python, deterministic)."""

    def __init__(self) -> None:
        self.weights: List[float] = []
        self._fit_n = 0

    def fit(self, pairs: Sequence[Tuple[MetaOutcome, float]]) -> None:
        """Fit on ``(outcome, oracle_quality)`` pairs. The labels are the
        oracle's, never the judge's own. With fewer than 4 pairs, or a fit
        that collapses to a singular normal matrix, the judge keeps no weights
        and :meth:`__call__` falls back to the safe ``auc`` reward -- a fixed
        weight vector would flatten every outcome to the same score."""
        n = len(pairs)
        self._fit_n = n
        if n < 4:
            self.weights = []
            return
        x = [outcome_features(o) for o, _ in pairs]
        y = [float(l) for _, l in pairs]
        weights = _least_squares(x, y)
        self.weights = weights if weights is not None else []
        self._fit_n = n

    def __call__(self, outcome: MetaOutcome) -> float:
        if not self.weights:
            return float(auc(outcome))
        feats = outcome_features(outcome)
        value = sum(w * f for w, f in zip(self.weights, feats))
        return min(1.0, max(0.0, value))

    @property
    def fitted(self) -> bool:
        return bool(self.weights)

    @property
    def n_labels(self) -> int:
        return self._fit_n

    def agreement(self, pairs: Sequence[Tuple[MetaOutcome, float]]) -> float:
        """Normalised Kendall tau between the judge's scores and the oracle's
        labels on ``pairs`` -- the measured "the judge learned truth" number.
        Ties count half (a constant judge cannot claim agreement). ``None``-safe:
        ``0.5`` with too few pairs (no signal is not a claim)."""
        if len(pairs) < 2:
            return 0.5
        judge = [self(o) for o, _ in pairs]
        labels = [float(l) for _, l in pairs]
        pos = 0.0
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                dj = (judge[i] > judge[j]) - (judge[i] < judge[j])
                dl = (labels[i] > labels[j]) - (labels[i] < labels[j])
                if dj == dl and dj != 0:
                    pos += 1.0
                elif dj == 0 or dl == 0:
                    pos += 0.5
        total = len(pairs) * (len(pairs) - 1) / 2
        return pos / total if total else 0.5


def _least_squares(x: List[List[float]], y: List[float]) -> Optional[List[float]]:
    """Solve the normal equations ``(X^T X) w = X^T y`` in pure Python; ``None``
    when the matrix is singular (too few or too correlated examples)."""
    m = len(x[0]) if x else 0
    a = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    for row, label in zip(x, y):
        for i in range(m):
            for j in range(m):
                a[i][j] += row[i] * row[j]
            b[i] += row[i] * label
    return _solve(a, b)


def _solve(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """Gaussian elimination with partial pivoting; ``None`` when the system is
    singular (too few varied examples) so the caller can fall back to the safe
    reward instead of a degenerate weight vector."""
    n = len(b)
    aug = [a[i] + [b[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for r in range(col + 1, n):
            factor = aug[r][col] / aug[col][col]
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (aug[i][n] - sum(aug[i][j] * x[j] for j in range(i + 1, n))) \
            / aug[i][i]
    return x


@dataclass(frozen=True)
class SelfRewardingReport:
    """What a :func:`self_rewarding_meta_run` did: the evolved value, the
    judge's measured agreement with the oracle on validation outcomes, whether
    the judge actually fitted (or fell back to ``auc``), how many oracle labels
    it trained on, and the value's oracle quality vs the seed's on the held-out
    problems."""

    seed_value: str
    value: str
    judge_agreement: float
    judge_labels: int
    judge_fitted: bool
    oracle_gain: float


def self_rewarding_meta_run(
    problems: Union[Sequence[Problem], Mapping[str, Problem]],
    val_problems: Union[Sequence[Problem], Mapping[str, Problem]],
    *,
    slot: str,
    spec: SlotSpec,
    oracle: MetaReward,
    propose: Optional[Callable[[str, Task, str, float], Optional[str]]] = None,
    model: Optional[Any] = None,
    seeds: Sequence[int] = (0, 1),
    val_seeds: Sequence[int] = (2, 3),
    judge_sample: int = 6,
    self_reward_rounds: int = 1,
    **evolve_kwargs: Any,
) -> Tuple[SelfRewardingReport, EvolutionResult]:
    """Evolve a decision slot with a judge that improves against truth.

    Round 1 evolves the slot with the plain ``auc`` reward and collects the
    candidate outcomes; ``judge_sample`` of them, drawn **round-robin across
    the candidates** (the seed and every value the run has produced), are
    scored by ``oracle`` and fit into :class:`LearnableMetaJudge`. Each further
    ``self_reward_rounds`` re-evolves with the judge as the meta reward and
    refits it on the **accumulated** oracle-labelled pairs -- so the judge's
    labels span several candidates and it improves as the run goes. At the end
    the judge's agreement with the oracle is measured on validation outcomes it
    never trained on, and the evolved value's oracle quality is compared with
    the seed's on the held-out problems.

    The judge never scores the validation: ``oracle`` labels only training
    outcomes, and the report's agreement is measured on held-out ones -- the
    same isolation every other development set in the framework gets.
    """
    if propose is None and model is None:
        raise ValueError("self_rewarding_meta_run() needs propose= or model=")
    named = (dict(problems) if isinstance(problems, Mapping)
             else {f"p{i}": p for i, p in enumerate(problems)})
    val_named = (dict(val_problems) if isinstance(val_problems, Mapping)
                 else {f"p{i}": p for i, p in enumerate(val_problems)})
    seed_value = spec.render(spec.initial())
    result = meta_evolve(
        problems, slot=slot, spec=spec, propose=propose, model=model,
        meta_reward=auc, seeds=seeds, **evolve_kwargs)
    value = result.rendered or seed_value

    judge = LearnableMetaJudge()
    current_value = value
    candidates = [seed_value, current_value]
    all_pairs: List[Tuple[MetaOutcome, float]] = []
    for _ in range(self_reward_rounds):
        all_pairs.extend(_sample_oracle_pairs(named, spec, candidates, seeds,
                                              oracle, judge_sample))
        judge.fit(all_pairs)
        result = meta_evolve(
            problems, slot=slot, spec=spec, propose=propose, model=model,
            meta_reward=judge, seeds=seeds, **evolve_kwargs)
        current_value = result.rendered or seed_value
        if current_value not in candidates:
            candidates.append(current_value)

    agreement_pairs = _oracle_pairs(val_named, spec, current_value, val_seeds,
                                    oracle)
    agreement = judge.agreement(agreement_pairs)
    seed_oracle = _mean_oracle(val_named, spec, seed_value, val_seeds, oracle)
    value_oracle = _mean_oracle(val_named, spec, current_value, val_seeds, oracle)

    report = SelfRewardingReport(
        seed_value=seed_value, value=current_value, judge_agreement=agreement,
        judge_labels=judge.n_labels, judge_fitted=judge.fitted,
        oracle_gain=value_oracle - seed_oracle)
    try:
        result.meta = {
            "judge_agreement": agreement, "judge_labels": judge.n_labels,
            "judge_fitted": judge.fitted,
            "oracle_gain": value_oracle - seed_oracle,
        }
    except AttributeError:
        pass
    return report, result


def _sample_oracle_pairs(
    named: Mapping[str, Problem], spec: SlotSpec, values: Sequence[str],
    seeds: Sequence[int], oracle: MetaReward, sample: int,
) -> List[Tuple[MetaOutcome, float]]:
    """Oracle-labelled outcomes drawn round-robin **across the candidate
    values**, so one candidate cannot crowd out the rest: each value
    contributes ``sample // len(values)`` pairs (first come first served within
    it), and the remainder tops up from the newest value."""
    if not values or sample < 1:
        return []
    per = max(1, sample // len(values))
    pairs: List[Tuple[MetaOutcome, float]] = []
    for value in values:
        taken = 0
        for outcome, label in _oracle_pairs(named, spec, value, seeds, oracle):
            if taken >= per or len(pairs) >= sample:
                break
            pairs.append((outcome, label))
            taken += 1
        if len(pairs) >= sample:
            break
    if len(pairs) < sample:
        for outcome, label in _oracle_pairs(named, spec, values[-1], seeds,
                                            oracle):
            pairs.append((outcome, label))
            if len(pairs) >= sample:
                break
    return pairs[:sample]


def _oracle_pairs(
    named: Mapping[str, Problem], spec: SlotSpec, value: str, seeds: Sequence[int],
    oracle: MetaReward,
) -> List[Tuple[MetaOutcome, float]]:
    policy = spec.compile(value)
    out = []
    for problem in named.values():
        for seed in seeds:
            outcome = problem(policy, seed)
            out.append((outcome, float(oracle(outcome))))
    return out


def _mean_oracle(named: Mapping[str, Problem], spec: SlotSpec, value: str,
                 seeds: Sequence[int], oracle: MetaReward) -> float:
    pairs = _oracle_pairs(named, spec, value, seeds, oracle)
    return statistics.fmean([l for _, l in pairs]) if pairs else 0.0
