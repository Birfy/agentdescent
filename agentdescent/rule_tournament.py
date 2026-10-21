"""Rule tournament: candidates play against each other, ratings decide promotion.

The population (#428) ranks rules by a single archived reward; nothing ever
pitted candidates *against* each other. This module is the self-play idea
(DeepSeek-R1's verifiable-reward RL; self-playing agents) applied to the
archive: a round-robin tournament where each rule is evaluated on the *other*
rule's challenge problems — its own problem rebuilt from its descriptor via
`problem_from_archive` — and scored by a critic (an oracle, or the
:class:`~agentdescent.self_rewarding.LearnableMetaJudge`). A rule that wins on
its rival's ground is a better rule, and Elo ratings convert those matches into
a promotion order. `tournament_promote` writes the winner back to the archive,
so the next session seeds from the rule the tournament said was best.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .consolidation import ConsolidationRecord, ConsolidationStore
from .evolution import EvolutionResult, Task
from .meta import (
    MetaReward, MetaRunReport, Problem, SlotSpec, _archived_rules,
    archive_rule, meta_validate, problem_descriptor, problem_from_archive,
)

__all__ = ["EloRating", "MatchResult", "TournamentReport", "rule_tournament",
           "tournament_promote"]


@dataclass
class EloRating:
    """A minimal Elo rating over pairwise matches (K-factor 32)."""

    initial: float = 1000.0
    k: float = 32.0

    @staticmethod
    def expected(a: float, b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))

    def update(self, ratings: Dict[str, float], winner: str, loser: str,
               drawn: bool = False) -> None:
        rw, rl = ratings[winner], ratings[loser]
        e_w = self.expected(rw, rl)
        score_w = 0.5 if drawn else 1.0
        ratings[winner] = rw + self.k * (score_w - e_w)
        ratings[loser] = rl + self.k * ((1.0 - score_w) - (1.0 - e_w))


@dataclass(frozen=True)
class MatchResult:
    """One pairwise match: who played whom, on whose ground, and who won.

    ``ground`` is the ``run_id`` whose challenge problem both rules were scored
    on -- a match is only a comparison if both sides face the same ground."""

    a: str
    b: str
    score_a: float          # a's mean critic score on ``ground``
    score_b: float          # b's mean critic score on ``ground``
    outcome: str            # "a" / "b" / "draw"
    ground: str = ""


@dataclass(frozen=True)
class TournamentReport:
    """The ratings a tournament produced, and the matches that moved them."""

    ratings: Dict[str, float]
    matches: Tuple[MatchResult, ...]
    winner: str


def rule_tournament(
    store: ConsolidationStore,
    rules: Sequence[ConsolidationRecord],
    *,
    slot: str,
    spec: SlotSpec,
    critic: MetaReward,
    challenges: Optional[Mapping[str, Problem]] = None,
    tasks_for: Optional[Callable[[Any], Sequence[Task]]] = None,
    run: Optional[Callable[[str, Task], str]] = None,
    reward: Optional[Callable[[Task, str], float]] = None,
    propose: Optional[Callable[[str, Task, str, float], Optional[str]]] = None,
    strategy: Any = None,
    seeds: Sequence[int] = (0,),
    elo_k: float = 32.0,
    **inner_defaults: Any,
) -> TournamentReport:
    """Run a round-robin between the rules, rated by Elo.

    Every pair plays **twice, once on each rule's ground, and both rules are
    scored on the same ground each time**: rule A and rule B both face rule
    A's challenge problem, then both face rule B's. Scoring each side on the
    *other* side's problem would hand the match to whoever's opponent owned the
    easier problem; same-ground scoring makes ``score_a`` and ``score_b`` a
    real comparison. Higher mean score wins the match (a tie is a draw), and
    Elo converts the matches into ratings. ``winner`` is the highest-rated
    rule's ``run_id``.
    """
    rule_ids = {r.run_id for r in rules}
    if len(rules) < 2:
        ratings = {r.run_id: 1000.0 for r in rules}
        return TournamentReport(ratings=ratings, matches=(),
                                winner=rules[0].run_id if rules else "")
    if challenges is None:
        if None in (tasks_for, run, reward, propose, strategy):
            raise ValueError(
                "rule_tournament() needs challenges= (a run_id -> Problem map) "
                "or the domain callables to rebuild problems")
        challenges = {
            r.run_id: problem_from_archive(
                store, r.run_id, tasks_for=tasks_for, spec=spec, run=run,
                reward=reward, propose=propose, strategy=strategy,
                **inner_defaults)
            for r in rules}
    elif not rule_ids <= set(challenges):
        raise ValueError("challenges must cover every rule's run_id")

    values = {r.run_id: spec.compile(spec.render(dict(r.state or {})))
              for r in rules}
    ratings = {r.run_id: 1000.0 for r in rules}
    elo = EloRating(k=elo_k)
    matches: List[MatchResult] = []

    def score_on(run_id_a: str, problem_b: Problem) -> float:
        total = 0.0
        n = 0
        for seed in seeds:
            outcome = problem_b(values[run_id_a], seed)
            total += float(critic(outcome))
            n += 1
        return total / n if n else 0.0

    for i in range(len(rules)):
        for j in range(i + 1, len(rules)):
            a, b = rules[i], rules[j]
            for ground in (a.run_id, b.run_id):
                s_a = score_on(a.run_id, challenges[ground])
                s_b = score_on(b.run_id, challenges[ground])
                if s_a > s_b:
                    outcome = "a"
                    elo.update(ratings, a.run_id, b.run_id)
                elif s_b > s_a:
                    outcome = "b"
                    elo.update(ratings, b.run_id, a.run_id)
                else:
                    outcome = "draw"
                    elo.update(ratings, a.run_id, b.run_id, drawn=True)
                matches.append(MatchResult(a=a.run_id, b=b.run_id, score_a=s_a,
                                           score_b=s_b, outcome=outcome,
                                           ground=ground))

    winner = max(ratings, key=lambda rid: (ratings[rid], rid))
    return TournamentReport(ratings=ratings, matches=tuple(matches), winner=winner)


def tournament_promote(
    store: ConsolidationStore,
    rules: Sequence[ConsolidationRecord],
    *,
    slot: str,
    spec: SlotSpec,
    critic: MetaReward,
    run_id: str,
    config: Optional[Dict[str, Any]] = None,
    artifact_id: str = "meta-rule",
    challenges: Optional[Mapping[str, Problem]] = None,
    tasks_for: Optional[Callable[[Any], Sequence[Task]]] = None,
    run: Optional[Callable[[str, Task], str]] = None,
    reward: Optional[Callable[[Task, str], float]] = None,
    propose: Optional[Callable[[str, Task, str, float], Optional[str]]] = None,
    strategy: Any = None,
    seeds: Sequence[int] = (0,),
    elo_k: float = 32.0,
    **inner_defaults: Any,
) -> Tuple[TournamentReport, Optional[str]]:
    """Run the tournament and promote the winner into the archive.

    The winner is validated against the seed on the rivals' challenge problems
    (:func:`~agentdescent.meta.meta_validate`) before anything is written: a
    tournament win on the shared grounds is only an adoption if it also
    transfers to problems the winner did not choose, so the promoted record
    carries real ``validation`` / ``mean_gain`` and is only archived when the
    winner is at least as good as the seed. Returns ``(report, promoted_run_id)``
    (``None`` when fewer than two rules were submitted or the winner fails
    validation)."""
    if challenges is None and None not in (tasks_for, run, reward, propose,
                                           strategy):
        challenges = {
            r.run_id: problem_from_archive(
                store, r.run_id, tasks_for=tasks_for, spec=spec, run=run,
                reward=reward, propose=propose, strategy=strategy,
                **inner_defaults)
            for r in rules}
    report = rule_tournament(
        store, rules, slot=slot, spec=spec, critic=critic,
        challenges=challenges, tasks_for=tasks_for, run=run, reward=reward,
        propose=propose, strategy=strategy, seeds=seeds, elo_k=elo_k,
        **inner_defaults)
    if len(rules) < 2:
        return report, None
    winner = next(r for r in rules if r.run_id == report.winner)
    if challenges is None:
        return report, None                     # no held-out problems to validate on
    seed_value = spec.render(spec.initial())
    winner_value = winner.rendered or spec.render(dict(winner.state or {}))
    rivals = {k: v for k, v in challenges.items() if k != winner.run_id}
    if not rivals:
        return report, None            # no rival ground to validate on
    validation = meta_validate(spec, seed_value, winner_value, rivals,
                               seeds=seeds, meta_reward=critic)
    mean_gain = statistics.fmean([d["gain"] for d in validation.values()]) \
        if validation else 0.0
    if mean_gain <= 0:
        return report, None                     # fails the trust layer: no adoption
    mean_score = _winner_mean(report, winner.run_id)
    result = EvolutionResult(
        state=dict(winner.state or {}), rendered=winner.rendered or "",
        final_reward=mean_score, history=[], ledger_log=[])
    meta_report = MetaRunReport(
        slot=slot, seed_value=seed_value, value=result.rendered,
        validation=validation, mean_gain=mean_gain, transfer=None,
        verdict="adopt",
        reason=(f"won the tournament with Elo "
                f"{report.ratings[winner.run_id]:.1f} and validates on the "
                f"rivals' problems (mean gain {mean_gain:.3f})"))
    archive_rule(store, meta_report, result, run_id=run_id, slot=slot,
                 spec=spec, config=config,
                 problem=winner.problem or problem_descriptor(slot),
                 artifact_id=artifact_id)
    return report, run_id


def _winner_mean(report: TournamentReport, winner: str) -> float:
    scores = [m.score_a if m.a == winner else m.score_b for m in report.matches
              if winner in (m.a, m.b)]
    return sum(scores) / len(scores) if scores else 0.0
