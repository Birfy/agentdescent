"""Rule tournament has to be right where candidates play against each other.

Every pair plays twice on each other's ground, scored by a critic, and Elo
converts the matches into ratings; the winner is promoted. The fixtures pin the
surfaces that make this honest: Elo moves ratings correctly (win / loss /
draw), a dominant rule beats its rival on the rival's own ground, a tournament
needs opponents or an explanation, the promotion writes the winner back so the
next seeding ranks it first, and the rebuild path runs on real archived
descriptors.
"""

import pytest

from agentdescent import (
    ConsolidationRecord,
    ConsolidationStore,
    EloRating,
    EvolutionResult,
    MetaOutcome,
    ParamSlot,
    SingleSlot,
    Task,
    auc,
    problem_descriptor,
    rule_tournament,
    tournament_promote,
)
from agentdescent.meta import _archived_rules
from agentdescent.sampling import DifficultyWeighted


def _sampler_slot():
    return ParamSlot(factory=lambda c: DifficultyWeighted(c=c),
                     params={"c": 0.2}, bounds={"c": (0.05, 1.0)})


CONFIG = {"dataset": "cap-arithmetic", "scorer": "exact"}


def _archive_rule(store, run_id, c, reward):
    store.archive(
        EvolutionResult(state={"c": f"{c:.1f}"},
                        rendered=f"# Policy parameters\nc: {c:.1f}",
                        final_reward=reward, history=[], ledger_log=[]),
        run_id=run_id, strategy="ParamSlot", config=CONFIG,
        artifact_id="meta-rule",
        problem=problem_descriptor("task_sampler", rounds=5, n_workers=1))


def _dominator(threshold=0.5):
    def problem(value, seed):
        good = getattr(value, "c", 0.2) >= threshold
        return MetaOutcome(curve=[1.0 if good else 0.1],
                           final=1.0 if good else 0.1)
    return problem


def test_elo_updates_ratings_correctly():
    elo = EloRating(k=32)
    r = {"a": 1000.0, "b": 1000.0}
    elo.update(r, "a", "b")
    assert r["a"] > 1000.0 > r["b"]
    assert round(r["a"] + r["b"], 6) == 2000.0        # rating mass is conserved
    draw = {"x": 1000.0, "y": 1000.0}
    elo.update(draw, "x", "y", drawn=True)
    assert draw["x"] == pytest.approx(1000.0)


def test_a_dominant_rule_wins_its_rival_s_ground():
    store = ConsolidationStore("/tmp/__tourney_test__")   # unused: challenges given
    a = {"run-a": _dominator(0.5), "run-b": _dominator(0.5)}
    report = rule_tournament(
        store, [_fake_rule("run-a", 0.8), _fake_rule("run-b", 0.2)],
        slot="task_sampler", spec=_sampler_slot(), critic=auc, challenges=a)
    assert report.winner == "run-a"
    assert report.ratings["run-a"] > report.ratings["run-b"]
    assert report.matches[0].outcome == "a"
    assert report.matches[0].score_a > report.matches[0].score_b


def test_matches_are_played_on_the_same_ground():
    # "good" beats "bad" by +0.1 on every problem, but good's own challenge is
    # easy enough that bad scores 0.6 there. Old cross-ground scoring handed the
    # match to bad (score_a=0.2, score_b=0.6); same-ground matches give good both.
    store = ConsolidationStore("/tmp/__tourney_test__")

    def good_ground(value, seed):
        good = getattr(value, "c", 0.2) >= 0.5
        return MetaOutcome(curve=[0.7 if good else 0.6],
                           final=0.7 if good else 0.6)

    def bad_ground(value, seed):
        good = getattr(value, "c", 0.2) >= 0.5
        return MetaOutcome(curve=[0.3 if good else 0.2],
                           final=0.3 if good else 0.2)

    challenges = {"run-good": good_ground, "run-bad": bad_ground}
    report = rule_tournament(
        store, [_fake_rule("run-good", 0.8), _fake_rule("run-bad", 0.2)],
        slot="task_sampler", spec=_sampler_slot(), critic=auc,
        challenges=challenges)
    assert report.winner == "run-good"
    assert all(m.outcome == "a" for m in report.matches)
    assert {m.ground for m in report.matches} == {"run-good", "run-bad"}


def test_a_missing_challenge_is_refused_not_crashed():
    store = ConsolidationStore("/tmp/__tourney_test__")
    # covers run-b but not run-a, plus an extra key: the old proper-subset check
    # let this through and then KeyError'd mid-tournament
    challenges = {"run-b": _dominator(0.5), "extra": _dominator(0.5)}
    with pytest.raises(ValueError):
        rule_tournament(
            store, [_fake_rule("run-a", 0.8), _fake_rule("run-b", 0.2)],
            slot="task_sampler", spec=_sampler_slot(), critic=auc,
            challenges=challenges)


def _fake_rule(run_id, c):
    return ConsolidationRecord(
        run_id=run_id, artifact_id="meta-rule", strategy="ParamSlot",
        config=CONFIG, state={"c": f"{c:.1f}"},
        rendered=f"# Policy parameters\nc: {c:.1f}", final_reward=0.5,
        rounds=1)


def test_a_tournament_needs_opponents_or_an_explanation(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    with pytest.raises(ValueError):
        rule_tournament(
            store, [_fake_rule("run-a", 0.8), _fake_rule("run-b", 0.2)],
            slot="task_sampler", spec=_sampler_slot(), critic=auc)  # no challenges, no domain


def test_tournament_promote_ranks_the_winner_first(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-a", 0.8, 0.7)
    _archive_rule(store, "run-b", 0.2, 0.9)      # higher archived reward, loses the tourney
    rules = _archived_rules(store, slot="task_sampler", spec=_sampler_slot(),
                            config=CONFIG, artifact_id="meta-rule")
    challenges = {"run-a": _dominator(0.5), "run-b": _dominator(0.5)}
    report, promoted = tournament_promote(
        store, rules, slot="task_sampler", spec=_sampler_slot(), critic=auc,
        run_id="champion", config=CONFIG, challenges=challenges)
    assert report.winner == "run-a"
    assert promoted == "champion"
    ranked = _archived_rules(store, slot="task_sampler", spec=_sampler_slot(),
                             config=CONFIG, artifact_id="meta-rule")
    assert ranked[0].run_id == "champion"       # the tournament's choice ranks first
    assert store.find("champion").final_reward >= 0.9


def test_promotion_refuses_a_winner_that_fails_validation(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-a", 0.8, 0.7)
    _archive_rule(store, "run-b", 0.2, 0.9)
    rules = _archived_rules(store, slot="task_sampler", spec=_sampler_slot(),
                            config=CONFIG, artifact_id="meta-rule")

    def flat(value, seed):
        return MetaOutcome(curve=[0.5], final=0.5)

    challenges = {"run-a": flat, "run-b": flat}
    report, promoted = tournament_promote(
        store, rules, slot="task_sampler", spec=_sampler_slot(), critic=auc,
        run_id="champion", config=CONFIG, challenges=challenges)
    assert report.winner in ("run-a", "run-b")  # all draws; Elo still names one
    assert promoted is None                     # but the trust layer refuses it
    assert store.find("champion") is None       # nothing fake was archived


def test_promotion_ignores_the_winner_s_own_ground(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-a", 0.8, 0.7)
    _archive_rule(store, "run-b", 0.2, 0.9)
    rules = _archived_rules(store, slot="task_sampler", spec=_sampler_slot(),
                            config=CONFIG, artifact_id="meta-rule")

    def ground_a(value, seed):          # run-a's own ground: favors good
        good = getattr(value, "c", 0.2) >= 0.5
        return MetaOutcome(curve=[0.9 if good else 0.1],
                           final=0.9 if good else 0.1)

    def ground_b(value, seed):          # the only rival ground: inverted
        good = getattr(value, "c", 0.2) >= 0.5
        return MetaOutcome(curve=[0.2 if good else 0.5],
                           final=0.2 if good else 0.5)

    challenges = {"run-a": ground_a, "run-b": ground_b}
    report, promoted = tournament_promote(
        store, rules, slot="task_sampler", spec=_sampler_slot(), critic=auc,
        run_id="champion", config=CONFIG, challenges=challenges)
    assert report.winner == "run-a"     # run-a wins Elo (second match)
    assert promoted is None             # but it loses to the seed on the only
    assert store.find("champion") is None   # rival ground -- its own easy
                                        # ground did not carry the adoption


def test_the_rebuild_path_runs_on_real_descriptors(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    _archive_rule(store, "run-a", 0.8, 0.7)
    _archive_rule(store, "run-b", 0.2, 0.9)
    rules = _archived_rules(store, slot="task_sampler", spec=_sampler_slot(),
                            config=CONFIG, artifact_id="meta-rule")

    def _cap(r):
        import re
        m = re.search(r"cap:(\d+)", r)
        return int(m.group(1)) if m else 0
    def run(r, t): return str(t.meta["gold"]) if t.meta.get("a", 0) <= _cap(r) else "?"
    def reward(t, o): return 1.0 if str(o).strip() == str(t.meta["gold"]).strip() else 0.0
    def propose(r, t, o, w):
        if w >= 0.999:
            return "cap:0"
        a = t.meta.get("a", 0)
        if a > _cap(r):
            return f"cap:{a}"
        return None
    def tasks_for(rec):
        order = [1, 2, 3, 4, 8, 5, 6, 7]
        return [Task(id=f"t{i}", prompt=f"item {i}",
                     meta={"gold": str(2 * a), "a": a}) for i, a in enumerate(order)]
    # rebuild path: domain callables instead of explicit challenges
    report = rule_tournament(
        store, rules, slot="task_sampler", spec=_sampler_slot(), critic=auc,
        tasks_for=tasks_for, run=run, reward=reward, propose=propose,
        strategy=SingleSlot(initial_value="cap:0"))
    assert len(report.matches) == 2      # one pair, two same-ground matches
    assert report.winner in ("run-a", "run-b")


def test_fewer_than_two_rules_is_no_tournament(tmp_path):
    store = ConsolidationStore(str(tmp_path / "archive"))
    report = rule_tournament(
        store, [_fake_rule("only", 0.8)], slot="task_sampler",
        spec=_sampler_slot(), critic=auc)
    assert report.matches == ()
    assert report.winner == "only"
