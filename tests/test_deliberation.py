"""Deliberation must surface weighted agreement, not just a plurality."""

import pytest

from agentdescent.deliberation import (
    Argument,
    DebateConfig,
    Deliberation,
    JudgeVote,
)


def _debaters():
    return {
        "pro": lambda topic, history, round_: Argument(
            "pro", "pro", f"pro {round_}", evidence=(topic,), confidence=0.8),
        "con": lambda topic, history, round_: Argument(
            "con", "con", f"con {round_}", evidence=(topic,), confidence=0.6),
    }


def test_majority_reaches_consensus_after_min_rounds():
    judges = [lambda *_: "pro", lambda *_: "pro", lambda *_: "con"]
    report = Deliberation(DebateConfig(rounds=3, min_rounds=1,
                                       consensus_threshold=0.6)).run(
        "topic", _debaters(), judges)
    assert report.winner == "pro"
    assert report.agreement == pytest.approx(2 / 3)
    assert report.consensus
    assert len(report.rounds) == 1
    assert len(report.rounds[0].arguments) == 2


def test_confidence_and_weight_can_override_a_plurality():
    judges = [
        lambda *_: JudgeVote("judge-0", "pro", confidence=0.9, weight=1),
        lambda *_: JudgeVote("judge-1", "con", confidence=0.1, weight=1),
        lambda *_: JudgeVote("judge-2", "con", confidence=0.1, weight=1),
    ]
    report = Deliberation(DebateConfig(rounds=1, consensus_threshold=0.5)).run(
        "topic", _debaters(), judges)
    assert report.winner == "pro"
    assert report.confidence > 0.8


def test_named_judge_votes_are_adopted_by_the_panel():
    judges = [
        lambda *_: JudgeVote("alice", "pro", confidence=0.8),
        lambda *_: JudgeVote("bob", "con", confidence=0.2),
    ]
    report = Deliberation(DebateConfig(rounds=1, consensus_threshold=0.5)).run(
        "topic", _debaters(), judges)
    assert report.winner == "pro"
    assert [vote.judge for vote in report.rounds[0].votes] == ["judge-0", "judge-1"]


def test_zero_weight_is_refused_so_confidence_cannot_divide_by_zero():
    with pytest.raises(ValueError):
        JudgeVote("j", "pro", weight=0.0)
    with pytest.raises(ValueError):
        JudgeVote("j", "pro", weight=True)
    deliberation = Deliberation(DebateConfig(rounds=1))
    with pytest.raises(ValueError):
        deliberation.run("t", _debaters(),
                         [lambda *_: JudgeVote("j", "pro", weight=0.0)])


def test_no_consensus_runs_all_rounds():
    judges = [lambda *_: "pro", lambda *_: "con"]
    report = Deliberation(DebateConfig(rounds=2, min_rounds=1,
                                       consensus_threshold=0.9)).run(
        "topic", _debaters(), judges)
    assert not report.consensus
    assert len(report.rounds) == 2
    assert report.stopped_reason == "rounds"


def test_invalid_arguments_votes_and_config_are_refused():
    with pytest.raises(ValueError):
        Argument("", "pro", "claim")
    with pytest.raises(ValueError):
        Argument("a", "pro", "claim", confidence=2)
    with pytest.raises(ValueError):
        JudgeVote("", "pro")
    with pytest.raises(ValueError):
        JudgeVote("j", "pro", confidence=2)
    with pytest.raises(ValueError):
        JudgeVote("j", "pro", weight=-1)
    with pytest.raises(ValueError):
        JudgeVote("j", "pro", confidence=True)
    with pytest.raises(ValueError):
        DebateConfig(rounds=True)
    with pytest.raises(ValueError):
        DebateConfig(rounds=0)
    with pytest.raises(ValueError):
        DebateConfig(rounds=2, min_rounds=3)
    deliberation = Deliberation()
    with pytest.raises(ValueError):
        deliberation.run("t", {}, [lambda *_: "pro"])
    with pytest.raises(ValueError):
        deliberation.run("t", _debaters(), [])
    with pytest.raises(ValueError):
        deliberation.run("t", {"x": lambda *_: Argument("y", "x", "c")},
                         [lambda *_: "x"])
    with pytest.raises(ValueError):
        deliberation.run("t", _debaters(), [lambda *_: {"confidence": 0.5}])
