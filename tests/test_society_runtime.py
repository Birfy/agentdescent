"""The society runtime must stop on agreement, not on the first message."""

from types import SimpleNamespace

import pytest

from agentdescent.society_runtime import (
    Contribution,
    RoleSpec,
    SocietyConfig,
    SocietyRuntime,
)


def _roles(events):
    def propose(task, board, history):
        events.append("propose")
        return Contribution("p", "proposal", content="answer", score=0.8, cost=1)

    def critic(task, board, history):
        events.append("critic")
        return Contribution("c", "critique", content="looks fine", cost=1)

    def verify(task, board, history):
        events.append("verify")
        return Contribution("v", "verdict", approved=True, cost=2)

    return [
        RoleSpec("p", "proposer", propose),
        RoleSpec("c", "critic", critic),
        RoleSpec("v", "verifier", verify, weight=2),
    ]


def test_consensus_stops_after_a_verifier_approval():
    events = []
    runtime = SocietyRuntime(_roles(events), SocietyConfig(max_turns=5))
    report = runtime.run("task")
    assert report.consensus
    assert report.stopped_reason == "consensus"
    assert report.turns == 1
    assert report.cost == 4
    assert report.role_credit["v"] > report.role_credit["c"]
    assert report.score == pytest.approx(0.8)


def test_target_score_and_string_actor_normalisation():
    def propose(_task, _board, _history):
        return "candidate"

    runtime = SocietyRuntime([
        RoleSpec("p", "proposer", propose),
    ], SocietyConfig(require_verifier=False, target_score=0.5, max_turns=2))
    report = runtime.run("task", score_fn=lambda _task, _history: 0.9)
    assert report.consensus
    assert report.stopped_reason == "target"
    assert report.proposals[0].content == "candidate"


def test_budget_and_max_turns_stop():
    def slow(task, board, history):
        return Contribution("p", "proposal", content="answer", cost=5)

    budget = SocietyRuntime([RoleSpec("p", "proposer", slow)],
                            SocietyConfig(require_verifier=False, max_cost=4,
                                          max_turns=5))
    report = budget.run("task")
    assert report.stopped_reason == "budget"
    assert report.cost == 5

    stalled = SocietyRuntime([RoleSpec("p", "proposer", lambda *_: None)],
                             SocietyConfig(require_verifier=False, max_turns=3))
    assert stalled.run("task").stopped_reason == "stalled"


def test_shapley_attribution_is_available_when_a_score_is_replayable():
    def role(name, payload):
        return RoleSpec(name, "proposer", lambda *_: Contribution(
            name, "proposal", content=payload))

    runtime = SocietyRuntime([
        role("a", "a"),
        role("b", "b"),
    ], SocietyConfig(require_verifier=False, max_turns=1))

    def score(contributions):
        ids = {c.content for c in contributions}
        return 2.0 if {"a", "b"} <= ids else 0.0

    report = runtime.run("task", attribute_score=score)
    assert report.shapley is not None
    assert report.shapley.gain == 2.0
    assert [row.credit for row in report.shapley.ranking] == [1.0, 1.0]


def test_decision_log_is_optional_and_isolated():
    class Log:
        def __init__(self):
            self.rows = []

        def record(self, **kwargs):
            self.rows.append(kwargs)

    log = Log()
    runtime = SocietyRuntime(_roles([]), SocietyConfig(max_turns=2),
                             decision_log=log)
    runtime.run("task")
    assert log.rows[0]["kind"] == "society"

    class Broken:
        def record(self, **kwargs):
            raise RuntimeError("disk full")

    assert SocietyRuntime(_roles([]), SocietyConfig(max_turns=2),
                          decision_log=Broken()).run("task").consensus


def test_invalid_roles_and_config_are_refused():
    with pytest.raises(ValueError):
        SocietyRuntime([])
    with pytest.raises(ValueError):
        SocietyRuntime([RoleSpec("p", "proposer", lambda *_: None),
                        RoleSpec("p", "verifier", lambda *_: None)])
    with pytest.raises(ValueError):
        SocietyRuntime([RoleSpec("c", "critic", lambda *_: None)])
    with pytest.raises(ValueError):
        SocietyRuntime([RoleSpec("p", "proposer", lambda *_: None)])
    with pytest.raises(ValueError):
        RoleSpec("p", "magic", lambda *_: None)
    with pytest.raises(ValueError):
        RoleSpec("p", "proposer", lambda *_: None, budget=0)
    with pytest.raises(ValueError):
        RoleSpec("p", "proposer", lambda *_: None, weight=0)
    with pytest.raises(ValueError):
        SocietyConfig(max_turns=0)
    with pytest.raises(ValueError):
        SocietyConfig(max_cost=0)
    with pytest.raises(ValueError):
        SocietyConfig(consensus_threshold=2)


def test_revised_proposal_can_reach_consensus():
    state = {"proposals": 0, "verdicts": 0}

    def propose(_task, _board, _history):
        state["proposals"] += 1
        return Contribution("p", "proposal", content=f"p{state['proposals']}")

    def verify(_task, _board, _history):
        state["verdicts"] += 1
        return Contribution("v", "verdict",
                            approved=state["verdicts"] > 1)

    runtime = SocietyRuntime([
        RoleSpec("p", "proposer", propose),
        RoleSpec("v", "verifier", verify),
    ], SocietyConfig(max_turns=3))
    report = runtime.run("task")
    assert report.consensus
    assert report.stopped_reason == "consensus"
    assert report.proposals[-1].content == "p2"


def test_string_verifier_is_a_verdict_and_can_consensus():
    runtime = SocietyRuntime([
        RoleSpec("p", "proposer", lambda *_: "candidate"),
        RoleSpec("v", "verifier", lambda *_: "looks good"),
    ], SocietyConfig(max_turns=2))
    report = runtime.run("task")
    assert report.consensus
    assert report.verdicts[0].kind == "verdict"
    assert report.verdicts[0].approved is True


def test_contribution_role_and_kind_are_bound_to_the_actor():
    bad_kind = SocietyRuntime([
        RoleSpec("p", "proposer", lambda *_: Contribution(
            "p", "verdict", approved=True)),
    ], SocietyConfig(require_verifier=False, max_turns=1))
    with pytest.raises(ValueError):
        bad_kind.run("task")

    bad_role = SocietyRuntime([
        RoleSpec("p", "proposer", lambda *_: Contribution(
            "alice", "proposal", content="x")),
    ], SocietyConfig(require_verifier=False, max_turns=1))
    with pytest.raises(ValueError):
        bad_role.run("task")


def test_critic_objection_blocks_consensus():
    runtime = SocietyRuntime([
        RoleSpec("p", "proposer", lambda *_: "candidate"),
        RoleSpec("c", "critic", lambda *_: Contribution(
            "c", "critique", content="WRONG", approved=False)),
        RoleSpec("v", "verifier", lambda *_: Contribution(
            "v", "verdict", approved=True)),
    ], SocietyConfig(max_turns=1))
    report = runtime.run("task")
    assert not report.consensus
    assert report.stopped_reason == "max-turns"


def test_role_credit_includes_the_proposer_and_blackboard_is_not_mutated():
    board = {"contributions": []}
    runtime = SocietyRuntime(_roles([]), SocietyConfig(max_turns=2))
    report = runtime.run("task", blackboard=board)
    assert report.role_credit["p"] > 0
    assert board["contributions"] == []


def test_score_fn_is_called_once_per_turn_and_checked():
    calls = []

    def score(_task, _history):
        calls.append(1)
        return 0.1

    runtime = SocietyRuntime([
        RoleSpec("p", "proposer", lambda *_: "candidate"),
    ], SocietyConfig(require_verifier=False, max_turns=2, target_score=0.5))
    runtime.run("task", score_fn=score)
    assert len(calls) == 2


def test_string_verdicts_use_an_explicit_leading_token():
    def run(text):
        runtime = SocietyRuntime([
            RoleSpec("p", "proposer", lambda *_: "candidate"),
            RoleSpec("v", "verifier", lambda *_: text),
        ], SocietyConfig(max_turns=1))
        return runtime.run("task").consensus

    assert not run("Unacceptable.")
    assert not run("I disapprove.")
    assert not run("Never approve this.")
    assert run("Approved, no issues found.")
    assert run("LGTM, nothing to fix")


def test_role_order_must_start_with_a_proposer():
    with pytest.raises(ValueError):
        SocietyRuntime([
            RoleSpec("v", "verifier", lambda *_: "approved"),
            RoleSpec("p", "proposer", lambda *_: "candidate"),
        ])
