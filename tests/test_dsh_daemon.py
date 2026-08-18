"""The sidecar, driven by a fake harness over a real bridge.

Nothing here stubs the engine: `evolve()` runs for real, with its ledger,
aggregator and acceptance test. What is faked is the *harness* -- the far side
that executes rollouts, scores them and runs the model -- because that is the
side the plugin owns and the side this protocol has to be right about.
"""

from __future__ import annotations

import json
import os
import threading

import pytest

from agentdescent.dsh.bridge import PROTOCOL_VERSION, Peer
from agentdescent.dsh.daemon import ENGINE_NAME, Engine, _clamp

START = "---\ndescription: total a column\n---\n\nCOLUMN: id\n"
FIXED = "---\ndescription: total a column\n---\n\nCOLUMN: amount\n"

EDIT = "<EDITS>" + json.dumps({
    "rationale": "id is a row number, not a quantity",
    "edits": [{"path": "SKILL.md", "content": FIXED}],
}) + "</EDITS>"


class FakeHarness:
    """The plugin's side: loads the artifact, runs rollouts, scores, completes."""

    def __init__(self, *, tasks=4, tokens_per_rollout=0):
        self.published = []
        self.progress = []
        self.rollouts = 0
        self.tokens_per_rollout = tokens_per_rollout
        self.n_tasks = tasks
        self.block = None

    def handlers(self):
        return {
            "artifact.load": self.load,
            "tasks.fetch": self.fetch,
            "rollout.execute": self.execute,
            "reward.score": self.score,
            "llm.complete": self.complete,
            "artifact.publish": self.publish,
            "log.progress": self.progress.append,
        }

    def load(self, _params):
        return {"SKILL.md": START}

    def fetch(self, params):
        return [{"id": f"t{i}", "prompt": "total the column", "meta": {"gold": "ok"}}
                for i in range(min(self.n_tasks, int(params["count"])))]

    def execute(self, params):
        if self.block is not None:
            self.block.wait(10)
        self.rollouts += 1
        body = params["state"].get("SKILL.md", "")
        # The rollout genuinely reads the candidate: that is what makes this a
        # measurement of the artifact rather than of the model's priors.
        output = "correct" if "COLUMN: amount" in body else "wrong"
        return {"output": output, "finishReason": "completed", "steps": 1,
                "tokens": self.tokens_per_rollout}

    def score(self, params):
        return 1.0 if params["run"]["output"] == "correct" else 0.0

    def complete(self, _params):
        return {"text": EDIT, "tokens": 0}

    def publish(self, params):
        self.published.append(params)
        return None


def wire(harness, **engine_kwargs):
    """An Engine and a fake harness, connected by two real pipes."""
    e_read, h_write = os.pipe()
    h_read, e_write = os.pipe()
    errors = []
    engine_peer = Peer(os.fdopen(e_read, "r"), os.fdopen(e_write, "w"),
                       on_error=errors.append)
    harness_peer = Peer(os.fdopen(h_read, "r"), os.fdopen(h_write, "w"),
                        harness.handlers(), on_error=errors.append)
    engine = Engine(engine_peer, **engine_kwargs)
    engine_peer.start()
    harness_peer.start()
    return engine, harness_peer, engine_peer, errors


def wait_for(harness_peer, run_id, phases, timeout=90.0):
    deadline = threading.Event()
    for _ in range(int(timeout * 20)):
        status = harness_peer.call("run.status", {"runId": run_id}, timeout=10)
        if status is not None and status["phase"] in phases:
            return status
        deadline.wait(0.05)
    raise AssertionError(f"run {run_id} never reached {phases}")


SPEC = {"artifact": "skill:total", "taskSource": "manual", "scorer": "gold",
        "rounds": 3, "workers": 1, "taskCount": 4}


def test_handshake_reports_the_protocol_and_the_engine():
    engine, harness_peer, engine_peer, _ = wire(FakeHarness())
    try:
        shake = harness_peer.call("handshake", {}, timeout=10)
        assert shake["protocol"] == PROTOCOL_VERSION
        assert shake["engine"] == ENGINE_NAME
        assert isinstance(shake["version"], str)
    finally:
        harness_peer.close(), engine_peer.close()


def test_a_whole_run_improves_the_artifact_and_publishes_it():
    """The end-to-end claim: the harness executes, the engine optimises, and the
    accepted state comes back over the same wire."""
    harness = FakeHarness()
    engine, harness_peer, engine_peer, errors = wire(harness)
    try:
        run_id = harness_peer.call("run.start", SPEC, timeout=10)
        status = wait_for(harness_peer, run_id, {"done", "failed"})
        assert status["phase"] == "done", status.get("error")
        assert harness.rollouts > 0
        assert len(harness.published) == 1
        published = harness.published[0]
        assert published["artifact"] == "skill:total"
        assert "COLUMN: amount" in published["state"]["SKILL.md"]
        assert published["commit"]["heldOutAfter"] >= published["commit"]["heldOutBefore"]
    finally:
        harness_peer.close(), engine_peer.close()


def test_the_ledger_head_and_history_follow_a_committed_run():
    harness = FakeHarness()
    engine, harness_peer, engine_peer, _ = wire(harness)
    try:
        run_id = harness_peer.call("run.start", SPEC, timeout=10)
        wait_for(harness_peer, run_id, {"done", "failed"})
        head = harness_peer.call("ledger.head", {"artifact": "skill:total"}, timeout=10)
        assert head is not None and head["version"] == 1
        assert "COLUMN: amount" in head["state"]["SKILL.md"]
        history = harness_peer.call("ledger.history", {"artifact": "skill:total"}, timeout=10)
        assert len(history) == 1 and history[0]["runId"] == run_id
    finally:
        harness_peer.close(), engine_peer.close()


def test_progress_arrives_as_notifications_during_the_run():
    harness = FakeHarness()
    engine, harness_peer, engine_peer, _ = wire(harness)
    try:
        run_id = harness_peer.call("run.start", SPEC, timeout=10)
        wait_for(harness_peer, run_id, {"done", "failed"})
        assert harness.progress, "no log.progress ever arrived"
        assert all(p["runId"] == run_id for p in harness.progress)
        assert all("round" in p for p in harness.progress)
    finally:
        harness_peer.close(), engine_peer.close()


def test_an_unknown_artifact_load_failure_is_reported_not_raised():
    class Broken(FakeHarness):
        def load(self, _params):
            raise RuntimeError("no such artifact")

    engine, harness_peer, engine_peer, _ = wire(Broken())
    try:
        run_id = harness_peer.call("run.start", SPEC, timeout=10)
        status = wait_for(harness_peer, run_id, {"failed"})
        assert "no such artifact" in status["error"]
    finally:
        harness_peer.close(), engine_peer.close()


def test_a_task_source_that_returns_nothing_fails_loudly():
    """There is no run to make without tasks, and a silent zero-task run would
    report 'done' having measured nothing."""
    engine, harness_peer, engine_peer, _ = wire(FakeHarness(tasks=0))
    try:
        run_id = harness_peer.call("run.start", SPEC, timeout=10)
        status = wait_for(harness_peer, run_id, {"failed"})
        assert "no run to make without tasks" in status["error"]
    finally:
        harness_peer.close(), engine_peer.close()


def test_cancel_stops_the_run_at_the_next_rollout():
    harness = FakeHarness()
    harness.block = threading.Event()
    engine, harness_peer, engine_peer, _ = wire(harness)
    try:
        run_id = harness_peer.call("run.start", SPEC, timeout=10)
        wait_for(harness_peer, run_id, {"running", "starting"})
        assert harness_peer.call("run.cancel", {"runId": run_id}, timeout=10)["cancelled"]
        harness.block.set()                       # let the in-flight rollout finish
        status = wait_for(harness_peer, run_id, {"cancelled", "done", "failed"})
        # Cooperative by contract: no *further* work, not "stopped now".
        assert status["phase"] == "cancelled"
        assert harness.published == []
    finally:
        harness.block.set()
        harness_peer.close(), engine_peer.close()


def test_cancelling_an_unknown_run_says_so_instead_of_pretending():
    engine, harness_peer, engine_peer, _ = wire(FakeHarness())
    try:
        assert harness_peer.call("run.cancel", {"runId": "run-99"},
                                 timeout=10)["cancelled"] is False
        assert harness_peer.call("run.status", {"runId": "run-99"}, timeout=10) is None
    finally:
        harness_peer.close(), engine_peer.close()


def test_the_token_budget_is_a_ceiling_that_ends_the_run():
    """A budget that only warns is not a budget. The run stops and says why."""
    harness = FakeHarness(tokens_per_rollout=1000)
    engine, harness_peer, engine_peer, _ = wire(harness)
    try:
        run_id = harness_peer.call("run.start", {**SPEC, "tokenBudget": 1500}, timeout=10)
        status = wait_for(harness_peer, run_id, {"budget-exhausted", "done", "failed"})
        assert status["phase"] == "budget-exhausted"
        assert status["tokensSpent"] >= 1500
        assert harness.published == []
    finally:
        harness_peer.close(), engine_peer.close()


def test_a_run_that_improves_nothing_reports_done_without_publishing():
    """The gate declining is the expected outcome on a small held-out set, not a
    fault -- but it must not look like a commit."""
    class NoHelp(FakeHarness):
        def complete(self, _params):
            return {"text": "<EDITS>" + json.dumps({
                "rationale": "no change", "edits": [{"path": "SKILL.md", "content": START}],
            }) + "</EDITS>", "tokens": 0}

    harness = NoHelp()
    engine, harness_peer, engine_peer, _ = wire(harness)
    try:
        run_id = harness_peer.call("run.start", SPEC, timeout=10)
        status = wait_for(harness_peer, run_id, {"done", "failed"})
        assert status["phase"] == "done"
        assert harness.published == []
        assert harness_peer.call("ledger.head", {"artifact": "skill:total"}, timeout=10) is None
    finally:
        harness_peer.close(), engine_peer.close()


@pytest.mark.parametrize("value,expected", [
    (0.5, 0.5), (1.0, 1.0), (0.0, 0.0),
    (2.0, 1.0), (-1.0, 0.0),            # a scorer's contract violation is clamped
    (float("nan"), 0.0), ("nonsense", 0.0), (None, 0.0),
])
def test_a_reward_outside_the_contract_is_clamped_rather_than_inherited(value, expected):
    assert _clamp(value) == expected


def test_a_staged_commit_does_not_become_a_head():
    """An L1 change is published only after a human approves it.

    If the engine recorded a head anyway it would report a version the harness
    is not serving, and the next run would start from a state that never
    shipped -- neither of which shows up as an error.
    """
    class Staging(FakeHarness):
        def publish(self, params):
            self.published.append(params)
            return {"status": "pending", "pendingId": "pending-1",
                    "reason": "blast radius 0.6 is above the line"}

    harness = Staging()
    engine, harness_peer, engine_peer, _ = wire(harness)
    try:
        run_id = harness_peer.call("run.start", SPEC, timeout=10)
        status = wait_for(harness_peer, run_id, {"awaiting-review", "done", "failed"})
        assert status["phase"] == "awaiting-review", status.get("error")
        assert len(harness.published) == 1          # it was offered
        # ...and not recorded as landed.
        assert harness_peer.call("ledger.head", {"artifact": "skill:total"}, timeout=10) is None
        assert harness_peer.call("ledger.history", {"artifact": "skill:total"}, timeout=10) == []
    finally:
        harness_peer.close(), engine_peer.close()


def test_an_async_run_states_its_budget_instead_of_inheriting_a_20_second_default():
    """`evolve(asynchronous=True)` has no unbounded mode.

    Left implicit, `max_seconds=None` becomes 20 seconds and `rounds` silently
    becomes a rollout budget -- so a run that expired after twenty seconds
    reports the same shape as one that converged.
    """
    budget = Engine._budget({"asynchronous": True}, rounds=4, workers=2)
    assert budget["asynchronous"] is True
    assert budget["max_rollouts"] == 8              # rounds x workers, said outright
    assert budget["max_seconds"] > 20.0             # not the silent default

    assert Engine._budget({}, rounds=4, workers=2) == {}   # sync path untouched


def test_an_async_run_reaches_a_terminal_phase_like_any_other():
    harness = FakeHarness()
    engine, harness_peer, engine_peer, _ = wire(harness)
    try:
        run_id = harness_peer.call(
            "run.start", {**SPEC, "asynchronous": True, "maxRollouts": 8,
                          "maxSeconds": 30.0}, timeout=10)
        status = wait_for(harness_peer, run_id, {"done", "failed", "awaiting-review"})
        assert status["phase"] in {"done", "awaiting-review"}, status.get("error")
        assert harness.rollouts > 0
    finally:
        harness_peer.close(), engine_peer.close()


def test_every_phase_transition_is_announced_not_just_progress():
    """The harness keeps a synchronous `status()` for callers polling from render
    paths, and it can only be right if every transition arrives.

    Without this the last thing the harness ever heard was a progress line
    saying `running`, so a finished run showed as running forever -- and nothing
    could wait for one to end.
    """
    harness = FakeHarness()
    phases = []
    engine, harness_peer, engine_peer, _ = wire(harness)
    harness_peer.handle("log.phase", lambda status: phases.append(status["phase"]))
    try:
        run_id = harness_peer.call("run.start", SPEC, timeout=10)
        wait_for(harness_peer, run_id, {"done", "failed", "awaiting-review"})
        # Give the notification a moment to cross; it is fire-and-forget.
        for _ in range(100):
            if "done" in phases:
                break
            threading.Event().wait(0.02)
        assert "running" in phases
        assert phases[-1] == "done", phases
    finally:
        harness_peer.close(), engine_peer.close()


def test_a_cancelled_run_announces_its_ending_too():
    harness = FakeHarness()
    harness.block = threading.Event()
    phases = []
    engine, harness_peer, engine_peer, _ = wire(harness)
    harness_peer.handle("log.phase", lambda status: phases.append(status["phase"]))
    try:
        run_id = harness_peer.call("run.start", SPEC, timeout=10)
        wait_for(harness_peer, run_id, {"running", "starting"})
        harness_peer.call("run.cancel", {"runId": run_id}, timeout=10)
        harness.block.set()
        wait_for(harness_peer, run_id, {"cancelled", "done", "failed"})
        for _ in range(100):
            if "cancelled" in phases:
                break
            threading.Event().wait(0.02)
        assert "cancelled" in phases, phases
    finally:
        harness.block.set()
        harness_peer.close(), engine_peer.close()
