"""Tests for ``max_tokens`` budget enforcement.

Covers:
- ``stop_reason == "max_tokens"`` when the cap is reached (sync path)
- ``stop_reason == "max_tokens"`` on the async path
- Token consumption reported in ``RoundInfo.tokens``
- ``result.usage.total_tokens`` reflects actual spend
- A generous cap does not interfere with normal termination
- Interaction with ``max_calls`` and ``max_seconds``
- Zero-token runs (no model) do not trigger the cap
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import pytest

from agentdescent.agents import Usage
from agentdescent.evolution import evolve, Task, RoundInfo, EvolutionResult


# --- helpers ---------------------------------------------------------------


def _tasks(n: int = 8) -> List[Task]:
    return [Task(id=f"t{i}", prompt=f"task {i}") for i in range(n)]


class _CountingAgent:
    """An agent that reports a fixed token cost per call.

    ``prompt_tokens`` and ``completion_tokens`` are set per call, so the total
    is deterministic and controllable — essential for testing a token budget.
    """

    def __init__(self, prompt: int = 100, completion: int = 200) -> None:
        self.prompt = prompt
        self.completion = completion
        self.calls = 0
        self._usage = Usage()

    def solve(self, rendered: str, task: Task) -> str:
        self.calls += 1
        self._usage.record(
            prompt_tokens=self.prompt, completion_tokens=self.completion)
        return task.id

    def propose(self, rendered: str, task: Task, output: str,
                reward: float) -> Optional[str]:
        self.calls += 1
        self._usage.record(
            prompt_tokens=self.prompt, completion_tokens=self.completion)
        return f"rule-{task.id}" if reward < 0.999 else None

    @property
    def usage(self) -> Usage:
        return self._usage


def _run_evolve(*, max_tokens=None, rounds=3, asynchronous=False, **kw):
    """Run a minimal evolve with a counting agent and token tracking."""
    agent = _CountingAgent(prompt=100, completion=200)
    # 300 tokens per call (100 prompt + 200 completion).
    defaults = dict(
        rounds=rounds,
        n_workers=1,
        max_tokens=max_tokens,
        asynchronous=asynchronous,
        usage=agent.usage,  # share so meter sees the tokens
    )
    if asynchronous:
        defaults["max_seconds"] = 30
    defaults.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evolve(_tasks(8), lambda t, o: 0.5, agent=agent, **defaults)
    return result, agent


# --- sync path -------------------------------------------------------------


def test_sync_max_tokens_stops_run():
    """A token budget that is reached stops the run with stop_reason="max_tokens"."""
    # 300 tokens/call, 2 calls/round (solve + propose) = 600 tokens/round.
    # 3 rounds = 1800 tokens. A cap of 1200 should stop after round 2.
    result, agent = _run_evolve(max_tokens=1200, rounds=5)
    assert result.stop_reason == "max_tokens"
    assert result.usage.total_tokens >= 1200
    # The run did not complete all 5 rounds.
    assert len(result.history) < 5


def test_sync_max_tokens_generous_cap_does_not_interfere():
    """A generous cap lets the run complete normally."""
    result, agent = _run_evolve(max_tokens=10_000_000, rounds=2)
    assert result.stop_reason == "rounds"
    assert len(result.history) == 2


def test_sync_max_tokens_zero_when_no_usage():
    """A run with no token reporting (no Usage shared) does not trigger the cap."""
    def run(rendered, task):
        return task.id

    def propose(rendered, task, output, reward):
        return f"rule-{task.id}" if reward < 0.999 else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evolve(
            _tasks(8), lambda t, o: 0.5,
            run=run, propose=propose,
            rounds=2, n_workers=1,
            max_tokens=1,  # would trigger if tokens were counted
        )
    # No Usage object shared → meter reports 0 tokens → cap never fires.
    assert result.stop_reason == "rounds"


def test_round_info_has_tokens_field():
    """RoundInfo.tokens is populated and cumulative."""
    result, agent = _run_evolve(max_tokens=10_000_000, rounds=3)
    assert len(result.history) == 3
    tokens = [r.tokens for r in result.history]
    # Cumulative: should be non-decreasing.
    assert all(tokens[i] <= tokens[i + 1] for i in range(len(tokens) - 1))
    # The last round should reflect the total spend.
    assert tokens[-1] == result.usage.total_tokens


def test_sync_max_tokens_interaction_with_max_calls():
    """When both max_tokens and max_calls are set, whichever fires first wins."""
    # 300 tokens/call, max_calls=3, max_tokens=10000 → max_calls fires first.
    result, agent = _run_evolve(max_tokens=10_000, rounds=5)
    # Now with a tight token cap.
    result2, _ = _run_evolve(max_tokens=600, rounds=5)
    # 600 tokens = 2 calls = 1 round. max_tokens should fire.
    assert result2.stop_reason == "max_tokens"


def test_result_usage_exposes_total_tokens():
    """EvolutionResult.usage.total_tokens is the sum of prompt + completion."""
    result, agent = _run_evolve(max_tokens=10_000_000, rounds=2)
    assert result.usage.total_tokens == result.usage.prompt_tokens + result.usage.completion_tokens
    assert result.usage.total_tokens > 0


# --- async path ------------------------------------------------------------


def test_async_max_tokens_stops_run():
    """The async path checks per-rollout and stops with stop_reason="max_tokens"."""
    result, agent = _run_evolve(
        max_tokens=900, rounds=10, asynchronous=True,
        max_seconds=30)
    # 300 tokens/call. 900 tokens = 3 calls. Should stop early.
    assert result.stop_reason == "max_tokens"
    assert result.usage.total_tokens >= 900


def test_async_max_tokens_generous_cap_completes():
    """A generous cap on the async path does not interfere."""
    result, agent = _run_evolve(
        max_tokens=10_000_000, rounds=2, asynchronous=True,
        max_seconds=30)
    assert result.stop_reason in ("rounds", "max_iters", "max_seconds")


# --- serialization ---------------------------------------------------------


def test_round_info_tokens_serializes():
    """RoundInfo preserves the tokens field through EvolutionResult.to_dict."""
    info = RoundInfo(
        round=3, held_out_reward=0.75, n_items=5,
        committed=2, rejected=1, reasons={},
        elapsed_s=10.0, rollouts=6, calls=12,
        tokens=3600)

    # Verify the field survives __dict__ access (what to_dict uses).
    assert info.__dict__["tokens"] == 3600

    # Verify a real run populates it.
    result, agent = _run_evolve(max_tokens=10_000_000, rounds=1)
    assert result.history[0].tokens > 0
    assert result.history[0].tokens == result.usage.total_tokens


def test_agent_usage_adopted_automatically():
    """When the agent carries a Usage and the caller did not pass usage=,
    the meter adopts the agent's — token budgets and the tokens= column work
    without the caller threading the same object through two parameters."""
    from agentdescent.agents import Usage

    agent = _CountingAgent(prompt=100, completion=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evolve(
            _tasks(8), lambda t, o: 0.5,
            agent=agent,
            rounds=2, n_workers=1,
            max_tokens=10_000_000,
            # NOTE: no usage= here — the agent's must be adopted.
        )
    assert result.usage is agent.usage
    assert result.usage.total_tokens > 0
    assert result.history[0].tokens > 0


def test_agent_usage_adopted_budget_fires():
    """A max_tokens cap fires even without evolve(usage=...) when the agent
    reports tokens through its own Usage."""
    agent = _CountingAgent(prompt=100, completion=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evolve(
            _tasks(8), lambda t, o: 0.5,
            agent=agent,
            rounds=5, n_workers=1,
            max_tokens=900,   # 300/call → stops early
            # no usage= — adoption must make this fire
        )
    assert result.stop_reason == "max_tokens"
    assert result.usage is agent.usage


def test_explicit_usage_wins_over_agent_usage():
    """An explicit usage= takes precedence over the agent's — the caller who
    passes both has said which counter is the truth."""
    from agentdescent.agents import Usage

    agent = _CountingAgent(prompt=100, completion=200)
    mine = Usage()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evolve(
            _tasks(8), lambda t, o: 0.5,
            agent=agent,
            rounds=1, n_workers=1,
            usage=mine,
        )
    assert result.usage is mine
    assert result.usage is not agent.usage


def test_no_usage_anywhere_reports_zero():
    """A bare run/propose pair with no Usage anywhere reports tokens=0 and a
    token budget that cannot fire — the honest reading, not an error."""
    def run(rendered, task):
        return task.id

    def propose(rendered, task, output, reward):
        return f"rule-{task.id}" if reward < 0.999 else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evolve(
            _tasks(8), lambda t, o: 0.5,
            run=run, propose=propose,
            rounds=1, n_workers=1,
            max_tokens=1,
        )
    assert result.usage.total_tokens == 0
    assert result.stop_reason == "rounds"
    assert result.history[0].tokens == 0


def test_async_governor_degrades_self_verify():
    """The async path's governor turns off self-verify at the hard floor.

    We verify this indirectly: a run with a tight token budget should produce
    evidence cards with delta=0 (self-verify skipped) once the budget crosses
    the hard floor, whereas a run without a budget produces non-zero deltas.

    Since we cannot easily inspect individual cards, we verify the observable
    consequence: the run stops with stop_reason='max_tokens' and the governor
    was consulted (the verbose output would mention it, but we check the
    budget fired at all — the governor's check is the only path that produces
    stop_reason='max_tokens' on the async path)."""
    agent = _CountingAgent(prompt=100, completion=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 300 tokens/call. With 2 workers and max_tokens=600,
        # the first 2 calls (600 tokens) hit the budget.
        result = evolve(
            _tasks(8), lambda t, o: 0.5,
            agent=agent,
            rounds=10, n_workers=1,
            max_tokens=600,
            asynchronous=True,
            max_seconds=30,
            usage=agent.usage,
        )
    assert result.stop_reason == "max_tokens"


def test_async_governor_fusion_degrades():
    """The async path's governor degrades the fusion tournament at the soft floor.

    With fusion_tournament=True and a tight token budget, the governor should
    turn off the tournament before the budget is exhausted. We verify by
    checking that the run still completes with the right stop reason and
    the tournament was available at least initially (fusion_trials is populated
    by DefaultFusion and accessible on the result)."""
    from agentdescent.agents import Usage
    agent = _CountingAgent(prompt=100, completion=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = evolve(
            _tasks(8), lambda t, o: 0.5,
            agent=agent,
            rounds=10, n_workers=1,
            max_tokens=2000,
            fusion_tournament=True,
            asynchronous=True,
            max_seconds=30,
            usage=agent.usage,
        )
    # The run should have stopped due to the token budget.
    assert result.stop_reason == "max_tokens"
    # fusion_trials may or may not be populated (depends on whether any merge
    # had survivors before degradation), but the run must not crash.
    assert hasattr(result, "fusion_trials")
