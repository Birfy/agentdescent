"""Knobs the async path used to drop in silence.

`evolve(asynchronous=True)` forwarded some arguments and quietly ignored others,
so a run could look configured while behaving as the default. Each is now either
honoured or warned about.
"""
import warnings

import pytest

from agentdescent.evolution import SingleSlot, Task, evolve
from agentdescent.parallel import TensorParallel

TASKS = [Task(id=str(i), prompt="p", meta={"gold": "x"}) for i in range(6)]


def _flat_run(**kw):
    """A run that can never improve -- so only a stop rule ends it early."""
    return evolve(TASKS, lambda t, o: 0.5, run=lambda a, t: "x",
                  propose=lambda *a: None, strategy=SingleSlot(initial_value="v"),
                  asynchronous=True, n_workers=2, rounds=200, **kw)


def test_patience_reaches_the_async_runtime():
    """It was accepted by evolve() and never forwarded -- the run ignored it."""
    from agentdescent.async_evolve import async_evolve
    import inspect
    assert "patience" in inspect.signature(async_evolve).parameters


def test_patience_stops_a_stalled_async_run():
    res = _flat_run(patience=2, max_seconds=20.0)
    # Stopped on patience, not on the 20s wall clock.
    assert res.error is None
    assert len(res.history) <= 6, f"ran {len(res.history)} sweeps despite patience=2"


def test_async_warns_about_knobs_it_cannot_honour():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _flat_run(parallel=TensorParallel(n_sections=2), max_concurrency=4,
                  round_timeout=30, max_seconds=1.0)
    names = ("parallel", "max_concurrency", "round_timeout")
    dropped = {m for x in w for m in names if f"ignores {m}=" in str(x.message)}
    assert dropped == set(names)


def test_no_evolve_argument_is_dropped_without_notice():
    """Guards the whole surface: a new evolve() knob must be forwarded or warned.

    `patience=` was silently dropped for exactly as long as nobody checked."""
    import inspect, re
    from agentdescent.evolution import evolve as _evolve
    src = inspect.getsource(_evolve)
    forwarded = set(re.findall(r"(\w+)=", src.split("return async_evolve(")[1]
                                              .split(")\n")[0]))
    warned = set(re.findall(r'\("(\w+)", ', src))
    # Forwarded under another name: `max_rollouts` is what the barrier-free path
    # has always called `max_iters`, so it reaches async_evolve as that.
    remapped = {"tasks", "reward", "rounds", "max_seconds", "asynchronous",
                "max_rollouts"}
    leaked = set(inspect.signature(_evolve).parameters) - forwarded - warned - remapped
    assert not leaked, f"silently dropped by the async path: {sorted(leaked)}"


def test_default_async_call_is_quiet():
    """The warning must fire on an explicit value, not on every async run."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _flat_run(max_seconds=1.0)
    assert not [x for x in w if "ignores" in str(x.message)]
