"""Reusable backend faults, each modelling a real failure mode.

Every resilience bug found so far surfaced only under a *real* fault -- a dead
socket, a wedged thread, a process that would not exit -- and each was found by a
throwaway script that then disappeared. These primitives make the same faults
cheap to apply, so a change that regresses one is caught by the suite instead of
by the next person who thinks to look.

Most wrap a `run(rendered, task)` and return a drop-in replacement.
`flaky_propose` wraps the *reflector* instead, and `ledger_dies_after` patches the
ledger -- because faulting only `run` left three classes of real failure
unmodelled, and each of them had already caused a bug.
"""
from __future__ import annotations

import itertools
import random
import threading
import time
from typing import Callable

OK = lambda rendered, task: "wrong"      # scores poorly, so `propose` keeps firing


def never_works(message: str = "401 unauthorized") -> Callable:
    """A misconfigured backend: wrong key, wrong URL. Must fail fast and loudly."""
    def run(rendered, task):
        raise RuntimeError(message)
    return run


def flaky(rate: float = 1 / 3, message: str = "429 rate limited",
          inner: Callable = OK, seed: int = 0) -> Callable:
    """A throttled backend: `rate` of calls fail, seeded so runs are reproducible.

    The important case, and the one that used to kill runs. Every worker shares
    one backend, so shedding workers cannot relieve it.

    Reproducible **counts**, not assignment: the seeded draw order is fixed, so a
    run always fails the same number of times, but under `max_concurrency > 1`
    which call receives which draw depends on thread interleaving.

    Seeded random rather than "every Nth call" on purpose. A strict modulo makes
    faults *periodic*, which can construct situations no real backend produces:
    with `i % 3`, a held-out measurement that makes 4 consecutive calls always
    covers an index divisible by 3, so it can never succeed -- the run then looks
    broken when nothing is. Real throttling is not periodic.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate is a probability in [0, 1], got {rate!r} -- this "
                         f"used to be a 'fail every N calls' count")
    rng, lock = random.Random(seed), threading.Lock()

    def run(rendered, task):
        with lock:
            fail = rng.random() < rate
        if fail:
            raise RuntimeError(message)
        return inner(rendered, task)
    return run


def dies_after(n: int, message: str = "500 backend gone",
               inner: Callable = OK) -> Callable:
    """A backend that works, then stops: credit exhausted, endpoint pulled."""
    counter, lock = itertools.count(), threading.Lock()

    def run(rendered, task):
        with lock:
            i = next(counter)
        if i >= n:
            raise RuntimeError(message)
        return inner(rendered, task)
    return run


def recovers_after(n: int, message: str = "503 transient",
                   inner: Callable = OK) -> Callable:
    """An outage that ends. Nothing should retire permanently over it."""
    counter, lock = itertools.count(), threading.Lock()

    def run(rendered, task):
        with lock:
            i = next(counter)
        if i < n:
            raise RuntimeError(message)
        return inner(rendered, task)
    return run


def wedged(seconds: float = 30.0, task_id: str = "0",
           inner: Callable = OK) -> Callable:
    """One rollout that hangs. Python cannot kill it, so it outlives its round."""
    def run(rendered, task):
        if task.id == task_id:
            time.sleep(seconds)
        return inner(rendered, task)
    return run


def returns_nothing(value=None) -> Callable:
    """A backend that **succeeds** and answers with nothing usable.

    Every other fault here raises, so this whole class had no primitive -- and it
    is the one the codebase itself calls the most insidious: `LLMAgent.propose`
    carries a dedicated counter and warning for it ("nearly always a starved
    reasoning model: the token budget went to internal reasoning and no visible
    content came back"), and `claude()`'s docstring records 4 of 8 reflection
    prompts returning nothing at a 1024-token cap.

    It raises nothing, so it bypasses every retry, backoff and error-tolerance
    path in the engine and lands as a *clean* run: `error=None`, a full `history`,
    and `outcomes()` reporting `below-threshold` -- which is documented to mean
    "the reflector is the problem", the opposite of the truth.

    ``value=None`` models an OpenAI-compatible endpoint returning JSON `null`;
    ``""`` models a starved reasoning model.
    """
    def run(rendered, task):
        return value
    return run


def ledger_dies_after(n: int, message: str = "index.lock exists") -> Callable:
    """Make the *ledger* fail after ``n`` git calls. Returns a context manager.

    The one component with no retry anywhere, the sole writer, on the critical
    path of every round -- and nothing faulted it. Its failure escaping as an
    exception (rather than the documented partial result) was found only by
    injecting this by hand, which is the workflow these primitives exist to replace.
    """
    import contextlib

    from agentdescent import ledger as ledger_mod

    @contextlib.contextmanager
    def patched():
        original, calls = ledger_mod._git, {"n": 0}

        def failing(repo, *args):
            calls["n"] += 1
            if calls["n"] > n:
                raise ledger_mod.GitError(f"git {args[0]} failed in {repo}: {message}")
            return original(repo, *args)

        ledger_mod._git = failing
        try:
            yield calls
        finally:
            ledger_mod._git = original
    return patched()


def flaky_propose(rate: float = 1 / 3, message: str = "429 rate limited",
                  seed: int = 0) -> Callable:
    """A **reflector** that fails while the solver stays healthy.

    Every fault above wraps `run`, and the matrix pairs them with a reflector that
    never fails -- but in any real configuration `propose` is a *separate backend
    call*, often to a different model (`reflector(claude(model="claude-haiku-4-5"))`
    is the documented pattern). A reflector outage is an ordinary two-provider
    failure, and it is the case where the run keeps producing rollouts and quietly
    stops learning.
    """
    rng, lock = random.Random(seed), threading.Lock()

    def propose(rendered, task, output, score):
        with lock:
            fail = rng.random() < rate
        if fail:
            raise RuntimeError(message)
        return f"rule-{task.id}"
    return propose


def slow(seconds: float = 0.05, inner: Callable = OK) -> Callable:
    """Uniform latency -- makes concurrency observable in wall-clock."""
    def run(rendered, task):
        time.sleep(seconds)
        return inner(rendered, task)
    return run
