"""Reusable backend faults, each modelling a real failure mode.

Every resilience bug found so far surfaced only under a *real* fault -- a dead
socket, a wedged thread, a process that would not exit -- and each was found by a
throwaway script that then disappeared. These primitives make the same faults
cheap to apply, so a change that regresses one is caught by the suite instead of
by the next person who thinks to look.

Each fault wraps a `run(rendered, task)` and returns a drop-in replacement.
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


def slow(seconds: float = 0.05, inner: Callable = OK) -> Callable:
    """Uniform latency -- makes concurrency observable in wall-clock."""
    def run(rendered, task):
        time.sleep(seconds)
        return inner(rendered, task)
    return run
