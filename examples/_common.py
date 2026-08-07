"""The command-line contract shared by the faithful algorithm ports.

Algorithm-specific vocabulary stays in each port. In particular, iteration
flags such as ``--rounds``, ``--generations``, ``--iterations``, and ``--steps``
must not be normalised here: they are part of the upstream algorithm's language.

Declaring a flag in one place is only half of a contract -- the code that
*honours* it has to live here too, or a port can grow a ``--yes`` it never reads
and every test still passes. So the behaviours behind the shared flags are
functions, not prose: ``confirm`` for ``--yes``, ``completion_for`` for
``--provider``/``--model``, ``worker_count`` for ``--serial``, and the early
``--dry-run`` return that each port's ``main`` performs before touching data.

``score_tasks`` is here for the same reason: every port had written the same
sequential held-out loop, and every port paid the same silent wall-clock for it.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from agentdescent.agents import Usage, claude, openai_compatible


PROVIDER_CHOICES = ("claude", "openai", "glm")
# Providers served by the OpenAI-compatible adapter; 'glm' is a legacy alias.
OPENAI_COMPATIBLE = ("openai", "glm")
DEFAULT_MODEL = "claude-haiku-4-5"


def add_standard_args(
    parser: argparse.ArgumentParser,
    *,
    model_default: Optional[str] = DEFAULT_MODEL,
    model_help: str = "model id",
    max_seconds_default: float = 30.0,
) -> argparse.ArgumentParser:
    """Add the provider/runtime flags shared by every algorithm port.

    Defaults that describe an algorithm's measured setup remain caller-owned.
    DGM, for example, deliberately defaults ``model`` to ``None`` because its
    surrogate can run without an API, while async wall-clock budgets differ by
    workload.
    """
    parser.add_argument(
        "--provider",
        default="claude",
        choices=PROVIDER_CHOICES,
        help=("claude, or any OpenAI-compatible endpoint (DeepSeek, GLM, "
              "vLLM, ...) via OPENAI_BASE_URL + OPENAI_API_KEY; 'glm' is a "
              "legacy alias"),
    )
    parser.add_argument("--model", default=model_default, help=model_help)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--async",
        dest="asynchronous",
        action="store_true",
        help="run barrier-free (async_evolve)",
    )
    parser.add_argument(
        "--async-ratio", type=int, default=3, help="staleness lag budget")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=max_seconds_default,
        help="async wall-clock budget",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show the plan without loading data or accessing the network",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip confirmation before real model API calls",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="run the upstream algorithm's own semantics: one worker, no merge",
    )
    parser.add_argument(
        "--budget-rollouts",
        type=int,
        default=None,
        help=("total rollouts, held fixed as workers vary -- required to compare "
              "--serial against a parallel run (see budget_kwargs)"),
    )
    return parser


def budget_kwargs(args: argparse.Namespace) -> dict:
    """``max_rollouts=`` for ``evolve()``, or nothing when no budget was asked for.

    **A speedup measured without this is not a speedup.** Six of the seven ports
    pass a fixed ``rounds`` and let ``n_workers`` multiply it, so an ``N=8`` arm
    performs *eight times* the rollouts of the ``--serial`` arm. Comparing their
    wall-clocks then reports eight times the model spend as parallel efficiency,
    and comparing their final quality credits the extra spend to parallelism.
    That is the confound :mod:`agentdescent.baselines` exists to remove, and
    ``docs/results.md`` already carries a warning that a speedup table cannot
    distinguish merging from sampling.

    OpenEvolve is the exception and got it right on its own: it derives
    ``rounds = iterations // workers``, so its total work is fixed and workers
    only change how it is divided. That is why its speedup row means something
    different from the other six unless they are budgeted -- which is a fact the
    matrix has to state, not one a reader should have to find.

    The engine has enforced this since ``evolve(max_rollouts=)`` shipped, and no
    port passed it. Left ``None`` by default, because a port run on its own is
    not a comparison and should keep the configuration its own docs describe;
    the moment two arms are compared, both need it.

    The synchronous path checks at the round barrier, so an ``N`` -worker arm
    overshoots by up to ``N-1`` rollouts. ``result.rollouts`` is what was
    actually spent -- report that rather than the budget.
    """
    return ({"max_rollouts": args.budget_rollouts}
            if getattr(args, "budget_rollouts", None) else {})


def is_openai_compatible(args: argparse.Namespace) -> bool:
    """Whether ``--provider`` selects the OpenAI-compatible adapter."""
    return args.provider in OPENAI_COMPATIBLE


def confirm(args: argparse.Namespace) -> bool:
    """Whether the run may proceed to real model API calls.

    Honours ``--yes``, and treats a non-interactive stdin as consent so a port
    stays scriptable in CI. Prints ``aborted.`` when the answer is no, so the
    caller only has to ``return``.
    """
    if args.yes or not sys.stdin.isatty():
        return True
    if input("\nProceed with real API calls? [y/N] ").strip().lower() in ("y", "yes"):
        return True
    print("aborted.")
    return False


def score_tasks(solve, artifact: str, tasks, reward, *,
                concurrency: int = 8) -> float:
    """Score an artifact on a task list, concurrently.

    Every port reports a final held-out number by looping over the split one task
    at a time. That is the *reported metric*, so it runs after `evolve()` returns
    and outside everything the engine parallelises -- `eval_concurrency` bounds
    the gate and never reaches here. On a reasoning model at ~38s a call, a
    20-task split is thirteen minutes of wall-clock per scoring pass, in silence,
    and a sweep pays it twice per arm: once for `final_reward` and once for the
    test split.

    Concurrency changes no result. Each task is scored independently, `reward` is
    pure, and the sum is order-independent -- so this is wall-clock only, which is
    why it is a plain default rather than a knob a caller has to discover.

    Sequential below two tasks, so a small split does not pay for a pool.
    """
    tasks = list(tasks)
    if not tasks:
        return 0.0
    if len(tasks) < 2 or concurrency < 2:
        return sum(reward(t, solve(artifact, t)) for t in tasks) / len(tasks)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(concurrency, len(tasks))) as pool:
        outputs = list(pool.map(lambda t: solve(artifact, t), tasks))
    return sum(reward(t, o) for t, o in zip(tasks, outputs)) / len(tasks)


def worker_count(args: argparse.Namespace, requested: int) -> int:
    """The number of workers to run, after ``--serial``.

    Every port here parallelises an algorithm that was published as a **serial**
    loop, and until this flag existed none of them could run that loop. So every
    claim about parallelising them -- speedup, and more importantly whether the
    final quality survives -- had no baseline in the repository at all. A speedup
    without the serial arm is a measurement of nothing.

    One worker, and the merge that goes with it disappears: with a single
    proposal per step there is nothing to fuse and nothing to resolve, which is
    what makes this the upstream semantics rather than a narrow version of ours.

    Refused together with ``--async``, loudly. The barrier-free runtime's
    concurrency *is* ``n_workers``, so ``--serial --async`` is not the upstream
    algorithm: it is a one-worker asynchronous run, where a diff can still be
    proposed against a head the merger has since moved. Reporting that as the
    serial baseline would put staleness into the control arm, which is the one
    place it must not be.
    """
    if not getattr(args, "serial", False):
        return requested
    if getattr(args, "asynchronous", False):
        raise SystemExit(
            "--serial and --async are contradictory: the barrier-free runtime's "
            "concurrency is n_workers, so --serial --async is a one-worker "
            "asynchronous run rather than the upstream serial algorithm -- its "
            "diffs can still go stale against a moved head. Drop one of them.")
    return 1


def completion_for(args: argparse.Namespace, *, usage: Optional[Usage] = None,
                   **kwargs):
    """Build the ``Completion`` that ``--provider`` and ``--model`` select.

    Extra keyword arguments reach whichever factory is chosen, so pass only
    options both accept; branch in the caller for genuinely one-sided ones (ADAS
    does this for the OpenAI-only ``--timeout``).
    """
    if is_openai_compatible(args):
        return openai_compatible(model=args.model, usage=usage, **kwargs)
    return claude(model=args.model, usage=usage, **kwargs)
