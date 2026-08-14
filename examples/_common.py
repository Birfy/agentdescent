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
    async_ratio_default: int = 3,
    eval_concurrency_default: Optional[int] = 8,
    include_val_cap: bool = True,
) -> argparse.ArgumentParser:
    """Add the provider/runtime flags shared by every algorithm port.

    Defaults that describe an algorithm's measured setup remain caller-owned.
    DGM, for example, deliberately defaults ``model`` to ``None`` because its
    surrogate can run without an API, while async wall-clock budgets differ by
    workload.

    ``async_ratio_default`` is caller-owned for the same reason: the lag budget
    counts *artifact versions*, so how much wall-clock it buys depends entirely
    on how long the caller's rollout takes. Three is sensible when a rollout is
    a model call taking seconds; :mod:`examples._method_runner` sets one,
    because that is the value every row in ``bench/results/`` was measured at.

    ``eval_concurrency_default=None`` says "this runner has a rule of its own;
    only override it when the flag is given". The MethodPolicy runner needs
    that: its ``--serial`` control scores the gate one task at a time, and a
    plain default of 8 would quietly make the control arm partly parallel --
    the confound ``bench/matrix_run.py`` already documents for the other ports.

    ``include_val_cap=False`` withholds ``--val-cap`` from a port whose splits
    are frozen before the parser ever runs. Accepting it there would be exactly
    the defect this module exists to prevent: a flag that parses, reports
    nothing, and moves nothing.
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
        "--async-ratio", type=int, default=async_ratio_default,
        help=f"staleness lag budget (default {async_ratio_default})")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=max_seconds_default,
        help="async wall-clock budget",
    )
    parser.add_argument(
        "--pipelined-gate",
        action="store_true",
        help=("--async only: run a merge's measurement phase on its own threads "
              "instead of on the merger, so the merger keeps draining while the "
              "gate runs (see async_evolve(pipelined_gate=))"),
    )
    parser.add_argument(
        "--gate-workers", type=int, default=2,
        help="threads for --pipelined-gate (default 2)")
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
    if include_val_cap:
        parser.add_argument(
            "--val-cap",
            type=int,
            default=None,
            help=("cap the gate/D_pareto split without shrinking test -- the only "
                  "lever on the rollout/evaluation ratio (see capped_val)"),
        )
    parser.add_argument(
        "--eval-concurrency",
        type=int,
        default=eval_concurrency_default,
        help=("held-out evaluations in flight at once; wall-clock only"
              + (" -- unset, the runner's own rule applies"
                 if eval_concurrency_default is None
                 else f" (default {eval_concurrency_default})")),
    )
    parser.add_argument(
        "--reflective-merge",
        action="store_true",
        help=("merge contradicting diffs with a model instead of ranking them on "
              "the cheap layer (see agentdescent.fusion.reflective_merge)"),
    )
    parser.add_argument(
        "--eval-cache",
        default="",
        metavar="DIR",
        help=("memoise held-out evaluations to this directory, shared across "
              "processes (see eval_cache_kwargs). Off by default: a cache that "
              "outlives the run changes what a rerun measures"),
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help=("ask the backend not to emit reasoning tokens (Anthropic-shaped "
              "endpoints only). A throughput knob, and a quality one -- see "
              "completion_for"),
    )
    return parser


def merge_kwargs(args: argparse.Namespace, complete) -> dict:
    """``policies=`` for ``--reflective-merge``, or nothing.

    The default aggregator resolves a contradiction by **scoring both sides on
    the cheap layer and dropping the loser**. On a one-key artifact -- GEPA's
    ``InstructionSlot``, and so this dataset -- *every* pair of proposals
    contradicts, so that ranking runs on every pair, every round, and the merge
    step it feeds then has a single candidate and never merges at all.

    :func:`~agentdescent.fusion.reflective_merge` replaces both halves of that:
    :class:`~agentdescent.fusion.KeepContradictions` stops the pairwise scoring,
    and :class:`~agentdescent.fusion.ReflectiveFusion` asks a model for the
    **union of the deltas** and hands it straight to the acceptance gate. One
    model call in place of a sweep per pair, and the round's proposals survive
    together instead of all but one being dropped.

    It is a change to *how diffs are merged*, which is the column
    ``docs/port-fidelity.md`` calls changeable -- but it is not free of meaning:
    the union that goes forward is not the pairwise winner, so a row measured
    with this on has to say so.
    """
    if not getattr(args, "reflective_merge", False):
        return {}
    from agentdescent import Policies
    from agentdescent.fusion import reflective_merge
    return {"policies": Policies(**reflective_merge(complete))}


def capped_val(trainval, val_frac: float, val_cap: Optional[int]):
    """Trim the gate's split without touching train or test.

    The loaders split at fixed ratios, so a ``--fetch``/``--pool`` knob moves all
    three at once -- and the two that matter pull in **opposite** directions.
    **val** is what a run costs: every admitted candidate is scored across it, so
    it multiplies by every proposal. **test** is what a run can *resolve*: it is
    scored once, at the end, and a split under 20 tasks cannot separate anything.
    Chasing a 20-task test split by raising the fetch buys a 20-task gate nobody
    asked for.

    Measured on GEPA/HotpotQA: a rollout costs 3 model calls and the D_pareto
    sweep it triggers costs ``len(val)``. At ``val=20`` the rollout is 3/23 of the
    work, so the part ``n_workers`` can parallelise is 13% of the run and the
    speedup ceiling is ~1.13x **whatever the budget** -- both sides scale
    together. Halving val is the only lever that moves that ratio.

    ``evolve()`` splits by position -- the last ``held_out_frac`` of the sequence
    -- so capping is a slice of the tail plus a recomputed fraction. Train keeps
    everything the cap drops rather than discarding it.

    The gate gets noisier, and that is a real cost rather than a free one:
    acceptance decisions rest on fewer tasks. Lifted from
    ``bench/baselines_run.py``, where the equal-budget work needed exactly this.
    """
    if not val_cap:
        return list(trainval), val_frac
    n = len(trainval)
    n_val = max(1, round(n * val_frac))
    if n_val <= val_cap:
        return list(trainval), val_frac
    train, val = list(trainval[:n - n_val]), list(trainval[n - n_val:])
    kept, spare = val[:val_cap], val[val_cap:]
    tasks = train + spare + kept          # the cap's leftovers become train
    return tasks, len(kept) / len(tasks)


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


def report_engine(result) -> None:
    """Print the engine's own counters in the one format the matrix parses.

    ``bench/matrix_run.py`` reads a port through its stdout, and until this line
    existed the ports printed *their* numbers (test score, usage) but never the
    engine's: measured rollouts rather than the budget, and staleness with its
    denominator. The async arm's quality column is unreadable without the
    latter -- "the score dropped" with no ``stale_discarded`` beside it cannot
    distinguish a lag-budget problem from a reflector problem, which need
    opposite fixes. One shared implementation, because seven hand-rolled copies
    of a parsed format is how the ACE ``model usage  :`` two-space bug happened.
    """
    print(f"engine counters: rollouts={result.rollouts}  "
          f"stale_considered={result.stale_considered}  "
          f"stale_discarded={result.stale_discarded}  "
          f"redispatched={result.redispatched}", flush=True)
    # The stage profile, on the same one-line format and for the same reason the
    # line above exists: without it "the async arm got faster" cannot be told
    # apart from "the gate stopped running". `merge_gate_seconds` is a subset of
    # `merge_seconds`, and `worker_starved_seconds` is summed across workers --
    # see `EvolutionResult.gate_share()` / `.merger_occupancy()`.
    print(f"stage profile: wallclock={result.wallclock:.1f}  "
          f"merge_seconds={result.merge_seconds:.1f}  "
          f"merge_gate_seconds={result.merge_gate_seconds:.1f}  "
          f"worker_starved_seconds={result.worker_starved_seconds:.1f}  "
          f"eval_seconds={result.eval_seconds:.1f}", flush=True)


def eval_cache_kwargs(args: argparse.Namespace) -> dict:
    """``policies=Policies(eval_cache=...)`` for ``--eval-cache``, or nothing.

    The engine has shipped :class:`~agentdescent.evalcache.FileCache` -- "two
    processes on one machine stop paying twice for the same gate" -- and every
    port used the in-process default, so every gate was re-paid on every run.

    ADAS is the clearest case: its seed archive is seven *hand-designed* programs
    scored over the whole validation split before round 0, and each program is
    itself several model calls, so a run begins by spending several hundred calls
    on an evaluation whose inputs never change. A three-arm, three-seed sweep pays
    that nine times, and a calibration session pays it once per attempt.

    Off by default, and that is not timidity: a persistent cache makes a rerun
    return the first run's numbers. That is exactly what you want while sizing a
    configuration and exactly what you do not want when the question is run-to-run
    variance -- which, on these workloads, is often the question. Turn it on for
    a sweep whose cells share a dataset and a seed; leave it off when measuring
    spread.
    """
    directory = getattr(args, "eval_cache", "")
    if not directory:
        return {}
    from agentdescent import Policies
    from agentdescent.evalcache import FileCache
    return {"policies": Policies(eval_cache=FileCache(directory))}


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

    ``--no-thinking`` becomes ``thinking={"type": "disabled"}`` on the Anthropic
    path, where the endpoint's reasoning preamble dominates latency. Measured on
    GLM-5.2 behind an Anthropic-shaped endpoint: a GEPA reflective-proposal call
    went 23.5s -> 5.4s and 783 output tokens -> 37, and both replies were usable
    instructions -- the shorter one simply terser. **That is a change to the
    model's output, not only its speed**, so a run that uses it has to say so:
    it belongs in the reported configuration next to the model id, and it must
    be identical across every arm of a comparison or the arms are not comparable.

    Silently ignored on the OpenAI-compatible path rather than translated. The
    equivalent knob there is per-vendor (``enable_thinking``,
    ``chat_template_kwargs``, a ``/no_think`` prefix), and guessing wrong would
    leave reasoning *on* while the run reports it off -- which is the failure
    that matters, since the wall-clock would then be attributed to the scheduler.
    """
    if is_openai_compatible(args):
        return openai_compatible(model=args.model, usage=usage, **kwargs)
    if getattr(args, "no_thinking", False):
        kwargs.setdefault("thinking", {"type": "disabled"})
    # --timeout reaches this path too: claude()'s 120s default assumes Claude
    # latencies, and a thinking model behind an Anthropic-shaped endpoint
    # (GLM-5.2 on GSM-Hard) legitimately exceeds it -- a baseline eval died at
    # 120s x 3 retries before any evolution happened.
    if getattr(args, "timeout", None):
        kwargs.setdefault("timeout", float(args.timeout))
    return claude(model=args.model, usage=usage, **kwargs)
