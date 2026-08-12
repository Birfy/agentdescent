"""Comparing parallel configurations, under conditions that make it mean something.

The question is whether parallelism buys quality per unit of time and money. The
easy way to answer it produces a table that looks like evidence and is not, so
the constraints below are enforced here rather than described in a footnote.

**Budget in model calls, not rounds.** Configurations differ in how many calls a
round makes. Fixing rounds hands the wider configuration more model, and then
reports the extra model as a win for parallelism.

**Quality on a set the gate never saw.** `evolve()` optimises against its
held-out split and stops when that split says so. Reporting the same number is
reporting a training score.

**Several seeds, and an interval.** This repository has already published a
point estimate that moved 4.8 points between two runs of one configuration
(`docs/algo-ace.md` says so in as many words). One number per configuration is
not a result.

**One environment, recorded.** Two configurations measured under different
toolchains differ by more than the variable being studied. A row can carry the
fingerprint it ran under, and a run whose acceptance saw mixed environments is
marked rather than averaged in. `bench.run` leaves both empty on purpose: it runs
a single environment, and a fingerprint invented for a comparison that has only
one of them would be decoration.

None of this makes a conclusion come out positive. `docs/results.md` carries a
row reading "lift not yet measured" for exactly that reason, and a harness whose
output can only support one answer is not measuring anything.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from agentdescent import EvolutionResult

__all__ = ["Config", "Measurement", "Summary", "run_config", "summarise", "to_markdown"]


@dataclass(frozen=True)
class Config:
    """One point in the configuration space, and nothing about the workload.

    Deliberately narrow: the whole design is that only these vary. A config that
    could also change the model or the data would let a comparison move two
    things at once, which is the failure this file exists to prevent.
    """

    name: str
    n_workers: int = 1
    asynchronous: bool = False
    eval_concurrency: int = 1
    #: Where rollouts run. `None` is in-process threads.
    executor: Optional[str] = None
    #: Which sandbox provider. `None` is a local workspace.
    sandbox: Optional[str] = None
    #: The barrier-free path's lag budget, in **versions**. How much wall-clock a
    #: version represents depends entirely on how long a rollout takes, so a
    #: value tuned for one workload is not a value at all for another -- which is
    #: why it is a dimension of the matrix rather than a constant in it.
    async_ratio: int = 3
    #: Off across this matrix, unlike the engine default. Every async row here
    #: exists to measure what the lag budget alone does, and a rollout in this
    #: workload is a dictionary lookup -- with commit-resync on, a worker is
    #: never more than one lookup behind head, eta never leaves 0 and the
    #: `async_ratio` column stops varying anything.
    resync_on_commit: bool = False
    staleness: str = "guarded"

    def as_row(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Measurement:
    """What one seed of one configuration produced."""

    config: str
    seed: int
    #: Wall-clock and rollouts to reach the quality bar. `None` means it never
    #: did -- which is a result, not a missing value, and must not be filled in
    #: with the total.
    time_to_quality: Optional[float]
    cost_to_quality: Optional[int]
    wallclock: float
    rollouts: int
    calls: int
    stale_rate: float
    duplicate_rate: float
    sandbox_wait_s: float
    sandbox_setup_s: float
    sandboxes_created: int
    #: Quality on the split the gate never saw.
    test_quality: Optional[float]
    #: Non-zero means acceptance compared measurements from different
    #: environments, and this row's quality number needs a caveat.
    env_mismatch: int = 0
    fingerprint: str = ""

    @classmethod
    def of(cls, config: str, seed: int, result: EvolutionResult, target: float,
           test_quality: Optional[float] = None, fingerprint: str = "") -> "Measurement":
        return cls(
            config=config, seed=seed,
            time_to_quality=result.time_to_quality(target),
            cost_to_quality=result.cost_to_quality(target),
            wallclock=result.wallclock, rollouts=result.rollouts,
            calls=result.usage.calls,
            stale_rate=result.stale_rate(), duplicate_rate=result.duplicate_rate(),
            sandbox_wait_s=result.sandbox_wait_s,
            sandbox_setup_s=result.sandbox_setup_s,
            sandboxes_created=result.sandboxes_created,
            test_quality=test_quality, fingerprint=fingerprint)


@dataclass
class Summary:
    """Several seeds of one configuration, as an interval."""

    config: str
    seeds: int
    time_to_quality: Optional[Sequence[float]] = None
    cost_to_quality: Optional[Sequence[float]] = None
    test_quality: Optional[Sequence[float]] = None
    #: How many seeds reached the bar at all. A median over the ones that did,
    #: without this, silently reports the easy seeds.
    reached: int = 0
    stale_rate: float = 0.0
    duplicate_rate: float = 0.0
    sandbox_share: float = 0.0
    heterogeneous: bool = False

    def as_row(self) -> Dict[str, Any]:
        return asdict(self)


def _interval(values: Sequence[float]) -> Optional[List[float]]:
    """(min, median, max). Not a confidence interval, and not labelled as one.

    Three seeds cannot support a real one. Reporting the spread that was actually
    observed is honest; reporting a t-interval over three points is arithmetic
    dressed as statistics."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    return [min(values), statistics.median(values), max(values)]


def run_config(config: Config, workload: Callable[[Config, int], EvolutionResult],
               *, seeds: Sequence[int], target: float,
               test_eval: Optional[Callable[[EvolutionResult], float]] = None,
               fingerprint: str = "") -> List[Measurement]:
    """Run one configuration across seeds and collect what each produced."""
    out: List[Measurement] = []
    for seed in seeds:
        result = workload(config, seed)
        quality = test_eval(result) if test_eval is not None else None
        out.append(Measurement.of(config.name, seed, result, target, quality,
                                  fingerprint))
    return out


def summarise(measurements: Sequence[Measurement]) -> List[Summary]:
    """Group by configuration and report intervals rather than point estimates."""
    by_config: Dict[str, List[Measurement]] = {}
    for m in measurements:
        by_config.setdefault(m.config, []).append(m)

    summaries = []
    for name, group in by_config.items():
        reached = [m for m in group if m.time_to_quality is not None]
        total_wall = sum(m.wallclock for m in group) or 1.0
        sandbox = sum(m.sandbox_wait_s + m.sandbox_setup_s for m in group)
        summaries.append(Summary(
            config=name, seeds=len(group), reached=len(reached),
            time_to_quality=_interval([m.time_to_quality for m in reached]),
            cost_to_quality=_interval([m.cost_to_quality for m in reached]),
            test_quality=_interval([m.test_quality for m in group
                                    if m.test_quality is not None]),
            stale_rate=statistics.median([m.stale_rate for m in group]),
            duplicate_rate=statistics.median([m.duplicate_rate for m in group]),
            sandbox_share=sandbox / total_wall,
            heterogeneous=any(m.env_mismatch for m in group)))
    return summaries


def to_markdown(summaries: Sequence[Summary]) -> str:
    """A table that shows what was not reached, rather than omitting it."""
    lines = ["| config | seeds | reached | time-to-quality (min/med/max) | "
             "cost-to-quality | test quality | stale% | dup% | sandbox% |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        def fmt(interval, spec="{:.2f}"):
            if not interval:
                return "—"
            return " / ".join(spec.format(v) for v in interval)

        note = " ⚠︎" if s.heterogeneous else ""
        lines.append(
            f"| {s.config}{note} | {s.seeds} | {s.reached}/{s.seeds} | "
            f"{fmt(s.time_to_quality)} | {fmt(s.cost_to_quality, '{:.0f}')} | "
            f"{fmt(s.test_quality, '{:.3f}')} | {s.stale_rate:.0%} | "
            f"{s.duplicate_rate:.0%} | {s.sandbox_share:.0%} |")
    if any(s.heterogeneous for s in summaries):
        lines.append("")
        lines.append("⚠︎ acceptance compared measurements from different "
                     "environments; that row's quality is not comparable.")
    if any(s.reached < s.seeds for s in summaries):
        lines.append("")
        lines.append("A configuration that did not reach the bar on every seed "
                     "has no time-to-quality for those seeds. The interval covers "
                     "the ones that did, and `reached` says how many that was.")
    return "\n".join(lines)
