"""The deterministic workloads behind the merge golden trace.

Shared by the generator (`tools/gen_merge_golden.py`) and the test that compares
against it, so the two cannot drift into measuring different things.

The workloads are chosen to *reach the merge decisions*, not merely to finish:

* `append` gives several workers different, complementary proposals in the same
  round, which is what the fusion tournament is for;
* `slot` gives them the same key with different values, which is a
  contradiction and the only way conflict resolution runs at all.

A trace that never conflicts and never fuses would pass any refactor of the code
that does those things.
"""

from typing import Any, Dict, List

from agentdescent import AppendRules, SingleSlot, Task, evolve
from agentdescent.aggregator import Aggregator

__all__ = ["WORKLOADS", "recording_factory", "run_workload", "trace_of"]


def recording_factory(sink: List[Dict[str, Any]]):
    """An `aggregator_factory` that records every merge decision, unchanged.

    `RoundInfo` carries outcome *categories* and nothing about conflicts or
    fusion, so a trace built from the result alone cannot see the two decisions
    this refactor extracts first. Wrapping `step()` observes them without
    altering a thing -- the aggregator is the shipped one, and the reports are
    the ones the engine goes on to act on."""

    def factory(ledger, verifier, audit, config, policy):
        agg = Aggregator(ledger, verifier, audit, config, staleness_policy=policy)
        real_step = agg.step

        def step():
            reports = real_step()
            for r in reports:
                sink.append({
                    "category": r.category,
                    "fused": bool(r.fused),
                    "considered": r.considered,
                    "survived_staleness": r.survived_staleness,
                    "discarded_stale": r.discarded_stale,
                    "conflicts_dropped": r.conflicts_dropped,
                    "committed_version": r.committed_version,
                    "prob_improve": round(float(r.prob_improve), 6),
                })
            return reports

        agg.step = step          # type: ignore[assignment]
        return agg

    return factory


def _tasks(n: int, n_cats: int = 4) -> List[Task]:
    """Tasks in categories, so a rule learned on one generalises to its siblings.

    Without that, a proposal made on a training task cannot move the held-out
    score, nothing ever commits, and the trace records a run that never reached
    the decisions it is meant to guard."""
    out = []
    for i in range(n):
        cat = f"c{i % n_cats}"
        out.append(Task(id=f"t{i}", prompt=f"q{i}",
                        meta={"gold": f"a{cat}", "cat": cat}))
    return out


def _reward(task: Task, output: str) -> float:
    return 1.0 if output == task.meta["gold"] else 0.0


def _run_append(rendered: str, task: Task) -> str:
    """Solved once a rule for this task's category is present."""
    return task.meta["gold"] if f"rule:{task.meta['cat']}" in rendered else "wrong"


def _propose_append(rendered: str, task: Task, output: str, reward: float) -> str:
    return f"rule:{task.meta['cat']}"


def _run_slot(rendered: str, task: Task) -> str:
    """One slot holds one category's rule: only that category is solved."""
    return task.meta["gold"] if task.meta["cat"] in rendered else "wrong"


def _propose_slot(rendered: str, task: Task, output: str, reward: float) -> str:
    # every worker writes the same key with a different value -> contradiction
    return f"answer for {task.meta['cat']}"


WORKLOADS: Dict[str, Dict[str, Any]] = {
    "append": {
        # More categories than a single round of workers can cover, so the run
        # takes several merges to converge instead of one -- a trace with one
        # decision in it guards almost nothing.
        "n_tasks": 24,
        "n_cats": 8,
        "strategy": "append",
        "kwargs": dict(rounds=14, n_workers=3, max_concurrency=1,
                       held_out_frac=0.5, eval_concurrency=4, seed=0),
    },
    "slot": {
        "n_tasks": 12,
        "n_cats": 4,
        "strategy": "slot",
        "kwargs": dict(rounds=10, n_workers=3, max_concurrency=1,
                       held_out_frac=0.5, eval_concurrency=2, seed=7),
    },
}


def run_workload(name: str):
    """Run one workload; returns ``(result, merge_reports)``."""
    spec = WORKLOADS[name]
    if spec["strategy"] == "append":
        strategy, run, propose = AppendRules(), _run_append, _propose_append
    else:
        strategy, run, propose = SingleSlot(), _run_slot, _propose_slot
    merges: List[Dict[str, Any]] = []
    tasks = _tasks(spec["n_tasks"], spec.get("n_cats", 4))
    result = evolve(tasks, _reward, run=run, propose=propose,
                    strategy=strategy, aggregator_factory=recording_factory(merges),
                    **spec["kwargs"])
    return result, merges


def trace_of(result, merges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The decision sequence, and nothing that varies with the machine.

    Deliberately excludes `elapsed_s`, `rollouts` and anything path-shaped: a
    fixture that embeds a `$TMPDIR` path or a duration goes red in CI for
    reasons unrelated to what it is guarding, and the response to that is
    always to weaken the fixture."""
    rounds = [
        {
            "round": h.round,
            "reward": round(h.held_out_reward, 6),
            "size": h.n_items,
            "committed": h.committed,
            "rejected": h.rejected,
            "reasons": dict(sorted(h.reasons.items())),
        }
        for h in result.history
    ]
    return {"rounds": rounds, "merges": merges,
            "final_reward": round(result.final_reward, 6),
            "stop_reason": result.stop_reason,
            "outcomes": dict(sorted(result.outcomes().items()))}
