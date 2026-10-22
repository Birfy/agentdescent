#!/usr/bin/env python3
"""End-to-end self-evolution: run the whole chain and measure every rung.

The framework's claim is recursive self-improvement, and it is spread across
several rungs: an archive of adopted rules, a next session seeded from the best,
a population of lineages, an exam that re-judges adoptions with truth, a
portfolio that reallocates spend, harnesses that compete for a domain, and a
chain audit that traces one adoption's evidence. Each rung has its own module
and its own tests; **nothing here proves they compose**. This bench runs the
whole chain on one deterministic arithmetic domain and reports each rung's
measured number — the evidence that the chain works end to end, offline, with
no model key:

    python -m bench.self_evolution            # human-readable per rung
    python -m bench.self_evolution --json     # machine-readable report

Stages (each must run and produce its measurement; any failure exits non-zero):

1. **Seed** — two ``continuous_meta_run`` sessions; the second seeds from the
   first. Reports the verdicts and the archive count.
2. **Population** — ``meta_population_run`` diversifies from the archive's rules
   and promotes the best. Reports the islands and what was promoted.
3. **Exam** — ``exam_archive`` re-judges adoptions with an independent oracle.
   Reports passed / flagged / demoted.
4. **Portfolio** — ``layers_from_archive`` + ``portfolio_report`` reallocate the
   next window. Reports the layers and their new budgets.
5. **Harness** — two algorithm ports compete on the domain; the archive is the
   leaderboard and ``select_harness`` routes on the measured winner.
6. **Chain audit** — ``chain_audit`` traces the newest adoption's evidence
   rings. Reports each ring's status.

The domain is a tiny capability-capped arithmetic task set held out ``cap:N``
rules solve; every run is offline and deterministic, so the numbers are stable
across machines (the temporary directories differ, the measurements do not).
Small held-out warnings are silenced -- the bench *wants* a tiny, fast domain.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from agentdescent import (
    ConsolidationStore,
    MetaOutcome,
    ParamSlot,
    SingleSlot,
    Task,
    chain_audit,
    compose_run,
    continuous_meta_run,
    evolve,
    exam_archive,
    final_reward,
    harness,
    harness_leaderboard,
    layers_from_archive,
    meta_population_run,
    portfolio_report,
    select_harness,
)
from agentdescent.sampling import DifficultyWeighted

__all__ = ["run_bench", "main", "Stage", "BenchReport"]


# -- the deterministic capability domain ----------------------------------------


def _cap(rendered: str) -> int:
    m = re.search(r"cap:(\d+)", rendered)
    return int(m.group(1)) if m else 0


def _c_of(rendered: str) -> float:
    m = re.search(r"^c: ([0-9.]+)", rendered, re.M)
    return float(m.group(1)) if m else 0.2


def run(rendered: str, task: Task) -> str:
    return str(task.meta["gold"]) if task.meta.get("a", 0) <= _cap(rendered) else "?"


def reward(task: Task, output: str) -> float:
    return 1.0 if str(output).strip() == str(task.meta["gold"]).strip() else 0.0


def propose(rendered: str, task: Task, output: str, reward: float) -> str:
    if reward >= 0.999:
        return "cap:0"
    a = task.meta.get("a", 0)
    if a > _cap(rendered):
        return f"cap:{a}"
    return None


def tasks(prefix: str, n: int = 8) -> List[Task]:
    order = [1, 2, 3, 4, 8, 5, 6, 7][:n]
    return [Task(id=f"{prefix}{i}", prompt=f"item {i}",
                 meta={"gold": str(2 * a), "a": a}) for i, a in enumerate(order)]


def slot() -> ParamSlot:
    return ParamSlot(factory=lambda c: DifficultyWeighted(c=c),
                     params={"c": 0.2}, bounds={"c": (0.05, 1.0)})


def nudge_up(rendered: str, task: Task, output: str, reward: float) -> str:
    c = _c_of(rendered)
    if reward < 0.7:
        return f"c: {min(1.0, c + 0.1):.2f}"
    return f"c: {c:.2f}"


def helpful(value: Any, seed: int) -> MetaOutcome:
    if getattr(value, "c", 0.2) >= 0.3:
        return MetaOutcome(curve=[0.5, 1.0], final=1.0)
    return MetaOutcome(curve=[0.1, 0.2], final=0.2)


def helpful_val(value: Any, seed: int) -> MetaOutcome:
    return helpful(value, seed)


CONFIG = {"dataset": "bench-arithmetic", "scorer": "exact"}
INNER = {"rounds": 5, "n_workers": 1, "target_reward": 1.0}

#: The inner problem pairs every session runs on (train / val, disjoint).
TRAIN = {"ta": helpful, "tb": helpful}
VAL = {"va": helpful_val}


# -- the report -----------------------------------------------------------------


@dataclass
class Stage:
    """One rung's result: did it run, its measurement, and a one-line summary."""

    name: str
    ok: bool
    detail: Dict[str, Any] = field(default_factory=dict)
    note: str = ""


@dataclass
class BenchReport:
    """The whole chain: every stage, and whether every rung produced its number."""

    stages: List[Stage] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return bool(self.stages) and all(stage.ok for stage in self.stages)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "all_ok": self.all_ok,
            "stages": {
                stage.name: {"ok": stage.ok, "detail": stage.detail,
                             "note": stage.note}
                for stage in self.stages
            },
        }


def _record(report: BenchReport, name: str,
            fn: Callable[[], Tuple[bool, Dict[str, Any], str]]) -> None:
    """Run one stage, recording a failure (never raising out of the bench)."""
    try:
        ok, detail, note = fn()
    except Exception as error:  # noqa: BLE001 - a broken rung is the measurement
        ok, detail, note = False, {}, f"{type(error).__name__}: {error}"
    report.stages.append(Stage(name=name, ok=ok, detail=detail, note=note))


# -- the stages -----------------------------------------------------------------


def _session_kwargs(rounds: int) -> Dict[str, Any]:
    """The domain wiring every seed/population session shares."""
    return dict(
        slot="task_sampler", spec=slot(), propose=nudge_up,
        tasks_for=lambda rec: tasks(rec.run_id), run=run, reward=reward,
        inner_propose=propose, inner_strategy=SingleSlot(initial_value="cap:0"),
        config=CONFIG, inner=dict(INNER), seeds=(0, 1, 2, 3),
        val_seeds=(10, 11), rounds=rounds, n_workers=1,
        problems=TRAIN, val_problems=VAL)


def _stage_seed(store: ConsolidationStore, rounds: int):
    verdicts: List[Dict[str, Any]] = []
    for run_id in ("run-1", "run-2"):
        report, _ = continuous_meta_run(
            store=store, run_id=run_id,
            repo_path=tempfile.mkdtemp(prefix="bench-seed-"),
            **_session_kwargs(rounds))
        verdicts.append({"run_id": run_id, "verdict": report.verdict,
                         "c": round(_c_of(report.value), 3)})
    count = store.count()
    ok = count >= 1 and any(v["verdict"] == "adopt" for v in verdicts)
    note = f"{count} rule(s) after {len(verdicts)} session(s)"
    return ok, {"verdicts": verdicts, "archive_count": count}, note


def _stage_population(store: ConsolidationStore, rounds: int, islands: int):
    report, _ = meta_population_run(
        store=store, islands=islands, train_runs=2, val_runs=1,
        repo_dir=tempfile.mkdtemp(prefix="bench-pop-repos-"),
        **_session_kwargs(rounds))
    ok = len(report.islands) >= 1
    note = (f"{len(report.islands)} island(s), promoted "
            f"{report.promoted_from} (archive now {report.archive_count})")
    return ok, {"islands": len(report.islands),
                "promoted_from": report.promoted_from,
                "archive_count": report.archive_count}, note


def _stage_exam(store: ConsolidationStore):
    exam = exam_archive(
        store, slot="task_sampler", spec=slot(), oracle=final_reward,
        tasks_for=lambda rec: tasks(rec.run_id), run=run, reward=reward,
        propose=propose, strategy=SingleSlot(initial_value="cap:0"),
        config=CONFIG, min_gain=0.0, max_flags=2)
    examined = exam.passed + exam.flagged + exam.demoted
    ok = examined >= 1
    return ok, {"passed": exam.passed, "flagged": exam.flagged,
                "demoted": exam.demoted, "examined": examined}, exam.reason


def _stage_portfolio(store: ConsolidationStore):
    layers = layers_from_archive(store)
    report = portfolio_report(layers, min_samples=1)
    ok = len(layers) >= 1
    note = (f"{len(layers)} layer(s); "
            f"reallocated {report.reallocated}")
    return ok, {
        "layers": [layer.name for layer in layers],
        "reallocated": report.reallocated,
        "total_budget": report.total_budget,
    }, note


def _cap_harness(name: str, rounds: int):
    def runner(task_set: Sequence[Task], repo: str):
        return evolve(list(task_set), reward, run=run, propose=propose,
                      strategy=SingleSlot(initial_value="cap:0"),
                      rounds=rounds, n_workers=1, repo_path=repo,
                      target_reward=1.0)
    return harness(name, runner, description=f"cap rule, {rounds} rounds")


def _stage_harness(store: ConsolidationStore):
    ports = [_cap_harness("quick", 2), _cap_harness("deep", 6)]
    compose_run(store, tasks("h", 8), ports, CONFIG, mode="all",
                repo_dir=tempfile.mkdtemp(prefix="bench-harness-"))
    board = harness_leaderboard(store, CONFIG)
    chosen, reason = select_harness(store, ports, CONFIG)
    ok = set(board) == {"quick", "deep"}
    best = {name: info.get("best") for name, info in board.items()}
    return ok, {"best": best, "selected": chosen.name, "reason": reason}, reason


def _stage_chain(store: ConsolidationStore):
    rules = [rec for rec in store.iter_archive()
             if rec.artifact_id == "meta-rule"]
    newest = max(rules, key=lambda rec: rec.created_at)
    chain = chain_audit(store, newest.run_id)
    rings = {link.layer: link.status for link in chain.links}
    # a NOT VERIFIED audit is a measurement, not a broken rung; the stage is
    # "ok" when the audit ran and returned rings for the adoption
    ok = bool(rings)
    return ok, {"run_id": newest.run_id, "verified": chain.verified,
                "rings": rings}, chain.reason


# -- the runner -----------------------------------------------------------------


def run_bench(*, rounds: int = 2, islands: int = 2) -> BenchReport:
    """Run the whole chain on a fresh archive and return every rung's number."""
    store = ConsolidationStore(tempfile.mkdtemp(prefix="agentdescent-bench-"))
    report = BenchReport()
    _record(report, "seed", lambda: _stage_seed(store, rounds))
    _record(report, "population",
            lambda: _stage_population(store, rounds, islands))
    _record(report, "exam", lambda: _stage_exam(store))
    _record(report, "portfolio", lambda: _stage_portfolio(store))
    _record(report, "harness", lambda: _stage_harness(store))
    _record(report, "chain", lambda: _stage_chain(store))
    return report


def _print_human(report: BenchReport) -> None:
    for stage in report.stages:
        print(f"[{'ok  ' if stage.ok else 'FAIL'}] {stage.name}: {stage.note}")
        for key, value in stage.detail.items():
            print(f"        {key}: {value}")
    verdict = "PASS — every rung produced its number" if report.all_ok \
        else "FAIL — a rung did not run or produced no measurement"
    print(verdict)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=2,
                        help="outer rounds per session (default 2)")
    parser.add_argument("--islands", type=int, default=2,
                        help="population islands (default 2)")
    parser.add_argument("--json", action="store_true",
                        help="print only the machine-readable report")
    args = parser.parse_args(argv)

    # The domain is deliberately tiny (fast, offline); the held-out warnings are
    # expected and would drown the report.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = run_bench(rounds=args.rounds, islands=args.islands)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False,
                         default=str))
    else:
        _print_human(report)
    return 0 if report.all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
