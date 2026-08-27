"""Re-measure finished AlgoTune programs the way AlgoTune measures them.

This is where the port's reported numbers are made comparable with upstream's,
and it is deliberately a separate step from the search. The split between the
two is: **the search's own measurement only has to rank candidates, the reported
one has to match AlgoTune.** So the defaults here are upstream's own config
(``AlgoTuner/config/config.yaml``) rather than the search's:

===================  ================================  ======================
knob                 upstream                          here
===================  ================================  ======================
instances            ``dataset.test_size: 100``        ``--instances 100``
timed runs           ``benchmark.eval_runs: 10``       ``--repeats 10``
per-instance time    minimum of those runs             minimum
per-instance score   ``baseline_ms / solver_ms``       same
task score           arithmetic mean over instances    same
one invalid answer   whole task scores nothing         same
warm-up              the previous instance             the previous instance
===================  ================================  ======================

Running that inside the search is not affordable -- at ten runs over a hundred
instances a single gate decision is minutes -- so the run stays cheap and the
comparison is made here, on the programs it produced.

This re-runs no search. A finished run's result file already carries, per task,
the root program and the best program the tree found; this scores those two on a
freshly drawn held-back split of the requested size and reports the ratio. The
seeds are disjoint from every seed the search saw -- ``Suite`` reserves a block
per shard wide enough for the larger split -- so widening the measurement cannot
accidentally measure the sets the search was allowed to optimise against.

    python -m tools.algotune_rescore bench/results/era-algotune-run.json \\
        --instances 100 --output bench/results/era-algotune-run-aligned.json

What it does *not* fix: this host is not AlgoTune's host, the instances are
drawn from the same generator rather than being their published ones, and their
agent's budget is nothing like this one's. Aligning the sample size removes the
one difference that was purely an artefact of how the port is configured.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from examples.era._era_algotune import evaluate_source, prepare_suite


def geometric_mean(values: Iterable[float]) -> Optional[float]:
    usable = [float(v) for v in values
              if v is not None and math.isfinite(float(v)) and float(v) > 0]
    if not usable:
        return None
    return math.exp(sum(math.log(v) for v in usable) / len(usable))


def _tasks_of(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = payload.get("tasks") or []
    missing = [row["task"] for row in rows if not row.get("best_program")]
    if missing:
        raise SystemExit(
            f"{', '.join(missing)} carry no best_program; this needs a result "
            f"file from a run that completed those tasks")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path,
                        help="era_algotune result file(s) to re-measure")
    parser.add_argument("--instances", type=int, default=100,
                        help="held-back problems per task (AlgoTune reports on 100)")
    parser.add_argument("--test-shards", type=int, default=2,
                        help=("held-back sets to split --instances across. Each set "
                              "is one sandboxed process, so this bounds how long a "
                              "single process runs"))
    parser.add_argument("--repeats", type=int, default=10,
                        help=("timed runs per program per problem, after a warm-up. "
                              "10 is upstream's `benchmark.eval_runs`, the count it "
                              "scores with; the search itself runs fewer because it "
                              "pays this on every rollout and only needs to rank"))
    parser.add_argument("--candidate-timeout", type=float, default=600.0,
                        help=("wall-clock for one held-back set. Larger than the "
                              "search's, because a set here is fifty problems "
                              "rather than two"))
    parser.add_argument("--seed", type=int, default=None,
                        help="override the seed the result file recorded")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    per_shard = max(1, args.instances // max(1, args.test_shards))
    instances = per_shard * args.test_shards

    entries: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    started = time.monotonic()
    for path in args.results:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config") or {}
        for row in _tasks_of(payload):
            task = row["task"]
            suite = prepare_suite(
                task,
                seed=args.seed if args.seed is not None else int(config.get("seed", 0)),
                shards=int(config.get("shards", 6)),
                test_shards=args.test_shards,
                problems=int(config.get("problems", 2)),
                test_problems=per_shard,
                size_scale=float(config.get("size_scale", 1.0)),
            )
            shards = suite.test_range()
            scored: Dict[str, Any] = {"task": task, "source": str(path),
                                      "instances": instances}
            for label, code in (("root", suite.initial_program),
                                ("best", row["best_program"])):
                valid, metrics, error = evaluate_source(
                    code, suite=suite, shards=shards,
                    timeout=args.candidate_timeout, repeats=args.repeats)
                scored[label] = {
                    "speedup": metrics.get("speedup"),
                    "problems": metrics.get("problems"),
                    "valid_problems": metrics.get("valid_problems"),
                    "baseline_ms": metrics.get("baseline_ms"),
                    "candidate_ms": metrics.get("candidate_ms"),
                    "valid": valid,
                    "error": error,
                }
                if not valid:
                    failures.append({"task": task, "program": label, "error": error})
            scored["gain"] = (
                (scored["best"]["speedup"] - scored["root"]["speedup"])
                if scored["best"]["speedup"] is not None
                and scored["root"]["speedup"] is not None else None)
            entries.append(scored)
            print(f"{task:34s} root {scored['root']['speedup']}  "
                  f"best {scored['best']['speedup']}", flush=True)

    out = {
        "experiment": "AlgoTune re-measurement on an aligned held-back split",
        "instances_per_task": instances,
        "test_shards": args.test_shards,
        "repeats": args.repeats,
        "sources": [str(p) for p in args.results],
        "wall_seconds": time.monotonic() - started,
        "summary": {
            "tasks": len(entries),
            "geometric_mean_root": geometric_mean(
                e["root"]["speedup"] for e in entries),
            "geometric_mean_best": geometric_mean(
                e["best"]["speedup"] for e in entries),
            "improved": sum(1 for e in entries if (e["gain"] or 0.0) > 0.0),
        },
        "tasks": entries,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=str) + "\n",
                           encoding="utf-8")
    print(f"\n{len(entries)} task(s) on {instances} instances each: "
          f"geometric mean {out['summary']['geometric_mean_root']} -> "
          f"{out['summary']['geometric_mean_best']}, "
          f"wall={out['wall_seconds']:.0f}s, output={args.output}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
