"""The parallelisation matrix: every port, serial against N workers, equal budget.

One row per algorithm, two arms per row, `--budget-rollouts` pinned to the same
value on both. That pin is the whole point. Six of the seven ports pass a fixed
iteration count and let `n_workers` multiply it, so an unbudgeted `N=8` arm runs
eight times the rollouts of the `--serial` arm -- measured on the engine at
`rounds=24`: 192 rollouts against 24. A wall-clock read across that gap is eight
times the model spend reported as parallel efficiency, and the quality column
beside it credits the extra spend to parallelism.

**Results are written after every cell**, because a full sweep is many hours of
real model time and a sweep that only reports at the end reports nothing when it
is interrupted. Re-running skips cells already in the output file, so a stopped
sweep resumes rather than restarts.

    python -m bench.matrix_run --budget 8 --width 8 --seeds 0,1,2 \
        --provider claude --model GLM-5.2 --json bench/results/matrix.json --yes

What it does *not* do is decide whether a number is publishable. `--seeds` below
three cannot support a spread, the wall-clock is this machine's and this
endpoint's, and a speedup measured against an endpoint that serialises concurrent
requests is a measurement of that endpoint. Read `notes` on every row.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: One entry per matrix row: the module, the flags that size it, and the flag its
#: worker count hides behind. The names differ because each port keeps its
#: upstream vocabulary, which is part of being a faithful port -- so this table is
#: the only place that translation lives.
ROWS = [
    dict(name="ace", module="examples.ace.ace_context_evolution",
         dataset="FiNER-139", width_flag="--workers",
         size=["--pool", "60", "--top-k", "24"]),
    dict(name="gepa", module="examples.gepa.gepa_prompt_evolution",
         dataset="HotpotQA", width_flag="--workers",
         # `--rounds 9999`: the ports' own iteration defaults sit BELOW the
         # budget and bind first -- measured: GEPA's `rounds=10` stopped the
         # serial arm at 10 rollouts while the 8-wide arm (2 rounds) spent all
         # 16, an unequal-budget comparison wearing an equal-budget flag. The
         # budget is the ceiling; the iteration knob must not be the floor.
         size=["--fetch", "80", "--val-cap", "8", "--reflective-merge",
               "--seed-instruction", "", "--rounds", "9999"]),
    dict(name="evoskill", module="examples.evoskill.evoskill_skill_discovery",
         dataset="OfficeQA/FinQA", width_flag="--workers", size=[]),
    dict(name="skillopt", module="examples.skillopt.skillopt_skill_training",
         dataset="SearchQA", width_flag="--minibatch",
         size=["--train", "16", "--val", "8"]),
    dict(name="adas", module="examples.adas.adas_meta_agent_search",
         dataset="MGSM", width_flag="--workers",
         size=["--langs", "en", "--per-lang", "8"]),
    # DGM's *objective* is a surrogate -- real SWE-bench needs the Docker harness,
    # which is a documented departure of the port. Its *proposals* are a real
    # model like every other row: `--model` builds a completion and only its
    # absence falls back to the deterministic self-improver. Withholding the model
    # here would have made one row of the matrix a different experiment.
    dict(name="dgm", module="examples.dgm.dgm_self_improve",
         dataset="SWE-bench (surrogate objective)",
         width_flag="--selfimprove-size", size=[]),
    dict(name="openevolve", module="examples.openevolve.openevolve_program_evolution",
         dataset="function minimization", width_flag="--workers",
         size=["--task-count", "8"], needs="bwrap"),
]

#: Pulled out of each port's own stdout rather than re-derived, so a row reports
#: what the port reported. `rollouts` is the measured count, not the budget: the
#: synchronous path checks at the round barrier and a wide arm can overshoot.
PATTERNS = {
    # ACE prints `model usage  :` with two spaces; the tighter pattern
    # silently dropped the call count on that row while the token counts
    # beside it parsed, which reads as "this port makes no calls".
    "calls": r"model usage\s*:\s*([\d,]+) calls",
    "prompt_tokens": r"([\d,]+) prompt",
    "completion_tokens": r"([\d,]+) completion",
    "model_seconds": r"([\d.]+)s in the model",
    "test": r"test (?:EM|score|hard-EM|accuracy|resolve-rate)\s*:\s*([\d.]+)",
    "val_end": r"->\s*([\d.]+)",
    "stopped": r"stopped\s*:\s*(\w+)",
    # Model seconds inside calls that failed -- retry waits, hung connections.
    # `wall_seconds - failure_seconds` is the wall-clock net of endpoint weather,
    # which is what a speedup between two arms should be computed from when one
    # arm was unlucky. Only printed by ports once Usage carries it.
    "failure_seconds": r"failed \(([\d.]+)s lost\)",
}


def _parse(text: str) -> Dict:
    """Last match per pattern, numeric where the pattern captures a number.

    It converted every capture unconditionally, so `stopped: max_rollouts` --
    a word, by design, since `stop_reason` is the engine's one signal that a paid
    run ended on a budget rather than converging -- crashed the whole sweep on
    `int("max_rollouts")` *after* the first cell had finished paying for itself.
    A results harness that discards a completed cell is worse than one that never
    ran it.
    """
    out: Dict = {}
    for key, pattern in PATTERNS.items():
        found = re.findall(pattern, text)
        if not found:
            continue
        raw = found[-1].replace(",", "")
        try:
            out[key] = float(raw) if "." in raw else int(raw)
        except ValueError:
            out[key] = raw
    return out


def _cell(row: dict, arm: str, seed: int, args) -> Dict:
    """Run one cell and return what it cost and what it reached."""
    cmd = [sys.executable, "-u", "-m", row["module"], "--yes",
           "--seed", str(seed), "--budget-rollouts", str(args.budget)]
    cmd += row["size"] + ["--eval-concurrency", str(args.eval_concurrency)]
    if arm == "serial":
        cmd += ["--serial"]
    else:
        cmd += [row["width_flag"], str(args.width)]
    if not row.get("offline"):
        cmd += ["--provider", args.provider, "--model", args.model]

    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                              timeout=args.timeout)
        text = proc.stdout + proc.stderr
        failed = proc.returncode != 0
    except subprocess.TimeoutExpired as exc:
        text = (exc.stdout or "") + (exc.stderr or "")
        if isinstance(text, bytes):
            text = text.decode("utf-8", "replace")
        failed = True
    wall = time.time() - started

    if args.logs:
        # The parsed row answers "what did it cost"; only the transcript answers
        # "what did it do". Kept per cell, because a sweep this long is not
        # re-runnable just to look at one.
        os.makedirs(args.logs, exist_ok=True)
        log = os.path.join(args.logs, f"{row['name']}-{arm}-seed{seed}.log")
        with open(log, "w", encoding="utf-8") as handle:
            handle.write(" ".join(cmd) + "\n\n" + text)

    cell = dict(row=row["name"], dataset=row["dataset"], arm=arm, seed=seed,
                width=1 if arm == "serial" else args.width,
                budget=args.budget, wall_seconds=round(wall, 1), **_parse(text))
    if failed:
        # Keep the tail rather than a boolean: a row that failed is a finding, and
        # "it did not run" without the reason is the least useful cell of all.
        cell["error"] = text.strip().splitlines()[-1][:300] if text.strip() else "no output"
    return cell


def _load(path: Optional[str]) -> List[Dict]:
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            return json.load(handle).get("cells", [])
    return []


def _save(path: Optional[str], cells: List[Dict], args) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"budget_rollouts": args.budget, "width": args.width,
                   "model": args.model, "provider": args.provider,
                   "cells": cells}, handle, indent=2)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--budget", type=int, default=8,
                   help="rollouts, pinned on both arms of every row")
    p.add_argument("--width", type=int, default=8, help="workers in the parallel arm")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--rows", default="", help="comma-separated subset of row names")
    p.add_argument("--provider", default="claude")
    p.add_argument("--model", default="GLM-5.2")
    p.add_argument("--eval-concurrency", type=int, default=16,
                   help="held-out evaluations in flight; wall-clock only")
    p.add_argument("--timeout", type=float, default=7200.0, help="seconds per cell")
    p.add_argument("--json", default="bench/results/matrix.json")
    p.add_argument("--logs", default="bench/results/matrix-logs",
                   help="per-cell transcripts; the parsed row is not the run")
    p.add_argument("--yes", action="store_true")
    args = p.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    wanted = {r.strip() for r in args.rows.split(",") if r.strip()}
    rows = [r for r in ROWS if not wanted or r["name"] in wanted]

    cells = _load(args.json)
    done = {(c["row"], c["arm"], c["seed"]) for c in cells}
    arms = ("serial", "parallel")
    todo = [(r, a, s) for r in rows for s in seeds for a in arms
            if (r["name"], a, s) not in done]

    print(f"matrix: {len(rows)} rows x {len(arms)} arms x {len(seeds)} seeds "
          f"= {len(rows) * len(arms) * len(seeds)} cells, {len(done)} already done, "
          f"{len(todo)} to run")
    print(f"budget: {args.budget} rollouts on BOTH arms; parallel arm width "
          f"{args.width}\n")
    if not args.yes:
        print("pass --yes to run (this spends real model time)")
        return

    for index, (row, arm, seed) in enumerate(todo, 1):
        if row.get("needs") and not any(
                os.access(os.path.join(d, row["needs"]), os.X_OK)
                for d in os.environ.get("PATH", "").split(os.pathsep)):
            cell = dict(row=row["name"], dataset=row["dataset"], arm=arm, seed=seed,
                        error=f"requires {row['needs']}, not on this host")
            print(f"[{index}/{len(todo)}] {row['name']:11} {arm:8} seed={seed}  "
                  f"SKIPPED ({cell['error']})")
        else:
            print(f"[{index}/{len(todo)}] {row['name']:11} {arm:8} seed={seed}  "
                  f"started {time.strftime('%H:%M:%S')}", flush=True)
            cell = _cell(row, arm, seed, args)
            note = (f"FAILED {cell['error'][:60]}" if "error" in cell
                    else f"{cell.get('wall_seconds', 0):.0f}s  "
                         f"{cell.get('calls', '?')} calls  "
                         f"test={cell.get('test', '?')}")
            print(f"    -> {note}", flush=True)
        cells.append(cell)
        _save(args.json, cells, args)      # after every cell, not at the end

    print(f"\nwrote {args.json} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
