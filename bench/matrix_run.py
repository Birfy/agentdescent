"""The parallelisation matrix: every port, serial / parallel / async, equal budget.

One row per algorithm, three arms per row, `--budget-rollouts` pinned to the
same value on all of them. That pin is the whole point. Six of the eight ports
pass a fixed
iteration count and let `n_workers` multiply it, so an unbudgeted `N=8` arm runs
eight times the rollouts of the `--serial` arm -- measured on the engine at
`rounds=24`: 192 rollouts against 24. A wall-clock read across that gap is eight
times the model spend reported as parallel efficiency, and the quality column
beside it credits the extra spend to parallelism.

The `async` arm is the same width barrier-free (`--async`), where a diff can be
proposed against a head the merger has since moved. Its `--max-seconds` is set
to the cell timeout so the wall-clock never binds before the rollout budget --
the async runtime has no unbounded mode, and its own defaults (tens of seconds)
would otherwise silently become the stop condition, reporting a wall-clock cap
as a converged run. Read `stale_discarded`/`stale_considered` on every async
cell: a quality drop with a high stale rate is the lag budget, not the
algorithm, and those need opposite fixes.

The serial control runs with `--eval-concurrency 1`. `--serial` only lowers
`n_workers`, so the control used to score its gate and its test split
concurrently -- measured on GEPA, 3849s of model time inside a 2148s wall-clock
on a *one-worker* run. A control that is already 1.8x parallel makes every
speedup beside it the leftover parallelism rather than the parallelism.

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
         # `--top-k 24` was below the port's own demonstrated floor: its docs
         # measure 1.000 at 10 (nothing to learn) and no admitted bullet at 40.
         # 120 is the first value where the Curator's gate passes anything, so a
         # row run below it reports "parallelism changed nothing" about a
         # configuration where *nothing* changes anything.
         #
         # `--pool 3200` is the *difficulty*, not just the split. FiNER is
         # stratified by concept, and a small pool never surfaces the rare ones:
         # at `--pool 1600` only ~120 distinct concepts appear, so `--top-k 139`
         # and `--top-k 120` select the identical 211 tasks and the baseline sits
         # at 0.875 -- high enough that `solved_threshold` answers most rollouts
         # before a reflection is ever requested. Measured: 32 rollouts there
         # produced `+0/-0` for most rounds and val never moved. At `--pool 3200`
         # the baseline is 0.667 and two rollouts already commit two bullets.
         # A row measured on a saturated configuration reports "parallelism
         # changed nothing" about a setup where nothing changes anything.
         #
         # `--val-cap 24` is the cost lever: the 118-task val is scored in full
         # for every candidate the Curator admits. Capped to 16 the gate goes
         # quiet again (val flat at 0.688 with bullets committed); 24 is the
         # smallest cap measured that still resolves the lift.
         size=["--pool", "3200", "--top-k", "139", "--val-cap", "64",
               "--grow-and-refine", "--staleness", "reflective",
               "--rounds", "9999"]),
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
         # `--iterations 9999` for the same reason GEPA passes `--rounds 9999`:
         # the port's own default (6) sits below the budget and stops the run
         # first, so the budget would be a ceiling nothing reaches.
         #
         # `--reflective-merge` because admission costs a **full validation
         # sweep per child** here -- upstream evaluates each one before the
         # frontier decides -- so an N-wide sweep costs N sweeps of val, and
         # that is the entire cost of going wider. Declared in SEMANTICS: the
         # frontier rule is untouched, the number of children it sees is not.
         dataset="OfficeQA/FinQA", width_flag="--workers",
         size=["--iterations", "9999", "--reflective-merge"]),
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
         # `--tasks`, not `--task-count`: the run function's parameter is named
         # task_count and the flag never was, so this row crashed argparse on
         # its first live cell.
         # No `needs="bwrap"` any more: candidate isolation has a Seatbelt
         # backend on macOS, and the runner's own `setrlimit` calls are what
         # impose the CPU and memory limits on either platform.
         size=["--tasks", "8"]),

    # The eleven declarative MethodPolicy ports. They reach the same three
    # arms through `examples._method_runner.standard_main`, and their budget
    # is already equal by construction: `rounds = candidates // workers` on
    # the synchronous path and `max_iters = candidates` on the async one, so
    # width divides the work rather than multiplying it. `--budget-rollouts`
    # is mapped onto `--candidates` in the runner -- one quantity, two
    # vocabularies -- and must be divisible by `--width`.
    dict(name="promptbreeder", module="examples.promptbreeder.promptbreeder_genetic_prompts",
         dataset="compact QA (mechanism microport)", width_flag="--workers", size=[], method_policy=True),
    dict(name="aflow", module="examples.aflow.aflow_workflow_search",
         dataset="compact QA (mechanism microport)", width_flag="--workers", size=[], method_policy=True),
    dict(name="reflexion", module="examples.reflexion.reflexion_episodic_memory",
         dataset="compact QA (mechanism microport)", width_flag="--workers", size=[], method_policy=True),
    dict(name="self_refine", module="examples.self_refine.self_refine_feedback_loop",
         dataset="compact QA (mechanism microport)", width_flag="--workers", size=[], method_policy=True),
    dict(name="voyager", module="examples.voyager.voyager_skill_library",
         dataset="skill library (environment analogue)", width_flag="--workers", size=[], method_policy=True),
    dict(name="skillweaver", module="examples.skillweaver.skillweaver_web_apis",
         dataset="web APIs (environment analogue)", width_flag="--workers", size=[], method_policy=True),
    dict(name="absolute_zero", module="examples.absolute_zero.absolute_zero_selfplay",
         dataset="self-play (inference analogue)", width_flag="--workers", size=[], method_policy=True),
    dict(name="r_zero", module="examples.r_zero.r_zero_challenger_solver",
         dataset="challenger/solver (inference analogue)", width_flag="--workers", size=[], method_policy=True),
    dict(name="agent0", module="examples.agent0.agent0_tool_curriculum",
         dataset="calculator tasks (inference analogue)", width_flag="--workers", size=[], method_policy=True),
    dict(name="sica", module="examples.sica.sica_self_edit",
         dataset="self-edit (analogue)", width_flag="--workers", size=[], method_policy=True),
    dict(name="godel_agent", module="examples.godel_agent.godel_agent_self_modify",
         dataset="self-edit (analogue)", width_flag="--workers", size=[], method_policy=True),
]

#: The expected answer in the "semantics changed" column, for every row and every
#: arm: parallelising is allowed to change *when* work is scheduled and *when*
#: diffs are merged, and nothing else (`docs/port-fidelity.md`).
SCHEDULING_ONLY = "rollout scheduling and merge timing only"

#: What each arm of each row changed about the published algorithm, keyed
#: `row -> arm`. This lives beside :data:`ROWS` rather than in a doc because a
#: column filled in after the fact is filled in by whoever is writing the
#: results, from memory, once the numbers are already on the page -- and a blank
#: or optimistic entry there makes the whole table unreadable rather than
#: slightly incomplete. `bench/matrix_report.py` refuses to render a row that has
#: no entry here, and `tests/test_matrix_semantics.py` refuses a row added to
#: `ROWS` without one.
#:
#: The `serial` arm's entry describes the control itself: it is the upstream loop
#: for six of the eight, and where it is *not* -- DGM -- saying so is the point,
#: because a speedup is measured against whatever is written here.
SEMANTICS = {
    "ace": {
        "serial": "upstream ACE: one worker, no merge",
        "parallel": SCHEDULING_ONLY,
        "async": SCHEDULING_ONLY,
    },
    "gepa": {
        "serial": "upstream GEPA (single instruction module), one worker",
        # `--reflective-merge` is in this row's `size`, so it is on in every arm
        # including the control -- which is what keeps the arms comparable. On
        # one worker there is nothing to merge and it is inert; from two workers
        # up it changes what the Pareto pool *contains* (one merged candidate per
        # round instead of one per worker) while leaving the selection rule
        # itself untouched. That is a merge-timing change by the letter of the
        # rule and a pool-contents change in effect, so it is named rather than
        # folded into SCHEDULING_ONLY.
        "parallel": SCHEDULING_ONLY + "; `--reflective-merge` admits one merged "
                    "candidate per round instead of one per worker (Pareto "
                    "selection unchanged, its input is not)",
        "async": SCHEDULING_ONLY + "; `--reflective-merge` as above",
    },
    "evoskill": {
        "serial": "upstream EvoSkill: bounded top-K aggregate frontier "
                  "(`manager.py:update_frontier`), one worker",
        "parallel": SCHEDULING_ONLY + "; `--reflective-merge` offers the frontier "
                    "one fused candidate per sweep instead of one per worker "
                    "(`update_frontier` and the parent draw unchanged, the number "
                    "of children they see is not)",
        # Was: the port swapped in `SgdSkillAggregator` on `asynchronous=True`,
        # which replaced the frontier with a single checkpoint and validated per
        # batch instead of per candidate -- the admission rule, which
        # `docs/port-fidelity.md` lists under "must not change". Found by
        # auditing the arms rather than by running them, and now fixed: the
        # frontier runs on every path and the SGD variant is opt-in behind
        # `--sgd-descent`.
        "async": SCHEDULING_ONLY + "; `--reflective-merge` as above",
    },
    "skillopt": {
        "serial": "upstream ReflACT: one edit batch per step, strict gate",
        "parallel": SCHEDULING_ONLY + "; `--minibatch` is this port's name for "
                    "the worker count, not upstream's minibatch of tasks. "
                    "`--reflective-merge` scores one fused patch per step, which "
                    "is upstream's shape -- a ReflACT step emits one patch of up "
                    "to `lr` edits and evaluates it once -- rather than a "
                    "departure from it",
        "async": SCHEDULING_ONLY + "; `--minibatch` and `--reflective-merge` as above",
    },
    "adas": {
        "serial": "upstream Meta Agent Search over the DSL substrate "
                  "(the substrate is this port's standing departure), one worker",
        # `--reflective-merge` is deliberately NOT passed to this row, and the
        # reason is the archive. ADAS is keep-all: every proposed design is
        # appended, and the meta-agent's entire conditioning signal is that
        # archive (designs plus fitness). Fusing a round's N designs into one
        # would remove N-1 archive entries per round, which shrinks what the next
        # generation is designed against -- the algorithm's central mechanism,
        # not its merge timing. The usual justification does not apply either: the
        # aggregator here drops nothing, so there is no "all but one discarded"
        # for fusion to recover. And a proposal is a whole agent *program*, not a
        # delta, so "merge these two designs" is itself an ADAS-shaped design
        # operation smuggled in below the archive.
        "parallel": SCHEDULING_ONLY,
        "async": SCHEDULING_ONLY,
    },
    "dgm": {
        # Not a clean control, and the matrix has to say so before any speedup is
        # read off this row: `--serial` reaches n_workers through
        # `--selfimprove-size`, and for DGM that is the population, so the control
        # arm is a population of one rather than upstream's default.
        "serial": "**degenerate control**: `--serial` sets `selfimprove_size=1`, "
                  "a population of one -- upstream's archive search with a "
                  "single child per generation, not its default configuration",
        "parallel": SCHEDULING_ONLY,
        "async": SCHEDULING_ONLY,
    },
    "openevolve": {
        "serial": "upstream OpenEvolve; `rounds = iterations // workers` already "
                  "fixed total work, so this row was equal-budget before "
                  "`--budget-rollouts` existed",
        "parallel": SCHEDULING_ONLY,
        "async": SCHEDULING_ONLY,
    },

    # The eleven MethodPolicy ports. Their merge behaviour is not a matrix
    # knob: `MethodPolicy.reflective` is declared by each method, because
    # for these the merger is part of the definition rather than an
    # execution choice -- so `--reflective-merge` is deliberately not passed
    # to them and their rows carry the clean string.
    "promptbreeder": {
        "serial": "PromptBreeder as a declared `mechanism_microport` -- see "
                  "docs/port-fidelity.md for what that class claims and does "
                  "not. **Serial here is not a one-worker loop**: these express "
                  "it as `max_concurrency=1` over the same `candidates // "
                  "workers` rounds, so the arm runs identical work without "
                  "concurrency rather than a narrower algorithm",
        "parallel": SCHEDULING_ONLY,
        "async": SCHEDULING_ONLY,
    },
    "aflow": {
        "serial": "AFlow as a declared `mechanism_microport` -- see "
                  "docs/port-fidelity.md for what that class claims and does "
                  "not. **Serial here is not a one-worker loop**: these express "
                  "it as `max_concurrency=1` over the same `candidates // "
                  "workers` rounds, so the arm runs identical work without "
                  "concurrency rather than a narrower algorithm",
        "parallel": SCHEDULING_ONLY,
        "async": SCHEDULING_ONLY,
    },
    "reflexion": {
        "serial": "Reflexion as a declared `mechanism_microport` -- see "
                  "docs/port-fidelity.md for what that class claims and does "
                  "not. **Serial here is not a one-worker loop**: these express "
                  "it as `max_concurrency=1` over the same `candidates // "
                  "workers` rounds, so the arm runs identical work without "
                  "concurrency rather than a narrower algorithm",
        "parallel": SCHEDULING_ONLY,
        "async": SCHEDULING_ONLY,
    },
    "self_refine": {
        "serial": "Self-Refine as a declared `mechanism_microport` -- see "
                  "docs/port-fidelity.md for what that class claims and does "
                  "not. **Serial here is not a one-worker loop**: these express "
                  "it as `max_concurrency=1` over the same `candidates // "
                  "workers` rounds, so the arm runs identical work without "
                  "concurrency rather than a narrower algorithm",
        "parallel": SCHEDULING_ONLY,
        "async": SCHEDULING_ONLY,
    },
    "voyager": {
        "serial": "Voyager as a declared `environment_analogue` -- see "
                  "docs/port-fidelity.md for what that class claims and does "
                  "not. **Serial here is not a one-worker loop**: these express "
                  "it as `max_concurrency=1` over the same `candidates // "
                  "workers` rounds, so the arm runs identical work without "
                  "concurrency rather than a narrower algorithm",
        "parallel": SCHEDULING_ONLY,
        "async": SCHEDULING_ONLY,
    },
    "skillweaver": {
        "serial": "SkillWeaver as a declared `environment_analogue` -- see "
                  "docs/port-fidelity.md for what that class claims and does "
                  "not. **Serial here is not a one-worker loop**: these express "
                  "it as `max_concurrency=1` over the same `candidates // "
                  "workers` rounds, so the arm runs identical work without "
                  "concurrency rather than a narrower algorithm",
        "parallel": SCHEDULING_ONLY,
        "async": SCHEDULING_ONLY,
    },
    "absolute_zero": {
        "serial": "Absolute Zero as a declared `inference_analogue` -- see "
                  "docs/port-fidelity.md for what that class claims and does "
                  "not. **Serial here is not a one-worker loop**: these express "
                  "it as `max_concurrency=1` over the same `candidates // "
                  "workers` rounds, so the arm runs identical work without "
                  "concurrency rather than a narrower algorithm",
        "parallel": SCHEDULING_ONLY,
        "async": SCHEDULING_ONLY,
    },
    "r_zero": {
        "serial": "R-Zero as a declared `inference_analogue` -- see "
                  "docs/port-fidelity.md for what that class claims and does "
                  "not. **Serial here is not a one-worker loop**: these express "
                  "it as `max_concurrency=1` over the same `candidates // "
                  "workers` rounds, so the arm runs identical work without "
                  "concurrency rather than a narrower algorithm",
        "parallel": SCHEDULING_ONLY,
        "async": SCHEDULING_ONLY,
    },
    "agent0": {
        "serial": "Agent0 as a declared `inference_analogue` -- see "
                  "docs/port-fidelity.md for what that class claims and does "
                  "not. **Serial here is not a one-worker loop**: these express "
                  "it as `max_concurrency=1` over the same `candidates // "
                  "workers` rounds, so the arm runs identical work without "
                  "concurrency rather than a narrower algorithm",
        "parallel": SCHEDULING_ONLY,
        "async": SCHEDULING_ONLY,
    },
    "sica": {
        "serial": "SICA as a declared `self_edit_analogue` -- see "
                  "docs/port-fidelity.md for what that class claims and does "
                  "not. **Serial here is not a one-worker loop**: these express "
                  "it as `max_concurrency=1` over the same `candidates // "
                  "workers` rounds, so the arm runs identical work without "
                  "concurrency rather than a narrower algorithm",
        "parallel": SCHEDULING_ONLY,
        "async": SCHEDULING_ONLY,
    },
    "godel_agent": {
        "serial": "Godel Agent as a declared `self_edit_analogue` -- see "
                  "docs/port-fidelity.md for what that class claims and does "
                  "not. **Serial here is not a one-worker loop**: these express "
                  "it as `max_concurrency=1` over the same `candidates // "
                  "workers` rounds, so the arm runs identical work without "
                  "concurrency rather than a narrower algorithm",
        "parallel": SCHEDULING_ONLY,
        "async": SCHEDULING_ONLY,
    },
}

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
    # The engine's own counters, printed by `examples._common.report_engine` in
    # one shared format. `engine_rollouts` is measured spend where `rollouts`
    # above is what the port's stdout happened to say; the stale pair is the
    # async arm's quality-drop diagnosis, numerator WITH denominator.
    "engine_rollouts": r"engine counters: rollouts=([\d,]+)",
    "stale_considered": r"stale_considered=([\d,]+)",
    "stale_discarded": r"stale_discarded=([\d,]+)",
    "redispatched": r"redispatched=([\d,]+)",
    # The stage profile from the same `report_engine` line. Without these a
    # width sweep can say the arm stopped scaling but not *where* it stopped:
    # `merge_seconds / wallclock` is merger occupancy and `merge_gate_seconds`
    # is the part of it that is gate evaluation, which is the term a saturation
    # model needs. `worker_starved_seconds` is the confirmation from the other
    # side -- a busy merger with no starved workers has hidden itself.
    "wallclock": r"stage profile: wallclock=([\d.]+)",
    "merge_seconds": r"merge_seconds=([\d.]+)",
    "merge_gate_seconds": r"merge_gate_seconds=([\d.]+)",
    "worker_starved_seconds": r"worker_starved_seconds=([\d.]+)",
    "eval_seconds": r"eval_seconds=([\d.]+)",
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
    # `--serial` only lowers `n_workers`; it does not touch `--eval-concurrency`,
    # so the "serial" arm was still scoring its gate and its test split
    # concurrently. Measured on GEPA: 3849s of model time inside a 2148s
    # wall-clock on a one-worker run -- a 1.8x compression that can only come
    # from overlapped evaluation. The control arm was therefore already
    # partly parallel, and every speedup measured against it was the *remaining*
    # parallelism rather than the parallelism.
    #
    # The published loop scores one task at a time, so the control does too.
    # It is a real cost -- the serial arm gets slower and the speedups get
    # larger -- which is why the value used is recorded in the output rather
    # than hidden here.
    eval_conc = args.serial_eval_concurrency if arm == "serial" else args.eval_concurrency
    cmd = [sys.executable, "-u", "-m", row["module"], "--yes",
           "--seed", str(seed), "--budget-rollouts", str(args.budget)]
    cmd += row["size"] + ["--eval-concurrency", str(eval_conc)]
    if args.no_thinking:
        cmd += ["--no-thinking"]
    if arm == "serial":
        cmd += ["--serial"]
    else:
        cmd += [row["width_flag"], str(args.width)]
    if row.get("method_policy"):
        # These size their work as `candidates // workers` rounds, so the worker
        # count is part of the budget arithmetic rather than only of the
        # schedule -- and their `--serial` sets `max_concurrency=1` without
        # touching it. Passing the width on every arm is therefore what keeps
        # the budget equal: leave it off the serial arm and that arm silently
        # runs the 2-worker default, i.e. a different number of rounds over the
        # same candidate budget.
        if arm == "serial":
            cmd += [row["width_flag"], str(args.width)]
    if arm == "async":
        # The cell timeout as the wall-clock budget: the async runtime refuses to
        # run unbounded, and the ports' own --max-seconds defaults (15-45s) would
        # otherwise stop the arm long before `--budget-rollouts` binds -- the one
        # comparison this matrix exists to make.
        cmd += ["--async", "--async-ratio", str(args.async_ratio),
                "--max-seconds", str(args.timeout)]
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
                # Recorded per cell, not only in the header: it is the second
                # concurrency in the run and it differs BY ARM, so a reader who
                # only has the cells can still tell what the control was.
                eval_concurrency=eval_conc,
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
                   "async_ratio": args.async_ratio,
                   "eval_concurrency": args.eval_concurrency,
                   "serial_eval_concurrency": args.serial_eval_concurrency,
                   "no_thinking": bool(args.no_thinking),
                   "model": args.model, "provider": args.provider,
                   "cells": cells}, handle, indent=2)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--budget", type=int, default=8,
                   help="rollouts, pinned on both arms of every row")
    p.add_argument("--width", type=int, default=8, help="workers in the parallel arm")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--rows", default="", help="comma-separated subset of row names")
    p.add_argument("--arms", default="serial,parallel,async",
                   help="comma-separated subset of arms to run")
    p.add_argument("--async-ratio", type=int, default=3,
                   help="staleness lag budget for the async arm")
    p.add_argument("--provider", default="claude")
    p.add_argument("--model", default="GLM-5.2")
    p.add_argument("--eval-concurrency", type=int, default=16,
                   help="held-out evaluations in flight on the parallel and "
                        "async arms; wall-clock only")
    p.add_argument("--serial-eval-concurrency", type=int, default=1,
                   help="the same, for the serial control -- 1 by default "
                        "because the published loop scores one task at a time, "
                        "and a control that evaluates concurrently is already "
                        "partly parallel (see _cell)")
    p.add_argument("--no-thinking", action="store_true",
                   help="pass --no-thinking to every cell: ask the backend to "
                        "skip reasoning tokens. Changes model output, not just "
                        "speed -- it lands in the recorded configuration")
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
    known = ("serial", "parallel", "async")
    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    if bad := [a for a in arms if a not in known]:
        raise SystemExit(f"unknown arm(s) {bad}; choose from {known}")
    todo = [(r, a, s) for r in rows for s in seeds for a in arms
            if (r["name"], a, s) not in done]

    # `run_port` refuses a candidate budget that is not a multiple of the worker
    # count, and it refuses it *after* the sweep has started paying for earlier
    # cells. Checked here instead, before anything runs.
    mp = [r["name"] for r in rows if r.get("method_policy")]
    if mp and (args.budget % args.width or args.budget < args.width):
        raise SystemExit(
            f"--budget {args.budget} must be a positive multiple of --width "
            f"{args.width} for the MethodPolicy rows ({', '.join(mp[:3])}...), "
            f"which size their work as `candidates // workers` rounds")

    print(f"matrix: {len(rows)} rows x {len(arms)} arms x {len(seeds)} seeds "
          f"= {len(rows) * len(arms) * len(seeds)} cells, {len(done)} already done, "
          f"{len(todo)} to run")
    print(f"budget: {args.budget} rollouts on EVERY arm; parallel/async width "
          f"{args.width}, async lag budget {args.async_ratio}")
    print(f"eval concurrency: {args.eval_concurrency} on the parallel/async arms, "
          f"{args.serial_eval_concurrency} on the serial control"
          f"{'; reasoning tokens disabled' if args.no_thinking else ''}\n")
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
