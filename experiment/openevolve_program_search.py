"""A compact OpenEvolve program-search experiment with sandboxed evaluation.

It keeps OpenEvolve's defining mechanics: Python source is the genome, an LLM
proposes mutations from a parent plus archive inspirations, every candidate is
executed by an external evaluator, and a quality-diversity archive supplies the
next parents. Full-program rewrites are used instead of diffs because the genome
is intentionally tiny.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import resource
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from agentdescent.agents import Usage

from experiment._common import (
    REPORT_DIR,
    ROOT,
    add_model_args,
    confirm_paid_run,
    make_completion,
    require_api_environment,
    usage_dict,
    utc_now,
    write_json,
)


UPSTREAM_COMMIT = "411fb59c886c18704caaffb611e17cf9e7d824d2"
GLOBAL_MIN_X = -1.704
GLOBAL_MIN_Y = 0.678
GLOBAL_MIN_VALUE = -1.519
ALLOWED_IMPORTS = {"bisect", "heapq", "itertools", "math", "random", "statistics"}
FORBIDDEN_CALLS = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
    "__import__",
}
RUNNER = Path(__file__).with_name("_openevolve_runner.py")

INITIAL_PROGRAM = '''"""Initial random-search genome for the function-minimization task."""

def search_algorithm(objective, budget, rng, bounds):
    """Return the best (x, y) found within a strict objective-call budget."""
    low, high = bounds
    best_x = rng.uniform(low, high)
    best_y = rng.uniform(low, high)
    best_value = objective(best_x, best_y)
    for _ in range(budget - 1):
        x = rng.uniform(low, high)
        y = rng.uniform(low, high)
        value = objective(x, y)
        if value < best_value:
            best_x, best_y, best_value = x, y, value
    return best_x, best_y
'''


@dataclass
class Program:
    program_id: str
    iteration: int
    island: int
    parent_id: Optional[str]
    code: str
    change_summary: str
    metrics: Dict[str, Any]
    valid: bool
    error: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser, max_tokens=2048, temperature=0.7)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--objective-budget", type=int, default=200)
    parser.add_argument("--archive-size", type=int, default=12)
    parser.add_argument("--islands", type=int, default=3)
    parser.add_argument("--exploitation-ratio", type=float, default=0.7)
    parser.add_argument("--candidate-timeout", type=float, default=15.0)
    parser.add_argument("--max-code-length", type=int, default=20000)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR / "openevolve-small-result.json",
    )
    parser.add_argument(
        "--best-program",
        type=Path,
        default=ROOT / "experiment" / "openevolve_best_program.py",
    )
    return parser


def objective_value(x: float, y: float) -> float:
    return math.sin(x) * math.cos(y) + math.sin(x * y) + (x * x + y * y) / 20.0


def validate_source(source: str, max_length: int = 20000) -> Tuple[bool, str]:
    if not source.strip():
        return False, "empty source"
    if len(source) > max_length:
        return False, f"source length {len(source)} exceeds {max_length}"
    if "\x00" in source:
        return False, "source contains a NUL byte"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc.msg} at line {exc.lineno}"

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not any(node.name == "search_algorithm" for node in functions):
        return False, "missing search_algorithm function"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return False, f"import {alias.name!r} is not allowed"
        elif isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".")[0] not in ALLOWED_IMPORTS:
                return False, f"import from {node.module!r} is not allowed"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, f"dunder attribute {node.attr!r} is not allowed"
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_CALLS:
            return False, f"name {node.id!r} is not allowed"

    allowed_top_level = (
        ast.Expr,
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.Assign,
        ast.AnnAssign,
    )
    for node in tree.body:
        if not isinstance(node, allowed_top_level):
            return False, f"top-level {type(node).__name__} is not allowed"
        if isinstance(node, ast.Expr) and not (
            isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ):
            return False, "only a module docstring may be a top-level expression"
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if not isinstance(value, (ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Set)):
                return False, "top-level assignments must be literal constants"

    compact = re.sub(r"\s+", "", source)
    for forbidden in ("-1.704", "0.678", "-1.519"):
        if forbidden in compact:
            return False, "hard-coding the evaluator's known optimum is not allowed"
    return True, ""


def _current_user_task_count() -> int:
    """Count Linux tasks owned by this uid for an RLIMIT_NPROC floor."""
    uid = os.getuid()
    total = 0
    try:
        entries = list(os.scandir("/proc"))
    except OSError:
        return 0
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            status = Path(entry.path, "status").read_text(encoding="utf-8")
            owner = next(
                int(line.split()[1])
                for line in status.splitlines()
                if line.startswith("Uid:")
            )
            if owner == uid:
                total += sum(1 for _ in os.scandir(Path(entry.path, "task")))
        except (OSError, StopIteration, ValueError):
            continue
    return total


def _limit_candidate_process(timeout: float, nproc_limit: int):
    def set_limits() -> None:
        os.setsid()
        cpu_seconds = max(2, int(math.ceil(timeout)))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        try:
            # RLIMIT_NPROC counts every thread owned by this WSL user, including
            # the editor and test runner. The parent computes a floor from the
            # current host load so bwrap keeps a small, bounded creation budget.
            resource.setrlimit(resource.RLIMIT_NPROC, (nproc_limit, nproc_limit))
        except (ValueError, OSError):
            pass

    return set_limits


def _bubblewrap_command(candidate: Path, trials: int, budget: int, seed: int) -> List[str]:
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("Bubblewrap (bwrap) is required for candidate isolation")
    command = [
        bwrap,
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/lib",
        "/lib",
    ]
    if Path("/lib64").exists():
        command.extend(("--ro-bind", "/lib64", "/lib64"))
    command.extend(
        (
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--ro-bind",
            str(candidate),
            "/candidate.py",
            "--ro-bind",
            str(RUNNER),
            "/runner.py",
            "--unshare-net",
            "--die-with-parent",
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/bin",
            "/usr/bin/python3",
            "-I",
            "/runner.py",
            "/candidate.py",
            "--trials",
            str(trials),
            "--budget",
            str(budget),
            "--seed",
            str(seed),
        )
    )
    return command


def _zero_metrics(error: str) -> Dict[str, Any]:
    return {
        "value_score": 0.0,
        "distance_score": 0.0,
        "reliability_score": 0.0,
        "combined_score": 0.0,
        "avg_value": None,
        "avg_distance": None,
        "avg_runtime_seconds": None,
        "avg_objective_calls": None,
        "successful_trials": 0,
        "total_trials": 0,
        "error": error,
    }


def official_metrics(trials: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    successes = [trial for trial in trials if trial.get("success")]
    total = len(trials)
    if not successes or total == 0:
        return _zero_metrics("all trials failed")
    values = [objective_value(float(trial["x"]), float(trial["y"])) for trial in successes]
    distances = [
        math.hypot(float(trial["x"]) - GLOBAL_MIN_X, float(trial["y"]) - GLOBAL_MIN_Y)
        for trial in successes
    ]
    avg_value = sum(values) / len(values)
    avg_distance = sum(distances) / len(distances)
    value_stddev = math.sqrt(sum((value - avg_value) ** 2 for value in values) / len(values))
    distance_stddev = math.sqrt(
        sum((distance - avg_distance) ** 2 for distance in distances) / len(distances)
    )
    value_score = 1.0 / (1.0 + abs(avg_value - GLOBAL_MIN_VALUE))
    distance_score = 1.0 / (1.0 + avg_distance)
    reliability_score = len(successes) / total
    if avg_distance < 0.5:
        multiplier = 1.5
    elif avg_distance < 1.5:
        multiplier = 1.2
    elif avg_distance < 3.0:
        multiplier = 1.0
    else:
        multiplier = 0.7
    base_score = 0.5 * value_score + 0.3 * distance_score + 0.2 * reliability_score
    return {
        "value_score": value_score,
        "distance_score": distance_score,
        "reliability_score": reliability_score,
        "solution_quality_multiplier": multiplier,
        "combined_score": base_score * multiplier,
        "avg_value": avg_value,
        "value_stddev": value_stddev,
        "best_value": min(values),
        "worst_value": max(values),
        "avg_distance": avg_distance,
        "distance_stddev": distance_stddev,
        "best_distance": min(distances),
        "worst_distance": max(distances),
        "avg_runtime_seconds": sum(float(trial["seconds"]) for trial in successes)
        / len(successes),
        "avg_objective_calls": sum(int(trial["objective_calls"]) for trial in successes)
        / len(successes),
        "successful_trials": len(successes),
        "total_trials": total,
        "error": "",
    }


def evaluate_source(
    source: str,
    *,
    trials: int,
    budget: int,
    seed: int,
    timeout: float,
    max_length: int = 20000,
) -> Tuple[bool, Dict[str, Any], str, List[Dict[str, Any]]]:
    valid, error = validate_source(source, max_length)
    if not valid:
        return False, _zero_metrics(error), error, []
    with tempfile.TemporaryDirectory(prefix="agentdescent-openevolve-") as directory:
        candidate = Path(directory) / "candidate.py"
        candidate.write_text(source, encoding="utf-8")
        command = _bubblewrap_command(candidate, trials, budget, seed)
        nproc_limit = max(512, _current_user_task_count() + 64)
        for sandbox_attempt in range(3):
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=_limit_candidate_process(timeout, nproc_limit),
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = process.communicate()
                error = f"candidate timed out after {timeout:.1f}s"
                return False, _zero_metrics(error), error, []
            transient_namespace_error = (
                process.returncode != 0
                and "Creating new namespace failed: Resource temporarily unavailable"
                in stderr
            )
            if transient_namespace_error and sandbox_attempt < 2:
                time.sleep(0.25 * (sandbox_attempt + 1))
                continue
            break
    if process.returncode != 0:
        error = f"sandbox exited {process.returncode}: {(stderr or stdout).strip()[:500]}"
        return False, _zero_metrics(error), error, []
    try:
        payload = json.loads(stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        error = f"invalid sandbox JSON: {exc}; output={stdout.strip()[:300]!r}"
        return False, _zero_metrics(error), error, []
    if not payload.get("ok"):
        error = str(payload.get("error") or "sandbox runner failed")
        return False, _zero_metrics(error), error, payload.get("trials", [])
    trial_rows = payload.get("trials", [])
    metrics = official_metrics(trial_rows)
    valid = metrics["successful_trials"] > 0
    return valid, metrics, metrics.get("error", ""), trial_rows


def extract_program(response: str) -> Tuple[str, str]:
    program_match = re.search(r"<PROGRAM>\s*(.*?)\s*</PROGRAM>", response, re.I | re.S)
    if program_match:
        code = program_match.group(1).strip()
    else:
        fence = re.search(r"```(?:python)?\s*(.*?)```", response, re.I | re.S)
        code = (fence.group(1) if fence else response).strip()
    summary_match = re.search(
        r"<CHANGE_SUMMARY>\s*(.*?)\s*</CHANGE_SUMMARY>", response, re.I | re.S
    )
    summary = summary_match.group(1).strip() if summary_match else ""
    return code, summary


def _code_tokens(source: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?", source.lower()))


def code_distance(left: str, right: str) -> float:
    a, b = _code_tokens(left), _code_tokens(right)
    return 1.0 - len(a & b) / len(a | b) if a or b else 0.0


def prune_archive(programs: Sequence[Program], size: int) -> List[Program]:
    valid = [program for program in programs if program.valid]
    if len(valid) <= size:
        return sorted(valid, key=lambda p: p.metrics["combined_score"], reverse=True)
    ranked = sorted(valid, key=lambda p: p.metrics["combined_score"], reverse=True)
    kept = [ranked.pop(0)]
    while ranked and len(kept) < size:
        max_score = max(program.metrics["combined_score"] for program in valid) or 1.0

        def utility(program: Program) -> float:
            quality = program.metrics["combined_score"] / max_score
            diversity = min(code_distance(program.code, other.code) for other in kept)
            return 0.75 * quality + 0.25 * diversity

        choice = max(ranked, key=utility)
        kept.append(choice)
        ranked.remove(choice)
    return sorted(kept, key=lambda p: p.metrics["combined_score"], reverse=True)


def _mutation_prompt(
    parent: Program,
    best: Program,
    inspiration: Program,
    *,
    iteration: int,
    budget: int,
    trials: int = 10,
) -> str:
    return f'''You are the mutation model in an OpenEvolve-style program search.

Improve a Python search algorithm for this objective on x,y in [-5,5]:
f(x,y) = sin(x)*cos(y) + sin(x*y) + (x^2+y^2)/20.

The evaluator runs {trials} deterministic seeds with a strict budget of {budget} calls to
the supplied objective. It rewards low average objective value, proximity to the
same global basin across seeds, and reliability. Never hard-code a known optimum.

Contract and safety constraints:
- Return a complete Python module defining exactly this callable interface:
  search_algorithm(objective, budget, rng, bounds) -> (x, y)
- Use objective(x, y) for scoring and never exceed budget calls.
- Use only the standard library modules math, random, statistics, heapq, bisect,
  or itertools. Do not access files, processes, the environment, or the network.
- Keep all returned coordinates inside bounds.
- Generalize across RNG seeds. Spend the fixed budget more intelligently rather
  than merely increasing loops.

Iteration: {iteration}

PARENT METRICS:
{json.dumps(parent.metrics, sort_keys=True)}

PARENT PROGRAM:
<PARENT_PROGRAM>
{parent.code}
</PARENT_PROGRAM>

GLOBAL BEST METRICS:
{json.dumps(best.metrics, sort_keys=True)}

DIVERSE INSPIRATION METRICS:
{json.dumps(inspiration.metrics, sort_keys=True)}

DIVERSE INSPIRATION PROGRAM:
<INSPIRATION_PROGRAM>
{inspiration.code}
</INSPIRATION_PROGRAM>

Propose one substantive algorithmic mutation. Return exactly:
<PROGRAM>
complete Python source
</PROGRAM>
<CHANGE_SUMMARY>one concise sentence</CHANGE_SUMMARY>'''


def _program_id(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def _serialize_program(program: Program, *, include_trials=None) -> Dict[str, Any]:
    payload = asdict(program)
    if include_trials is not None:
        payload["trials"] = include_trials
    return payload


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.iterations < 1 or args.trials < 1 or args.objective_budget < 2:
        raise ValueError("iterations/trials must be positive and objective-budget must be >= 2")
    if args.archive_size < 1 or args.islands < 1:
        raise ValueError("archive-size and islands must be positive")
    if not 0.0 <= args.exploitation_ratio <= 1.0:
        raise ValueError("exploitation-ratio must be in [0, 1]")
    require_api_environment(args.provider)

    started = time.monotonic()
    started_at = utc_now()
    usage = Usage()
    completion = make_completion(args, usage)
    rng = random.Random(args.seed)

    initial_valid, initial_metrics, initial_error, initial_trials = evaluate_source(
        INITIAL_PROGRAM,
        trials=args.trials,
        budget=args.objective_budget,
        seed=args.seed,
        timeout=args.candidate_timeout,
        max_length=args.max_code_length,
    )
    if not initial_valid:
        raise RuntimeError(f"initial program failed evaluation: {initial_error}")
    initial = Program(
        _program_id(INITIAL_PROGRAM),
        0,
        0,
        None,
        INITIAL_PROGRAM,
        "uniform random search baseline",
        initial_metrics,
        True,
    )
    archive = [initial]
    history = [_serialize_program(initial, include_trials=initial_trials)]

    result: Dict[str, Any] = {
        "experiment": "OpenEvolve-style function minimization",
        "status": "running",
        "started_at": started_at,
        "upstream": {
            "repository": "https://github.com/algorithmicsuperintelligence/openevolve",
            "commit": UPSTREAM_COMMIT,
            "example": "examples/function_minimization",
        },
        "config": {
            "provider": args.provider,
            "model": args.model,
            "thinking": args.thinking,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "api_timeout": args.api_timeout,
            "seed": args.seed,
            "iterations": args.iterations,
            "trials": args.trials,
            "objective_budget": args.objective_budget,
            "archive_size": args.archive_size,
            "islands": args.islands,
            "exploitation_ratio": args.exploitation_ratio,
            "candidate_timeout": args.candidate_timeout,
            "max_code_length": args.max_code_length,
            "evolution_mode": "full-program rewrite",
            "sandbox": "bubblewrap + no network + clear environment + rlimits + AST gate",
        },
        "baseline": history[0],
        "history": history,
    }
    write_json(args.output, result)

    for iteration in range(1, args.iterations + 1):
        island = (iteration - 1) % args.islands
        island_pool = [program for program in archive if program.island == island] or archive
        if rng.random() < args.exploitation_ratio:
            parent = max(island_pool, key=lambda program: program.metrics["combined_score"])
        else:
            parent = rng.choice(island_pool)
        best = max(archive, key=lambda program: program.metrics["combined_score"])
        inspiration = max(archive, key=lambda program: code_distance(parent.code, program.code))

        raw = completion(
            _mutation_prompt(
                parent,
                best,
                inspiration,
                iteration=iteration,
                budget=args.objective_budget,
                trials=args.trials,
            )
        ).strip()
        code, summary = extract_program(raw)
        valid, metrics, error, trials = evaluate_source(
            code,
            trials=args.trials,
            budget=args.objective_budget,
            seed=args.seed,
            timeout=args.candidate_timeout,
            max_length=args.max_code_length,
        )
        program = Program(
            _program_id(code),
            iteration,
            island,
            parent.program_id,
            code,
            summary,
            metrics,
            valid,
            error,
        )
        history.append(_serialize_program(program, include_trials=trials))
        if valid:
            archive = prune_archive([*archive, program], args.archive_size)
        best = max(archive, key=lambda item: item.metrics["combined_score"])
        result.update(
            {
                "history": history,
                "archive_ids": [item.program_id for item in archive],
                "best_so_far_id": best.program_id,
                "usage": usage_dict(usage),
                "wall_seconds_so_far": round(time.monotonic() - started, 6),
            }
        )
        write_json(args.output, result)
        print(
            f"iteration {iteration}/{args.iterations}: valid={valid}, "
            f"score={metrics['combined_score']:.6f}, "
            f"best={best.metrics['combined_score']:.6f}"
        )

    best = max(archive, key=lambda program: program.metrics["combined_score"])
    args.best_program.parent.mkdir(parents=True, exist_ok=True)
    args.best_program.write_text(best.code.rstrip() + "\n", encoding="utf-8")
    baseline_score = initial.metrics["combined_score"]
    best_score = best.metrics["combined_score"]
    result.update(
        {
            "status": "completed",
            "completed_at": utc_now(),
            "best": _serialize_program(best),
            "best_program_path": str(args.best_program),
            "improvement": {
                "combined_score_absolute": best_score - baseline_score,
                "combined_score_percent": (
                    100.0 * (best_score - baseline_score) / baseline_score
                    if baseline_score
                    else None
                ),
                "avg_value_delta": best.metrics["avg_value"] - initial.metrics["avg_value"],
                "avg_distance_delta": (
                    best.metrics["avg_distance"] - initial.metrics["avg_distance"]
                ),
            },
            "archive_ids": [program.program_id for program in archive],
            "usage": usage_dict(usage),
            "wall_seconds": round(time.monotonic() - started, 6),
        }
    )
    write_json(args.output, result)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        print(
            "OpenEvolve dry run: "
            f"model={args.model}, thinking={args.thinking}, "
            f"iterations/calls={args.iterations}, "
            f"trials={args.trials}, objective_budget={args.objective_budget}, "
            f"archive={args.archive_size}, islands={args.islands}, output={args.output}"
        )
        return 0
    confirm_paid_run(args, "OpenEvolve function-minimization experiment")
    result = run(args)
    print(
        "completed: "
        f"score {result['baseline']['metrics']['combined_score']:.6f} -> "
        f"{result['best']['metrics']['combined_score']:.6f}; "
        f"calls={result['usage']['calls']}; output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
