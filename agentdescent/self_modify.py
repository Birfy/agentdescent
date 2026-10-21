"""Self-modifying framework: the framework evolves patches to its own code.

Everything else in this project evolves *artifacts* -- skills, prompts, decision
rules. This module evolves the framework's own source, gated by the framework's
own tests. A candidate is
the source of one module-level function (its own patch proposal, kept as text);
the gate is an AST walk that refuses imports, dunder access and the calls that
reach the interpreter; the tests are callables that must all pass before any
benchmark score counts (a patch that breaks a test is a zero, never a
regression with a number attached); and the benchmark measures what the change
is for (speed, cost, accuracy). ``improve_source`` runs the ordinary
``evolve()`` over ``(test, seed)`` tasks, so the framework searches its own
code under its own gate and reports the accepted patch, the tests it passed,
and the measured improvement over the seed.
"""

from __future__ import annotations

import ast
import difflib
import inspect
import re
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .evolution import EvolutionResult, Task, evolve
from .meta import MetaOutcome, SlotSpec, slot_reflector

__all__ = ["patch_source", "apply_patch", "gate_source", "evaluate_source",
           "SelfModifyReport", "improve_source"]


# -- the patch ------------------------------------------------------------------


def patch_source(old_source: str, new_source: str) -> str:
    """A unified diff between two sources -- the patch a human or a reviewer
    sees, and what an apply can reverse."""
    return "\n".join(difflib.unified_diff(
        old_source.splitlines(), new_source.splitlines(),
        fromfile="before", tofile="after", lineterm=""))


def apply_patch(source: str, patch: str) -> str:
    """Apply a :func:`patch_source` diff back to a source. ``ValueError`` when
    the patch does not apply cleanly -- a patch whose context does not match
    ``source`` is refused, never half-applied. Untouched lines outside a hunk's
    context are preserved, so a middle-of-file edit keeps the rest of the file."""
    hunks: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for line in patch.splitlines():
        header = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if header:
            if current is not None:
                hunks.append(current)
            current = {"old_start": int(header.group(1)), "ops": []}
        elif current is not None:
            if line.startswith("+"):
                current["ops"].append(("add", line[1:]))
            elif line.startswith("-"):
                current["ops"].append(("del", line[1:]))
            elif line.startswith(" "):
                current["ops"].append(("ctx", line[1:]))
    if current is not None:
        hunks.append(current)
    if not hunks:
        raise ValueError("patch contains no hunks")

    lines = source.splitlines()
    out = list(lines)
    # apply from the last hunk so earlier offsets stay valid
    for hunk in reversed(hunks):
        old = [text for kind, text in hunk["ops"] if kind in ("del", "ctx")]
        new = [text for kind, text in hunk["ops"] if kind != "del"]
        idx = _find_hunk(out, old, hunk["old_start"] - 1)
        if idx is None:
            raise ValueError("patch context does not match source")
        out[idx:idx + len(old)] = new
    result = "\n".join(out)
    return result + "\n" if out else result


def _find_hunk(lines: List[str], old: List[str], hint: int) -> Optional[int]:
    """The 0-based index where ``old`` (context + removed lines) matches,
    starting at the hunk's own line number (unified diff numbers are exact)."""
    if not old:
        return None
    for probe in range(max(0, hint), len(lines) - len(old) + 1):
        if lines[probe:probe + len(old)] == old:
            return probe
        if probe > hint + 3:      # a slide this big is no longer the same hunk
            break
    return None


def gate_source(source: str, *, function_name: str = "fn") -> ast.Module:
    """The candidate's gate: exactly one function of no arguments other than
    what its signature declares, built only from safe syntax. Imports, dunder
    access and the calls that reach the interpreter are refused -- the same
    walk ``compile_policy_source`` runs over every self-edit."""
    if not source or len(source) > 12_000:
        raise ValueError("candidate source is empty or too long")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError("candidate source is not valid Python") from error
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != function_name:
        raise ValueError(f"candidate source must define exactly one function, "
                         f"{function_name}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            raise ValueError("imports are not allowed in a self-edit")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder names are forbidden")
        if isinstance(node, ast.Attribute):
            # dunder attributes AND frame/code introspection (gi_frame,
            # cr_frame, ag_frame, f_globals, f_back, ...) are how a gated
            # source escapes to the interpreter
            if node.attr.startswith("__") or node.attr in (
                    "gi_frame", "gi_code", "gi_running", "cr_frame", "cr_code",
                    "ag_frame", "ag_code", "ag_await",
                    "f_back", "f_builtins", "f_code", "f_globals", "f_locals"):
                raise ValueError(f"forbidden attribute: {node.attr}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("exec", "eval", "open", "__import__",
                                     "getattr", "setattr", "globals", "locals"):
            raise ValueError(f"forbidden call: {node.func.id}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and node.value.startswith("__") and node.value.endswith("__"):
            raise ValueError("dunder string constants are forbidden")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise ValueError("global/nonlocal are forbidden")
    return tree


def _compile_fn(source: str, function_name: str = "fn") -> Callable:
    """Gate and compile the candidate into the function it declares, inside a
    namespace holding only safe builtins (arithmetic, comparisons, the mapping
    and sequence tools a scoring function needs -- never the interpreter)."""
    tree = gate_source(source, function_name=function_name)
    safe = {name: __builtins__[name] if isinstance(__builtins__, dict)
            else getattr(__builtins__, name)
            for name in ("abs", "all", "any", "bool", "dict", "divmod",
                         "enumerate", "float", "frozenset", "int", "isinstance",
                         "iter", "len", "list", "map", "max", "min", "next",
                         "range", "repr", "reversed", "round", "set", "slice",
                         "sorted", "str", "sum", "tuple", "zip",
                         "Exception", "ValueError", "TypeError", "KeyError",
                         "IndexError", "ZeroDivisionError",
                         "True", "False", "None", "staticmethod",
                         "classmethod", "property")
                         if name in (__builtins__ if isinstance(__builtins__, dict)
                                     else dir(__builtins__))}
    namespace: Dict[str, Any] = {"__builtins__": safe, "__name__": "candidate"}
    exec(compile(tree, "<self-edit>", "exec"), namespace, namespace)
    fn = namespace[function_name]
    if not callable(fn):
        raise ValueError(f"{function_name} did not compile to a callable")
    return fn


def evaluate_source(
    source: str,
    tests: Sequence[Callable[[Callable], Any]],
    benchmark: Callable[[Callable], float],
    *,
    function_name: str = "fn",
) -> Tuple[int, int, float]:
    """Run the candidate against the tests and the benchmark.

    Returns ``(passed, total, benchmark_score)``. Every test must pass for the
    benchmark to count -- a failing test makes the score ``0.0``, so a patch
    that breaks anything is a zero, never a regression with a number attached.
    A candidate that fails to compile or raises in a test is ``(0, total, 0.0)``
    -- the gate counts the refusal, it does not crash the run.
    """
    try:
        fn = _compile_fn(source, function_name=function_name)
    except Exception:  # noqa: BLE001 - module-level code and defaults run at
        return 0, len(tests), 0.0                      # compile time and can raise
                                                       # ZeroDivisionError, NameError,
                                                       # IndexError, ... -- count,
                                                       # don't crash
    passed = 0
    for test in tests:
        try:
            if test(fn):
                passed += 1
        except Exception:  # noqa: BLE001 - a failing test is a failure, not a crash
            pass
    try:
        score = float(benchmark(fn)) if passed == len(tests) else 0.0
    except Exception:  # noqa: BLE001
        score = 0.0
    if score != score or score < 0.0 or score > 1.0:
        score = 0.0
    return passed, len(tests), score


# -- the loop -------------------------------------------------------------------


@dataclass(frozen=True)
class SelfModifyReport:
    """What :func:`improve_source` accepted: the patch, its tests, its gain."""

    seed_source: str
    value: str
    patch: str
    seed_tests: int
    evolved_tests: int
    seed_benchmark: float
    evolved_benchmark: float
    gain: float


def improve_source(
    seed_source: str,
    tests: Sequence[Callable[[Callable], Any]],
    benchmark: Callable[[Callable], float],
    *,
    propose: Optional[Callable[[str, Task, str, float], Optional[str]]] = None,
    model: Optional[Any] = None,
    seeds: Sequence[int] = (0, 1, 2, 3),
    function_name: str = "fn",
    **evolve_kwargs: Any,
) -> Tuple[SelfModifyReport, EvolutionResult]:
    """Evolve the source of one function under its own tests and benchmark.

    A bespoke ``evolve()`` over ``(test, seed)`` tasks: the artifact is the
    source text, ``propose`` (or ``model`` for the shipped reflector) rewrites
    it, the gate refuses anything unsafe, the reward is the benchmark score with
    a hard test gate (a failing test is a zero). The report carries the accepted
    patch, the tests it passed, and the measured gain over the seed.

    The candidate's gate is the same AST walk every other self-edit in this
    framework runs: enough to keep a model's rewrite to *computing*, not enough
    to run untrusted code from a stranger.
    """
    if propose is None and model is None:
        raise ValueError("improve_source() needs propose= or model=")

    def run(rendered: str, task: Task) -> str:
        passed, total, score = evaluate_source(
            rendered, tests, benchmark, function_name=function_name)
        return MetaOutcome(curve=[score], final=score, rollouts=1,
                           detail={"passed": passed, "total": total}).to_json()

    def reward(task: Task, output: str) -> float:
        try:
            return float(MetaOutcome.from_json(output).final)
        except (TypeError, ValueError, KeyError):
            return 0.0

    spec = _source_spec(seed_source, function_name)
    reflector = _strip_fence(propose if propose is not None
                             else slot_reflector(model, spec))
    tasks = [Task(id=f"t{seed}", prompt=f"test batch {seed}",
                  meta={"seed": seed}) for seed in seeds]
    result = evolve(
        tasks, reward, run=run, propose=reflector, strategy=spec,
        initial_state={"value": seed_source.strip() + "\n"},
        self_verify=False, solved_threshold=1.1, **evolve_kwargs)
    value = result.rendered or seed_source.strip() + "\n"

    seed_passed, seed_total, seed_score = evaluate_source(
        seed_source, tests, benchmark, function_name=function_name)
    evolved_passed, evolved_total, evolved_score = evaluate_source(
        value, tests, benchmark, function_name=function_name)
    report = SelfModifyReport(
        seed_source=seed_source.strip() + "\n", value=value,
        patch=patch_source(seed_source, value),
        seed_tests=seed_passed, evolved_tests=evolved_passed,
        seed_benchmark=seed_score, evolved_benchmark=evolved_score,
        gain=evolved_score - seed_score)
    try:
        result.meta = {
            "seed_tests": seed_passed, "evolved_tests": evolved_passed,
            "seed_benchmark": seed_score, "evolved_benchmark": evolved_score,
            "gain": evolved_score - seed_score,
        }
    except AttributeError:
        pass
    return report, result


def _strip_fence(
    reflector: Callable[[str, Task, str, float], Optional[str]],
) -> Callable[[str, Task, str, float], Optional[str]]:
    """Models reply with the candidate wrapped in a ```python fence (often with
    prose around it); pull the fenced block out so the gated source is what
    reaches the artifact."""
    def wrapped(rendered: str, task: Task, output: str,
                reward: float) -> Optional[str]:
        reply = reflector(rendered, task, output, reward)
        if not reply:
            return reply
        text = str(reply).strip()
        fenced = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
        return fenced.group(1).strip() if fenced else text
    return wrapped


def _source_spec(seed_source: str, function_name: str) -> SlotSpec:
    """The single-slot strategy the source text rides on, with the description
    ``slot_reflector`` needs to drive the model-backed proposer."""
    from .strategies import SingleSlot

    class _SelfEditSlot(SingleSlot):
        def describe(self) -> str:
            return (
                "The artifact is the source of one module-level function "
                f"`{function_name}`. Rewrite the source to pass the tests and "
                "beat the benchmark. No imports, dunder access, or interpreter "
                "calls. Reply with the complete source in one ```python fence "
                "(the fence is stripped for you).")

    return _SelfEditSlot(initial_value=seed_source.strip() + "\n")
