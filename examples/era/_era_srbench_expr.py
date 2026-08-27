"""The answer format for the ERA LLM-SRBench task, and the metrics it is scored by.

A candidate on this task does not return predictions. It returns a **closed-form
expression** -- a string in the problem's own variable names -- and this module
is the only thing that turns that string into numbers. Two consequences, and
both of them are the reason it is a separate module:

* **It is the benchmark's definition of an answer.** LLM-SRBench is equation
  discovery, not regression: a nearest-neighbour table that predicts the test
  set perfectly has not discovered anything. The grammar here admits arithmetic,
  powers and a fixed list of elementary functions, and nothing else -- no
  comparisons, no indexing, no calls outside the list -- so what comes back is
  an equation or it is rejected.
* **It is the boundary the test split hides behind.** The candidate is handed
  the training data only. The held-out points are touched *after* ``discover``
  has returned, and they are only ever touched by the evaluator here, walking an
  AST it has already validated. A candidate cannot observe them, because the
  thing it hands back is not code that runs -- it is a formula this module
  interprets.

Imported by the host and loaded by path inside the sandbox, exactly as
``_era_integrals.py`` is by the integration runner, so both sides score with the
same code. numpy and the standard library only.

Metrics are the benchmark's own (arXiv:2504.10415, Sec. 3):

    Acc_tau = 1(max_i |(y_hat_i - y_i) / y_i| <= tau),
    NMSE    = sum_i (y_hat_i - y_i)^2 / sum_i (y_i - mean(y))^2

with ``tau = 0.1`` in the paper's tables and here.
"""

from __future__ import annotations

import ast
import math
import warnings
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np


#: Elementary functions a returned equation may call. Chosen to cover every
#: function that appears in the benchmark's own ground-truth expressions (sin,
#: cos, exp, log, sqrt, Abs, tanh) plus the obvious neighbours, and to stop
#: there. `where`, `heaviside` and the comparison operators are deliberately
#: absent: a piecewise answer with enough branches is a lookup table, which is
#: the one thing an equation-discovery benchmark must not accept.
FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "log1p": np.log1p,
    "sqrt": np.sqrt,
    "cbrt": np.cbrt,
    "abs": np.abs,
    "Abs": np.abs,
    "sign": np.sign,
}

#: Constants an equation may name.
CONSTANTS: Dict[str, float] = {"pi": math.pi, "e": math.e, "E": math.e}

#: The NMSE at which a problem is called solved to the storage precision of the
#: benchmark itself. The published samples are float32, so a ground-truth
#: expression re-evaluated on them lands at NMSE ~1e-13; a cap of 12 digits
#: keeps a lucky rounding from outscoring the truth.
DIGIT_CAP = 12.0

#: The paper's tolerance for Acc_tau.
TOLERANCE = 0.1

#: Upstream's cap on how many free constants an answer may carry
#: (`methods/llmsr/searcher.py`: ``MAX_NPARAMS = 10``, and
#: ``minimize(loss, [1.0]*MAX_NPARAMS, method='BFGS')``). It is not decoration:
#: a candidate that wants to interpolate with a twenty-term basis cannot, because
#: there are only ten constants to spend.
MAX_NPARAMS = 10

#: The function a program-format answer must define, with upstream's signature.
PROGRAM_ENTRY = "equation"

#: What a program-format answer may import. Upstream gates nothing at all -- it
#: `exec`s whatever the model wrote -- so this is stricter, but only against
#: accidents: the sandbox is the boundary either way, and nothing a
#: symbolic-regression answer legitimately does needs more than these.
PROGRAM_IMPORTS = {"numpy", "math"}

_PROGRAM_FORBIDDEN = {"eval", "exec", "compile", "open", "input", "globals",
                      "locals", "vars", "getattr", "setattr", "delattr"}


class ProgramError(ValueError):
    """The returned source is not a usable ``equation`` program."""


def validate_program(source: Any, variables: Sequence[str],
                     max_length: int = 8000) -> ast.Module:
    """Parse a program-format answer and refuse what should never run.

    This is the **upstream-aligned** answer format: a Python function body, free
    to branch, loop and call numpy, with its constants left as ``params[i]`` for
    the harness to fit. It is far more permissive than
    :func:`validate_expression`, and deliberately so -- the restricted grammar is
    this port's own tightening, and a comparison with the paper's numbers has to
    allow what the paper allowed.

    The cost is real and is stated rather than hidden: in this format the
    candidate's own code runs at scoring time, on the held-out **inputs** (never
    the targets). Expression format's guarantee -- that candidate code and
    held-out data are never both live -- does not hold here. Upstream has the
    same property; it calls `lambda_fn(X_id)` on exactly this kind of object.
    """
    if not isinstance(source, str):
        raise ProgramError(
            f"expected program source as a string, got {type(source).__name__}")
    text = source.strip()
    if not text:
        raise ProgramError("empty program")
    if len(text) > max_length:
        raise ProgramError(f"program length {len(text)} exceeds {max_length}")
    if "\x00" in text:
        raise ProgramError("program contains a NUL byte")
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise ProgramError(f"SyntaxError: {exc.msg} at line {exc.lineno}") from None

    entry = next((node for node in tree.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == PROGRAM_ENTRY), None)
    if entry is None:
        raise ProgramError(f"missing {PROGRAM_ENTRY} function")
    names = [arg.arg for arg in entry.args.args]
    expected = list(variables) + ["params"]
    if names != expected:
        raise ProgramError(
            f"{PROGRAM_ENTRY} takes {names}, expected {expected}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in PROGRAM_IMPORTS:
                    raise ProgramError(f"import {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if not node.module or node.module.split(".")[0] not in PROGRAM_IMPORTS:
                raise ProgramError(f"import from {node.module!r} is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ProgramError(f"dunder attribute {node.attr!r} is not allowed")
        elif isinstance(node, ast.Name) and node.id in _PROGRAM_FORBIDDEN:
            raise ProgramError(f"name {node.id!r} is not allowed")
    return tree


def compile_program(source: str, variables: Sequence[str]):
    """Turn a validated program-format answer into ``f(columns, params)``."""
    tree = validate_program(source, variables)
    namespace: Dict[str, Any] = {"np": np, "numpy": np, "math": math}
    code = compile(tree, "<equation>", "exec")
    exec(code, namespace)  # noqa: S102 - the gate above and the sandbox around it
    entry = namespace.get(PROGRAM_ENTRY)
    if not callable(entry):
        raise ProgramError(f"{PROGRAM_ENTRY} is not callable")

    def call(x: np.ndarray, params: np.ndarray) -> np.ndarray:
        columns = [np.asarray(x[:, i], dtype=np.float64)
                   for i in range(len(variables))]
        with np.errstate(all="ignore"):
            value = np.asarray(entry(*columns, params), dtype=np.float64)
        if value.ndim == 0:
            value = np.full(x.shape[0], float(value))
        if value.shape != (x.shape[0],):
            raise ProgramError(
                f"{PROGRAM_ENTRY} produced shape {value.shape}, "
                f"expected ({x.shape[0]},)")
        return value

    return call


def fit_program(call, x: np.ndarray, y: np.ndarray,
                n_params: int = MAX_NPARAMS) -> np.ndarray:
    """Fit an answer's constants the way upstream fits them, and no better.

    Transcribed from `methods/llmsr/searcher.py`::

        def loss(params):
            y_pred = equation(*X, params)
            return np.mean((y_pred - outputs) ** 2)
        result = minimize(loss, [1.0]*MAX_NPARAMS, method='BFGS')

    One BFGS run, from all ones, no restarts. Keeping that exactly is what makes
    this format's numbers comparable: fitting harder here would beat the paper's
    methods at constant recovery rather than at equation discovery.
    """
    from scipy.optimize import minimize  # lazy: only this format needs scipy

    def loss(params: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            try:
                prediction = call(x, params)
            except Exception:
                return float(np.inf)
            value = float(np.mean((prediction - y) ** 2))
        return value if math.isfinite(value) else float(np.inf)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(loss, [1.0] * n_params, method="BFGS")
    return np.asarray(result.x, dtype=np.float64)

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)


class ExpressionError(ValueError):
    """The returned string is not an equation this task accepts."""


def validate_expression(expression: str, variables: Sequence[str],
                        max_length: int = 4000) -> ast.Expression:
    """Parse an answer and refuse anything that is not a closed form.

    Every rejection is a scored zero for that problem rather than a failure of
    the program: a method that emits nonsense for one problem and a good
    equation for the next has earned the second one.
    """
    if not isinstance(expression, str):
        raise ExpressionError(
            f"expected a string expression, got {type(expression).__name__}")
    text = expression.strip()
    if not text:
        raise ExpressionError("empty expression")
    if len(text) > max_length:
        raise ExpressionError(f"expression length {len(text)} exceeds {max_length}")
    if "\x00" in text:
        raise ExpressionError("expression contains a NUL byte")
    # `^` is what a scientist writes and what sympy prints for a power; Python
    # reads it as xor, which the node whitelist would then reject with a message
    # about BitXor. Rewriting it is the one liberty taken with the text.
    text = text.replace("^", "**")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"SyntaxError: {exc.msg}") from None

    allowed_names = set(variables) | set(CONSTANTS)
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(
                f"{type(node).__name__} is not allowed in an equation")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ExpressionError("only direct calls to allowed functions")
            if node.func.id not in FUNCTIONS:
                raise ExpressionError(f"unknown function {node.func.id!r}")
            if node.keywords or len(node.args) != 1:
                raise ExpressionError(
                    f"{node.func.id}() takes exactly one positional argument")
        elif isinstance(node, ast.Name):
            if node.id in FUNCTIONS:
                continue
            if node.id not in allowed_names:
                raise ExpressionError(
                    f"unknown symbol {node.id!r}; the variables are "
                    f"{', '.join(variables)}")
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise ExpressionError("only numeric constants are allowed")
    return tree


def evaluate_expression(expression: str, variables: Sequence[str],
                        x: np.ndarray) -> np.ndarray:
    """Evaluate a validated expression on ``x`` (shape ``(n, len(variables))``).

    Overflow, division by zero and the log of a negative are ordinary events for
    a wrong equation, so they are silenced into ``inf``/``nan`` here and counted
    as non-finite points by :func:`score_predictions` rather than raised.
    """
    tree = validate_expression(expression, variables)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(variables):
        raise ExpressionError(
            f"data has shape {x.shape}, expected (n, {len(variables)})")
    namespace: Dict[str, Any] = dict(FUNCTIONS)
    namespace.update(CONSTANTS)
    for index, name in enumerate(variables):
        namespace[name] = x[:, index]
    code = compile(tree, "<equation>", "eval")
    with np.errstate(all="ignore"):
        # `{"__builtins__": {}}` and a whitelisted namespace, over an AST this
        # module has already walked. The AST is the guarantee; this is the belt.
        value = eval(code, {"__builtins__": {}}, namespace)  # noqa: S307
        result = np.asarray(value, dtype=np.float64)
        if result.ndim == 0:  # a constant equation is a legitimate answer
            result = np.full(x.shape[0], float(result))
    if result.shape != (x.shape[0],):
        raise ExpressionError(
            f"equation produced shape {result.shape}, expected ({x.shape[0]},)")
    return result


def score_predictions(prediction: np.ndarray, truth: np.ndarray) -> Dict[str, Any]:
    """The benchmark's metrics for one problem, plus what this port adds.

    ``nmse`` and ``acc`` count a non-finite prediction as a failed point, which
    is *not* what upstream's ``compute_output_base_metrics`` does -- it drops
    the non-finite points and scores the rest. Both are reported: ``nmse`` is
    this port's rule and what the search optimises, ``nmse_upstream`` is the
    rule the paper's tables were produced under, so a number here can be put
    beside a number there. The deviation is deliberate: an equation with a pole
    inside the test range should not be rewarded for the points either side of
    it, and the search would otherwise learn to place poles.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    finite = np.isfinite(prediction)
    n_points = int(truth.size)
    n_finite = int(finite.sum())
    spread = float(np.sum((truth - truth.mean()) ** 2))

    def _nmse(pred: np.ndarray, obs: np.ndarray) -> float:
        if obs.size == 0:
            return math.inf
        denominator = float(np.sum((obs - obs.mean()) ** 2))
        if denominator <= 0.0:
            return math.inf
        return float(np.sum((pred - obs) ** 2) / denominator)

    if n_finite:
        upstream = _nmse(prediction[finite], truth[finite])
    else:
        upstream = math.inf

    # Acc_tau divides by the target, so a target of exactly zero leaves the
    # paper's formula undefined at that point and no equation -- the ground
    # truth included -- can satisfy it. Five of LSR-Synth's 129 problems have
    # one, which caps Acc(0.1) there at 124/129. That is the published metric on
    # the published data, so it is counted and reported rather than patched.
    zero_targets = int(np.count_nonzero(truth == 0.0))
    if n_finite < n_points or spread <= 0.0:
        nmse = math.inf
        acc = 0
        max_relative = math.inf
    else:
        nmse = _nmse(prediction, truth)
        with np.errstate(all="ignore"):
            relative = np.abs((prediction - truth) / truth)
        relative = relative[np.isfinite(relative)]
        max_relative = float(relative.max()) if relative.size else math.inf
        acc = int(max_relative <= TOLERANCE and relative.size == n_points)
    return {
        "nmse": nmse,
        "nmse_upstream": upstream,
        "acc": acc,
        "digits": digits_of(nmse),
        "max_relative": max_relative,
        "points": n_points,
        "nonfinite_points": n_points - n_finite,
        "zero_targets": zero_targets,
    }


def digits_of(nmse: float) -> float:
    """``-log10(NMSE)``, clipped to ``[0, DIGIT_CAP]`` -- the search's signal.

    Acc_0.1 is an indicator per problem and NMSE spans twelve orders of
    magnitude, so neither is usable on its own as the number a tree search ranks
    nodes by: the first is flat almost everywhere, the second is dominated by
    whichever problem failed worst. This is the integrals task's move -- score
    the log, cap it at the precision of the data -- and it is monotone in NMSE,
    so it never disagrees with the metric it summarises.
    """
    value = float(nmse)
    if not math.isfinite(value) or value <= 0.0:
        return DIGIT_CAP if value == 0.0 else 0.0
    return max(0.0, min(DIGIT_CAP, -math.log10(value)))


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pool per-problem results into the numbers a run reports.

    Pooled over problems rather than averaged per shard, for the reason the
    sibling tasks pool theirs: the shards are equal-sized draws from one
    benchmark, and a mean of means would weight a shard rather than a problem.
    """
    scored = [row for row in rows if row.get("id") is not None]
    if not scored:
        return {
            "mean_digits": None,
            "median_nmse": None,
            "acc_0.1": None,
            "problems": 0,
        }

    def _median(values: List[float]) -> float:
        values = sorted(values)
        middle = len(values) // 2
        if len(values) % 2:
            return values[middle]
        return 0.5 * (values[middle - 1] + values[middle])

    digits = [float(row["id"]["digits"]) for row in scored]
    nmse = [float(row["id"]["nmse"]) for row in scored]
    nmse_upstream = [float(row["id"]["nmse_upstream"]) for row in scored]
    acc = [int(row["id"]["acc"]) for row in scored]
    ood_rows = [row["ood"] for row in scored if row.get("ood") is not None]

    payload: Dict[str, Any] = {
        "mean_digits": sum(digits) / len(digits),
        "median_nmse": _median(nmse),
        "median_nmse_upstream": _median(nmse_upstream),
        "acc_0.1": sum(acc) / len(acc),
        "solved": sum(1 for value in digits if value >= 6.0),
        "problems": len(scored),
        "failed": sum(1 for row in rows if row.get("id") is None),
    }
    if ood_rows:
        payload.update({
            "ood_mean_digits": sum(float(r["digits"]) for r in ood_rows) / len(ood_rows),
            "ood_median_nmse": _median([float(r["nmse"]) for r in ood_rows]),
            "ood_acc_0.1": sum(int(r["acc"]) for r in ood_rows) / len(ood_rows),
            "ood_problems": len(ood_rows),
        })
    return payload
