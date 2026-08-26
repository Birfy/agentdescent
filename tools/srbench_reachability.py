"""Measure the **ceiling** an LLM-SRBench answer format puts on a run.

    python -m tools.srbench_reachability --dataset lsr_transform \
        --answer-format program

A run that scores 37% has two possible explanations, and they call for opposite
work: either the search is not finding equations it could have found, or the
*answer format* cannot express them at the accuracy the metric demands. This
tool settles which, by handing the grader the one thing a run never sees -- the
ground truth -- rewritten in the format the run had to answer in, and scoring
it exactly as the run's answers were scored.

Three numbers per problem, and the gaps between them are the diagnosis:

``expression``
    the ground truth with its constants intact, evaluated by this port's AST
    walker. This is the data check: anything below ``Acc(0.1) = 1`` here means
    the samples and the published truth disagree, which is a bug in the loader,
    not a property of the format.
``program``
    the ground truth with its **fitted coefficients** replaced by ``params[i]``
    and filled in by the grader's own single ``BFGS`` from all ones -- the
    benchmark's protocol, the one upstream uses. A drop from ``expression`` to
    here is the *fitting protocol's* ceiling, and it is paid by every method
    measured under it, this port and LLM-SR alike.
``restarts``
    the same program, fitted by many BFGS runs from scattered starts. A drop
    from here to ``program`` is the single-start optimiser specifically; a drop
    that survives restarts is the parameterisation itself.

Exponents are left literal. ``x**2`` is structure, not a constant to fit, and
holing it out would measure a harder problem than the benchmark poses.

The count that matters is how many problems reach ``Acc(0.1) = 1`` in the
``program`` column: that is the highest score any method answering in that
format can obtain, and a run well below it is search-limited rather than
format-limited.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from examples.era import _era_srbench as sr
from examples.era import _era_srbench_expr as ex

#: Values that carry structure rather than a magnitude. Holing these out would
#: hand the optimiser a sign or an identity to discover, which is not what the
#: benchmark's `params` are for.
STRUCTURAL = (0, 1, -1)


def hole_coefficients(gt_expression: str, variables: Sequence[str],
                      max_params: int = ex.MAX_NPARAMS) -> Tuple[str, int]:
    """Rewrite a ground truth as `equation(..., params)`, upstream's format.

    Every distinct numeric coefficient becomes a ``params[i]``; exponents and
    the structural values stay literal. A truth with no coefficient at all still
    gets one, because upstream's format always fits *something* and a form with
    no free scale is a stricter claim than the benchmark makes.
    """
    import sympy as sp

    # The problem's own names first: `beta`, `gamma`, `zeta` and friends are
    # sympy *functions*, and sympifying without binding them turns a variable
    # into a function object three lines later.
    local = {name: sp.Symbol(name) for name in variables}
    expr = sp.sympify(gt_expression, locals=local)

    holes: Dict[Any, Any] = {}
    order: List[Any] = []

    def walk(node):
        if isinstance(node, sp.Pow):                 # base is fitted, exponent is not
            return sp.Pow(walk(node.base), node.exp, evaluate=False)
        if isinstance(node, sp.Number):
            if node in (sp.Integer(v) for v in STRUCTURAL):
                return node
            if node not in holes:
                if len(order) >= max_params:
                    return node
                holes[node] = sp.Symbol(f"__P{len(order)}")
                order.append(node)
            return holes[node]
        return node.func(*[walk(a) for a in node.args]) if node.args else node

    body = sp.printing.pycode(walk(expr)).replace("math.", "np.")
    count = len(order)
    for index in range(count):
        body = body.replace(f"__P{index}", f"params[{index}]")
    if count == 0:
        body, count = f"params[0]*({body})", 1
    signature = ", ".join(list(variables) + ["params"])
    return f"def equation({signature}):\n    return {body}\n", count


def fit_with_restarts(call, x: np.ndarray, y: np.ndarray, *,
                      n_params: int = ex.MAX_NPARAMS, restarts: int = 24,
                      seed: int = 0) -> np.ndarray:
    """The same objective as `fit_program`, given many starts instead of one."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)

    def loss(params: Sequence[float]) -> float:
        with np.errstate(all="ignore"):
            try:
                predicted = call(x, np.asarray(params, dtype=np.float64))
            except Exception:
                return 1e18
        predicted = np.asarray(predicted, dtype=np.float64)
        if predicted.shape != y.shape or not np.all(np.isfinite(predicted)):
            return 1e18
        return float(np.mean((predicted - y) ** 2))

    starts = [np.ones(n_params)]
    for scale in (0.3, 1.0, 3.0, 10.0):
        starts.extend(rng.normal(0.0, scale, n_params)
                      for _ in range(max(1, restarts // 4)))
    best, best_params = np.inf, np.ones(n_params)
    for start in starts:
        try:
            found = minimize(loss, start, method="BFGS")
        except Exception:
            continue
        if found.fun < best:
            best, best_params = found.fun, np.asarray(found.x, dtype=np.float64)
    return best_params


def _digits(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(ex.score_predictions(predicted, target)["digits"])


def measure(problem, samples, *, train_points: int, restarts: int,
            seed: int) -> Dict[str, Any]:
    """The three columns for one problem, and never let one truth stop the sweep."""
    variables = list(problem.input_vars)
    row: Dict[str, Any] = {
        "problem_id": problem.problem_id,
        "subset": problem.subset,
        "variables": len(variables),
        "gt_expression": problem.gt_expression,
    }
    test_x = np.asarray(samples["test_x"], dtype=np.float64)
    test_y = np.asarray(samples["test_y"], dtype=np.float64)
    train_x = np.asarray(samples["train_x"], dtype=np.float64)[:train_points]
    train_y = np.asarray(samples["train_y"], dtype=np.float64)[:train_points]

    try:
        row["expression_digits"] = round(
            _digits(ex.evaluate_expression(problem.gt_expression, variables, test_x),
                    test_y), 4)
    except Exception as exc:
        row["expression_digits"] = None
        row["expression_error"] = f"{type(exc).__name__}: {exc}"[:160]

    try:
        source, n_params = hole_coefficients(problem.gt_expression, variables)
        row["params"] = n_params
        row["program"] = source
        call = ex.compile_program(source, variables)
        single = ex.fit_program(call, train_x, train_y)
        row["program_digits"] = round(_digits(call(test_x, single), test_y), 4)
        many = fit_with_restarts(call, train_x, train_y, restarts=restarts, seed=seed)
        row["restart_digits"] = round(_digits(call(test_x, many), test_y), 4)
    except Exception as exc:
        row["program_digits"] = row["restart_digits"] = None
        row["program_error"] = f"{type(exc).__name__}: {exc}"[:160]
    return row


def _reached(value: Optional[float]) -> bool:
    """`Acc(0.1) = 1` is exactly `digits >= 1`: a worst relative error under 10%."""
    return value is not None and value >= 1.0


def summarise(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    return {
        "problems": total,
        "expression_reachable": sum(_reached(r.get("expression_digits")) for r in rows),
        "program_reachable": sum(_reached(r.get("program_digits")) for r in rows),
        "restart_reachable": sum(_reached(r.get("restart_digits")) for r in rows),
        "lost_to_single_start": sum(
            1 for r in rows
            if _reached(r.get("restart_digits")) and not _reached(r.get("program_digits"))),
        "lost_to_parameterisation": sum(
            1 for r in rows
            if _reached(r.get("expression_digits")) and not _reached(r.get("restart_digits"))),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="lsr_transform",
                        help="a subset name, `lsr_synth`, `lsr_transform` or `all`")
    parser.add_argument("--problems", type=int, default=0,
                        help="cap the sweep, stratified across subsets (0 = every problem)")
    parser.add_argument("--train-points", type=int, default=4000,
                        help="training rows the constants are fitted on")
    parser.add_argument("--restarts", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="", help="write the per-problem rows here")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    warnings.filterwarnings("ignore")
    rows = []
    for problem, samples in sr.load_catalogue(args.dataset, problems=args.problems,
                                              seed=args.seed):
        row = measure(problem, samples, train_points=args.train_points,
                      restarts=args.restarts, seed=args.seed)
        rows.append(row)
        print(f"{row['problem_id']:38s} expr={str(row.get('expression_digits')):>8}"
              f"  program={str(row.get('program_digits')):>8}"
              f"  restarts={str(row.get('restart_digits')):>8}", flush=True)
    summary = summarise(rows)
    print("\n" + json.dumps(summary, indent=1))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump({"dataset": args.dataset, "summary": summary,
                       "per_problem": rows}, handle, indent=1)
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
