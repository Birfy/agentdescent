"""Score an LLM-SRBench run's **symbolic accuracy**, the paper's third metric.

    python -m tools.score_symbolic_accuracy RESULT.json --provider claude \\
        --model glm-5.2 --yes

The two metrics an LLM-SRBench run reports on its own -- ``Acc(0.1)`` and NMSE --
are data fidelity: they ask whether the answer *predicts* the held-out samples.
Symbolic accuracy asks whether it **is the equation**, and it is the column that
separates discovering a law from interpolating one. Measured on this port's own
answers, that distinction is not academic: the whole-category protocol returns
nine-term library fits that reach NMSE 1e-13 on truths of two terms.

It is a separate tool rather than part of a run for three reasons. It needs the
**ground truth**, which must never come near the search. It needs a model call
per problem, which a sandboxed evaluator cannot make. And it should be
re-runnable with a different judge against an unchanged result file, because the
judge is part of the metric.

How faithful this is
--------------------
The paper (arXiv:2504.10415, §3) specifies the metric but its prompt lives in an
appendix figure that the arXiv HTML render does not carry, so the prompt below
is written to the paper's description rather than transcribed from it:

    "GPT-4o evaluates mathematical equivalence by comparing the symbolic form of
    the predicted hypothesis versus the ground-truth equation after removing
    parameters and constants."

Two further deviations, both stated in the output file:

* **The judge is not GPT-4o.** It is whatever ``--model`` names. A different
  judge is a different metric, so a number from here sits beside the paper's
  with that said, not inside its table.
* **Not every problem can be scored.** 36 ``chem_react`` and 44 ``phys_osc``
  ground truths are damaged in the published copy this port reads -- a mangled
  constant, and symbolic templates with unbound parameters -- so they are
  reported as ``not_scorable`` rather than guessed at. That leaves 49 of 129
  LSR-Synth problems and all 111 LSR-Transform problems.

Before the judge is asked, a **deterministic check** runs: both sides have their
numeric literals replaced by free symbols and sympy is asked whether the
difference simplifies to zero. Where that settles the question it is used
directly and the judge is not called -- an exact answer beats a probable one,
and it puts a floor under the metric that does not depend on any model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentdescent.agents import Usage  # noqa: E402
from examples._common import completion_for, confirm  # noqa: E402


JUDGE_PROMPT = """You are judging whether two mathematical expressions describe \
the same equation.

Ignore all numeric constants and fitted parameters entirely. Two expressions are \
EQUIVALENT if one can be turned into the other by choosing different values for \
its constants, and by ordinary algebra and trigonometric identities. They are \
NOT equivalent if they have different functional structure -- different terms, \
different functions of the variables, a different arrangement of the variables.

Variables: {variables}

Ground-truth equation:
  {truth}

Discovered equation:
  {answer}

Answer with exactly one word on the first line, EQUIVALENT or DIFFERENT, then one \
sentence of justification on the second line."""


def normalise(text: str, variables: List[str],
              fitted: Optional[List[float]] = None) -> str:
    """Strip the notation the two sides do not share.

    Four differences, none of them about the mathematics:

    * ``A(t)`` is sympy's function-call form for a state variable and the same
      symbol is a plain column here;
    * ``^`` is a power in one convention and xor in the other;
    * a program-format answer arrives as source, so its ``return`` line is what
      carries the equation; and
    * a program-format answer leaves its constants as ``params[i]`` holes.
      **Those are substituted with the values the harness actually fitted**,
      because a skeleton compared against a concrete ground truth can only be
      guessed at, while `c*sqrt(1 - (m_0/m)**2)*(-1.0000)` against
      `-c*sqrt(1 - m_0**2/m**2)` is a question arithmetic can answer. Without
      fitted values the holes become free symbols and the judge decides.
    """
    text = text.strip()
    if "def equation" in text:
        returns = [line.strip()[len("return "):]
                   for line in text.splitlines() if line.strip().startswith("return ")]
        if returns:
            text = returns[-1]
    for name in variables:
        text = text.replace(f"{name}(t)", name)

    def hole(match: "re.Match[str]") -> str:
        index = int(match.group(1))
        if fitted is not None and index < len(fitted):
            return f"({fitted[index]!r})"
        return f"__p{index}"

    text = re.sub(r"params\s*\[\s*(\d+)\s*\]", hole, text)
    return text.replace("^", "**").replace("np.", "").strip()


def _free_of_constants(expression: str, variables: List[str]):
    """The expression with every numeric literal replaced by a free symbol."""
    import sympy

    counter = {"n": 0}

    def swap(match: "re.Match[str]") -> str:
        counter["n"] += 1
        return f"__c{counter['n']}"

    # A numeric literal, but not the digits inside an identifier such as `x1`.
    masked = re.sub(r"(?<![A-Za-z_0-9.])\d+\.?\d*(?:[eE][-+]?\d+)?", swap, expression)
    symbols = {name: sympy.Symbol(name, positive=True) for name in variables}
    symbols.update({f"__c{i}": sympy.Symbol(f"__c{i}")
                    for i in range(1, counter["n"] + 1)})
    return sympy.sympify(masked, locals=symbols), symbols


def numerically_equivalent(truth: str, answer: str, variables: List[str],
                           samples: int = 240, tolerance: float = 1e-6,
                           seed: int = 0) -> Optional[bool]:
    """Do the two expressions agree as *functions*, at points nobody fitted on?

    Once a program-format answer has its fitted constants substituted back in,
    both sides are concrete expressions, and two concrete expressions that agree
    to eight digits at a few hundred scattered points are the same function.
    That is mathematical equivalence, which is what the paper's metric is after,
    and it settles the cases where the search recovered the equation outright.

    Sampled over a wide positive range rather than over the training domain: an
    answer that agrees only where it was fitted is exactly what this must not
    accept. ``None`` when the comparison cannot be made -- an unparsable side, or
    too few points where both are finite -- so the judge decides rather than the
    problem scoring a miss.
    """
    try:
        import numpy
        import sympy
        symbols = [sympy.Symbol(name, positive=True) for name in variables]
        left = sympy.sympify(truth, locals=dict(zip(variables, symbols)))
        right = sympy.sympify(answer, locals=dict(zip(variables, symbols)))
        f_left = sympy.lambdify(symbols, left, "numpy")
        f_right = sympy.lambdify(symbols, right, "numpy")
    except Exception:
        return None
    rng = numpy.random.default_rng(seed)
    columns = [rng.uniform(0.3, 3.0, size=samples) for _ in variables]
    try:
        with numpy.errstate(all="ignore"):
            a = numpy.asarray(f_left(*columns), dtype=float) * numpy.ones(samples)
            b = numpy.asarray(f_right(*columns), dtype=float) * numpy.ones(samples)
    except Exception:
        return None
    usable = numpy.isfinite(a) & numpy.isfinite(b)
    if usable.sum() < samples // 4:
        return None
    scale = numpy.maximum(numpy.abs(a[usable]), 1e-12)
    return bool(numpy.max(numpy.abs(a[usable] - b[usable]) / scale) < tolerance)


def deterministic_verdict(truth: str, answer: str,
                          variables: List[str]) -> Optional[bool]:
    """``True`` where equivalence can be *shown*, ``None`` where it cannot.

    A **conservative accelerator, not a second metric**. Two routes, both of
    which only ever return a positive:

    * **symbolic** -- mask every numeric literal as its own free symbol and ask
      sympy whether the difference, or the ratio, collapses. This catches
      `1/(2*sqrt(pi))` against `1/sqrt(4*pi)`.
    * **numeric** -- with the fitted constants substituted back in, check whether
      the two agree as functions at a few hundred scattered points.

    Anything neither route settles goes to the judge. That is not a gap in the
    implementation: an answer with the right structure and the wrong constants is
    equivalent under the paper's definition and cannot be shown so by either
    route, which is why the paper uses a model at all.
    """
    numeric = numerically_equivalent(truth, answer, variables)
    if numeric is True:
        return True
    try:
        import sympy
        left, _ = _free_of_constants(truth, variables)
        right, _ = _free_of_constants(answer, variables)
    except Exception:
        return None
    try:
        if sympy.simplify(left - right) == 0:
            return True
        ratio = sympy.simplify(left / right)
        if ratio.is_number or not ratio.free_symbols & {
                sympy.Symbol(v, positive=True) for v in variables}:
            return True  # the same equation up to an overall constant
    except Exception:
        return None
    return None


def judge(complete, truth: str, answer: str, variables: List[str]) -> Tuple[bool, str]:
    reply = complete(JUDGE_PROMPT.format(
        variables=", ".join(variables), truth=truth, answer=answer)).strip()
    head = reply.splitlines()[0].strip().upper() if reply else ""
    return head.startswith("EQUIVALENT"), reply[:400]


def scorable(truth: str, variables: Optional[List[str]] = None) -> bool:
    """Is this ground truth intact enough to be compared with anything?

    Two defects in the published copy disqualify a problem, and neither is a
    property of any answer, so a problem carrying one is *not* a miss:

    * the mangled constant (``0.189…_z``), which does not parse; and
    * the symbolic template, whose ``F0``/``beta``/``omega0`` are free symbols
      with no values attached -- so the "ground truth" names quantities the data
      does not contain and no answer could mention.

    The second is caught by checking that every free symbol in the truth is one
    of the problem's own variables. That needs the variable list; without one
    only the first check runs, which is why callers pass it.
    """
    if not truth or not truth.strip():
        return False
    if re.search(r"\d_[a-zA-Z]", truth):
        return False
    if variables is None:
        return True
    try:
        import sympy
        expression = sympy.sympify(normalise(truth, list(variables)))
    except Exception:
        return False
    known = {str(sympy.Symbol(name)) for name in variables} | {"pi", "E"}
    return all(str(symbol) in known for symbol in expression.free_symbols)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, help="a --per-problem result file")
    parser.add_argument("--provider", default="claude", choices=("claude", "openai", "glm"))
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the first N scorable problems")
    parser.add_argument("--output", type=Path,
                        help="where to write the scored file (default: alongside)")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    rows = payload.get("per_problem") or []
    candidates = [row for row in rows
                  if scorable(row.get("gt_expression", ""),
                              list(row.get("input_vars") or []))]
    print(f"Metric   : symbolic accuracy (arXiv:2504.10415 §3), judge={args.model}")
    print(f"Problems : {len(rows)} in the run, {len(candidates)} with an intact "
          f"ground truth, {len(rows) - len(candidates)} not scorable")
    if args.dry_run:
        print("[dry-run] no judge was called.")
        return 0
    if not confirm(args):
        return 0

    usage = Usage()
    complete = completion_for(args, usage=usage, max_tokens=args.max_tokens,
                              timeout=args.timeout, temperature=args.temperature)
    scored: List[Dict[str, Any]] = []
    asked = settled = 0
    for index, row in enumerate(candidates):
        if args.limit and index >= args.limit:
            break
        variables = list(row.get("input_vars") or [])
        truth = normalise(row["gt_expression"], variables)
        answer = normalise((row.get("best") or {}).get("equation") or "", variables,
                           fitted=row.get("fitted_params"))
        verdict: Dict[str, Any] = {
            "problem_id": row["problem_id"], "subset": row["subset"],
            "gt_expression": row["gt_expression"],
            "answer": answer[:400],
        }
        if not answer:
            verdict.update(equivalent=False, decided_by="no answer")
        else:
            proof = deterministic_verdict(truth, answer, variables)
            if proof is True:
                settled += 1
                verdict.update(equivalent=True, decided_by="sympy")
            else:
                asked += 1
                equivalent, reply = judge(complete, truth, answer, variables)
                verdict.update(equivalent=equivalent, decided_by="judge",
                               judge_reply=reply)
        scored.append(verdict)
        print(f"[{len(scored):3d}/{len(candidates)}] {row['problem_id']:34s} "
              f"{'EQUIVALENT' if verdict['equivalent'] else 'different':11s} "
              f"({verdict['decided_by']})", flush=True)

    by_subset: Dict[str, List[Dict[str, Any]]] = {}
    for verdict in scored:
        by_subset.setdefault(verdict["subset"], []).append(verdict)
    summary = {
        "judge_model": args.model,
        "judge_is_not_gpt4o": ("The paper uses GPT-4o. A different judge is a "
                               "different metric; this number sits beside the "
                               "paper's rather than inside its table."),
        "prompt_provenance": ("Written to the paper's §3 description -- compare "
                              "symbolic form after removing parameters and "
                              "constants -- because the appendix figure holding "
                              "the original prompt is not in the arXiv HTML "
                              "render."),
        "problems_in_run": len(rows),
        "scorable": len(candidates),
        "not_scorable": len(rows) - len(candidates),
        "scored": len(scored),
        "decided_by_sympy": settled,
        "decided_by_judge": asked,
        "symbolic_accuracy": (
            sum(1 for v in scored if v["equivalent"]) / len(scored)) if scored else None,
        "by_subset": {
            name: {"n": len(group),
                   "symbolic_accuracy": sum(1 for v in group if v["equivalent"]) / len(group)}
            for name, group in sorted(by_subset.items())},
        "judge_usage": {"calls": usage.calls, "tokens": usage.total_tokens},
    }
    out = args.output or args.result.with_name(args.result.stem + "-symbolic.json")
    out.write_text(json.dumps({"summary": summary, "verdicts": scored},
                              indent=2, default=str) + "\n", encoding="utf-8")
    accuracy = summary["symbolic_accuracy"]
    print(f"completed: SA = {100 * accuracy:.1f}% over {len(scored)} scorable problems "
          f"({settled} settled by sympy, {asked} by the judge), output={out}"
          if accuracy is not None else "completed: nothing scorable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
