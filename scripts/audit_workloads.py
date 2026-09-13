#!/usr/bin/env python3
"""What a workload *is*: how it is scored, how a row becomes a task, how its
errors are named.

One module, one direction. `audit_modes.py` used to hold the error-mode
functions and import its scoring primitives *from* `audit_phase0.py`, which
made the dependency circular -- and so the `Workload` object could not hold the
one field the diagnosis needs. The result was an **eighth** table keyed by
workload name, `ERROR_MODES`, which is exactly the structure collapsing the
first seven was meant to remove: a sixth workload could still be
half-registered, failing at `assess()` after a paid run had finished.

Inverting the dependency fixes it at the shape. Everything that is per-workload
now lives on one object here, `audit_phase0.py` is the Phase 0 *experiment*, and
a workload cannot be registered in pieces.

An **error mode** is the species Good--Turing counts: a coarse label for what a
person would have to fix, deliberately coarser than the record and coarser than
the task. Two answers that echo their question are the same bug however
different the questions were, and counting them as two would make the
improvement pool look like it was still learning when it had stopped.
"""

from __future__ import annotations

import ast
import hashlib
import os
import pathlib
import random
import re
import string
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import (Any, Callable, Dict, List, Optional, Sequence)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentdescent import Task  # noqa: E402
from agentdescent.audit import AuditRecord, AuditStore  # noqa: E402


def read_verdict(reply: str, *, on_unparsed: Optional[Callable[[], None]] = None
                 ) -> float:
    """A judge's YES/NO reply as a score.

    One copy. This was three -- `audit_phase0`, `audit_judge_repair` and
    `audit_evolve_judge` each had the same six lines, and the only thing any of
    them did differently was count the unparsed replies, which is what
    ``on_unparsed`` is for.

    **An unparseable reply is not a 0.** Scoring it wrong would blame the
    artifact for the judge's failure to answer the question it was asked.
    """
    head = (reply or "").strip().upper()
    if head.startswith("YES"):
        return 1.0
    if head.startswith("NO"):
        return 0.0
    if on_unparsed is not None:
        on_unparsed()
    return 1.0 if "YES" in head else 0.0


def refuse_to_overwrite(path: pathlib.Path, flag: str = "--out") -> pathlib.Path:
    """Stop before a run, not after it, if its report already exists.

    An experiment writes its report at the end, so a collision discovered there
    costs the whole run -- here, ten minutes of model calls. Checked up front it
    costs nothing.

    Refusing rather than auto-suffixing on purpose. A silently-numbered second
    file is easy not to notice, and the failure that matters is quoting one run
    while looking at another's numbers. `audit_phase0.py` put the workload in
    the name for the same reason; it did not stop two runs of the *same*
    workload on the same day, which is this.
    """
    if path.exists():
        raise SystemExit(
            f"{path} already exists -- a previous run wrote it.\n"
            f"Pass {flag} to write somewhere else, or move the old one aside. "
            f"Refusing rather than overwriting: an experiment result that "
            f"silently replaces another is the one mistake nobody can catch "
            f"later from the file.")
    return path


def resolved_records(path: Any) -> List[AuditRecord]:
    """Every resolved pair in an audit JSONL.

    `AuditStore` already reads the file and already applies the last-occurrence
    -per-`record_id` rule, so the two hand-rolled readers this replaces were
    reimplementing the store's own parser next to it -- verified
    record-for-record identical before they were deleted. What is left is the
    filter, which is the only thing any caller wanted on top.
    """
    return [r for r in AuditStore(str(path)).all()
            if r.oracle_score is not None]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}


def normalize(text: str) -> str:
    """The SQuAD/HotpotQA normalisation: lowercase, strip punctuation and articles."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(w for w in text.split() if w not in _ARTICLES)


def exact_match(task: Task, output: str) -> float:
    """**The oracle** for the free-text workloads: the benchmark's own definition."""
    return 1.0 if normalize(output) == normalize(task.meta["gold"]) else 0.0


#: A number, with optional thousands separators and decimal part. Leading `$`
#: and trailing `%` are outside the match on purpose -- the oracle compares
#: quantities, and `$18` and `18` are the same quantity.
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def final_number(text: str) -> Optional[float]:
    """The last number in ``text``, or ``None``.

    "The last number in the completion" is the GSM8K convention, and it is
    convention rather than truth: an answer ending ``18 (over 7 days)`` scores
    against 7. That is not a flaw to route around -- it is this workload's
    contribution to the :attr:`~agentdescent.audit.diagnose.Kind.AMBIGUOUS`
    bucket, where the judge is right and the oracle is wrong, and the two
    existing workloads produce that bucket only through name and date
    formatting. A third *kind* of ambiguity is most of why this workload is
    here.
    """
    found = _NUMBER.findall(text or "")
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:                          # e.g. a bare "1,,2"
        return None


def number_match(task: Task, output: str) -> float:
    """**The oracle** for GSM8K: the final number, compared as a quantity.

    Not `exact_match`: normalised string equality would score ``$18`` and
    ``18.00`` wrong against ``18``, and a workload whose oracle refuses its own
    correct answers measures the oracle, not the judge.
    """
    want = final_number(task.meta["gold"])
    got = final_number(output)
    if want is None or got is None:
        return 0.0
    return 1.0 if abs(want - got) < 1e-6 else 0.0


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(output: str) -> str:
    """The code in ``output``: the first fenced block, or the whole thing.

    A model asked for a function answers with prose around a fence about half
    the time. Executing the prose is a syntax error, which the oracle would
    score as a wrong answer -- so the oracle would be measuring markdown.
    """
    m = _FENCE.search(output or "")
    return (m.group(1) if m else (output or "")).strip()


#: Seconds a candidate's tests may run before it is called wrong. MBPP
#: solutions are small; a candidate that takes longer than this has hung.
CODE_TIMEOUT = 10.0


def run_tests(code: str, tests: Sequence[str], *,
              timeout: float = CODE_TIMEOUT) -> bool:
    """Execute ``code`` and then ``tests``, in a child process. All or nothing.

    .. warning:: **Process isolation, not a sandbox.**

       This runs model-authored Python with this user's permissions, exactly as
       :func:`agentdescent.runners.code_runner` does and with the same caveat:
       a trimmed environment, a scratch working directory and a hard timeout are
       not a security boundary. It is here because the alternative -- asking a
       model whether the code is correct -- is the thing being *measured*. Run
       it in a container for anything you would not run by hand.

    A timeout, a crash and a failed assert are all one answer: wrong. The
    distinction matters to whoever is fixing the candidate and not to an oracle,
    whose whole job is a bit.
    """
    program = code + "\n\n" + "\n".join(tests) + "\n"
    with tempfile.TemporaryDirectory() as work:
        path = os.path.join(work, "candidate.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(program)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
               "HOME": work, "TMPDIR": work,
               "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            done = subprocess.run([sys.executable, "-I", path], cwd=work,
                                  env=env, timeout=timeout,
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        except (subprocess.TimeoutExpired, OSError):
            return False
        return done.returncode == 0


def tests_pass(task: Task, output: str) -> float:
    """**The oracle** for MBPP: does the candidate pass the task's own asserts?

    The only oracle here that is not a comparison against a written answer, and
    the reason this workload exists: the judge is shown the *reference solution*
    and has to decide whether a differently-written candidate does the same
    thing, which it cannot settle by running anything. That is a judging task
    with real ambiguity in it, which the three numeric and short-answer
    workloads did not have.
    """
    tests = task.meta.get("tests") or []
    if not tests:
        return 0.0
    return 1.0 if run_tests(extract_code(output), tests) else 0.0



# ---------------------------------------------------------------------------
# Error modes
# ---------------------------------------------------------------------------

#: The oracle's own normalisation, under the name the mode functions use.
normalise = normalize

#: A mode function's second argument. It **is** the task, not a tuple built
#: from one: `task_index` already returns real `Task` objects, and the
#: `(question, reference, extra)` tuple this replaces dropped `meta["tests"]`
#: on the floor and gave slot `[2]` a different meaning per workload.
Context = Optional[Task]


# -- the predicates the hand-written rules are built from --------------------

def echoes_the_question(output: str, question: str) -> bool:
    return bool(output) and normalize(output) in normalize(question)


def far_shorter_than_reference(output: str, gold: str, ratio: float) -> bool:
    o, g = normalize(output), normalize(gold)
    return bool(o) and bool(g) and o != g and len(o) < ratio * len(g)


# -- one mode function per workload ------------------------------------------

def text_error_mode(record, task: Context) -> Optional[str]:
    """For the free-text workloads: HotpotQA, and BBH's non-label subtasks.

    Every branch here is a statement about *strings* -- length, containment,
    overlap with the question -- because that is the whole of what those two
    workloads' answers are.
    """
    if task is None:
        return None
    question, gold = task.prompt, task.meta.get("gold", "")
    out, ref = normalize(record.output), normalize(gold)
    if echoes_the_question(record.output, question):
        return "echoes-question"
    if out and ref and (out in ref or ref in out):
        return "substring-of-reference"
    if out and ref and len(out) < 0.6 * len(ref):
        return "far-shorter"
    if out and ref and len(out) > 1.6 * len(ref):
        return "far-longer"
    if not out:
        return "empty"
    return f"other:{record.task_id[:6]}"          # unclassified: its own mode


def gsm8k_error_mode(record, task: Context) -> Optional[str]:
    """For GSM8K, where the answer is a quantity and the output shows working.

    The modes are chosen so that the one this workload was added for --
    ``wrong-number-with-working``, the judge marking a wrong final number right
    because the derivation reads correctly -- is **separable** from the three
    ways of being wrong that the other workloads already cover. A mode function
    that lumped them would add labels and no information, which is the exact
    failure that saturated the pool at 0.0169.

    ``gold-not-final`` is the one to read as a criticism of the *oracle*: the
    reference number is in the output and the convention of taking the last
    number did not find it. Those belong in the AMBIGUOUS bucket and must not be
    "fixed" -- driving them out means teaching the judge to read the last number,
    which is what having a judge was supposed to avoid.
    """
    if task is None:
        return None
    gold, worked = task.meta.get("gold", ""), task.meta.get("worked")
    want = final_number(gold)
    got = final_number(record.output)
    if got is None:
        return "no-number"
    if want is not None and abs(want - got) < 1e-6:
        # The oracle and the judge can still disagree here -- the oracle reads
        # the last number and this record reached the mode function because the
        # two scores differ, so the disagreement is the judge's.
        return "judge-rejected-right-number"
    if want is not None and _mentions(record.output, want):
        return "gold-not-final"
    if want is not None and _is_rounding(want, got):
        return "rounded"
    if want is not None and _equivalent_fraction(record.output, want):
        return "equivalent-fraction"
    if _has_working(record.output, worked):
        return "wrong-number-with-working"
    return "wrong-number-bare"


#: ``14053029 2/3`` and ``2/3``: a mixed number or a bare fraction.
_FRACTION = re.compile(r"(?:(-?[\d,]+)\s+)?(-?[\d,]+)\s*/\s*([\d,]+)")


def _equivalent_fraction(output: str, want: float) -> bool:
    """Does ``output`` state ``want`` exactly, as a fraction the oracle cannot read?

    Both disagreements in the GSM-Hard Phase 0 run were this shape or its
    cousin, and neither was the judge's fault. ``14053029 2/3`` **is**
    ``14053029.666666666``; the oracle reads the last number and got ``3``. The
    judge said the answer was right and was exactly right.

    So this is not a judge error mode at all -- it is the *oracle* failing to
    parse, and filing it under ``wrong-number-with-working`` (which is where it
    landed, because a mixed number contains a ``/``) would count an oracle bug
    as a judge slip and send the improvement pool's budget after it.

    Only the exactly-checkable case. The run's other disagreement was
    ``98,826 hours, 37 minutes, and 35 seconds`` against a gold of
    ``5929597.583333333`` minutes -- also correct, also unreadable by the
    oracle, and not detectable without guessing at units. That one stays
    unclassified rather than being guessed at, and the report says so.
    """
    for whole, num, den in _FRACTION.findall(output or ""):
        try:
            d = float(den.replace(",", ""))
            if d == 0.0:
                continue
            value = float(num.replace(",", "")) / d
            if whole:
                w = float(whole.replace(",", ""))
                value = w + (value if w >= 0 else -value)
        except ValueError:
            continue
        if abs(value - want) < 1e-6:
            return True
    return False


def _is_rounding(want: float, got: float) -> bool:
    """Is ``got`` ``want`` rounded or truncated at some decimal place?

    Its own mode because it is its own bug, with its own fix, and a *third*
    verdict on who was wrong. Found in the GSM-Hard probe: gold
    ``14053029.666666666`` answered ``14053029.666``, and gold
    ``5489.5466666667`` answered ``5489``. The arithmetic was right and the
    write-up was short; a judge saying yes to those has the better case, so
    they belong with the AMBIGUOUS disagreements rather than with
    ``wrong-number-with-working``, which is a genuine slip and the thing worth
    fixing.

    Only for a non-integral reference. Against an integral one every ``round``
    is the identity, and ``8`` answered ``6`` would be filed as a rounding.
    """
    if want == int(want):
        return False
    for places in range(0, 7):
        scale = 10.0 ** places
        if abs(round(want, places) - got) < 1e-9:
            return True
        if abs(int(want * scale) / scale - got) < 1e-9:   # truncated, not rounded
            return True
    return False


def _mentions(output: str, want: float) -> bool:
    """Does ``want`` appear anywhere in ``output``, not only at the end?"""
    for token in _NUMBER.findall(output or ""):
        try:
            if abs(float(token.replace(",", "")) - want) < 1e-6:
                return True
        except ValueError:
            continue
    return False


#: Arithmetic in the output is what "working" means here, and an arithmetic
#: operator is the cheapest signal for it that does not depend on the model's
#: prose style. Deliberately not a length threshold: a long wrong sentence is
#: not a derivation, and calling it one would put the bare case in the bucket
#: this workload was added to isolate.
_OPERATORS = ("=", "+", "*", "/", " - ", "×", "÷")


def _has_working(output: str, worked: Any) -> bool:
    del worked                    # the reference derivation, kept for callers
    return any(op in (output or "") for op in _OPERATORS)


def code_error_mode(record, task: Context) -> Optional[str]:
    """For MBPP, where the judge reads code it cannot run.

    The modes are about **what the judge was looking at when it got it wrong**,
    because that is what a rubric clause could address. The numeric workloads'
    modes were about the answer's shape; there is no useful shape here, and
    grouping by "how long was the code" would be the frequency-not-value mistake
    the coverage sampler exists to avoid.

    ``no-code`` and ``does-not-parse`` are separated from the rest on purpose: a
    judge saying yes to something that is not a program is a different failure
    from a judge saying yes to a program that is wrong, and only the second is
    about reading code.
    """
    if task is None:
        return None
    code = extract_code(record.output)
    if not code:
        return "no-code"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "does-not-parse"
    defs = [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not defs:
        return "no-function"
    if any(_is_stub(fn) for fn in defs):
        return "stub"
    reference = task.meta.get("gold", "")
    if _names_of(code) & _names_of(reference):
        # Same helper or function name as the reference: the judge was reading
        # something that looks like the answer, which is the interesting case.
        return "looks-like-the-reference"
    return "different-approach"


def _is_stub(fn) -> bool:
    """A body that is only `pass`, `...`, a docstring, or a bare `return`."""
    for node in fn.body:
        if isinstance(node, ast.Pass):
            continue
        if isinstance(node, ast.Return) and node.value is None:
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        return False
    return True


def _names_of(code: str) -> frozenset:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return frozenset()
    return frozenset(n.name for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))



# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

def _hotpot_task(row: Dict) -> Optional[Task]:
    """One HotpotQA row as a task, or ``None`` if it is unusable.

    The **only** place a HotpotQA task id is constructed. Building an id in the
    loader and re-deriving it in an analysis is two pieces of code that can
    drift, and they did: see :func:`task_index`.
    """
    answer = (row.get("answer") or "").strip()
    question = (row.get("question") or "").strip()
    if not answer or not question or not row.get("id"):
        return None
    return Task(id=str(row["id"]), prompt=question,
                meta={"gold": answer, "expected": answer})


def _bbh_task(subtask: str, row: Dict) -> Optional[Task]:
    """One BBH row as a task. The only place a BBH task id is constructed.

    Hashed from the question, not the row's position. BBH has no id column, and
    a positional id is only meaningful together with the ``limit`` and ``seed``
    that produced the shuffle -- so a records file written today could not be
    re-joined to its gold answers tomorrow without reproducing the exact call.
    That is the durability the whole audit rests on (a wet-lab result comes back
    next week to a process that has only the JSONL), and it was broken here.
    """
    answer = str(row.get("target") or "").strip()
    question = str(row.get("input") or "").strip()
    if not answer or not question:
        return None
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:10]
    return Task(id=f"{subtask}:{digest}", prompt=question,
                meta={"gold": answer, "expected": answer, "subtask": subtask})


#: BBH subtasks, chosen for the *shape* of their answers rather than for
#: difficulty. The phenomenon under study is a judge accepting what exact match
#: refuses, so a subtask whose only correct string is ``True`` or ``False``
#: contributes nothing: the two scorers cannot disagree there. These can.
#:
#: ``date_understanding`` and ``salient_translation_error_detection`` want a
#: label -- ``(B)`` -- where a model naturally answers with the content;
#: ``object_counting`` wants ``8`` where a model may write ``eight``;
#: ``word_sorting`` wants an exact sequence a model may punctuate. The Yes/No
#: pair is in for contrast: it is most of BBH, and a workload where the two
#: scorers *cannot* differ is worth having in the same measurement.
BBH_SUBTASKS = (
    "date_understanding",
    "object_counting",
    "word_sorting",
    "salient_translation_error_detection",
    "causal_judgement",
    "sports_understanding",
)


def _gsm8k_task(row: Dict) -> Optional[Task]:
    """One GSM8K row as a task. The only place a GSM8K task id is constructed.

    The gold kept in ``meta`` is the **final number**, not the worked solution
    the dataset stores with it. The judge is shown ``meta["gold"]``, and a judge
    shown the reasoning would be grading against a derivation rather than an
    answer -- which is a different experiment, and an easier one.
    """
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "")
    if "####" not in answer or not question:
        return None
    gold = answer.rsplit("####", 1)[1].strip()
    if final_number(gold) is None:
        return None
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:10]
    return Task(id=f"gsm8k:{digest}", prompt=question,
                meta={"gold": gold, "expected": gold,
                      # The derivation, for the diagnosis only. `error_mode`
                      # needs it to tell "wrong number" from "right number the
                      # oracle could not find", and nothing scored ever sees it.
                      "worked": answer.rsplit("####", 1)[0].strip()})


def _gsm_hard_task(row: Dict) -> Optional[Task]:
    """One GSM-Hard row as a task. The only place a GSM-Hard task id is constructed."""
    question = str(row.get("input") or "").strip()
    gold = str(row.get("target") or "").strip()
    if not question or final_number(gold) is None:
        return None
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:10]
    return Task(id=f"gsm_hard:{digest}", prompt=question,
                meta={"gold": gold, "expected": gold})


def _mbpp_task(row: Dict) -> Optional[Task]:
    """One MBPP row as a task. The only place an MBPP task id is constructed.

    The prompt carries the **first assert** as well as the description, which is
    the MBPP convention and not a hint: the description does not name the
    function, so without it every candidate fails on the name and the oracle
    measures naming rather than correctness.

    ``gold`` is the dataset's reference solution, because ``gold`` is what the
    judge is shown -- and "does this candidate do the same thing as that
    reference" is the ambiguous judging task this workload is here to create.
    The asserts go in ``tests`` and only the oracle sees them; a judge shown the
    tests could evaluate them in its head and would be doing the oracle's job.
    """
    text = str(row.get("text") or row.get("prompt") or "").strip()
    code = str(row.get("code") or "").strip()
    tests = [str(t) for t in (row.get("test_list") or []) if str(t).strip()]
    if not text or not code or not tests:
        return None
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
    return Task(id=f"mbpp:{digest}",
                prompt=f"{text}\n\nYour function must satisfy: {tests[0]}",
                meta={"gold": code, "expected": code, "tests": tests})


@dataclass(frozen=True)
class Workload:
    """Everything that is per-workload, in one place.

    This was **seven** parallel dicts keyed by workload name -- the oracle, its
    label, the loader, the indexer, the dry-run near-miss, the dry-run wrong
    answers, and the report label. Seven tables that had to agree, kept in
    agreement by a test that checked they had the same keys.

    The test was not enough, and the way it failed is the argument for this
    class: `gsm_hard` shipped with six entries and no indexer, the test passed
    because it predated the indexer table, and nothing failed until something
    asked for that workload's gold answers -- after a paid run had finished.
    One object per workload cannot be half-registered.
    """

    #: What the report calls the workload.
    label: str
    #: ``(task, output) -> 1.0 | 0.0``. Ground truth.
    oracle: Callable[[Task, str], float]
    #: What the report calls the oracle. A report naming the wrong oracle is
    #: the one error in a measurement nobody can catch later from the file.
    oracle_label: str
    #: ``(record, task) -> the error mode``, or ``None`` when it cannot say.
    #: On the object rather than in a table beside it: this was `ERROR_MODES`,
    #: an eighth dict keyed by workload name, and it could go missing on a new
    #: workload without anything failing until `assess()` -- after a paid run.
    modes: Callable[[AuditRecord, Optional[Task]], Optional[str]]
    #: An answer the judge forgives and this oracle refuses, for `--dry-run`.
    #: It has to be a near-miss *for this oracle*: `The answer is 18.` is one
    #: for exact match and simply correct for `number_match`.
    near_miss: Callable[[str], str]
    #: Wrong answers for `--dry-run`, several shapes rather than one constant.
    #: A single `"unknown"` gives the whole dry run one error mode, so
    #: `P(new error mode)` comes back 0.0 and anything gated on coverage
    #: refuses to run -- which reads as a finding and is an artefact.
    wrong: Sequence[Callable[[str, random.Random], str]]
    #: The dataset a row comes from, and the row -> Task rule.
    dataset: str = ""
    split: str = ""
    config: str = ""
    to_task: Optional[Callable[[Dict], Optional[Task]]] = None
    #: BBH alone needs these: it is one dataset config per subtask, so neither
    #: the sample nor the index is a single `hf_rows` call.
    sample_with: Optional[Callable[[int, int], List[Task]]] = None
    index_with: Optional[Callable[[int], Dict[str, Task]]] = None

    def rows(self, limit: int) -> List[Dict]:
        from agentdescent.dataloader import hf_rows

        return hf_rows(self.dataset, self.split, config=self.config,
                       limit=limit)

    def sample(self, n: int, *, seed: int = 0) -> List[Task]:
        """``n`` tasks, shuffled. A *sample*, and that is the point.

        Which tasks come back depends on ``n`` and ``seed``, so this is the
        wrong thing to rebuild a past run's context with -- see `task_index`.
        """
        if self.sample_with is not None:
            return self.sample_with(n, seed)
        rows = self.rows(max(n * 2, 40))
        random.Random(seed).shuffle(rows)
        out: List[Task] = []
        for row in rows:
            task = self.to_task(row)
            if task is not None:
                out.append(task)
                if len(out) >= n:
                    break
        return out

    def index(self, rows: int) -> Dict[str, Task]:
        """``task_id -> Task`` over a *window*, with no sampling."""
        if self.index_with is not None:
            return self.index_with(rows)
        out: Dict[str, Task] = {}
        for row in self.rows(rows):
            task = self.to_task(row)
            if task is not None:
                out[task.id] = task
        return out


def _bbh_sample(n: int, seed: int) -> List[Task]:
    """BBH, sampled across subtasks rather than down one of them.

    A single subtask is one answer shape, and the residual this experiment
    measures is mostly a property of the shape. Spreading the draw is what makes
    it a second *workload* and not a second sample of the first.
    """
    from agentdescent.dataloader import hf_rows

    rng = random.Random(seed)
    per = max(1, n // len(BBH_SUBTASKS))
    tasks: List[Task] = []
    for name in BBH_SUBTASKS:
        rows = hf_rows("lukaemon/bbh", "test", config=name,
                       limit=max(per * 2, 20))
        rng.shuffle(rows)
        taken = 0
        for row in rows:
            task = _bbh_task(name, row)
            if task is None:
                continue
            tasks.append(task)
            taken += 1
            if taken >= per:
                break
    rng.shuffle(tasks)
    return tasks[:n]


def _bbh_index(rows: int) -> Dict[str, Task]:
    from agentdescent.dataloader import hf_rows

    out: Dict[str, Task] = {}
    for name in BBH_SUBTASKS:
        for row in hf_rows("lukaemon/bbh", "test", config=name,
                           limit=max(40, rows // len(BBH_SUBTASKS))):
            task = _bbh_task(name, row)
            if task is not None:
                out[task.id] = task
    return out


def _restated(gold: str) -> str:
    return f"The answer is {gold}."


def _trailing_context(gold: str) -> str:
    """A near-miss for an oracle that reads the **last** number."""
    return f"Working through it, the answer is {gold} (over 7 days)."


def _plausibly_broken(gold: str) -> str:
    """The reference solution with one operator flipped, for `--dry-run`.

    Not `"unknown"` and not a stub: the near-miss has to be something the
    *judge* forgives, and a judge forgives code that looks like the reference.
    Flipping a comparison keeps the shape and breaks the asserts.
    """
    for a, b in (("<=", "<"), (">=", ">"), ("==", "!="), ("+", "-")):
        if a in gold:
            return gold.replace(a, b, 1)
    return gold + "\n# (returns the wrong branch)"


_UNKNOWN = (lambda gold, rng: "unknown",)
_WRONG_NUMBERS = (
    lambda gold, rng: str(rng.randint(1, 400)),
    lambda gold, rng: (f"{rng.randint(2, 9)} * {rng.randint(2, 9)} = "
                       f"{rng.randint(1, 400)}, so that is the answer."),
    lambda gold, rng: "I could not work this out.",
)
_NUMERIC_ORACLE = "the final number in the answer, compared as a quantity"
_STRING_ORACLE = "normalized exact match against the reference"

#: One entry per workload. `--workload` chooses from these keys.
WORKLOADS: Dict[str, Workload] = {
    "hotpot": Workload(
        label="HotpotQA validation",
        oracle=exact_match, oracle_label=_STRING_ORACLE,
        modes=text_error_mode, near_miss=_restated, wrong=_UNKNOWN,
        dataset="hotpotqa/hotpot_qa", split="validation", config="distractor",
        to_task=_hotpot_task),
    "bbh": Workload(
        label=f"BIG-Bench Hard across {len(BBH_SUBTASKS)} subtasks",
        oracle=exact_match, oracle_label=_STRING_ORACLE,
        modes=text_error_mode, near_miss=_restated, wrong=_UNKNOWN,
        sample_with=_bbh_sample, index_with=_bbh_index),
    "gsm8k": Workload(
        label="GSM8K test, scored on the final number",
        oracle=number_match, oracle_label=_NUMERIC_ORACLE,
        modes=gsm8k_error_mode, near_miss=_trailing_context,
        wrong=_WRONG_NUMBERS, dataset="openai/gsm8k", split="test", config="main",
        to_task=_gsm8k_task),
    "gsm_hard": Workload(
        label="GSM-Hard (GSM8K with large numbers), scored on the final number",
        oracle=number_match, oracle_label=_NUMERIC_ORACLE,
        modes=gsm8k_error_mode, near_miss=_trailing_context,
        wrong=_WRONG_NUMBERS, dataset="reasoning-machines/gsm-hard", split="train", config="default",
        to_task=_gsm_hard_task),
    "mbpp": Workload(
        label="MBPP, scored by executing each task's asserts",
        oracle=tests_pass,
        oracle_label="the task's own asserts, executed against the candidate",
        modes=code_error_mode, near_miss=_plausibly_broken,
        wrong=(lambda gold, rng: "def solve():\n    pass",
               lambda gold, rng: "I could not work out an implementation.",
               lambda gold, rng: "def solve(:\n  return"),
        dataset="google-research-datasets/mbpp", split="test", config="full",
        to_task=_mbpp_task),
}


def task_index(workload: str, *, rows: int = 400) -> Dict[str, Task]:
    """``task_id -> Task`` for re-joining records to their gold answers.

    **Not** ``{t.id: t for t in <loader>(n)}``, and the difference is the point.
    A loader draws a *sample*: it shuffles and truncates, so which tasks come
    back depends on ``n`` and ``seed``. An index does not sample -- it reads a
    window of the dataset and keys every row by the same id rule.

    Measured the hard way: rebuilding HotpotQA context with 200 tasks failed to
    join **90 of 177** records from a run that had drawn 80, because a 400-row
    shuffle and a 160-row shuffle under the same seed are different shuffles.
    The ids were never the problem; drawing a different sample was.
    """
    try:
        return WORKLOADS[workload].index(rows)
    except KeyError:
        raise ValueError(f"unknown workload {workload!r}") from None


