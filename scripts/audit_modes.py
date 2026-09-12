#!/usr/bin/env python3
"""Error-mode signatures, one per workload, in one place.

An *error mode* is the species Good--Turing counts: a coarse label for **what a
person would have to fix**, deliberately coarser than the record and coarser
than the task. Two answers that echo their question are the same bug however
different the questions were, and counting them as two modes would make the
improvement pool look like it was still learning when it had stopped.

This module exists because there is now more than one workload and the mode
function is the third thing that is per-workload, after the oracle and the
dry-run near-miss. The first two were single copies that quietly became wrong
when a second workload arrived. A **second copy** of a mode function would be
worse than either: the coverage number decides whether the improvement pool
still gets budget, and two definitions that drift would answer that question
differently in two files with no way to tell which ran.

The context values are ``(question, reference, extra)``. ``extra`` is whatever
the workload's diagnosis needs and nothing scored ever sees -- for GSM8K it is
the dataset's worked solution, which distinguishes "the judge was seduced by
correct-looking working" from "there was no working to be seduced by", and those
are different bugs with different fixes.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.audit_phase0 import final_number, normalize  # noqa: E402

#: The oracle's own normalisation, imported rather than reimplemented. This was
#: a second copy for one commit, and the copy replaced punctuation with a space
#: where the original deletes it -- so ``cat's`` normalised to ``cat s`` here and
#: ``cats`` there.
normalise = normalize

Context = Tuple[str, str, Any]


# -- the predicates the hand-written rules are built from --------------------

def echoes_the_question(output: str, question: str) -> bool:
    return bool(output) and normalise(output) in normalise(question)


def far_shorter_than_reference(output: str, gold: str, ratio: float) -> bool:
    o, g = normalise(output), normalise(gold)
    return bool(o) and bool(g) and o != g and len(o) < ratio * len(g)


# -- one mode function per workload ------------------------------------------

def text_error_mode(record, ctx: Optional[Context]) -> Optional[str]:
    """For the free-text workloads: HotpotQA, and BBH's non-label subtasks.

    Every branch here is a statement about *strings* -- length, containment,
    overlap with the question -- because that is the whole of what those two
    workloads' answers are.
    """
    if ctx is None:
        return None
    question, gold = ctx[0], ctx[1]
    out, ref = normalise(record.output), normalise(gold)
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


def gsm8k_error_mode(record, ctx: Optional[Context]) -> Optional[str]:
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
    if ctx is None:
        return None
    gold, worked = ctx[1], ctx[2]
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
    from scripts.audit_phase0 import _NUMBER

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


#: Which mode function a workload's records are read with. BBH shares the text
#: one: its answers are short strings and labels, and the label failure it
#: actually exhibits is a property of the *judge's* rubric rather than of the
#: output's shape, so it shows up as `substring-of-reference` here and in the
#: rubber-stamp table in the Phase 0 report, which is where it belongs.
ERROR_MODES = {
    "hotpot": text_error_mode,
    "bbh": text_error_mode,
    "gsm8k": gsm8k_error_mode,
    "gsm_hard": gsm8k_error_mode,
}


def context_for(workload: str, *, limit: int = 200) -> Dict[str, Context]:
    """``task_id -> (question, reference, extra)``, built from `task_index`.

    Goes through `task_index` rather than a loader so the ids match the ones the
    records were written with. Rebuilding a past run's sample with a loader
    failed to join 90 of 177 records once, because a loader shuffles and
    truncates and a different ``n`` is a different shuffle.
    """
    from scripts.audit_phase0 import task_index

    return {tid: (task.prompt, task.meta.get("gold", ""),
                  task.meta.get("worked"))
            for tid, task in task_index(workload, rows=limit).items()}
