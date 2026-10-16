"""The number reader behind ``last_number`` / ``numeric_close`` / ``command_scorer`` (#288).

The old pattern ``-?\\d[\\d,]*\\.?\\d*`` read ``1.5e-3`` as ``-3``, ``.5`` as
``5``, ``pages 3-4`` as ``-4`` and ``1,2,3`` as ``123`` -- each a silently wrong
reward, not an error.
"""
import sys

import pytest

from agentdescent.evolution import Task
from agentdescent.rewards import (GraderError, _last_number_in, command_scorer,
                                  last_number, numeric_close)


@pytest.mark.parametrize("text, want", [
    # the regressions from #288
    ("1.5e-3", 0.0015),
    ("6.02e23", 6.02e23),
    ("6.02E+23", 6.02e23),
    ("1.e-3", 0.001),
    ("2.E5", 2e5),
    ("1,234.e2", 123400.0),
    ("Answer: 1e-05", 1e-05),
    (".5", 0.5),
    ("-.25", -0.25),
    ("pages 3-4", 4.0),
    ("x-3", 3.0),
    ("(3)-2", 2.0),
    ("list 1,2,3", 3.0),
    ("1,2345", 2345.0),
    # behaviour that must not change
    ("72", 72.0),
    ("#### 72", 72.0),
    ("The answer is 72.", 72.0),
    ("$1,234.50", 1234.5),
    ("1,234,567", 1234567.0),
    ("12345", 12345.0),
    ("x = -3", -3.0),
    ("-7", -7.0),
    ("+5", 5.0),
    ("答案是72", 72.0),
    ("答案是-3", -3.0),
    ("first 10 then 20", 20.0),
    ("3em", 3.0),
    # a figure glued to an ellipsis is still a figure; a version is not three
    ("Total...42", 42.0),
    ("Answer:...42", 42.0),
    ("so...7.5", 7.5),
    ("v1.2.3", 1.2),
])
def test_last_number_in_reads_the_written_figure(text, want):
    assert _last_number_in(text) == pytest.approx(want)


@pytest.mark.parametrize("text", ["", "no digits here", "1e999", "-1e999", "1.e999"])
def test_last_number_in_has_no_finite_number(text):
    assert _last_number_in(text) is None


def _task(gold):
    return Task(id="t", prompt="q", meta={"gold": gold})


def test_last_number_scientific_gold_needs_the_right_mantissa():
    reward = last_number()
    assert reward(_task("6.02e23"), "It is 6.02e23.") == 1.0
    # used to match: both sides read as 23
    assert reward(_task("6.02e23"), "It is 23.") == 0.0
    assert reward(_task("0.0015"), "1.5e-3") == 1.0


def test_last_number_leading_decimal_matches_its_value():
    assert last_number()(_task("0.5"), "The probability is .5") == 1.0
    assert last_number()(_task("5"), "The probability is .5") == 0.0


def test_last_number_does_not_read_a_range_as_a_negative():
    assert last_number()(_task("4"), "See pages 3-4") == 1.0
    assert last_number()(_task("-4"), "See pages 3-4") == 0.0


def test_numeric_close_with_exponents():
    assert numeric_close(tolerance=0.01)(_task("1.0e-3"), "about 0.001") == 1.0


def _grader(code):
    return command_scorer([sys.executable, "-c", code])


def test_command_scorer_reads_python_repr_of_a_small_score():
    # print(1e-05) is how Python writes 0.00001; it used to read as -5 -> 0.0
    assert _grader("print(1e-05)")(_task("x"), "out") == pytest.approx(1e-05)
    assert _grader("print(.25)")(_task("x"), "out") == pytest.approx(0.25)


def test_command_scorer_refuses_an_overflowing_score():
    # 1e999 is inf; it used to clamp to a perfect 1.0
    with pytest.raises(GraderError):
        _grader("print('1e999')")(_task("x"), "out")
    # a trailing decimal point before the exponent must not split it into 999
    with pytest.raises(GraderError):
        _grader("print('1.e999')")(_task("x"), "out")


def test_command_scorer_reads_a_score_after_an_ellipsis():
    assert _grader("print('score...0.8')")(_task("x"), "out") == pytest.approx(0.8)


def test_last_number_overflowing_gold_is_a_loud_error():
    with pytest.raises(ValueError, match="contains no number.*finite"):
        last_number()(_task("1e999"), "1e999")
