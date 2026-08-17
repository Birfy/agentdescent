"""`rewards.command` -- the rung of the objective ladder a user can reach
without writing Python.

The gold-answer scorers need an expected string. This one needs a command that
already exists, which is what people actually have for the objectives they care
about: it compiles, it runs against my database, it passes these tests.

Two behaviours here are choices rather than conveniences, and both are about not
lying to the optimiser: a timeout is a failing sample, a missing binary is not a
sample at all.
"""

from __future__ import annotations

import sys

import pytest

from agentdescent.rewards import command


class Task:
    def __init__(self, id="t", meta=None):
        self.id, self.meta = id, meta or {}


PY = sys.executable


def test_exit_zero_is_one_and_nonzero_is_zero():
    good = command([PY, "-c", "raise SystemExit(0)"])
    bad = command([PY, "-c", "raise SystemExit(3)"])
    assert good(Task(), "anything") == 1.0
    assert bad(Task(), "anything") == 0.0


def test_the_output_arrives_as_a_file_through_the_placeholder():
    check = command([PY, "-c",
                     "import sys; sys.exit(0 if open(sys.argv[1]).read() == 'hello' else 1)",
                     "{output}"])
    assert check(Task(), "hello") == 1.0
    assert check(Task(), "goodbye") == 0.0


def test_the_output_also_arrives_on_stdin_so_a_filter_needs_no_placeholder():
    check = command([PY, "-c", "import sys; sys.exit(0 if sys.stdin.read() == 'ok' else 1)"])
    assert check(Task(), "ok") == 1.0
    assert check(Task(), "no") == 0.0


def test_a_string_is_split_without_a_shell():
    """`shlex.split`, not `shell=True`.

    A user's objective is their own command on their own machine, but running it
    through a shell would make `>` in an argument truncate a file rather than
    reach the program. Metacharacters stay literal; a shell is opt-in.
    """
    assert command(f"{PY} -c pass")(Task(), "") == 1.0
    literal = command([PY, "-c", "import sys; sys.exit(0 if sys.argv[1] == '|' else 1)", "|"])
    assert literal(Task(), "") == 1.0


def test_an_explicit_shell_still_works_when_asked_for():
    check = command(["bash", "-lc", "test -s {output}"])
    assert check(Task(), "non-empty") == 1.0
    assert check(Task(), "") == 0.0


def test_a_timeout_scores_zero_rather_than_raising():
    """A check that hangs on this output is a failing check.

    Raising would drop the sample instead of scoring it, which biases the run
    toward whichever candidates happen not to hang the checker.
    """
    slow = command([PY, "-c", "import time; time.sleep(30)"], timeout=0.4)
    assert slow(Task(), "whatever") == 0.0


def test_a_missing_executable_raises_instead_of_scoring_zero():
    """A configuration mistake is not a verdict on the candidate.

    Scoring it 0.0 would make every rollout fail identically -- which looks
    exactly like a model that cannot do the task, and sends the user hunting for
    the wrong bug.
    """
    with pytest.raises(FileNotFoundError, match="agentdescent-no-such-binary"):
        command(["agentdescent-no-such-binary"])(Task(), "x")


def test_ok_widens_the_accepted_exit_codes():
    check = command([PY, "-c", "raise SystemExit(2)"], ok=(0, 2))
    assert check(Task(), "") == 1.0


def test_an_empty_command_is_rejected_at_construction():
    """Fail where the mistake is, not once per rollout."""
    with pytest.raises(ValueError):
        command([])
    with pytest.raises(ValueError):
        command("")


def test_the_reward_is_a_plain_task_output_callable():
    """Same contract as every other scorer here, so `evolve` takes it directly."""
    check = command([PY, "-c", "pass"])
    assert check(Task(id="anything", meta={"gold": "unused"}), None) == 1.0
