"""Ready-made scorers for the common cases.

A `reward` is `(task, output) -> float in [0, 1]`, and writing one is easy --
but almost everyone writes the *same* three, and gets the same details wrong:
thousands separators, a trailing period, a model that answers in a sentence, an
LLM that says "The answer is 42." when the gold is "42".

Every scorer here reads the expected answer from ``task.meta`` (``gold_key``,
default ``"gold"``), which is also where the reflector looks -- see
:func:`agentdescent.evolution.reflector`.

    from agentdescent.rewards import last_number
    evolve(tasks, last_number(), agent=agent)

Bring your own for anything else; these are a convenience, not a contract.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
from typing import Callable, Mapping, Optional, Sequence, Union

__all__ = ["exact_match", "contains", "last_number", "numeric_close", "command"]

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def _gold(task, gold_key: str) -> str:
    meta = getattr(task, "meta", None) or {}
    if gold_key not in meta:
        raise KeyError(
            f"task {getattr(task, 'id', '?')!r} has no meta[{gold_key!r}]. The "
            f"built-in scorers read the expected answer from Task.meta -- pass "
            f"gold_key= if yours is under a different name.")
    return str(meta[gold_key])


def _normalise(text: str) -> str:
    """Casefold, collapse whitespace, drop surrounding punctuation."""
    return re.sub(r"\s+", " ", (text or "").strip().strip(".!?\"' ")).casefold()


def exact_match(gold_key: str = "gold", *, normalise: bool = True) -> Callable:
    """1.0 when the output equals the gold answer.

    ``normalise`` casefolds, collapses whitespace and strips surrounding
    punctuation -- without it a model that ends its answer with a period scores
    zero, which looks like a reasoning failure and is not one.
    """
    def reward(task, output) -> float:
        want, got = _gold(task, gold_key), output or ""
        if normalise:
            want, got = _normalise(want), _normalise(got)
        return 1.0 if got == want else 0.0
    return reward


def contains(gold_key: str = "gold", *, normalise: bool = True) -> Callable:
    """1.0 when the gold answer appears anywhere in the output.

    The forgiving option for models that answer in a sentence. It is also the
    easiest to fool -- a gold of ``"2"`` is inside ``"12"`` -- so prefer
    :func:`last_number` for numeric answers.
    """
    def reward(task, output) -> float:
        want, got = _gold(task, gold_key), output or ""
        if normalise:
            want, got = _normalise(want), _normalise(got)
        return 1.0 if want and want in got else 0.0
    return reward


def last_number(gold_key: str = "gold", *, tolerance: float = 0.0) -> Callable:
    """1.0 when the **last** number in the output matches the gold number.

    The right default for arithmetic word problems (GSM8K and friends): models
    show their working, so the answer is the last number, not the first.

    The **gold is read the same way**, which matters more than it sounds: a
    dataset's answer column is often the whole worked solution ending in the
    figure (GSM8K's ends ``"#### 72"``). Parsing that column as a bare number
    fails, and the failure is silent -- every item scores 0 and it reads as a
    hopeless model rather than a scorer mismatch. Taking the last number from both
    sides handles ``"72"``, ``"#### 72"`` and ``"The answer is 72."`` alike.

    Handles thousands separators and a leading currency symbol. ``tolerance``
    compares with a relative tolerance -- use it when the gold is rounded.
    """
    def reward(task, output) -> float:
        want = _last_number_in(_gold(task, gold_key))
        if want is None:
            raise ValueError(
                f"task {getattr(task, 'id', '?')!r}: meta[{gold_key!r}] contains no "
                f"number, so last_number() can never match it. Use exact_match() or "
                f"contains(), or point gold_key= at the right column.")
        found = _NUMBER.findall(output or "")
        got = _to_float(found[-1]) if found else None
        if got is None:
            return 0.0
        if tolerance <= 0:
            return 1.0 if got == want else 0.0
        return 1.0 if abs(got - want) <= tolerance * max(1.0, abs(want)) else 0.0
    return reward


def numeric_close(gold_key: str = "gold", *, tolerance: float = 0.01) -> Callable:
    """:func:`last_number` with a relative tolerance -- for rounded answers."""
    return last_number(gold_key, tolerance=tolerance)


def command(run: Union[str, Sequence[str]], *, timeout: float = 30.0,
            cwd: Optional[str] = None, env: Optional[Mapping[str, str]] = None,
            ok: Sequence[int] = (0,)) -> Callable:
    """1.0 when a command of yours **exits 0** on the output.

    The scorers above need a gold answer. This one needs a *judgement*, which is
    what people actually have: "the SQL has to run against my database", "it has
    to compile", "it has to pass these tests". None of those is an expected
    string, all of them are a command that already exists::

        evolve(tasks, command("psql --quiet -f {output}"), agent=agent)
        evolve(tasks, command(["ruff", "check", "{output}"]), agent=agent)

    ``{output}`` in any argument is replaced with the path to a file holding
    that rollout's output; the same text also arrives on **stdin**, so a filter
    that reads stdin needs no placeholder at all.

    A string is split with :func:`shlex.split` and run **without a shell**, so
    ``|``, ``>`` and ``&&`` are not operators -- they arrive as literal
    arguments. Ask for a shell explicitly when you want one::

        command(["bash", "-lc", "psql -f {output} | grep -q OK"])

    **A timeout scores 0.0 rather than raising.** A check that hangs on this
    output *is* a failing check, and raising would drop the sample instead of
    scoring it -- which biases the run toward whichever candidates happen not to
    hang the checker.

    **A missing executable raises.** That is a configuration mistake, not a
    verdict on the candidate, and scoring it 0.0 would make every rollout fail
    identically -- indistinguishable from a model that cannot do the task at all.
    The same reasoning as :func:`last_number` raising on an unparseable gold.

    ``ok`` widens the set of exit codes that count as success, for the tools that
    signal "clean" with something other than 0.
    """
    template = shlex.split(run) if isinstance(run, str) else [str(a) for a in run]
    if not template:
        raise ValueError("command() needs a command to run; got an empty one.")
    accepted = tuple(ok)

    def reward(task, output) -> float:
        text = output or ""
        with tempfile.TemporaryDirectory(prefix="agentdescent-check-") as scratch:
            path = os.path.join(scratch, "output.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            argv = [arg.replace("{output}", path) for arg in template]
            try:
                proc = subprocess.run(argv, cwd=cwd, env=dict(env) if env else None,
                                      input=text, capture_output=True, text=True,
                                      timeout=timeout)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"command() cannot run {template[0]!r}: {exc.strerror}. Every "
                    f"rollout would score 0.0, which is indistinguishable from a "
                    f"model that cannot do the task -- so this raises instead.") from exc
            except subprocess.TimeoutExpired:
                return 0.0
        return 1.0 if proc.returncode in accepted else 0.0
    return reward


def _last_number_in(text: str) -> Optional[float]:
    """The last number in a string, or None if it holds none."""
    found = _NUMBER.findall(str(text or ""))
    return _to_float(found[-1]) if found else None


def _to_float(text: str) -> Optional[float]:
    try:
        return float(str(text).replace(",", "").lstrip("$£€").strip())
    except (TypeError, ValueError):
        return None
