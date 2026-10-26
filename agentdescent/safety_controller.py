"""Deployment safety for self-modifications: kill switch, canary, rollback.

An acceptance gate decides whether to commit; a deployment controller decides
whether to *keep* a change live. These are deliberately separate. This module
gives a product the three operations an operator needs:

* **KillSwitch** -- a file that any process can check before starting another
  session. Engaging it is an operational action, not a code change.
* **canary** -- a paired sign test on incumbent vs candidate per-task scores.
  Promote only when the improvement is significant and above the effect bar;
  roll back when the degradation is; otherwise hold for more evidence.
* **rollback_plan** -- a pure diff between the current and previous artifact
  states: files to write and delete, no destructive application. The caller
  decides when to execute it.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

__all__ = ["CanaryResult", "KillSwitch", "canary", "rollback_plan"]


class KillSwitch:
    """A file-based global stop for further self-improvement sessions.

    The file's content is the operator's reason. ``engaged()`` is safe to call
    from a loop; ``engage``/``release`` are the operator actions.
    """

    def __init__(self, path: str) -> None:
        self.path = os.path.abspath(os.path.expanduser(path))

    def engaged(self) -> bool:
        # exists, not isfile: a directory at the path is still a stop signal.
        return os.path.exists(self.path)

    def reason(self) -> str:
        try:
            with open(self.path, encoding="utf-8") as fh:
                return fh.read().strip()
        except OSError:
            return ""

    def engage(self, reason: str = "operator stop") -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = f"{self.path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(reason + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    def release(self) -> None:
        try:
            if os.path.isdir(self.path):
                os.rmdir(self.path)
            else:
                os.remove(self.path)
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class CanaryResult:
    """The paired sign test and the deployment verdict."""

    verdict: str          # "promote" / "hold" / "rollback"
    mean_gain: float
    wins: int
    losses: int
    p_value: float
    reason: str
    missing: Tuple[str, ...] = ()


def _sign_test_p(wins: int, losses: int) -> float:
    """One-sided exact sign-test p-value for more wins than losses."""
    n = wins + losses
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / (2 ** n)


def canary(incumbent: Mapping[str, float], candidate: Mapping[str, float], *,
           min_effect: float = 0.0, max_regression: float = 0.0,
           alpha: float = 0.05) -> CanaryResult:
    """Compare candidate vs incumbent per task and recommend promote/hold/rollback.

    Only tasks present in both mappings count; a task the candidate did not
    score is reported as ``missing`` and blocks promotion. ``min_effect`` is the
    mean gain required to promote, ``max_regression`` the mean loss required to
    roll back (default 0 -- any significant loss). A tie is not a win.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    for name, value in (("min_effect", min_effect),
                        ("max_regression", max_regression)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or value < 0):
            raise ValueError(f"{name} must be a finite number >= 0")
    values = [float(value) for value in (*incumbent.values(), *candidate.values())]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("canary scores must be finite")
    missing = tuple(sorted(set(incumbent) - set(candidate)))
    diffs = [float(candidate[key]) - float(incumbent[key])
             for key in incumbent.keys() & candidate.keys()]
    if not diffs:
        return CanaryResult("hold", 0.0, 0, 0, 1.0,
                            "no tasks in common between incumbent and candidate",
                            missing=missing)
    mean = sum(diffs) / len(diffs)
    wins = sum(1 for value in diffs if value > 0)
    losses = sum(1 for value in diffs if value < 0)
    p = _sign_test_p(wins, losses)
    if not missing and mean > min_effect and p < alpha:
        return CanaryResult("promote", mean, wins, losses, p,
                            f"candidate gains {mean:+.3f}; sign test p={p:.3f}",
                            missing=missing)
    if mean < -max_regression:
        reverse = _sign_test_p(losses, wins)
        if reverse < alpha:
            return CanaryResult(
                "rollback", mean, wins, losses, reverse,
                f"candidate loses {mean:+.3f}; reverse sign test p={reverse:.3f}",
                missing=missing)
    note = (f"; {len(missing)} task(s) missing from the candidate"
            if missing else "")
    return CanaryResult(
        "hold", mean, wins, losses, p,
        f"no significant effect (mean {mean:+.3f}, p={p:.3f}){note}; keep observing",
        missing=missing)


def rollback_plan(current: Mapping[str, str],
                  previous: Mapping[str, str]) -> Dict[str, object]:
    """Files to restore/delete to return ``current`` to ``previous``.

    Pure and non-destructive: ``write`` maps restored paths to their previous
    content; ``delete`` lists paths only in ``current``. The caller applies it with the same
    materialiser it used to write the artifact.
    """
    write = {path: previous[path] for path in previous
             if current.get(path) != previous[path]}
    delete = tuple(sorted(path for path in current if path not in previous))
    return {"write": write, "delete": delete,
            "noop": not write and not delete}
