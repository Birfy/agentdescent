"""Compute-aware stopping: stop when more calls stop buying reward.

``evolve`` already stops on a target, patience, rounds, wall clock, calls,
tokens and diminishing returns. A campaign or a long RSI run usually wants a
different question answered: *is the next call still paying for itself?* This
module turns the recent reward/call curve into a ``stop_when`` callable.

    evolve(..., stop_when=MarginalGainStop(min_gain_per_call=1e-4, patience=3))

The policy is deliberately stateful and read-only: it receives the same
``RoundInfo`` the engine passes to ``on_round``, keeps the last ``window``
points, and stops once ``patience`` consecutive measurements show that the
marginal gain per call is below the bar. ``reason`` says which measurement
triggered it, for a caller's own report.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Tuple

__all__ = ["MarginalGainStop"]


class MarginalGainStop:
    """Stop when reward gained per additional call falls below a bar.

    ``min_gain_per_call`` is the economic bar and is sufficient on its own.
    ``min_gain`` is an optional absolute-gain floor combined with OR: a
    measurement is low when the per-call gain is below its bar *or* the recent
    absolute gain is at most ``min_gain``. ``patience`` consecutive low
    measurements stop the run. ``window`` is how far back the gain is measured,
    in round records.

    The value compared is this round's ``held_out_reward``, not the best so
    far, so a noisy dip can read as a negative gain. The object is stateful;
    call :meth:`reset` before reusing it for a new run.
    """

    def __init__(self, *, min_gain: float = 0.0,
                 min_gain_per_call: float = 1e-4,
                 patience: int = 2, window: int = 3) -> None:
        for name, value in (("min_gain", min_gain),
                            ("min_gain_per_call", min_gain_per_call)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value)) or value < 0):
                raise ValueError(f"{name} must be a finite number >= 0")
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValueError("patience must be an int >= 1")
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError("window must be an int >= 1")
        self.min_gain = float(min_gain)
        self.min_gain_per_call = float(min_gain_per_call)
        self.patience = int(patience)
        self.window = int(window)
        self._points: Deque[Tuple[int, float]] = deque(maxlen=self.window + 1)
        self._low = 0
        self.reason = ""

    def _calls(self, info) -> int:
        return int(getattr(info, "calls", 0) or 0)

    def __call__(self, info) -> bool:
        point = (self._calls(info),
                 float(getattr(info, "held_out_reward", 0.0) or 0.0))
        if self._points and point[0] <= self._points[-1][0]:
            # A round that did not spend a call (or a caller driving the
            # policy by hand) cannot establish a per-call slope.
            self.reason = "no additional calls since the last measurement"
            return False
        self._points.append(point)
        if len(self._points) < 2:
            return False
        old_calls, old_reward = self._points[0]
        calls, reward = self._points[-1]
        delta_calls = calls - old_calls
        gain = reward - old_reward
        per_call = gain / delta_calls if delta_calls > 0 else float("inf")
        low = (per_call < self.min_gain_per_call
               or gain <= self.min_gain)
        self._low = self._low + 1 if low else 0
        if self._low >= self.patience:
            self.reason = (
                f"reward gained {gain:+.4f} over {delta_calls} call(s) "
                f"({per_call:.2e} per call) for {self._low} measurement(s)")
            return True
        return False

    def reset(self) -> None:
        """Forget the previous run's points and streak."""
        self._points.clear()
        self._low = 0
        self.reason = ""
