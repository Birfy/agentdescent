# Copyright 2026 The rrsi Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is adapted from google-research/rrsi/rrsi/history.py at
# be50316e1db05914068a973f322770ef08ed7ba1. It is distributed under the
# Apache License, Version 2.0.
r"""The edit history L_t and the summaries the paper computes from it.

    L_t = {(t_i, l_i, h_i, d_i, Delta S_i, Delta C_i, a_i) : i <= n_t}   Eq. (history)
    T_t = {l_i : i <= n_t}                                                (tried components)
    g_t(l) = max{Delta S_i : l_i = l, t - t_i <= n_prune}, max(empty) = -inf   Eq. (yield)
    U_t = K \ T_t,  sigma_t = 1[S_t - S_{t-w} <= delta]                   Eq. (explore)
    B_t = {l in T_t : g_t(l) <= 0}                                         Eq. (prune)

One JSONL record per EDIT. A candidate harness that bundles n edits (n <= b_t)
receives one measurement (Delta S, Delta C, a) and every edit in the bundle
carries it; as b_t anneals to 1 a record becomes evidence about a single
component. Candidates dropped before measurement (critic reject, smoke fail,
invalid evaluation) are recorded with ``delta_S = None`` and do not enter T_t
or g_t, though :meth:`History.attempted` still counts them.

Upstream behaviour preserved, including the parts that surprise:

* ``yield_g`` is ``-inf`` for a tried component with no measured record inside
  the prune window, so such a component is in ``B_t`` (``-inf <= 0``).
* ``append_candidate`` with no edits writes one placeholder record
  (``edit_id="C1"``, no component, ``bundle=0``).
* ``replace_round`` by default drops every per-edit record of round ``t``.
* ``stall_flag`` is 0 until ``w`` rounds exist and for ``t`` past the trajectory.

Defensive boundaries (AgentDescent convention, not upstream parity):

* Non-finite ``delta_S`` / ``delta_C`` / ``S`` / ``C`` are rejected; upstream
  would write a non-standard ``NaN`` into the JSONL.
* A torn final line (a crash mid-append, including one cut inside a multibyte
  UTF-8 character) is ignored by :meth:`History.records` and dropped by the
  next append, so the log stays loadable. A malformed line anywhere else --
  invalid JSON, invalid UTF-8, or a JSON value that is not an object -- raises
  ``ValueError`` with its line number: that is corruption, not an interrupted
  write.
* :meth:`History.replace_round` rewrites the file atomically (fsynced, file
  mode preserved).
* Numeric fields accept any ``numbers.Real`` (numpy scalars included) except
  ``bool``.

This module is an offline mechanism microport. It is not wired into
``evolve()``, a proposal policy, or a resumable run record.
"""

from __future__ import annotations

import json
import math
import numbers
import os
import stat
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Union

from .components import K

__all__ = ["MEASURED_OUTCOMES", "History", "exploration", "stall_flag"]

MEASURED_OUTCOMES = ("ACCEPTED", "REJECTED", "LOST")


def _finite(name: str, value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    # numbers.Real admits numpy scalars (np.float32 is not a float subclass).
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a number or None, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


class History:
    """The append-only per-edit JSONL ledger ``L_t``."""

    def __init__(self, path: Union[Path, str]):
        self.path = Path(path)

    # ------------------------------------------------------------ storage --
    def records(self) -> List[dict]:
        if not self.path.exists():
            return []
        # Split bytes before decoding: a crash can cut the tail inside a
        # multibyte UTF-8 character, and that must read as a torn line too.
        data = self.path.read_bytes()
        lines = data.split(b"\n")
        torn = not data.endswith(b"\n")     # last element is an unterminated tail
        out = []
        for i, raw in enumerate(lines):
            try:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    raise ValueError("not a JSON object")
            except (UnicodeDecodeError, ValueError) as exc:   # JSONDecodeError too
                if torn and i == len(lines) - 1:
                    break                    # interrupted append; not a record
                raise ValueError(f"{self.path}:{i + 1}: malformed history record") from exc
            out.append(rec)
        return out

    def _drop_torn_tail(self) -> None:
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        if not data or data.endswith(b"\n"):
            return
        tail = data[data.rfind(b"\n") + 1:]
        try:
            json.loads(tail.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            with open(self.path, "r+b") as fh:
                fh.truncate(len(data) - len(tail))
        else:
            with open(self.path, "ab") as fh:   # complete record, just unterminated
                fh.write(b"\n")

    def append(self, rec: dict) -> None:
        rec = dict(rec)
        rec.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))
        line = json.dumps(rec, ensure_ascii=False, allow_nan=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._drop_torn_tail()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)

    def append_candidate(self, t: int, variant: str, edits: Optional[List[dict]],
                         outcome: str, delta_S: Optional[float],
                         delta_C: Optional[float], accepted: bool,
                         S: Optional[float], C: Optional[float],
                         diff: Optional[str], detail: str = "") -> None:
        """Write the per-edit records of one candidate harness H'."""
        delta_S = _finite("delta_S", delta_S)
        delta_C = _finite("delta_C", delta_C)
        S = _finite("S", S)
        C = _finite("C", C)
        for e in edits or [{"id": "C1", "component": None, "hypothesis": None}]:
            self.append({
                "t": t, "variant": variant, "edit_id": e.get("id"),
                "component": e.get("component"),
                "hypothesis": e.get("hypothesis") or e.get("mechanism"),
                "targets_mode": e.get("targets_mode"),
                "predicted_affected": e.get("predicted_affected"),
                "diff": diff,
                "delta_S": None if delta_S is None else round(delta_S, 6),
                "delta_C": None if delta_C is None else round(delta_C, 6),
                "accepted": bool(accepted),
                "outcome": outcome,
                "S": None if S is None else round(S, 6),
                "C": None if C is None else round(C, 1),
                "bundle": len(edits or []),
                "detail": detail[:600] if detail else "",
            })

    def replace_round(self, t: int, keep: Callable[[dict], bool] = lambda r: False) -> None:
        """Drop the per-edit records of round ``t`` (a re-adjudication rewrites
        them); records satisfying ``keep`` survive. Written atomically."""
        recs = [r for r in self.records()
                if not (r.get("t") == t and r.get("edit_id")) or keep(r)]
        body = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n"
                       for r in recs)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".history-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(body)
                fh.flush()
                os.fsync(fh.fileno())
            if self.path.exists():           # mkstemp is 0600; keep the file's mode
                os.chmod(tmp, stat.S_IMODE(os.stat(self.path).st_mode))
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def has(self, t: int, variant: str) -> bool:
        return any(r.get("t") == t and r.get("variant") == variant
                   for r in self.records())

    # ---------------------------------------------------------- summaries --
    def measured(self) -> List[dict]:
        return [r for r in self.records() if r.get("delta_S") is not None
                and r.get("component") in K]

    def tried(self) -> Set[str]:
        """T_t: components with at least one measured edit."""
        return {r["component"] for r in self.measured()}

    def attempted(self) -> Dict[str, int]:
        """Edits per component that were shipped at all, measured or not."""
        counts = {c: 0 for c in K}
        for r in self.records():
            if r.get("component") in counts and r.get("edit_id"):
                counts[r["component"]] += 1
        return counts

    def accepted_edits(self) -> Dict[str, List[dict]]:
        """Machinery currently in the incumbent, per component (accepted edits)."""
        out: Dict[str, List[dict]] = {c: [] for c in K}
        for r in self.records():
            if r.get("accepted") and r.get("component") in out:
                out[r["component"]].append(
                    {"t": r["t"], "edit_id": r.get("edit_id"),
                     "hypothesis": r.get("hypothesis"),
                     "delta_S": r.get("delta_S")})
        return out

    def incumbent_component_counts(self) -> Dict[str, int]:
        return {c: len(v) for c, v in self.accepted_edits().items()}

    def yield_g(self, t: int, n_prune: int) -> Dict[str, float]:
        """g_t(l) for every tried component; ``-inf`` when nothing recent."""
        g = {c: -math.inf for c in self.tried()}
        for r in self.measured():
            if t - int(r["t"]) <= n_prune:
                c = r["component"]
                g[c] = max(g.get(c, -math.inf), float(r["delta_S"]))
        return g

    def prune_set(self, t: int, n_prune: int) -> List[dict]:
        """B_t with, for each component, the accepted machinery to remove."""
        g = self.yield_g(t, n_prune)
        acc = self.accepted_edits()
        out = []
        for c in sorted(g):
            if g[c] <= 0:
                out.append({"component": c,
                            "recent_best_gain": (None if g[c] == -math.inf
                                                 else round(g[c], 5)),
                            "accepted_edits_in_incumbent": acc.get(c, [])})
        return out

    # ------------------------------------------------------- for prompts --
    def render(self, n: int = 40) -> List[dict]:
        """Compact view of the most recent records for the proposer's context.

        Measured outcomes dominate: gate failures without a measurement are
        kept only for their most recent few, because a wall of aborts is a
        feedback loop, not evidence."""
        recs = self.records()
        kept, unmeasured = [], 0
        for r in reversed(recs):
            if r.get("delta_S") is None:
                unmeasured += 1
                if unmeasured > 4:
                    continue
            kept.append(r)
            if len(kept) >= n:
                break
        keys = ("t", "variant", "edit_id", "component", "hypothesis",
                "targets_mode", "delta_S", "delta_C", "accepted", "outcome",
                "bundle", "detail")
        return [{k: r.get(k) for k in keys if r.get(k) is not None}
                for r in reversed(kept)]


# --------------------------------------------------------------- explore --
def stall_flag(trajectory: Sequence[float], t: int, w: int, delta: float) -> int:
    """sigma_t = 1[S_t - S_{t-w} <= delta]; 0 while fewer than ``w`` rounds exist."""
    if t < w or t >= len(trajectory) or t - w < 0:
        return 0
    return int(trajectory[t] - trajectory[t - w] <= delta)


def exploration(t: int, stall: int, tried: Set[str], m_draft: int) -> dict:
    """E_t = (sigma_t, U_t, m_draft) plus the text handed to the proposer."""
    untried = [c for c in K if c not in tried]
    if stall and untried:
        text = (f"STALL: the incumbent has not moved by more than the noise band "
                f"over the last rounds (sigma_t = 1). {m_draft} candidate slot(s) "
                f"this round are RESERVED for exploratory edits on components the "
                f"run has never exercised: {untried}. A variant holding a reserved "
                f"slot must put at least one edit on one of those components.")
    elif untried:
        text = (f"Components not yet exercised in this run: {untried}. Not "
                f"mandatory this round (sigma_t = 0), but evidence about them is "
                f"still missing.")
    else:
        text = "Every component in K has been exercised at least once."
    return {"sigma": stall, "untried": untried, "m_draft": m_draft, "text": text}
