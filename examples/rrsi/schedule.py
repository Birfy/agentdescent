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
# This file is adapted from google-research/rrsi/rrsi/schedule.py at
# be50316e1db05914068a973f322770ef08ed7ba1. It is distributed under the
# Apache License, Version 2.0.
"""Cosine-annealed L0 edit budget (paper Eq. anneal).

    b_t = ceil( b_min + (b_max - b_min) * 1/2 (1 + cos(pi t / T)) ),  t = 0..T-1

Early rounds may bundle several coordinated edits in one candidate; late rounds
become sparse and attributable. The constraint bounds ||z_t||_0, the number of
independent edits active in one proposal, and nothing else: the set of
mechanisms the harness may eventually contain is not restricted.

This module is an offline mechanism microport. It is not wired into
``evolve()``, a proposal policy, or a resumable run record.
"""

from __future__ import annotations

import math
from typing import List


def _validate_bounds(b_min: int, b_max: int) -> None:
    """Raise ``ValueError`` unless ``1 <= b_min <= b_max`` are plain ints."""
    if not isinstance(b_min, int) or isinstance(b_min, bool):
        raise ValueError("b_min must be an int")
    if not isinstance(b_max, int) or isinstance(b_max, bool):
        raise ValueError("b_max must be an int")
    if b_min < 1 or b_max < 1:
        raise ValueError("b_min and b_max must be >= 1")
    if b_min > b_max:
        raise ValueError("b_min must be <= b_max")


def edit_budget(t: int, T: int, b_min: int, b_max: int) -> int:
    """Return ``b_t`` for round ``t`` (0-indexed) of a ``T``-round run.

    Upstream behaviour preserved:

    * ``T <= 0`` returns ``b_max`` (degenerate horizon).
    * ``t`` is clamped into ``[0, T]``.
    * The float is rounded to 9 decimals before ``ceil`` so ``t = T`` does not
      spuriously bump the budget off floating error.

    Defensive validation (AgentDescent convention, not upstream parity):
    ``b_min`` / ``b_max`` must be positive integers with ``b_min <= b_max``.
    """
    _validate_bounds(b_min, b_max)
    if not isinstance(T, int) or isinstance(T, bool):
        raise ValueError("T must be an int")
    if not isinstance(t, int) or isinstance(t, bool):
        raise ValueError("t must be an int")

    if T <= 0:
        return int(b_max)
    t = max(0, min(int(t), int(T)))
    v = b_min + (b_max - b_min) * 0.5 * (1.0 + math.cos(math.pi * t / T))
    # Guard against 1.0000000002 -> 2 from floating error at t = T.
    return int(math.ceil(round(v, 9)))


def budget_table(T: int, b_min: int, b_max: int) -> List[int]:
    """Return ``[b_0, ..., b_{T-1}]`` for a ``T``-round run."""
    if not isinstance(T, int) or isinstance(T, bool):
        raise ValueError("T must be an int")
    if T < 0:
        raise ValueError("T must be >= 0")
    # Validate bounds up front: for T == 0 the comprehension is empty and
    # would otherwise silently accept invalid bounds.
    _validate_bounds(b_min, b_max)
    return [edit_budget(t, T, b_min, b_max) for t in range(T)]
