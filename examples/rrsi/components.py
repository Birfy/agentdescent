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
# This file is adapted from google-research/rrsi/rrsi/components.py at
# be50316e1db05914068a973f322770ef08ed7ba1. It is distributed under the
# Apache License, Version 2.0.
"""The editable component set K and its structural subset K_str.

The paper tags every candidate edit with the harness component it modifies
(evidence-aware credit assignment). ``K`` is the fixed vocabulary of those tags;
``K_STR`` are the components that add *machinery* (a tool, a skill file, a
memory store, an extra policy call) rather than changing text or constants.

A proposer declares each edit's component. :func:`normalize` keeps the
declaration only when the diff carries evidence for it, and otherwise recovers
the tag from the diff, so a mislabelled prompt tweak cannot pass as a skill.

Upstream behaviour worth knowing (preserved, not fixed): only the structural
components have generic evidence signals. A declared ``config`` /
``control_flow`` / ``output_plumbing`` / ``context_mgmt`` tag therefore
survives :func:`normalize` only when a *domain* signal backs it; without one
the edit is reclassified (to ``prompt`` unless a structural signal matches).

Defensive boundary (AgentDescent, not upstream parity): :func:`normalize`
checks :func:`text_only` before the declaration, so a string-literal / comment
only diff is always ``prompt``. Upstream keeps a declared structural tag when
the prose happens to contain a signal word, which lets a text edit claim a
reserved structural exploration slot.

This module is an offline mechanism microport. It is not wired into
``evolve()``, a proposal policy, or a resumable run record.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "GENERIC_SIGNALS",
    "K",
    "K_STR",
    "classify_diff",
    "has_evidence",
    "normalize",
    "novelty",
    "text_only",
]

Signals = Sequence[Tuple[str, Sequence[str]]]

K: Tuple[str, ...] = ("prompt", "control_flow", "config", "output_plumbing",
                      "context_mgmt", "client_tool", "skill", "memory", "subagent")
K_STR: Tuple[str, ...] = ("client_tool", "skill", "memory", "subagent")

#: Signals shared by every domain (structural substrate names).
GENERIC_SIGNALS: List[Tuple[str, List[str]]] = [
    ("memory",      [r"\bMemory\(", r"\.remember\(", r"\.recall\(", r"_STATE_DIR"]),
    ("skill",       [r"skills/", r"SkillRegistry", r"skill_use", r"skill_catalog",
                     r"upload_skills"]),
    ("client_tool", [r"ToolRegistry", r"register_tool", r"tool_spec", r"CLIENT_TOOLS"]),
    ("subagent",    [r"\bsubcall\(", r"sub_agent", r"subagent"]),
]

_STRING_LINE = re.compile(r'^[+-]\s*(?:[frb]?["\']|""")')


def _signals(domain_signals: Optional[Signals]) -> List[Tuple[str, Sequence[str]]]:
    return list(domain_signals or []) + list(GENERIC_SIGNALS)


def text_only(diff: str) -> bool:
    """True when every changed line is a string literal or a comment: a
    model-facing text edit, i.e. ``prompt``, whatever words the prose contains."""
    changed = [line for line in diff.splitlines()
               if (line.startswith("+") or line.startswith("-"))
               and not line.startswith("+++") and not line.startswith("---")
               and line[1:].strip()]
    if not changed:
        return False
    return all(_STRING_LINE.match(line) or line[1:].lstrip().startswith("#")
               for line in changed)


def classify_diff(diff: str, domain_signals: Optional[Signals] = None) -> str:
    """First matching component in (domain signals, then generic); default ``prompt``."""
    if text_only(diff):
        return "prompt"
    for component, pats in _signals(domain_signals):
        if any(re.search(p, diff) for p in pats):
            return component
    return "prompt"


def has_evidence(component: str, diff: str,
                 domain_signals: Optional[Signals] = None) -> bool:
    """Does the diff contain any signal of ``component``?

    A declared tag is kept only when the diff carries evidence for it: a
    proposer can name a skill/memory/tool/subagent edit without shipping one,
    or relabel a control-flow change as an untried component to satisfy a
    reserved exploration slot, and an unverified tag would corrupt ``T_t``,
    ``U_t`` and the novelty term."""
    for comp, pats in _signals(domain_signals):
        if comp == component and any(re.search(p, diff) for p in pats):
            return True
    return False


def normalize(declared: Optional[str], diff: str,
              domain_signals: Optional[Signals] = None) -> str:
    """The component an edit is recorded under: the declaration if the diff
    backs it, else :func:`classify_diff`."""
    d = (declared or "").strip().lower()
    if text_only(diff):
        # Not upstream: a prose/comment-only edit that merely *mentions* e.g.
        # ``SkillRegistry`` is a prompt edit, whatever it is declared as.
        return "prompt"
    if d in K and has_evidence(d, diff, domain_signals):
        return d
    return classify_diff(diff, domain_signals)


def novelty(edit_components: Iterable[str], incumbent_counts: Dict[str, int]) -> int:
    """nu(l'): how many STRUCTURAL components the candidate touches that the
    incumbent has never had an accepted edit on. Only ever tie-breaks a
    candidate inside the noise band (Alg. 2 line 5)."""
    return sum(1 for c in set(edit_components)
               if c in K_STR and incumbent_counts.get(c, 0) == 0)
