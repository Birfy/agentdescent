"""ACE's grow-and-refine playbook, shipped as a named algorithm (#384).

The mechanism is the paper's context object and Curator: a sectioned, itemised
playbook that **accumulates** content-addressed bullets (near-duplicates in the
same section are refused), and a Curator that applies a structural delta instead
of holding each bullet to a held-out gate. The example keeps its FiNER domain,
prompts and CLI and imports this implementation, so there is one copy of the
algorithm rather than one per entry point.

Use it from a spec as ``"algorithm": "ace"`` (``kind: text``).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..evolvable import Diff
from ..evolution import rule_id
from . import Algorithm, BuildContext, Overrides, Param, register

__all__ = ["ACEPlaybook", "GrowAndRefine", "jaccard"]

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    """Lexical overlap -- a dependency-free proxy for ACE's embedding de-dup."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

class ACEPlaybook:
    """ACE's context object: sectioned, itemised bullets, grown by delta only.

    State is ``{bullet_id: "SECTION :: text"}``. A proposal is ``"section ::
    lesson"``. `to_diff` **appends** a new content-addressed bullet, or returns
    ``None`` when the lesson duplicates (exact or near-duplicate) an existing one
    -- ACE's *grow-and-refine*. It never rewrites existing bullets, so the
    playbook accumulates detail instead of collapsing into a lossy summary."""

    def __init__(self, title: str = "# Playbook (ACE)", dedup_threshold: float = 0.8):
        self.title = title
        self.dedup_threshold = dedup_threshold

    def initial(self) -> Dict[str, str]:
        return {}

    @staticmethod
    def _split(entry: str) -> Tuple[str, str]:
        section, _, text = entry.partition(" :: ")
        return (section or "general"), text

    def render(self, state: Dict[str, str]) -> str:
        if not state:
            return f"{self.title}\n(empty -- no lessons yet)"
        by_section: Dict[str, List[str]] = {}
        for bid in sorted(state):
            section, text = self._split(state[bid])
            by_section.setdefault(section, []).append(f"  - [{bid}] {text}")
        out = [self.title]
        for section in sorted(by_section):
            out.append(f"## {section}")
            out.extend(by_section[section])
        return "\n".join(out)

    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]:
        m = re.match(r"\s*([\w\- /]+?)\s*::\s*(.+)", proposal, re.DOTALL)
        if m:
            section, text = m.group(1).strip().lower(), m.group(2).strip()
        else:
            section, text = "general", proposal.strip()
        if not text:
            return None
        entry = f"{section} :: {text}"
        bid = rule_id(entry)
        if bid in state:
            return None  # exact duplicate
        # grow-and-refine: reject a near-duplicate of an existing bullet in the
        # same section (proactive de-dup, ACE Section 3.3).
        for other in state.values():
            other_section, other_text = self._split(other)
            if other_section == section and jaccard(text, other_text) >= self.dedup_threshold:
                return None
        return Diff(diff_id=f"{author}:{bid}:{base_version}", target=target,
                    ops={bid: entry}, author=author)


class GrowAndRefine:
    """ACE's Curator at the acceptance seam: apply the delta, do not gate it.

    Upstream (`ace/core/curator.py`) validates the Reflector's operations
    *structurally* -- reasoning is a string, operations is a list -- and applies
    them. There is no held-out evaluation of a bullet before it enters the
    playbook, and no acceptance threshold; utility is tracked afterwards by
    per-bullet `helpful=X harmful=Y` counters, and size is managed by the
    de-duplication this port already performs in `ACEPlaybook.to_diff` plus a
    token budget (upstream default 80,000).

    The port previously used the engine's shipped `DefaultAcceptance` -- a Beta
    posterior requiring each bullet to raise held-out reward. That is a different
    algorithm, and on a 139-way concept space it is a *strictly* different one:
    a single bullet teaches one concept, so it can only move a validation split
    that happens to contain that concept. Measured on FiNER at `--pool 3200`,
    five committed bullets covered **4 of 32** validation tasks; widening the
    gate's sample to 64 dropped commits from 5 to **0**. The playbook therefore
    never accumulates, and ACE's whole claim is that it accumulates -- the
    reported +8.6% comes from a large playbook whose *aggregate* coverage is
    high, not from bullets that each prove themselves.

    So this is a fidelity fix, and the effect on the numbers is a consequence of
    it rather than the reason for it. What is given up is real and belongs in
    the row's notes: nothing now stops a wrong lesson entering the playbook,
    which is exactly the exposure upstream carries and manages with counters.
    """

    def accept(self, ctx):
        from agentdescent.policies import AcceptDecision, MergeContext
        base = MergeContext.rate(ctx.base_counts)
        cand = MergeContext.rate(ctx.cand_counts)
        return AcceptDecision(
            accept=True, category="",
            detail=f"grow-and-refine (upstream Curator applies; val {base:.3f} "
                   f"-> {cand:.3f} is recorded, not gated on)",
            observed_delta=cand - base)


# ACE separates the Reflector from the Curator "to avoid conflating what to learn
# with how to store it" (paper Section 3.2). The Curator is this framework's
# deterministic aggregator, so here we only need the Reflector prompt.
_GENERATOR_TMPL = (
    "You are an expert financial-analysis agent tagging figures in SEC filings "
    "with US-GAAP XBRL concepts.\n\n{artifact}\n\n"
    "Use the playbook above. Read the sentence and output ONLY the single XBRL "
    "tag for the amount wrapped in <<>>. Choose exactly one tag from the "
    "candidate list; output the tag name and nothing else.\n\n{prompt}"
)
_REFLECTOR_TMPL = (
    "You are the ACE Reflector. The agent mis-tagged an XBRL amount (score "
    "{reward:.2f}).\n\nPlaybook so far:\n{artifact}\n\nTask:\n{prompt}\n\n"
    "The agent answered:\n{output}\n\n"
    "Diagnose the specific reasoning error and distil ONE concise, general, "
    "reusable lesson that would fix this and similar cases. Format your answer "
    "as `section :: lesson` where `section` is the XBRL concept family (e.g. "
    "`revenue`, `debt`, `shares`). Output only that one line, or NONE."
)


def _build(ctx: BuildContext, p: Dict[str, object]) -> Overrides:
    return Overrides(
        knobs={"strategy": ACEPlaybook(title=str(p["title"]),
                                       dedup_threshold=float(p["dedup_threshold"]))},
        policies={"acceptance": GrowAndRefine()},
        notes=["algorithm ace: grow-and-refine admits every structural delta "
               "(no held-out gate per bullet) and de-duplicates near-identical "
               "lessons in the same section; utility counters are upstream's job, "
               "not this engine's."])


register(Algorithm(
    name="ace",
    summary="ACE's sectioned playbook: append content-addressed lessons, refuse "
            "near-duplicates, and grow-and-refine instead of gating each bullet on "
            "held-out reward.",
    kinds=("text",),
    fidelity="mechanism_microport",
    params=(
        Param("title", "str", "# Playbook (ACE)", "rendered playbook title"),
        Param("dedup_threshold", "float", 0.8,
              "Jaccard similarity at or above which a same-section lesson is a duplicate",
              minimum=0.0, maximum=1.0),
    ),
    build=_build,
    source="ACE: Agentic Context Engineering (Zhang et al., 2025)",
    extra_calls="none (one reflection call per failed rollout, as the default recipe)",
    doc="algo-ace.md"))
