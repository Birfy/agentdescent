"""The text strategies: what evolves, and how a proposal becomes a diff.

A strategy answers two questions and nothing else -- what the artifact *is* (a
flat ``{key: value}`` state) and what a proposal *means* (a :class:`Diff`).
Everything downstream -- merging, conflict resolution, acceptance, versioning --
is the aggregator's job and needs no cooperation.

**One module per strategy family, none of them in the engine.** These three lived
inside :mod:`agentdescent.evolution` while the file-tree strategy had a module of
its own, so the same concept was filed two different ways depending on when it was
written. The engine imports from here; ``from agentdescent.evolution import
AppendRules`` still works.

* here -- text artifacts: one slot, a playbook, keyed categories;
* :mod:`agentdescent.treestrategy` -- a **directory**, one key per file path.

The key space is the design decision, because it is what can be merged
concurrently: content hashes fuse almost everything, a single slot makes every
round a tournament, categories and file paths sit in between.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Sequence, runtime_checkable

from .evolvable import Diff

__all__ = ["Strategy", "AppendRules", "SingleSlot", "KeyedRules", "rule_id"]


def rule_id(text: str) -> str:
    """Content-address a proposal so identical proposals dedupe automatically."""
    return "r" + hashlib.sha1(text.strip().lower().encode()).hexdigest()[:10]


@runtime_checkable
class Strategy(Protocol):
    """Defines *what evolves and how* -- the representation and the merge rule.

    An artifact's state is a flat ``{key: value}`` dict (the diff op-space the
    aggregator resolves conflicts and fusion over). A strategy decides the
    initial state, how it renders, and how a proposal becomes a :class:`Diff`."""

    def initial(self) -> Dict[str, str]: ...

    def render(self, state: Dict[str, str]) -> str: ...

    def to_diff(self, state: Dict[str, str], proposal: str, author: str,
                base_version: int, target: str) -> Optional[Diff]: ...

    # Optional. A strategy that knows, ahead of time, every key it can write should
    # say so: that declared space is what tensor parallelism partitions into
    # sections. A strategy that content-addresses its keys (AppendRules) has no
    # such space, so it simply does not implement this -- and `evolve()` refuses to
    # pair it with TP rather than silently dropping most of its proposals.
    # def keys(self) -> Sequence[str]: ...


@dataclass
class AppendRules:
    """Accumulate a deduped list of rules/lessons (append-only, content-addressed).

    Identical proposals from different workers collapse to one; complementary
    rules are *fused* by the aggregator."""

    title: str = "# Playbook"

    def initial(self) -> Dict[str, str]:
        return {}

    def render(self, state: Dict[str, str]) -> str:
        if not state:
            return f"{self.title}\n(empty)"
        return "\n".join([self.title] + [f"- {state[k]}" for k in sorted(state)])

    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]:
        rid = rule_id(proposal)
        if rid in state:
            return None
        return Diff(diff_id=f"{author}:{rid}:{base_version}", target=target,
                    ops={rid: proposal}, author=author)


@dataclass
class SingleSlot:
    """The artifact **is one value**, and each accepted proposal replaces it.

    The most common thing anyone evolves -- a system prompt, an instruction, one
    document -- and until now every caller wrote this themselves (three of the
    shipped algorithm ports each rolled their own variant). Competing proposals
    contradict on the same key, so the aggregator resolves them on held-out score
    and the best replacement wins:

        evolve(tasks, reward, agent=agent,
               strategy=SingleSlot(initial_value="Answer concisely."))

    ``key`` names the slot in the artifact state and ``initial_value`` seeds it.
    ``min_chars`` is the shortest proposal worth taking, which guards against a
    reflector that replies with a terse non-answer; ``empty_render`` is what the
    artifact renders as before anything has been accepted."""

    initial_value: str = ""
    key: str = "value"
    empty_render: str = "(no instruction yet)"
    min_chars: int = 1

    def keys(self) -> Sequence[str]:
        """The artifact is one slot, so the key space has exactly one member.

        Declared so ``evolve()`` can reject ``TensorParallel`` up front: a single
        key cannot be split into disjoint sections, so every worker but one would
        be authorised for nothing."""
        return [self.key]

    def initial(self) -> Dict[str, str]:
        return {self.key: self.initial_value} if self.initial_value else {}

    def render(self, state: Dict[str, str]) -> str:
        return state.get(self.key) or self.empty_render

    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]:
        text = (proposal or "").strip()
        if len(text) < self.min_chars or state.get(self.key) == text:
            return None
        return Diff(diff_id=f"{author}:{self.key}:{base_version}", target=target,
                    ops={self.key: text}, author=author)


@dataclass
class KeyedRules:
    """One entry per *category*: competing proposals contradict and are resolved.

    Proposals look like ``"category: text"``. A new proposal for an existing
    category **overwrites** it, so two workers proposing different text for the
    same category produce a contradiction the aggregator resolves (keeping the
    one that scores better). Unknown categories fall back to append behaviour."""

    categories: Sequence[str]
    title: str = "# Config (by category)"

    def keys(self) -> Sequence[str]:
        """The declared categories -- the key space tensor parallelism partitions.

        Note that an *unrecognised* proposal still falls back to a content-addressed
        key, which is outside this space; under TP those are reported as
        ``section-violation`` rather than silently dropped."""
        return list(self.categories)

    def initial(self) -> Dict[str, str]:
        return {}

    def render(self, state: Dict[str, str]) -> str:
        if not state:
            return f"{self.title}\n(empty)"
        return "\n".join([self.title] + [f"## {k}\n{state[k]}" for k in sorted(state)])

    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]:
        m = re.match(r"\s*([\w\- ]+?)\s*:\s*(.+)", proposal, re.DOTALL)
        if m and m.group(1).strip().lower() in {c.lower() for c in self.categories}:
            key, value = m.group(1).strip().lower(), m.group(2).strip()
        else:
            key, value = rule_id(proposal), proposal.strip()
        if state.get(key) == value:
            return None
        return Diff(diff_id=f"{author}:{key}:{base_version}", target=target,
                    ops={key: value}, author=author)


