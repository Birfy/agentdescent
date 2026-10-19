"""Meta-evolution: the decision slots of `evolve()` as the artifact.

`evolve()` makes eight replaceable decisions -- which candidate the next batch
starts from, which task a rollout spends, whether a merged candidate commits,
and so on -- and each is a field of :class:`~agentdescent.policies.Policies`.
Every algorithm port plugs its mechanism into one of those fields (or, for a
tree search, into the ``aggregator_factory`` exit that wraps one). This module
lets a run evolve *the field itself*::

    result = meta_evolve(
        problems,                      # inner problems: (slot value, seed) -> MetaOutcome
        slot="selection",              # which decision evolves
        spec=priority_selection(),     # how a value is represented, gated, compiled
        propose=slot_reflector(model, priority_selection()),
        rounds=6, n_workers=4)
    policy = priority_selection().compile(result.rendered)   # -> Policies(selection=policy)

The shape is the ordinary engine applied one level up, and every piece of it is
the same object it always was:

* the **artifact** is the slot's value, held by a :class:`SlotSpec` -- a
  :class:`~agentdescent.strategies.Strategy` that also knows how to *compile* its
  rendered text into the object ``Policies`` takes (a ``SelectionPolicy``, a
  ``TaskSampler``, an ``AcceptancePolicy`` ...);
* a **task** is one inner :class:`Problem`: run a whole inner search or inner
  ``evolve()`` with the candidate value installed, at a fixed budget, from a
  seed, and return what happened as a :class:`MetaOutcome`;
* the **reward** is a function of that outcome -- :func:`auc` by default,
  because a decision rule cannot make a better answer exist, only find one
  sooner, and the final score at a fixed budget barely separates rules;
* **propose** reads the value and the outcome and rewrites the value;
* **governance** is L1: a slot value changes how *everything downstream* is
  searched, so every merge also passes the oracle.

Which slots may evolve is a governance decision, and it is drawn where the
central analogy draws it: the decision plane is the algorithm and may change;
the machinery -- verifier, ledger, executor, sandbox -- is the training code,
and is not self-modifiable. :data:`SLOTS` is that line.

Validation is the other half of the design. A rule that wins on the problems it
was evolved on has shown nothing yet; :func:`meta_validate` scores the value
before and after on problems the outer loop never saw -- other instances, or a
different benchmark entirely -- and reports the gain on both, so "a better
search rule" and "a fit to the training landscape" are two different rows.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import os
import random
import re
import statistics
import tempfile
import threading
import time
import types
from dataclasses import dataclass, field
from types import FunctionType
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Set, Tuple,
    Union, runtime_checkable,
)

from .agents import Completion
from .consolidation import ConsolidationRecord, ConsolidationStore, config_fingerprint
from .evolution import EvolutionResult, Task, evolve
from .evolvable import Diff, EvidenceCard
from .policies import (AcceptancePolicy, AcceptDecision, ConflictPolicy, FusionPolicy,
                       MergeContext, Policies, Promotion, PromotionPolicy,
                       ProposalContext, ProposalPolicy)
from .sampling import TaskSampler
from .selection import Candidate, SelectionContext, SelectionPolicy, SingleHead
from .staleness import StaleAction, StalenessPolicy
from .strategies import SingleSlot

__all__ = [
    "meta_parts",
    "SLOTS",
    "MetaOutcome",
    "Problem",
    "SlotSpec",
    "ParamSlot",
    "SourceSlot",
    "PRIORITY_SEED",
    "PrioritySelection",
    "compile_priority",
    "priority_selection",
    "SLOT_PROTOCOLS",
    "compile_policy_source",
    "policy_source",
    "seed_source",
    "evolve_problem",
    "cached_completion",
    "auc",
    "final_reward",
    "rollouts_to",
    "slot_reflector",
    "meta_evolve",
    "meta_validate",
    "transfer_ratio",
    "meta_run",
    "MetaRunReport",
    "problem_descriptor",
    "problem_from_archive",
    "archive_rule",
    "ContinuousMetaReport",
    "continuous_meta_run",
]


#: The decision plane -- the `Policies` fields a run may evolve. The machinery
#: fields (`verifier`, `ledger`, `executor`, `evaluator`, `eval_cache`,
#: `aggregator_factory`, `sandbox_*`) are the training code and stay frozen.
SLOTS: Tuple[str, ...] = (
    "selection", "task_sampler", "acceptance", "conflict", "fusion",
    "promotion", "staleness", "proposal",
)


# ---------------------------------------------------------------------------
# What an inner run reports, and how it becomes a number
# ---------------------------------------------------------------------------


@dataclass
class MetaOutcome:
    """What one inner run did under a candidate slot value.

    ``curve`` is the inner held-out reward after each sweep, in order -- the
    x-axis of every meta-reward here. ``final`` is what the inner run reported
    at the end, ``rollouts`` what it spent, and ``detail`` whatever the problem
    wants the reflector to see (a tree summary, the outcomes, an error).
    """

    curve: List[float] = field(default_factory=list)
    final: float = 0.0
    rollouts: int = 0
    detail: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(cls, result: EvolutionResult, **detail: Any) -> "MetaOutcome":
        """Read an inner :class:`~agentdescent.evolution.EvolutionResult`."""
        info = {"outcomes": result.outcomes(), "stop_reason": result.stop_reason,
                "error": result.error}
        info.update(detail)
        return cls(curve=[float(h.held_out_reward) for h in result.history],
                   final=float(result.final_reward), rollouts=int(result.rollouts),
                   detail=info)

    def best_so_far(self) -> List[float]:
        out, best = [], -math.inf
        for value in self.curve:
            best = max(best, value)
            out.append(best)
        return out

    def to_json(self) -> str:
        return json.dumps({"curve": self.curve, "final": self.final,
                           "rollouts": self.rollouts, "detail": self.detail},
                          separators=(",", ":"), default=str)

    @classmethod
    def from_json(cls, text: str) -> "MetaOutcome":
        payload = json.loads(text)
        return cls(curve=[float(v) for v in payload.get("curve", [])],
                   final=float(payload.get("final", 0.0)),
                   rollouts=int(payload.get("rollouts", 0)),
                   detail=dict(payload.get("detail", {})))


#: One inner problem: ``(compiled slot value, seed) -> MetaOutcome``. An inner
#: ``evolve()`` with the value installed (:func:`evolve_problem`), an ERA tree
#: search with ``selection=`` set, a synthetic landscape -- anything that runs a
#: whole search under the value and says how it went.
Problem = Callable[[Any, int], MetaOutcome]

MetaReward = Callable[[MetaOutcome], float]


def auc(outcome: MetaOutcome) -> float:
    """Mean best-so-far held-out reward over the inner run: how *fast* it rose.

    The default meta-reward. At a fixed budget two selection rules usually end
    at the same best; the area under the best-so-far curve is what the rule
    actually controlled. Zero when the run recorded no sweep."""
    curve = outcome.best_so_far()
    value = sum(curve) / len(curve) if curve else 0.0
    return min(1.0, max(0.0, value))


def final_reward(outcome: MetaOutcome) -> float:
    """The inner run's own final held-out reward, clipped to ``[0, 1]``."""
    return min(1.0, max(0.0, outcome.final))


def rollouts_to(target: float) -> MetaReward:
    """``1 / (1 + sweeps until the curve first reaches target)``; 0 if never.

    A time-to-quality reward: a rule that reaches the bar in one sweep scores
    0.5, in three sweeps 0.25, and one that never reaches it scores 0."""

    def reward(outcome: MetaOutcome) -> float:
        for index, value in enumerate(outcome.best_so_far()):
            if value >= target:
                return 1.0 / (1.0 + index)
        return 0.0

    return reward


# ---------------------------------------------------------------------------
# How a slot value is represented, gated, and compiled
# ---------------------------------------------------------------------------


@runtime_checkable
class SlotSpec(Protocol):
    """A :class:`~agentdescent.strategies.Strategy` that also compiles.

    ``compile(rendered)`` returns the object ``Policies(<slot>=...)`` takes, and
    must raise ``ValueError`` for a rendering it cannot turn into one -- that is
    the single validation point, and ``to_diff`` must apply the same rule so an
    unusable proposal never becomes a diff. ``describe()`` tells a reflector
    what the surface is and what shape a proposal must have.
    """

    def initial(self) -> Dict[str, str]: ...
    def render(self, state: Dict[str, str]) -> str: ...
    def to_diff(self, state: Dict[str, str], proposal: str, author: str,
                base_version: int, target: str) -> Optional[Diff]: ...
    def compile(self, rendered: str) -> Any: ...
    def describe(self) -> str: ...


_PARAM_LINE = re.compile(r"^\s*([A-Za-z_]\w*)\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$")


@dataclass
class ParamSlot:
    """The numeric hyper-parameters of any policy class, one key each.

    ``FlatPuct(c_puct=1.0, prior_exponent=0.0)``, ``Beam(k=1)``,
    ``DifficultyWeighted(temperature=...)`` -- a value is the constructor
    keywords, held as ``{name: number}``. A proposal is lines of ``name: value``;
    unknown names and out-of-bounds values are refused (counted, no diff).
    Two workers moving *different* parameters union-merge without a model
    call, and the same parameter twice contradicts and is resolved on
    held-out score -- the ordinary key-space argument.

    ``factory`` is called with the parsed keywords; ``bounds`` is optional and
    per parameter, ``(low, high)`` inclusive.
    """

    factory: Callable[..., Any]
    params: Mapping[str, float]
    bounds: Mapping[str, Tuple[float, float]] = field(default_factory=dict)
    title: str = "# Policy parameters"
    invalid_proposals: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def keys(self) -> Sequence[str]:
        return list(self.params)

    def initial(self) -> Dict[str, str]:
        return {name: repr(float(value)) for name, value in self.params.items()}

    def render(self, state: Dict[str, str]) -> str:
        return "\n".join([self.title] + [f"{k}: {state[k]}" for k in sorted(state)])

    def _parse(self, text: str) -> Dict[str, str]:
        ops: Dict[str, str] = {}
        for line in (text or "").splitlines():
            match = _PARAM_LINE.match(line)
            if not match:
                continue
            name, raw = match.group(1), match.group(2)
            if name not in self.params:
                raise ValueError(f"unknown parameter {name!r}")
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"{name} is not finite")
            low, high = self.bounds.get(name, (-math.inf, math.inf))
            if not low <= value <= high:
                raise ValueError(f"{name}={value} is outside [{low}, {high}]")
            ops[name] = repr(value)
        return ops

    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]:
        try:
            ops = {k: v for k, v in self._parse(proposal).items() if state.get(k) != v}
        except ValueError:
            ops = {}
        if not ops:
            with self._lock:
                self.invalid_proposals += 1
            return None
        key = "+".join(sorted(ops))
        return Diff(diff_id=f"{author}:{key}:{base_version}", target=target,
                    ops=ops, author=author)

    def compile(self, rendered: str) -> Any:
        values = self._parse(rendered)
        missing = [name for name in self.params if name not in values]
        if missing:
            raise ValueError(f"rendering lacks {missing}")
        return self.factory(**{name: float(values[name]) for name in self.params})

    def describe(self) -> str:
        lines = [f"- {name} (currently {value}"
                 + (f", within [{self.bounds[name][0]}, {self.bounds[name][1]}]"
                    if name in self.bounds else "") + ")"
                 for name, value in self.params.items()]
        return ("The value is a set of numeric parameters:\n" + "\n".join(lines)
                + "\nReply with one `name: value` line per parameter you change.")


def _unfence(text: str) -> str:
    """One fenced ``python`` block if there is one, else the text."""
    marker = "```"
    if marker not in (text or ""):
        return (text or "").strip()
    chunks = text.split(marker)
    for index in range(1, len(chunks), 2):
        block = chunks[index].strip()
        if block.lower().startswith("python"):
            return block[6:].lstrip("\r\n ")
    return chunks[1].strip()


@dataclass
class SourceSlot(SingleSlot):
    """One slot of validated source, compiled by ``build``.

    The general form: the slot's value is text, ``validate(text) -> text`` is
    the gate (raise ``ValueError`` to refuse), and ``build(text)`` turns the
    accepted text into the policy object. A code fence in a proposal is
    stripped. Contradicting proposals -- two rewrites of the one slot -- are
    resolved on held-out score, so every outer round is a tournament.
    """

    #: ``None`` means "strip whitespace and accept" and "the text is the value".
    #: Not callables, deliberately: a function-valued default renders with a
    #: memory address in the generated API reference, which then differs on
    #: every run.
    validate: Optional[Callable[[str], str]] = None
    build: Optional[Callable[[str], Any]] = None
    description: str = "The value is source text; reply with the complete revised text."
    invalid_proposals: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _validate(self, text: str) -> str:
        return self.validate(text) if self.validate is not None else text.strip()

    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]:
        try:
            value = self._validate(_unfence(proposal or ""))
        except ValueError:
            value = ""
        if not value:
            with self._lock:
                self.invalid_proposals += 1
            return None
        return super().to_diff(state, value, author, base_version, target)

    def compile(self, rendered: str) -> Any:
        # Unfenced first, like `to_diff` and `accepts`: without it a caller who
        # asked `accepts(text)`, was told yes, and then compiled the same text
        # got a ValueError -- the two answered different questions about one
        # string. A rendered artifact carries no fence, so this is a no-op there.
        value = self._validate(_unfence(rendered))
        return self.build(value) if self.build is not None else value

    def accepts(self, proposal: str) -> Tuple[bool, str]:
        """Would :meth:`to_diff` take this proposal? ``(accepted, reason)``.

        Side-effect free: it neither counts an invalid proposal nor builds a
        diff. It exists so a caller that wants to *report* on proposals -- a
        benchmark explaining why a run committed nothing -- does not have to
        re-implement the gate. One that did, and forgot the fence stripping
        `to_diff` does, reported every accepted proposal as rejected.
        """
        try:
            value = self._validate(_unfence(proposal or ""))
        except ValueError as error:
            return False, str(error)
        except Exception as error:  # noqa: BLE001 - a proposal must not raise past here
            return False, f"{type(error).__name__}: {error}"
        if not value:
            return False, "empty after validation"
        return True, ""

    def describe(self) -> str:
        return self.description


# ---------------------------------------------------------------------------
# The shipped selection spec: a tree search's priority rule as gated source
# ---------------------------------------------------------------------------


_PRIORITY_FUNCTION = "priority"
_PRIORITY_ARGS = ("rank", "visits", "total", "prior", "depth", "n_nodes")

#: Upstream ERA's flat PUCT -- ``rank + c * P(s,a) * sqrt(N) / (1 + n)`` with
#: ``c = 1`` and a uniform prior. :class:`PrioritySelection` on this source
#: expands the same node as ``FlatPuct(c_puct=1.0)`` at every step.
PRIORITY_SEED = """def priority(rank, visits, total, prior, depth, n_nodes):
    # Flat PUCT (ERA, futs.py): exploit by rank, explore by visit count.
    c = 1.0
    return rank + c * (1.0 / n_nodes) * math.sqrt(total) / (1 + visits)
"""

_PRIORITY_MAX_CHARS = 2_500

_PRIORITY_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
    ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Expr,
    ast.If, ast.IfExp, ast.BoolOp, ast.And, ast.Or, ast.Not,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UnaryOp, ast.USub, ast.UAdd,
    ast.Call, ast.Attribute, ast.Name, ast.Load, ast.Store, ast.Constant,
)
_PRIORITY_CALLS = {"min", "max", "abs", "float", "int"}
_PRIORITY_MATH = {"sqrt", "log", "log1p", "exp", "tanh", "pow"}

#: The grid every candidate must survive: the root before any expansion
#: (``visits=0, total=0``), a lone node, deep nodes, unrated priors.
_PRIORITY_GRID = [
    (0.5, 0, 0, 1.0, 0, 1),
    (0.0, 0, 1, 0.5, 0, 2),
    (1.0, 3, 7, 0.25, 1, 4),
    (0.33, 12, 40, 0.01, 6, 25),
    (0.9, 1, 300, 0.0, 12, 120),
]


def compile_priority(source: str) -> Callable[..., float]:
    """AST-gate ``source`` and return its ``priority`` function.

    Accepts exactly one function of the six named arguments made of
    arithmetic, comparisons, conditionals, locals, ``min``/``max``/``abs`` and
    ``math.sqrt/log/log1p/exp/tanh/pow``; refuses everything else, and refuses
    a rule that is not a finite number everywhere on a fixed grid of inputs
    that includes the root before any expansion. A rule that divides by
    ``visits`` is refused at proposal time rather than at the root.
    """
    if not source or len(source) > _PRIORITY_MAX_CHARS:
        raise ValueError("policy source is empty or too long")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError("policy source is not valid Python") from error
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != _PRIORITY_FUNCTION:
        raise ValueError(f"policy source must define exactly one function, {_PRIORITY_FUNCTION}")
    if any(not isinstance(node, ast.FunctionDef) for node in tree.body):
        raise ValueError("only the function may appear at module level")
    function = functions[0]
    if tuple(a.arg for a in function.args.args) != _PRIORITY_ARGS or function.args.vararg \
            or function.args.kwarg or function.args.kwonlyargs or function.args.defaults:
        raise ValueError(f"{_PRIORITY_FUNCTION} must take exactly {_PRIORITY_ARGS}")
    for node in ast.walk(tree):
        if not isinstance(node, _PRIORITY_NODES):
            raise ValueError(f"forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == "math"
                    and node.attr in _PRIORITY_MATH):
                raise ValueError("only math.sqrt/log/log1p/exp/tanh/pow may be used")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id not in _PRIORITY_CALLS:
                    raise ValueError(f"forbidden call: {func.id}")
            elif not isinstance(func, ast.Attribute):
                raise ValueError("forbidden call")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder names are forbidden")
    namespace: Dict[str, object] = {
        "__builtins__": {"min": min, "max": max, "abs": abs, "float": float, "int": int},
        "math": math,
    }
    exec(compile(tree, "<candidate-priority>", "exec"), namespace, namespace)
    compiled = namespace.get(_PRIORITY_FUNCTION)
    if not isinstance(compiled, FunctionType):
        raise ValueError(f"{_PRIORITY_FUNCTION} did not compile to a function")
    for point in _PRIORITY_GRID:
        try:
            value = compiled(*point)
        except Exception as error:  # noqa: BLE001 - reported as a shape failure
            raise ValueError(
                f"{_PRIORITY_FUNCTION}{point} raised {type(error).__name__}: {error}")
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or not math.isfinite(float(value)):
            raise ValueError(
                f"{_PRIORITY_FUNCTION}{point} returned {value!r}, not a finite number")
    return compiled  # type: ignore[return-value]


class PrioritySelection(SingleHead):
    """A :class:`~agentdescent.selection.SelectionPolicy` driven by a ``priority`` rule.

    The evolvable surface of a tree search is *which node is expanded next*,
    and this class is the fixed part around it: rank normalisation, prior
    normalisation, depth, the visit reservation up the parent chain, and the
    tie-break are its own; the rule is the compiled function's. A whole
    ``select`` would be able to return a dead node forever or skip the
    reservation and starve the root; a function of six numbers can only be
    wrong about priority, which is the thing being searched for.

    With :data:`PRIORITY_SEED` it is ``FlatPuct(c_puct=1.0, prior_exponent=0.0)``
    to the floating-point bit. The rule is handed the *rated* prior, normalised
    to sum to one (uniform when nobody is rated), so a candidate rule may use
    it; the seed does not, which is upstream's choice.
    """

    def __init__(self, source: str = PRIORITY_SEED) -> None:
        self.source = source
        self.priority = compile_priority(source)

    @staticmethod
    def _ranks(rows: Sequence[Candidate]) -> List[float]:
        if len(rows) == 1:
            return [0.5]
        scores = [-math.inf if c.score is None else c.score for c in rows]
        order = sorted(range(len(rows)), key=lambda i: scores[i])
        ranks = [0.0] * len(rows)
        for rank, index in enumerate(order):
            ranks[index] = rank / (len(rows) - 1)
        return ranks

    @staticmethod
    def _priors(rows: Sequence[Candidate]) -> List[float]:
        rated = [c.prior for c in rows
                 if c.prior is not None and math.isfinite(c.prior) and c.prior > 0]
        if not rated:
            return [1.0 / len(rows)] * len(rows)
        fallback = sum(rated) / len(rated)
        raw = [(c.prior if c.prior is not None and math.isfinite(c.prior) and c.prior > 0
                else fallback) for c in rows]
        total = sum(raw)
        return [v / total for v in raw] if total else [1.0 / len(rows)] * len(rows)

    @staticmethod
    def _depths(rows: Sequence[Candidate]) -> List[int]:
        by_version = {c.version: i for i, c in enumerate(rows)}
        depths: List[int] = []
        for row in rows:
            depth, seen, node = 0, set(), row
            while node.parent is not None and node.parent in by_version \
                    and node.parent not in seen:
                seen.add(node.parent)
                node = rows[by_version[node.parent]]
                depth += 1
            depths.append(depth)
        return depths

    def select(self, ctx: SelectionContext, n: int) -> Sequence[Candidate]:
        rows = list(ctx.candidates)
        if len(rows) <= 1:
            return super().select(ctx, n)
        ranks, priors, depths = self._ranks(rows), self._priors(rows), self._depths(rows)
        visits = [c.selected for c in rows]
        by_version = {c.version: i for i, c in enumerate(rows)}
        picked: List[Candidate] = []
        for _ in range(n):
            total = sum(visits)
            best, best_value = 0, -math.inf
            for i in range(len(rows)):
                value = float(self.priority(ranks[i], visits[i], total, priors[i],
                                            depths[i], len(rows)))
                if value > best_value:
                    best, best_value = i, value
            picked.append(rows[best])
            seen: Set[int] = set()
            node: Optional[int] = best
            while node is not None and node not in seen:
                seen.add(node)
                visits[node] += 1
                parent = rows[node].parent
                node = by_version.get(parent) if parent is not None else None
        return picked


_PRIORITY_DESCRIPTION = f"""The value is the SELECTION RULE of a flat tree search over candidates
(flat PUCT, as in ERA / AlphaZero without a policy network). The tree holds
every node ever expanded; on each iteration the node with the highest priority
is expanded once into a child, and a child may be a dead end (invalid, ranked
last forever). Visits are back-propagated to every ancestor.

Arguments, per node:
- rank: the node's score rank among all nodes, normalised to [0, 1] (1 = best);
- visits: expansions of this node's subtree so far (0 for a fresh node);
- total: sum of visits over all nodes;
- prior: an external rating of this node's promise, normalised to sum to 1
  across nodes (uniform when nobody is rated);
- depth: distance from the root;
- n_nodes: how many nodes the tree holds.

Keep exactly:
    def {_PRIORITY_FUNCTION}({", ".join(_PRIORITY_ARGS)}):
using only arithmetic, comparisons, if/else, local variables, min/max/abs and
math.sqrt/log/log1p/exp/tanh/pow. No imports, loops, or other functions. The
rule must return a finite number for every input, including visits=0 and
total=0. Reply with only the function in one ```python fence, with a one-line
comment saying what changed and why."""


def _priority_source(text: str) -> str:
    source = (text or "").strip()
    compile_priority(source)
    return source + "\n"


def priority_selection(seed: str = PRIORITY_SEED) -> SourceSlot:
    """The shipped spec for the ``selection`` slot of a tree search.

    A :class:`SourceSlot` whose gate is :func:`compile_priority` and whose
    ``build`` is :class:`PrioritySelection`; ``seed`` defaults to upstream
    ERA's flat PUCT."""
    return SourceSlot(initial_value=seed.strip() + "\n", validate=_priority_source,
                      build=PrioritySelection, description=_PRIORITY_DESCRIPTION)



# ---------------------------------------------------------------------------
# The general spec: any slot, as the source of a class satisfying its Protocol
# ---------------------------------------------------------------------------


#: The contract each slot's value must satisfy -- the engine's own Protocols,
#: all ``runtime_checkable``, so a compiled class is checked structurally.
SLOT_PROTOCOLS: Dict[str, type] = {
    "selection": SelectionPolicy,
    "task_sampler": TaskSampler,
    "acceptance": AcceptancePolicy,
    "conflict": ConflictPolicy,
    "fusion": FusionPolicy,
    "promotion": PromotionPolicy,
    "staleness": StalenessPolicy,
    "proposal": ProposalPolicy,
}

#: What a candidate class may import, by module name. These are bound into
#: its namespace, and an ``import`` of anything else is refused at the gate.
#:
#: ``random`` is the exception and is NOT the module -- see
#: :func:`_candidate_modules`.
_SOURCE_MODULES: Dict[str, Any] = {
    "math": math, "statistics": statistics, "json": json, "re": re,
}
for _name in ("random", "itertools", "collections", "functools", "dataclasses",
              "typing", "enum", "heapq", "bisect"):
    _SOURCE_MODULES[_name] = __import__(_name)


def _candidate_modules(source: str, rng_seed: Optional[int] = None) -> Dict[str, Any]:
    """The allowed modules, with ``random`` replaced by a **seeded** generator.

    A candidate policy is allowed to be stochastic -- "revisit the best archived
    head 20% of the time" is a reasonable rule and a model writes it -- but the
    process-wide `random` module makes such a rule a function of global
    interpreter state, and everything above rests on an inner run being a
    function of ``(value, seed)``. :func:`meta_validate` scores the seed value
    and the evolved value on the *same* problem and calls the difference the
    rule; if the rule rolls dice from a shared stream, that difference is partly
    the dice.

    Measured, and why this exists: the first evolved `selection` policy on
    GSM-Hard called ``random.random() < 0.2``. It did not bite -- the branch
    never changed an outcome in that run -- but nothing made that true.

    So `random` here is a `random.Random` **instance**, whose API is the module's
    for everything a scoring rule uses (`random`, `uniform`, `choice`, `shuffle`,
    `sample`, `randint`, `randrange`, `gauss`). It is seeded from the source text
    by default, so the same policy always draws the same stream and two different
    policies draw different ones -- and `SystemRandom`, which would escape this,
    is simply not an attribute of an instance.
    """
    if rng_seed is None:
        rng_seed = int.from_bytes(hashlib.sha256(source.encode("utf-8")).digest()[:8], "big")
    modules = dict(_SOURCE_MODULES)
    modules["random"] = _SeededRandom(rng_seed)
    return modules


class _SeededRandom:
    """A `random.Random` wearing enough of the module's face to stand in for it.

    Delegation rather than subclassing so that exactly two things are decided
    here: `Random` stays available -- a candidate that seeds its own generator
    (`random.Random(ctx.round)`) is the *well-behaved* case and must keep
    working -- and `SystemRandom` does not exist, because it reads the OS
    entropy pool and no seeding can make it reproducible. Everything else is
    the bound generator's own method, so `random.random()`, `random.choice(...)`
    and `random.shuffle(...)` all draw from this stream and nothing else does.
    """

    Random = random.Random

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def __getattr__(self, name: str) -> Any:
        # Only reached for names not found above, so `SystemRandom` raises
        # AttributeError here rather than returning the module's.
        return getattr(self._rng, name)
#: ...and the engine's own value types, so a rule can build what it returns.
def _aggregator_helpers() -> Dict[str, Any]:
    # Imported lazily: aggregator imports policies, which this module imports.
    from .aggregator import diffs_contradict, fuse_diffs
    return {"diffs_contradict": diffs_contradict, "fuse_diffs": fuse_diffs}


_SOURCE_TYPES: Dict[str, Any] = {
    "Candidate": Candidate, "SelectionContext": SelectionContext,
    "StaleAction": StaleAction, "AcceptDecision": AcceptDecision,
    "MergeContext": MergeContext, "Promotion": Promotion,
    "ProposalContext": ProposalContext, "Diff": Diff, "EvidenceCard": EvidenceCard,
}
_SOURCE_PACKAGES = {
    "agentdescent.selection": {"Candidate", "SelectionContext"},
    "agentdescent.staleness": {"StaleAction"},
    "agentdescent.policies": {"AcceptDecision", "MergeContext", "Promotion",
                              "ProposalContext"},
    "agentdescent.evolvable": {"Diff", "EvidenceCard"},
    "agentdescent.aggregator": {"diffs_contradict", "fuse_diffs"},
}
_SAFE_BUILTINS: Dict[str, Any] = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float",
        "frozenset", "int", "isinstance", "iter", "len", "list", "map", "max", "min",
        "next", "object", "pow", "print", "range", "repr", "reversed", "round", "set",
        "slice", "sorted", "str", "sum", "tuple", "zip", "True", "False", "None",
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
        "ZeroDivisionError", "StopIteration", "staticmethod", "classmethod", "property",
    )
    if name in (__builtins__ if isinstance(__builtins__, dict) else dir(__builtins__))
}
_FORBIDDEN_CALLS = {
    "exec", "eval", "compile", "open", "__import__", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "input", "breakpoint", "exit", "quit", "type",
    "super", "memoryview", "bytearray", "help", "dir", "id", "hash",
}
_SOURCE_MAX_CHARS = 12_000


def _gate_source(source: str, class_name: str) -> ast.Module:
    if not source or len(source) > _SOURCE_MAX_CHARS:
        raise ValueError("policy source is empty or too long")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError("policy source is not valid Python") from error
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
    if len(classes) != 1:
        raise ValueError(f"policy source must define exactly one class named {class_name}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in _SOURCE_MODULES:
                    raise ValueError(f"import of {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            allowed = _SOURCE_PACKAGES.get(node.module or "", None)
            if allowed is None:
                if (node.module or "") in _SOURCE_MODULES and node.level == 0:
                    continue
                raise ValueError(f"import from {node.module!r} is not allowed")
            for alias in node.names:
                if alias.name not in allowed:
                    raise ValueError(f"{node.module}.{alias.name} is not importable here")
        elif isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder names are forbidden")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder attributes are forbidden")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in _FORBIDDEN_CALLS:
            raise ValueError(f"forbidden call: {node.func.id}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise ValueError("global/nonlocal are forbidden")
    return tree


def _smoke_selection(policy: Any) -> None:
    rows = tuple(Candidate("a", v, score=s, selected=n, parent=p)
                 for v, s, n, p in [(0, 0.1, 3, None), (1, 0.7, 1, 0), (2, None, 0, 0)])
    ctx = SelectionContext(head=rows[0], candidates=rows, round=2, n_workers=2)
    for n in (1, 2):
        picked = list(policy.select(ctx, n))
        if not picked or len(picked) > n or any(c not in rows for c in picked):
            raise ValueError("select() must return 1..n candidates from ctx.candidates")
    solo = SelectionContext(head=rows[0], candidates=(rows[0],))
    if list(policy.select(solo, 1)) != [rows[0]]:
        raise ValueError("select() with one candidate must return it")


def _smoke_task_sampler(policy: Any) -> None:
    """`pick` must answer with a key from the list it was **just** handed.

    The first version of this checked one fixed key list, and that is not the
    contract: the engine hands a worker its own shard, so the list changes
    between calls, and a sampler that remembers a key and returns it later gets
    a `KeyError` out of the engine's `by_id` lookup. Measured -- every sampler
    a reflector proposed for this slot kept per-task state and returned a stale
    id, the inner run died after two rounds, and the gate had passed all of
    them because the ids never changed under the old smoke test.
    """
    #: Shards a worker could plausibly be handed, in an order that walks every
    #: branch a stateful sampler has: fresh keys, a disjoint shard, a shard whose
    #: keys it has **all** seen and scored (the fallback branch, and the one that
    #: crashed), and a shard it has never seen at all.
    #: Deliberately not in sorted order: a sampler that sorts the caller's list
    #: in place is mutating the engine's shard, and with a sorted fixture that
    #: is invisible.
    shards = [["t2", "t0", "t1"], ["t4", "t3"], ["t2", "t0", "t1"],
              ["t3"], ["u1", "u0"]]
    for round_index, keys in enumerate(shards):
        frozen = list(keys)
        for _ in range(len(frozen) + 1):     # several picks inside one shard
            choice = policy.pick(keys, round_index)
            if choice not in frozen:
                raise ValueError(
                    f"pick() returned {choice!r}, which is not in the keys it was "
                    f"given ({frozen}) -- the engine looks that id up in this "
                    "round's tasks and raises KeyError")
            if list(keys) != frozen:
                raise ValueError("pick() must not mutate the keys it is given")
            policy.record(choice, 1.0 if round_index % 2 else 0.0)


def _smoke_staleness(policy: Any) -> None:
    for eta, alpha, breaking in [(0, 1, False), (1, 1, False), (5, 1, False), (2, 1, True)]:
        if not isinstance(policy.decide(eta, alpha, breaking), StaleAction):
            raise ValueError("decide() must return a StaleAction")
    if not isinstance(getattr(policy, "name", None), str):
        raise ValueError("a staleness policy needs a string `name`")


class _SmokeArtifact:
    """The least an Evolvable can be, for a smoke test that needs one."""

    id = "smoke"
    version = 3
    blast_radius = 0.2

    def __init__(self, state: Optional[Dict[str, str]] = None) -> None:
        self.state = dict(state or {"a": "1"})

    def apply(self, diff: Diff) -> "_SmokeArtifact":
        return _SmokeArtifact({**self.state, **diff.ops})

    def diff(self, other: Any) -> Diff:
        return Diff("smoke-diff", self.id, dict(getattr(other, "state", {})))

    def evidence_eval(self, evidence: EvidenceCard) -> float:
        return 0.0


def _smoke_cards() -> List[EvidenceCard]:
    diffs = [Diff("d1", "smoke", {"a": "2"}, author="w1"),
             Diff("d2", "smoke", {"a": "3"}, author="w2"),      # contradicts d1
             Diff("d3", "smoke", {"b": "1"}, author="w3")]      # disjoint
    return [EvidenceCard(d, {"smoke": 3}, ["smoke"], trajectory_refs=[]) for d in diffs]


def _smoke_acceptance(policy: Any) -> None:
    art = _SmokeArtifact()
    for base, cand in [((8.0, 2.0), (9.0, 1.0)), ((5.0, 5.0), (2.0, 8.0)), ((0.0, 0.0), (0.0, 0.0))]:
        ctx = MergeContext(artifact=art, candidate=art.apply(_smoke_cards()[0].diff),
                           cards=_smoke_cards()[:1], base_counts=base, cand_counts=cand,
                           diff=_smoke_cards()[0].diff)
        decision = policy.accept(ctx)
        if not isinstance(decision, AcceptDecision):
            raise ValueError("accept() must return an AcceptDecision")


def _smoke_conflict(policy: Any) -> None:
    from .aggregator import diffs_contradict

    cards = _smoke_cards()
    result = policy.resolve(_SmokeArtifact(), cards)
    if not (isinstance(result, tuple) and len(result) == 2):
        raise ValueError("resolve() must return (kept_cards, dropped_count)")
    kept, dropped = result
    kept = list(kept)
    if not kept or any(c not in cards for c in kept) or not isinstance(dropped, int):
        raise ValueError("resolve() must keep at least one of the given cards and count the drops")
    for i, x in enumerate(kept):
        for y in kept[i + 1:]:
            if diffs_contradict(x.diff, y.diff):
                raise ValueError("resolve() must not keep two contradicting cards")
    single = list(policy.resolve(_SmokeArtifact(), cards[:1])[0])
    if single != cards[:1]:
        raise ValueError("resolve() of one card must keep it")


def _smoke_fusion(policy: Any) -> None:
    art = _SmokeArtifact()
    disjoint = [_smoke_cards()[0].diff, _smoke_cards()[2].diff]
    for diffs in (disjoint, disjoint[:1]):
        result = policy.select(art, list(diffs))
        if not (isinstance(result, tuple) and len(result) == 3):
            raise ValueError("select() must return (diff, candidate, fused)")
        chosen, candidate, fused = result
        if not isinstance(chosen, Diff) or not isinstance(fused, bool) \
                or not hasattr(candidate, "apply"):
            raise ValueError("select() must return a Diff, an Evolvable and a bool")
        if not set(chosen.ops) <= {k for d in diffs for k in d.ops}:
            raise ValueError("select() must not invent keys none of the diffs proposed")


def _smoke_promotion(policy: Any) -> None:
    from .aggregator import MergeReport

    quiet = MergeReport("smoke", None, False, 1, 1, 0, 0, 0.5, None, "", "below-threshold")
    committed = MergeReport("smoke", _smoke_cards()[0].diff, False, 1, 1, 0, 0, 0.9, 4, "", "committed")
    for reports in ([], [quiet], [committed], [quiet] * 3):
        out = list(policy.observe(reports))
        if any(not isinstance(p, Promotion) for p in out):
            raise ValueError("observe() must return Promotions")


def _smoke_proposal(policy: Any) -> None:
    ctx = ProposalContext(rendered="# playbook", task=None, output="42", reward=0.0)
    out = policy.propose(ctx)
    if isinstance(out, str) or any(not isinstance(p, str) for p in out):
        raise ValueError("propose() must return a sequence of strings")


_SMOKES: Dict[str, Callable[[Any], None]] = {
    "selection": _smoke_selection,
    "task_sampler": _smoke_task_sampler,
    "staleness": _smoke_staleness,
    "acceptance": _smoke_acceptance,
    "conflict": _smoke_conflict,
    "fusion": _smoke_fusion,
    "promotion": _smoke_promotion,
    "proposal": _smoke_proposal,
}


def compile_policy_source(slot: str, source: str, *, class_name: str = "Policy",
                          smoke: Optional[Callable[[Any], None]] = None,
                          rng_seed: Optional[int] = None) -> Any:
    """Gate ``source``, instantiate its ``class_name``, and check it fits ``slot``.

    The gate is structural, not semantic: an AST walk that refuses imports
    outside a fixed allowlist, dunder access, and the calls that reach the
    interpreter (``exec``, ``open``, ``getattr`` ...); then the class is built
    in a namespace holding only safe builtins, the allowed modules and the
    engine's value types, instantiated with no arguments, checked with
    ``isinstance`` against the slot's Protocol, and run through a smoke test --
    one is shipped for every slot, and ``smoke`` replaces it. Everything raises
    ``ValueError``, which is what the strategy's ``to_diff`` turns into "no
    diff, counted".

    This is not a sandbox. It is the same gate SICA and Gödel Agent run their
    self-edits behind: enough to keep a model's rewrite from doing anything but
    deciding, not enough to run untrusted code from a stranger.
    """
    if slot not in SLOT_PROTOCOLS:
        raise ValueError(f"{slot!r} is not an evolvable slot; choose one of {SLOTS}")
    tree = _gate_source(source, class_name)
    modules = _candidate_modules(source, rng_seed)
    builtins = dict(_SAFE_BUILTINS)
    builtins["__build_class__"] = __builtins__["__build_class__"] if isinstance(__builtins__, dict) \
        else __builtins__.__build_class__      # `class` statements need it

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        # An `import` inside a method reaches here; only the allowlist answers.
        if level == 0 and name in modules:
            return modules[name]
        if level == 0 and name in _SOURCE_PACKAGES and set(fromlist or ()) <= _SOURCE_PACKAGES[name]:
            pool = {**_SOURCE_TYPES, **_aggregator_helpers()}
            # SimpleNamespace, not `type(...)()`: a plain function placed in a
            # class body becomes a *bound method* on instantiation, so the
            # namespace object is prepended to every call. Two of the eleven
            # whitelisted names are functions -- `diffs_contradict` and
            # `fuse_diffs`, exactly the helpers a `conflict` or `fusion` policy
            # imports -- and a candidate using one was rejected with
            # "takes 2 positional arguments but 3 were given".
            return types.SimpleNamespace(**{n: pool[n] for n in fromlist})
        raise ImportError(f"import of {name!r} is not allowed")

    builtins["__import__"] = restricted_import
    namespace: Dict[str, Any] = {"__builtins__": builtins, "__name__": "candidate"}
    namespace.update(modules)
    namespace.update(_SOURCE_TYPES)
    namespace.update(_aggregator_helpers())
    try:
        exec(compile(tree, f"<candidate-{slot}>", "exec"), namespace, namespace)
        cls = namespace[class_name]
        policy = cls()
    except Exception as error:  # noqa: BLE001 - the candidate's failure, reported
        raise ValueError(f"policy source failed to build: {type(error).__name__}: {error}")
    if not isinstance(policy, SLOT_PROTOCOLS[slot]):
        raise ValueError(f"{class_name} does not satisfy {SLOT_PROTOCOLS[slot].__name__}")
    check = smoke or _SMOKES.get(slot)
    if check is not None:
        try:
            check(policy)
        except ValueError:
            raise
        except Exception as error:  # noqa: BLE001 - the candidate's failure, reported
            raise ValueError(f"{class_name} failed its smoke test: "
                             f"{type(error).__name__}: {error}")
    return policy


_SEED_SOURCES: Dict[str, str] = {
    "selection": """class Policy:
    # Every worker starts from the current head (the engine's default).
    def select(self, ctx, n):
        return [ctx.head] * n
""",
    "task_sampler": """class Policy:
    # Round robin over the shard (the engine's default).
    def pick(self, keys, round_index):
        return keys[round_index % len(keys)]

    def record(self, task_id, score):
        pass
""",
    "staleness": """class Policy:
    # Guarded: fresh diffs pass, a little lag is rebased, more is discarded.
    name = "guarded"

    def decide(self, eta, alpha, contract_breaking):
        if eta == 0:
            return StaleAction.ACCEPT
        if contract_breaking or eta > alpha:
            return StaleAction.DISCARD
        return StaleAction.REBASE
""",
    "acceptance": """class Policy:
    # Commit when the candidate's full held-out rate beats the base's.
    # (The engine default also draws a Beta posterior against an annealed
    # threshold; this is the simplest rule that satisfies the contract.)
    def accept(self, ctx):
        base = MergeContext.rate(ctx.base_counts)
        cand = MergeContext.rate(ctx.cand_counts)
        if cand > base:
            return AcceptDecision(True, "committed", "", 1.0, cand - base)
        return AcceptDecision(False, "below-threshold",
                              "held-out did not improve", 0.0, cand - base)
""",
    "conflict": """class Policy:
    # First come, first kept: a card that contradicts a kept one is dropped.
    # (The engine default scores the pair on the cheap layer and keeps the
    # better one; that needs the verifier, which bind(verifier) would hand you.)
    def resolve(self, artifact, cards):
        kept, dropped = [], 0
        for card in cards:
            if any(diffs_contradict(card.diff, k.diff) for k in kept):
                dropped += 1
            else:
                kept.append(card)
        return kept, dropped
""",
    "fusion": """class Policy:
    # The union of the surviving diffs goes to the gate (the engine default).
    def select(self, artifact, diffs):
        if len(diffs) == 1:
            return diffs[0], artifact.apply(diffs[0]), False
        union = fuse_diffs(list(diffs))
        return union, artifact.apply(union), True
""",
    "promotion": """class Policy:
    # Promote dev -> stable after K regression-free rounds; a commit or an
    # oracle rejection restarts the clock (the engine default's rule).
    def __init__(self):
        self.k = 3
        self.survival = {}

    def configure(self, config):
        self.k = int(config.promote_after_k)

    def observe(self, reports):
        by_id = {r.artifact_id: r for r in reports}
        out = []
        for aid in set(self.survival) | set(by_id):
            rep = by_id.get(aid)
            if rep is not None and (rep.committed_version is not None
                                    or rep.category == "oracle-rejected"):
                self.survival[aid] = 0
                continue
            self.survival[aid] = self.survival.get(aid, 0) + 1
            if self.survival[aid] >= self.k:
                out.append(Promotion(aid, str(self.survival[aid]) + " rounds survived"))
        return out
""",
    "proposal": """class Policy:
    # Propose nothing on a success; on a failure, one rule restating what the
    # grader wanted. A placeholder shape: the actor's own propose is the default.
    def propose(self, ctx):
        if ctx.reward >= 1.0:
            return []
        return ["On a task like this one, check the answer against the "
                "grader before finishing; the last attempt scored "
                + str(round(ctx.reward, 2)) + "."]
""",
}


def seed_source(slot: str) -> str:
    """A valid starting value for ``slot``, as candidate source.

    ``selection``, ``task_sampler``, ``staleness``, ``fusion`` and ``promotion``
    are the engine's default rule transcribed. ``acceptance`` and ``conflict``
    are the simplest rule that satisfies the contract -- the defaults read the
    verifier or a Beta posterior, which a seed cannot carry -- and the source
    says so in its comment, so a reflector starts from an honest description.
    ``proposal`` is a placeholder shape: the engine's default is the actor's
    own ``propose``, which is not a policy object."""
    if slot not in _SEED_SOURCES:
        raise KeyError(f"no shipped seed for {slot!r}; write one satisfying "
                       f"{SLOT_PROTOCOLS[slot].__name__}")
    return _SEED_SOURCES[slot]


def _protocol_surface(protocol: type) -> str:
    lines = []
    for name, member in vars(protocol).items():
        if name.startswith("_") or not callable(member):
            continue
        try:
            sig = str(inspect.signature(member))
        except (TypeError, ValueError):
            sig = "(...)"
        doc = (inspect.getdoc(member) or "").splitlines()
        lines.append(f"    def {name}{sig}" + (f"  # {doc[0]}" if doc else ""))
    return "\n".join(lines)


def policy_source(slot: str, seed: Optional[str] = None, *, class_name: str = "Policy",
                  smoke: Optional[Callable[[Any], None]] = None,
                  notes: str = "") -> SourceSlot:
    """The general spec: ``slot``'s value is the source of a class satisfying its Protocol.

    ``seed`` defaults to :func:`seed_source`. The gate is
    :func:`compile_policy_source`; ``describe()`` shows the reflector the
    Protocol's method signatures, so it knows what it may change and what it
    must keep. ``notes`` is appended for whatever the slot needs said -- what
    the candidates' ``score`` means in this domain, say.
    """
    if slot not in SLOT_PROTOCOLS:
        raise ValueError(f"{slot!r} is not an evolvable slot; choose one of {SLOTS}")
    seed = seed if seed is not None else seed_source(slot)
    protocol = SLOT_PROTOCOLS[slot]

    def validate(text: str) -> str:
        text = (text or "").strip()
        compile_policy_source(slot, text, class_name=class_name, smoke=smoke)
        return text + "\n"

    def build(text: str) -> Any:
        return compile_policy_source(slot, text, class_name=class_name, smoke=smoke)

    description = (
        f"The value is the Python source of one class named {class_name}, "
        f"filling the engine's `{slot}` slot. It must satisfy this protocol "
        f"({protocol.__name__}), with these methods and no constructor arguments:\n"
        f"{_protocol_surface(protocol)}\n"
        f"Allowed imports: {', '.join(sorted(_SOURCE_MODULES))}; also available: "
        f"{', '.join(sorted(_SOURCE_TYPES))}. No file, network, exec, getattr or "
        f"dunder access. Reply with the complete class in one ```python fence, "
        f"with a one-line comment saying what changed and why."
        + (f"\n\n{notes}" if notes else "")
    )
    validate(seed)
    return SourceSlot(initial_value=seed.strip() + "\n", validate=validate, build=build,
                      description=description)

def cached_completion(complete: Callable[[str], str], directory: str, *,
                      key_extra: str = "") -> Callable[[str], str]:
    """``prompt -> text``, memoised on disk. Makes an inner run reproducible.

    A :class:`~agentdescent.meta.Problem` is documented as
    ``(value, seed) -> MetaOutcome``, and the paired comparison
    :func:`~agentdescent.meta.meta_validate` makes rests on that: score the seed
    rule and the evolved rule on the same problem and seed, and the difference
    is the rule. The engine's ``eval_cache`` memoises the *gate*, and nothing
    memoised the **rollouts** -- so two runs of the same sampler could take
    different proposals and land on different instructions.

    Measured, and the reason this exists: validating the seed rule against
    **itself**, byte for byte, reported a gain of -0.0625 on one 20-task
    GSM-Hard window (one seed better, one worse). A noise floor that size sits
    on top of any effect the outer loop could find, and under L1 governance
    every tie is an oracle veto -- which is exactly what the first two live runs
    produced: three sweeps, `{'oracle-rejected': 3}`, nothing committed.

    Deterministic sampling (``temperature=0``) is necessary and was not
    sufficient: the endpoint returned the same text for the same prompt most of
    the time, not always. Caching the call is what closes it.

    The key covers the prompt and whatever ``key_extra`` names about the
    request (model, temperature, token budget) -- change any of those and the
    old entries are simply not found rather than silently reused.
    """
    os.makedirs(directory, exist_ok=True)
    lock = threading.Lock()
    stats = {"hits": 0, "misses": 0}

    def complete_cached(prompt: str) -> str:
        digest = hashlib.sha256(f"{key_extra}\0{prompt}".encode("utf-8")).hexdigest()
        path = os.path.join(directory, f"{digest}.json")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    entry = json.load(handle)
            except (OSError, ValueError):
                entry = None      # a half-written entry is a miss, not a failure
            cached = entry.get("response") if isinstance(entry, dict) else None
            # Only a well-formed entry is a hit -- counted after it parsed, so a
            # corrupt entry is a miss and only a miss, and the contract stays
            # ``prompt -> str`` even if someone hand-edited the file.
            if isinstance(cached, str):
                with lock:
                    stats["hits"] += 1
                return cached
        response = complete(prompt)
        with lock:
            stats["misses"] += 1
        _store(path, response)
        return response

    def _store(path: str, response: Any) -> None:
        """Best effort: a cache that cannot write must never cost the caller
        the response it already paid for.

        Encoded before the disk is touched (an unencodable response leaves no
        temp file), written whole then renamed (a concurrent reader never sees
        a partial entry), and the temp sibling is removed on every failure path
        so a flaky disk does not accumulate ``*.partial`` files.
        """
        if not isinstance(response, str):
            return
        try:
            blob = json.dumps({"response": response})
        except (TypeError, ValueError):
            return
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir=directory, suffix=".partial")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(blob)
            os.replace(tmp, path)
            tmp = None
        except OSError:
            pass
        finally:
            if tmp is not None:
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    complete_cached.stats = stats          # type: ignore[attr-defined]
    return complete_cached


# ---------------------------------------------------------------------------
# Inner problems
# ---------------------------------------------------------------------------


def evolve_problem(tasks: Sequence[Task], reward: Callable[[Task, str], float], *,
                   slot: str, base: Optional[Policies] = None,
                   **evolve_kwargs: Any) -> Problem:
    """An inner ``evolve()`` as a :class:`Problem`.

    Everything you would pass to :func:`~agentdescent.evolution.evolve` goes
    here; the candidate value is installed at ``Policies(<slot>=value)`` on top
    of ``base`` and the run's ``seed`` is the problem seed. ``verbose`` is
    forced off -- an inner run's round log inside an outer round log is noise.
    """
    if slot not in SLOTS:
        raise ValueError(f"{slot!r} is not an evolvable slot; choose one of {SLOTS}")
    base = base or Policies()
    kwargs = dict(evolve_kwargs)
    kwargs["verbose"] = False

    def problem(value: Any, seed: int) -> MetaOutcome:
        result = evolve(list(tasks), reward, seed=seed,
                        policies=base.merged_with(**{slot: value}), **kwargs)
        return MetaOutcome.from_result(result)

    return problem


# ---------------------------------------------------------------------------
# The outer loop
# ---------------------------------------------------------------------------


def slot_reflector(complete: Completion, spec: SlotSpec,
                   *, max_outcome_chars: int = 2_000) -> Callable[[str, Task, str, float], Optional[str]]:
    """A ``propose`` for :func:`meta_evolve`: one model call per failing rollout.

    The prompt is the spec's own :meth:`SlotSpec.describe`, the current value,
    the inner outcome and its reward; the reply is handed to the spec's gate.
    """

    def propose(rendered: str, task: Task, output: str, reward: float) -> Optional[str]:
        prompt = (
            "You are improving one decision rule of a search algorithm.\n\n"
            f"{spec.describe()}\n\n"
            f"Current value:\n```python\n{rendered}\n```\n\n"
            f"Inner problem: {task.prompt}\n"
            f"What the current value did there (JSON; `curve` is held-out reward "
            f"after each sweep):\n{output[:max_outcome_chars]}\n\n"
            f"Its meta-reward on this problem was {reward:.3f} (higher is better, "
            "1.0 is the ceiling).\n\n"
            "Propose ONE revised value that would raise that reward on problems "
            "like this one -- not on this instance alone."
        )
        return complete(prompt)

    return propose


def _outer_tasks(problems: Union[Sequence[Problem], Mapping[str, Problem]],
                 seeds: Sequence[int]) -> Tuple[List[Task], Dict[str, Problem]]:
    named: Dict[str, Problem] = (dict(problems) if isinstance(problems, Mapping)
                                 else {f"p{i}": p for i, p in enumerate(problems)})
    if not named:
        raise ValueError("meta_evolve() got no problems")
    # Seed-major, not problem-major, and this is not cosmetic: `evolve()` splits
    # train from held-out **by position**, so a list grouped by problem hands the
    # whole of the first problem to train and the whole of the last to the gate --
    # the search then never rolls out on the problem it is judged on. Measured on
    # two GSM-Hard windows x four seeds: every one of twelve rollouts landed on
    # `gsmhard-0`, and every acceptance decision was made on `gsmhard-1`.
    # Interleaving puts every problem on both sides of any positional cut.
    tasks = [Task(id=f"{name}:{seed}", prompt=f"{name} (seed {seed})",
                  meta={"problem": name, "seed": seed})
             for seed in seeds for name in named]
    return tasks, named


def meta_evolve(
    problems: Union[Sequence[Problem], Mapping[str, Problem]],
    *,
    slot: str,
    spec: SlotSpec,
    propose: Optional[Callable[[str, Task, str, float], Optional[str]]] = None,
    model: Optional[Completion] = None,
    meta_reward: Optional[MetaReward] = None,
    seeds: Sequence[int] = (0,),
    blast_radius: float = 0.6,
    artifact_id: str = "policy-slot",
    **evolve_kwargs: Any,
) -> EvolutionResult:
    """Evolve one decision slot of the engine against a set of inner problems.

    Parameters
    ----------
    problems:
        The inner problems, each ``(value, seed) -> MetaOutcome`` -- a list, or a
        mapping from a name to a problem (the name appears in task ids and in
        the reflector's prompt). :func:`evolve_problem` builds one from the
        arguments of an inner ``evolve()``.
    slot:
        Which :class:`~agentdescent.policies.Policies` field the value fills;
        one of :data:`SLOTS`. Recorded, and checked -- machinery fields refuse.
    spec:
        How a value is represented, gated and compiled -- a :class:`SlotSpec`
        such as :func:`priority_selection` or a :class:`ParamSlot`.
    propose, model:
        The reflector. Pass ``propose`` directly, or ``model`` to get
        :func:`slot_reflector` over the spec. One of the two is required.
    meta_reward:
        :class:`MetaOutcome` to ``[0, 1]``; ``None`` is :func:`auc`.
    seeds:
        Inner seeds per problem; each ``(problem, seed)`` pair is one outer
        task, so ``len(problems) * len(seeds)`` tasks in all, split into train
        and held-out by ``held_out_frac`` as ``evolve()`` always does.
    blast_radius, artifact_id:
        Governance. ``0.6`` is L1: the value is a harness and every merge also
        passes the oracle.
    **evolve_kwargs:
        Everything else :func:`~agentdescent.evolution.evolve` takes --
        ``rounds``, ``n_workers``, ``max_concurrency``, ``held_out_frac``,
        ``max_rollouts`` ... ``strategy``, ``run`` and ``reward`` are this
        function's and cannot be passed.

    Returns the ordinary :class:`~agentdescent.evolution.EvolutionResult`;
    ``spec.compile(result.rendered)`` is the evolved value, and
    ``result.rendered`` is what to hand :func:`meta_validate`.
    """
    for taken in ("strategy", "run", "reward", "agent"):
        if taken in evolve_kwargs:
            raise TypeError(f"meta_evolve() sets {taken}= itself")
    if propose is None:
        if model is None:
            raise ValueError("meta_evolve() needs propose= or model=")
        propose = slot_reflector(model, spec)
    tasks, reward, kwargs = meta_parts(
        problems, slot=slot, spec=spec, propose=propose, meta_reward=meta_reward,
        seeds=seeds, blast_radius=blast_radius, artifact_id=artifact_id)
    kwargs.update(evolve_kwargs)
    return evolve(tasks, reward, **kwargs)


def _check_meta_reward(score: Any) -> None:
    """A meta-reward that cannot score is a run of zeros, and looks like a null.

    ``reward()`` turns any failure into ``0.0`` -- deliberately, because a
    candidate whose outcome will not parse *has* earned nothing. That same
    catch swallows a **wiring** mistake: handing over a factory instead of the
    function it returns makes ``float(<closure>)`` raise, every rollout score
    0.0, and the run report a flat null with nothing in it to say why.

    Measured: ``rollouts_to`` is ``rollouts_to(target) -> MetaReward``, and
    naming it where ``auc`` goes scored every outcome 0.000. So try it once,
    here, on an outcome that is known good.
    """
    probe = MetaOutcome(curve=[0.0, 0.5, 1.0], final=1.0, rollouts=3)
    try:
        value = float(score(probe))
    except Exception as error:      # noqa: BLE001 - re-raised with the cause named
        raise ValueError(
            f"meta_reward {getattr(score, '__name__', score)!r} could not score a "
            f"MetaOutcome ({type(error).__name__}: {error}). A reward is "
            "`MetaOutcome -> float`; `rollouts_to` is a *factory* for one and needs "
            "its target, as `rollouts_to(0.9)`.") from error
    if value != value:
        raise ValueError(f"meta_reward {getattr(score, '__name__', score)!r} returned NaN")


def meta_parts(
    problems: Union[Sequence[Problem], Mapping[str, Problem]],
    *,
    slot: str,
    spec: SlotSpec,
    propose: Callable[[str, Task, str, float], Optional[str]],
    meta_reward: Optional[MetaReward] = None,
    seeds: Sequence[int] = (0,),
    blast_radius: float = 0.6,
    artifact_id: str = "policy-slot",
) -> Tuple[List[Task], Callable[[Task, str], float], Dict[str, Any]]:
    """The :func:`evolve` call :func:`meta_evolve` makes, taken apart.

    ``(tasks, reward, kwargs)``, so that ``evolve(tasks, reward, **kwargs)`` is
    the run. Split out because a spec-driven front end -- ``kind:
    "policy_slot"`` in :mod:`agentdescent.evolvespec` -- has to assemble the
    same call to hand the CLI a ``Composition`` it can plan, cost and detach,
    and two copies of this would drift. The seed is compiled here, before any
    rollout, so a slot whose own seed fails its gate is refused by ``plan``
    rather than at the first rollout.
    """
    if slot not in SLOTS:
        raise ValueError(f"{slot!r} is not an evolvable slot; choose one of {SLOTS}")
    tasks, named = _outer_tasks(problems, seeds)
    score = meta_reward or auc
    spec.compile(spec.render(spec.initial()))      # the seed must pass its own gate
    _check_meta_reward(score)

    def run(rendered: str, task: Task) -> str:
        try:
            value = spec.compile(rendered)
        except ValueError as error:
            return MetaOutcome(detail={"error": f"compile: {error}"}).to_json()
        outcome = named[task.meta["problem"]](value, int(task.meta["seed"]))
        return outcome.to_json()

    def reward(task: Task, output: str) -> float:
        try:
            return float(score(MetaOutcome.from_json(output)))
        except (TypeError, ValueError, json.JSONDecodeError):
            return 0.0

    kwargs: Dict[str, Any] = {
        "run": run, "propose": propose, "strategy": spec,
        "blast_radius": blast_radius, "artifact_id": artifact_id,
        "initial_state": spec.initial(),
        "self_verify": False, "solved_threshold": 1.1,
    }
    return tasks, reward, kwargs


def meta_validate(
    spec: SlotSpec,
    before: str,
    after: str,
    problems: Union[Sequence[Problem], Mapping[str, Problem]],
    *,
    seeds: Sequence[int] = (0,),
    meta_reward: Optional[MetaReward] = None,
) -> Dict[str, Dict[str, float]]:
    """Score ``before`` and ``after`` on problems the outer loop never saw.

    Paired by ``(problem, seed)``; per problem: the mean reward of each value,
    the mean paired gain, its standard deviation, and wins / losses. Pass the
    seed value's rendering as ``before`` and ``result.rendered`` as ``after``;
    pass problems and seeds disjoint from the outer run's, or the row measures
    the training set.
    """
    named = (dict(problems) if isinstance(problems, Mapping)
             else {f"p{i}": p for i, p in enumerate(problems)})
    old, new = spec.compile(before), spec.compile(after)
    score = meta_reward or auc
    report: Dict[str, Dict[str, float]] = {}
    for name, problem in named.items():
        base = [score(problem(old, s)) for s in seeds]
        cand = [score(problem(new, s)) for s in seeds]
        deltas = [b - a for a, b in zip(base, cand)]
        report[name] = {
            "n": len(seeds),
            "before": statistics.fmean(base),
            "after": statistics.fmean(cand),
            "gain": statistics.fmean(deltas),
            "gain_sd": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
            "wins": sum(d > 0 for d in deltas),
            "losses": sum(d < 0 for d in deltas),
        }
    return report


def transfer_ratio(report: Mapping[str, Mapping[str, float]], source: str,
                   target: str) -> Optional[float]:
    """Gain on ``target`` over gain on ``source``, from a :func:`meta_validate` report.

    Near 1 is a better rule; near 0 with a positive source gain is a fit to the
    problems it was evolved on; negative is a rule that traded generality for
    them. ``None`` when the source gain is nil -- nothing over nothing is not a
    transfer result."""
    src = report[source]["gain"]
    if abs(src) < 1e-9:
        return None
    return report[target]["gain"] / src


# ---------------------------------------------------------------------------
# The product surface: run, validate, decide
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetaRunReport:
    """What a :func:`meta_run` decided, and why.

    ``meta_evolve`` returns a raw ``EvolutionResult`` whose ``rendered`` is the
    evolved value; wiring it back and judging it is the caller's job. This
    report is that job, done: the rendered value (``value``; compile with
    ``spec.compile(report.value)``), the validation on problems the outer loop
    never saw, the transfer ratio (mean val gain / mean train gain), and an
    ``adopt`` / ``reject`` verdict with its reason. Attached to the run as
    ``result.meta`` so a CLI or a plugin can show the decision without
    recomputing it.
    """

    slot: str
    seed_value: str
    value: str
    validation: Dict[str, Dict[str, float]]
    mean_gain: float
    transfer: Optional[float]
    verdict: str
    reason: str


def meta_run(
    problems: Union[Sequence[Problem], Mapping[str, Problem]],
    val_problems: Union[Sequence[Problem], Mapping[str, Problem]],
    *,
    slot: str,
    spec: SlotSpec,
    propose: Optional[Callable[[str, Task, str, float], Optional[str]]] = None,
    model: Optional[Completion] = None,
    meta_reward: Optional[MetaReward] = None,
    seeds: Sequence[int] = (0,),
    val_seeds: Sequence[int] = (1, 2),
    min_gain: float = 0.0,
    transfer_floor: float = 0.0,
    **evolve_kwargs: Any,
) -> Tuple[MetaRunReport, EvolutionResult]:
    """Evolve one decision slot, validate it on unseen problems, and decide.

    ``meta_evolve`` on ``problems`` (the train set), then ``meta_validate`` of
    the evolved value on both the train problems (at ``seeds``) and
    ``val_problems`` (at ``val_seeds``), then the verdict:

    * ``adopt`` -- the evolved value's mean gain on the unseen problems clears
      ``min_gain`` **and** its transfer ratio (mean val gain / mean train gain)
      is not below ``transfer_floor``;
    * ``reject`` -- otherwise, with the reason (no gain / a fit that does not
      transfer).

    Returns ``(report, result)``. ``report.value`` is the rendered string;
    ``spec.compile(report.value)`` is the object to hand
    ``Policies(<slot>=...)``. ``result.meta`` carries the report so a CLI or
    plugin shows the decision.

    Governance: the caller must keep ``val_problems`` and ``val_seeds``
    disjoint from the training set (overlap raises). That is the same
    final-evaluation isolation every other development set in the framework
    gets.
    """
    if not val_problems:
        raise ValueError("meta_run() needs val_problems to judge the transfer")
    _require_disjoint(problems, val_problems, seeds, val_seeds)
    result = meta_evolve(
        problems, slot=slot, spec=spec, propose=propose, model=model,
        meta_reward=meta_reward, seeds=seeds, **evolve_kwargs)
    seed_value = spec.render(spec.initial())
    value = result.rendered or seed_value
    # Source side: gain on the problems that shaped the value (train + seeds).
    # Target side: gain on problems the outer loop never saw (val + val_seeds).
    train_report = meta_validate(
        spec, seed_value, value, problems,
        seeds=seeds, meta_reward=meta_reward)
    validation = meta_validate(
        spec, seed_value, value, val_problems,
        seeds=val_seeds, meta_reward=meta_reward)
    mean_gain = _mean_gain(validation)
    train_gain = _mean_gain(train_report)
    transfer = _transfer_from(train_gain, mean_gain)
    if mean_gain <= min_gain:
        verdict, reason = "reject", (
            f"evolved value gained {mean_gain:+.3f} on unseen problems, "
            f"not above min_gain {min_gain:+.3f}")
    elif transfer is not None and transfer < transfer_floor:
        verdict, reason = "reject", (
            f"transfer ratio {transfer:.3f} is below transfer_floor "
            f"{transfer_floor:.3f} (a fit that does not transfer)")
    else:
        verdict, reason = "adopt", (
            f"mean gain {mean_gain:+.3f} on unseen problems, "
            f"transfer {transfer if transfer is None else round(transfer, 3)}")
    report = MetaRunReport(
        slot=slot, seed_value=seed_value, value=value,
        validation=validation, mean_gain=mean_gain, transfer=transfer,
        verdict=verdict, reason=reason)
    result.meta = _report_dict(report)
    return report, result


def _named_problems(
    problems: Union[Sequence[Problem], Mapping[str, Problem]],
) -> Dict[str, Problem]:
    return (dict(problems) if isinstance(problems, Mapping)
            else {f"p{i}": p for i, p in enumerate(problems)})


def _require_disjoint(
    problems: Union[Sequence[Problem], Mapping[str, Problem]],
    val_problems: Union[Sequence[Problem], Mapping[str, Problem]],
    seeds: Sequence[int],
    val_seeds: Sequence[int],
) -> None:
    """Raise when the validation set is not isolated from training.

    Isolation is fail-closed on two axes:

    * **explicit names** — when both sides are mappings, shared keys raise
      (the same contract the existing named-overlap tests pin);
    * **problem identity** — the same ``Problem`` object must not appear in
      both sets. Sequence inputs get synthetic labels (``p0``, ``p1``, …)
      independently per side via :func:`_named_problems`; those labels are
      local and must *not* participate in the name check, or every pair of
      nonempty sequences falsely collides on ``p0``. Identity comparison is
      the explicit policy for genuine shared problems under the sequence API.
    """
    train_named = _named_problems(problems)
    val_named = _named_problems(val_problems)
    # Synthetic sequence labels are not a shared namespace — only explicit
    # mapping keys count as names for the overlap gate.
    if isinstance(problems, Mapping) and isinstance(val_problems, Mapping):
        overlap = set(train_named) & set(val_named)
        if overlap:
            raise ValueError(
                f"meta_run() val_problems overlap training problems: "
                f"{sorted(overlap)}; "
                "keep them disjoint so the verdict is not measured on the "
                "problems that shaped the value")
    shared_ids = ({id(p) for p in train_named.values()}
                  & {id(p) for p in val_named.values()})
    if shared_ids:
        train_hit = sorted(
            n for n, p in train_named.items() if id(p) in shared_ids)
        val_hit = sorted(
            n for n, p in val_named.items() if id(p) in shared_ids)
        raise ValueError(
            f"meta_run() val_problems overlap training problems: "
            f"shared problem identity (train={train_hit}, val={val_hit}); "
            "keep them disjoint so the verdict is not measured on the "
            "problems that shaped the value")
    shared_seeds = set(seeds) & set(val_seeds)
    if shared_seeds:
        raise ValueError(
            f"meta_run() val_seeds overlap training seeds: "
            f"{sorted(shared_seeds)}; "
            "keep them disjoint so the verdict is not measured on the seeds "
            "that shaped the value")


def _mean_gain(report: Mapping[str, Mapping[str, float]]) -> float:
    gains = [row["gain"] for row in report.values() if row.get("n", 0) > 0]
    return statistics.fmean(gains) if gains else 0.0


def _transfer_from(train_gain: float, val_gain: float) -> Optional[float]:
    """Mean val gain over mean train gain — the aggregate of :func:`transfer_ratio`.

    Near 1 is a better rule; near 0 with a positive train gain is a fit to the
    problems it was evolved on; negative is a rule that traded generality for
    them. ``None`` when the train gain is nil — nothing over nothing is not a
    transfer result (the same contract as :func:`transfer_ratio`).
    """
    if abs(train_gain) < 1e-9:
        return None
    return val_gain / train_gain


def _report_dict(report: MetaRunReport) -> Dict[str, Any]:
    return {
        "slot": report.slot,
        "seed_value": report.seed_value,
        "value": report.value,
        "validation": report.validation,
        "mean_gain": report.mean_gain,
        "transfer": report.transfer,
        "verdict": report.verdict,
        "reason": report.reason,
    }


# ---------------------------------------------------------------------------
# Continuous meta-learning: the archive drives the next session
# ---------------------------------------------------------------------------


def problem_descriptor(slot: str, **inner: Any) -> Dict[str, Any]:
    """The serializable shape of one inner :class:`Problem`, for an archive.

    ``{"slot": <slot>, "inner": {**inner}}``: ``slot`` names the decision
    field the inner run installed the candidate value into, and ``inner`` is
    everything else ``evolve_problem()`` was called with that is JSON-safe --
    ``rounds``, ``n_workers``, ``target_reward`` ... The parts an archive
    cannot hold (the tasks, and the domain's ``run``/``reward``/``propose``/
    ``strategy``) are not described; :func:`problem_from_archive` gets them
    from the caller, which is the "link, don't copy" rule of
    :class:`~agentdescent.consolidation.ConsolidationStore` applied to a
    problem instead of a trajectory."""
    return {"slot": slot, "inner": dict(inner)}


def problem_from_archive(
    store: ConsolidationStore,
    run_id: str,
    *,
    tasks_for: Callable[[ConsolidationRecord], Sequence[Task]],
    spec: SlotSpec,
    run: Callable[[str, Task], str],
    reward: Callable[[Task, str], float],
    propose: Callable[[str, Task, str, float], Optional[str]],
    strategy: Any,
    **defaults: Any,
) -> Problem:
    """Rebuild an inner :class:`Problem` from an archived rule's descriptor.

    :func:`archive_rule` writes a ``ConsolidationRecord.problem`` describing
    how the rule's inner problem was run -- the slot and the JSON-safe
    ``evolve()`` settings. This reads it back and returns the same
    :class:`Problem`, ``(value, seed) -> MetaOutcome``, with the caller
    supplying what an archive cannot hold: the tasks, via ``tasks_for(record)``
    (the archived configuration maps to task instances), and the domain's
    ``run``/``reward``/``propose``/``strategy``.

    Parameters
    ----------
    store:
        The archive holding the record.
    run_id:
        Which archived run to rebuild from. ``KeyError`` when it is not
        archived or carries no problem descriptor.
    tasks_for:
        Maps an archived :class:`~agentdescent.consolidation.ConsolidationRecord`
        to the inner tasks its problem ran on. The caller owns the data; the
        archive owns the semantics and the configuration.
    spec:
        The outer :class:`SlotSpec` -- used as a guard, not as an input to the
        inner run: the record's ``strategy`` must be this spec's class name,
        the same "adopt through the strategy that wrote the state" rule
        :func:`~agentdescent.consolidation.seed_from_archive` enforces. A
        record archived by a different spec is refused, never silently misread.
    run, reward, propose, strategy:
        The inner problem's own ``evolve()`` parts -- the domain functions.
    **defaults:
        Inner ``evolve()`` settings. Each overrides the archived value: the
        descriptor says how the problem was run *then*; the caller may want it
        run differently *now*.
    """
    record = store.find(run_id)
    if record is None or not record.problem:
        raise KeyError(f"no archived problem for run {run_id!r}")
    if record.strategy and record.strategy != type(spec).__name__:
        raise ValueError(
            f"record {run_id} was archived by {record.strategy!r}, not by "
            f"{type(spec).__name__}; refusing to misread its state")
    desc = record.problem
    slot = desc.get("slot")
    if slot not in SLOTS:
        raise ValueError(f"archived problem names unknown slot {slot!r}")
    inner = {**dict(desc.get("inner", {})), **defaults}
    return evolve_problem(
        list(tasks_for(record)), reward, slot=slot, run=run, propose=propose,
        strategy=strategy, **inner)


def archive_rule(
    store: ConsolidationStore,
    report: MetaRunReport,
    result: EvolutionResult,
    *,
    run_id: str,
    slot: str,
    spec: SlotSpec,
    config: Optional[Dict[str, Any]] = None,
    problem: Optional[Dict[str, Any]] = None,
    artifact_id: str = "meta-rule",
    trajectory_store: Optional[str] = None,
) -> ConsolidationRecord:
    """Write an adopted slot rule back to the archive.

    After :func:`meta_run` (or :func:`continuous_meta_run`) says ``adopt``,
    this is what makes the adoption durable: the evolved value (as the
    artifact state a later session seeds from), the measured reward, the
    configuration, and the problem descriptor that lets a later session
    rebuild the problems the rule was judged on. The next call on the same
    store then starts from a better rule and a known problem -- the loop that
    makes evolution *continue* instead of restart.

    ``report`` is the run's :class:`MetaRunReport` (its ``value``/``slot`` are
    the adopted rule), ``result`` the underlying ``EvolutionResult`` (its
    ``state``/``rendered`` are the evolved value). ``problem`` defaults to
    :func:`problem_descriptor` and should carry the inner settings --
    ``problem_descriptor(slot, rounds=..., n_workers=...)`` -- so
    :func:`problem_from_archive` can rebuild the problem exactly.
    """
    record = store.archive(
        result, run_id=run_id, artifact_id=artifact_id,
        strategy=type(spec).__name__, config=config,
        problem=problem or problem_descriptor(slot),
        trajectory_store=trajectory_store)
    return record


@dataclass(frozen=True)
class ContinuousMetaReport:
    """What one :func:`continuous_meta_run` session did, and why.

    Everything :class:`MetaRunReport` says -- the value, the validation against
    the value this session started from, the verdict -- plus the session facts:
    which archived rule seeded the run (``seeded_from``), and what the evolved
    value gained against the **fresh seed** (``fresh_gain``) -- the honest
    cross-session number, the rule you have now against the rule you had before
    the first session ever ran.
    """

    slot: str
    seed_value: str
    value: str
    validation: Dict[str, Dict[str, float]]
    mean_gain: float
    transfer: Optional[float]
    verdict: str
    reason: str
    seeded_from: Optional[str]
    fresh_seed_value: str
    fresh_gain: float
    fresh_transfer: Optional[float]
    archived: bool
    archived_run_id: Optional[str]
    archive_count: int


def _archived_rules(
    store: ConsolidationStore,
    *,
    slot: str,
    spec: SlotSpec,
    config: Optional[Dict[str, Any]],
    artifact_id: str,
) -> List[ConsolidationRecord]:
    """The archive's rule records for this slot and spec, best first.

    A "rule record" is one that carries a problem descriptor naming ``slot``.
    Same configuration fingerprint first, then measured reward -- the same
    ranking :func:`~agentdescent.consolidation.seed_from_archive` uses, so a
    rule evolved on a different domain is a *weaker* candidate rather than a
    disqualified one (its problem is still a valid unseen problem to train or
    validate against).
    """
    want = config_fingerprint(config)
    name = type(spec).__name__
    rules = []
    for rec in store.iter_archive():
        if rec.artifact_id != artifact_id:
            continue
        if rec.strategy and rec.strategy != name:
            continue
        if not rec.problem or rec.problem.get("slot") != slot:
            continue
        rules.append(rec)
    rules.sort(key=lambda r: (r.config_hash == want, r.final_reward), reverse=True)
    return rules


def _session_problems(
    store: ConsolidationStore,
    rules: Sequence[ConsolidationRecord],
    *,
    tasks_for: Callable[[ConsolidationRecord], Sequence[Task]],
    spec: SlotSpec,
    run: Callable[[str, Task], str],
    reward: Callable[[Task, str], float],
    propose: Callable[[str, Task, str, float], Optional[str]],
    strategy: Any,
    problems: Optional[Union[Sequence[Problem], Mapping[str, Problem]]],
    val_problems: Optional[Union[Sequence[Problem], Mapping[str, Problem]]],
    train_runs: int,
    val_runs: int,
    inner: Dict[str, Any],
) -> Tuple[Dict[str, Problem], Dict[str, Problem]]:
    """The session's train/val problems: archive-derived, caller-filled.

    The top ``train_runs`` rules supply the train problems and the next
    ``val_runs`` the validation ones -- disjoint records, so the outer loop
    never validates on a problem it trained on. When the archive cannot fill a
    set (fewer than ``train_runs`` records for the train set, or fewer than
    ``train_runs + val_runs`` for the validation one), the caller's
    ``problems`` / ``val_problems`` stand in -- an empty archive is an ordinary
    :func:`meta_run`.
    """
    named = (lambda ps: dict(ps) if isinstance(ps, Mapping)
             else {f"p{i}": p for i, p in enumerate(ps or [])})

    def build(rec: ConsolidationRecord) -> Problem:
        return problem_from_archive(
            store, rec.run_id, tasks_for=tasks_for, spec=spec, run=run,
            reward=reward, propose=propose, strategy=strategy, **inner)

    train: Dict[str, Problem] = (
        {f"arch:{rec.run_id}": build(rec) for rec in rules[:train_runs]}
        if len(rules) >= train_runs else {})
    val: Dict[str, Problem] = (
        {f"arch:{rec.run_id}": build(rec)
         for rec in rules[train_runs:train_runs + val_runs]}
        if len(rules) >= train_runs + val_runs else {})
    if not train:
        train = named(problems)
    if not val:
        val = named(val_problems)
    if not train:
        raise ValueError(
            "continuous_meta_run() got no train problems -- the archive has "
            "fewer than train_runs records and problems= was not given")
    if not val:
        raise ValueError(
            "continuous_meta_run() needs val_problems to judge the transfer "
            "-- the archive has too few records to fill val_runs")
    return train, val


def continuous_meta_run(
    store: ConsolidationStore,
    slot: str,
    spec: SlotSpec,
    propose: Optional[Callable[[str, Task, str, float], Optional[str]]] = None,
    *,
    tasks_for: Callable[[ConsolidationRecord], Sequence[Task]],
    run: Callable[[str, Task], str],
    reward: Callable[[Task, str], float],
    inner_propose: Callable[[str, Task, str, float], Optional[str]],
    inner_strategy: Any,
    config: Optional[Dict[str, Any]] = None,
    problems: Optional[Union[Sequence[Problem], Mapping[str, Problem]]] = None,
    val_problems: Optional[Union[Sequence[Problem], Mapping[str, Problem]]] = None,
    artifact_id: str = "meta-rule",
    train_runs: int = 2,
    val_runs: int = 1,
    inner: Optional[Dict[str, Any]] = None,
    model: Optional[Completion] = None,
    meta_reward: Optional[MetaReward] = None,
    seeds: Sequence[int] = (0,),
    val_seeds: Sequence[int] = (1, 2),
    min_gain: float = 0.0,
    transfer_floor: float = 0.0,
    run_id: Optional[str] = None,
    **evolve_kwargs: Any,
) -> Tuple[ContinuousMetaReport, EvolutionResult]:
    """One session of continuous meta-learning: evolve, judge, archive.

    ``meta_run`` evolves a slot value and decides; the archive makes that
    decision durable, and this call closes the loop between sessions:

    1. **Problems come from the archive.** The top ``train_runs`` archived
       rules' problems are the train set and the next ``val_runs`` the
       validation set -- rebuilt by :func:`problem_from_archive` from each
       record's descriptor plus ``tasks_for`` and the inner domain functions,
       so the outer loop never validates on a problem it trained on. An empty
       archive (or one too thin to fill a set) falls back to ``problems`` /
       ``val_problems`` -- the first session is an ordinary :func:`meta_run`.
    2. **The search starts where the archive left off.** The best archived rule
       for this slot, spec and configuration becomes the session's
       ``initial_state``, so the evolution *refines* last session's rule rather
       than rediscovering it; ``report.seed_value`` is that prior.
    3. **The gain is measured twice.** ``mean_gain`` is the evolved value
       against the prior (what this session had to beat); ``fresh_gain`` is it
       against the fresh seed ``spec.initial()`` (what the system had before
       the first session) -- the honest cross-session number.
    4. **Adoption is archived.** On ``adopt`` the new rule is written back via
       :func:`archive_rule` with its problem descriptor, so the next session
       starts from a better rule. ``report.archived`` says whether it was.

    Parameters
    ----------
    store:
        The :class:`~agentdescent.consolidation.ConsolidationStore` that makes
        the loop continuous: rules it already holds seed this session, and an
        adopted rule is written back to it.
    slot, spec, propose, model:
        As :func:`meta_run` -- which :class:`Policies` field evolves, how a
        value is represented/gated/compiled, and the reflector (pass ``propose``
        directly or ``model`` to get :func:`slot_reflector`).
    tasks_for:
        Maps an archived :class:`~agentdescent.consolidation.ConsolidationRecord`
        to the inner tasks of its problem -- the caller's data, read through
        the archive's configuration.
    run, reward, inner_propose, inner_strategy:
        The inner ``evolve()`` domain functions, passed through to
        :func:`problem_from_archive` for every rebuilt problem.
    config:
        This session's configuration (dataset/scorer/model/params), used to
        rank archived rules and written back with the adopted rule.
    problems, val_problems:
        The fallback inner problems, used when the archive cannot fill a set
        (an empty archive is an ordinary ``meta_run``). ``inner`` must describe
        how these were built for the archived descriptor to be faithful.
    artifact_id:
        The archive ``artifact_id`` of rule records.
    train_runs, val_runs:
        How many archived rules supply the train and validation problems.
    inner:
        The JSON-safe inner ``evolve()`` settings (``rounds``, ``n_workers``,
        ``target_reward`` ...): defaults for rebuilt problems and the
        descriptor archived with an adopted rule.
    model, meta_reward, seeds, val_seeds, min_gain, transfer_floor:
        As :func:`meta_run`.
    run_id:
        The archived ``run_id`` for the adopted rule; a timestamp id when
        omitted (pass one explicitly when you need to find it again).
    **evolve_kwargs:
        Everything else :func:`meta_evolve` takes, except ``initial_state``
        (this function sets it from the archive) and ``strategy``/``run``/
        ``reward`` (the inner domain's, passed above).

    Returns ``(report, result)``; ``report.value`` is the evolved value and
    ``report.verdict`` the adopt/reject decision.
    """
    for taken in ("strategy", "run", "reward", "agent", "initial_state"):
        if taken in evolve_kwargs:
            raise TypeError(f"continuous_meta_run() sets {taken}= itself")
    if propose is None and model is None:
        raise ValueError("continuous_meta_run() needs propose= or model=")
    inner = dict(inner or {})
    rules = _archived_rules(store, slot=slot, spec=spec, config=config,
                            artifact_id=artifact_id)
    train, val = _session_problems(
        store, rules, tasks_for=tasks_for, spec=spec, run=run, reward=reward,
        propose=inner_propose, strategy=inner_strategy,
        problems=problems, val_problems=val_problems,
        train_runs=train_runs, val_runs=val_runs, inner=inner)
    _require_disjoint(train, val, seeds, val_seeds)

    prior = rules[0] if rules else None
    prior_state = dict(prior.state) if prior is not None else None
    prior_value = (spec.render(prior_state) if prior_state
                   else spec.render(spec.initial()))
    fresh_value = spec.render(spec.initial())
    result = meta_evolve(
        train, slot=slot, spec=spec, propose=propose, model=model,
        meta_reward=meta_reward, seeds=seeds,
        initial_state=prior_state if prior_state is not None else spec.initial(),
        **evolve_kwargs)
    value = result.rendered or prior_value

    # gain vs the prior, and vs the fresh seed, on both the train and the
    # validation problems -- so the transfer is a real ratio, not a restated
    # val gain.
    validation = meta_validate(spec, prior_value, value, val,
                               seeds=val_seeds, meta_reward=meta_reward)
    fresh_validation = meta_validate(spec, fresh_value, value, val,
                                     seeds=val_seeds, meta_reward=meta_reward)
    train_validation = meta_validate(spec, prior_value, value, train,
                                     seeds=seeds, meta_reward=meta_reward)
    fresh_train_validation = meta_validate(spec, fresh_value, value, train,
                                           seeds=seeds, meta_reward=meta_reward)
    mean_gain = _mean_gain(validation)
    train_gain = _mean_gain(train_validation)
    transfer = _transfer_from(train_gain, mean_gain)
    fresh_gain = _mean_gain(fresh_validation)
    fresh_train_gain = _mean_gain(fresh_train_validation)
    fresh_transfer = _transfer_from(fresh_train_gain, fresh_gain)

    if mean_gain <= min_gain:
        verdict, reason = "reject", (
            f"evolved value gained {mean_gain:+.3f} on unseen problems "
            f"over the prior, not above min_gain {min_gain:+.3f}")
    elif transfer is not None and transfer < transfer_floor:
        verdict, reason = "reject", (
            f"transfer ratio {transfer:.3f} is below transfer_floor "
            f"{transfer_floor:.3f} (train gain {train_gain:+.3f}, "
            f"val gain {mean_gain:+.3f})")
    else:
        verdict, reason = "adopt", (
            f"mean gain {mean_gain:+.3f} on unseen problems over the prior, "
            f"transfer {transfer if transfer is None else round(transfer, 3)}")

    archived, archived_run_id = False, None
    if verdict == "adopt":
        rid = run_id or f"meta-{slot}-{int(time.time())}"
        report = MetaRunReport(
            slot=slot, seed_value=prior_value, value=value,
            validation=validation, mean_gain=mean_gain, transfer=transfer,
            verdict=verdict, reason=reason)
        archive_rule(store, report, result, run_id=rid, slot=slot, spec=spec,
                     config=config, problem=problem_descriptor(slot, **inner),
                     artifact_id=artifact_id)
        archived, archived_run_id = True, rid

    report = ContinuousMetaReport(
        slot=slot, seed_value=prior_value, value=value, validation=validation,
        mean_gain=mean_gain, transfer=transfer, verdict=verdict, reason=reason,
        seeded_from=prior.run_id if prior is not None else None,
        fresh_seed_value=fresh_value, fresh_gain=fresh_gain,
        fresh_transfer=fresh_transfer, archived=archived,
        archived_run_id=archived_run_id, archive_count=store.count())
    try:
        result.meta = _continuous_report_dict(report)
    except AttributeError:
        pass
    return report, result


def _continuous_report_dict(report: ContinuousMetaReport) -> Dict[str, Any]:
    return {
        "slot": report.slot,
        "seed_value": report.seed_value,
        "value": report.value,
        "validation": report.validation,
        "mean_gain": report.mean_gain,
        "transfer": report.transfer,
        "verdict": report.verdict,
        "reason": report.reason,
        "seeded_from": report.seeded_from,
        "fresh_seed_value": report.fresh_seed_value,
        "fresh_gain": report.fresh_gain,
        "fresh_transfer": report.fresh_transfer,
        "archived": report.archived,
        "archived_run_id": report.archived_run_id,
        "archive_count": report.archive_count,
    }
