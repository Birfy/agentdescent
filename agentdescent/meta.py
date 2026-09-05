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
import inspect
import json
import math
import re
import statistics
import threading
from dataclasses import dataclass, field
from types import FunctionType
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Set, Tuple,
    Union, runtime_checkable,
)

from .agents import Completion
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
    "auc",
    "final_reward",
    "rollouts_to",
    "slot_reflector",
    "meta_evolve",
    "meta_validate",
    "transfer_ratio",
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
        value = self._validate(rendered)
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
_SOURCE_MODULES: Dict[str, Any] = {
    "math": math, "statistics": statistics, "json": json, "re": re,
}
for _name in ("random", "itertools", "collections", "functools", "dataclasses",
              "typing", "enum", "heapq", "bisect"):
    _SOURCE_MODULES[_name] = __import__(_name)
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
    keys = ["t0", "t1", "t2"]
    for round_index in range(4):
        if policy.pick(keys, round_index) not in keys:
            raise ValueError("pick() must return one of the keys")
    policy.record("t1", 0.5)


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
                          smoke: Optional[Callable[[Any], None]] = None) -> Any:
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
    builtins = dict(_SAFE_BUILTINS)
    builtins["__build_class__"] = __builtins__["__build_class__"] if isinstance(__builtins__, dict) \
        else __builtins__.__build_class__      # `class` statements need it

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        # An `import` inside a method reaches here; only the allowlist answers.
        if level == 0 and name in _SOURCE_MODULES:
            return _SOURCE_MODULES[name]
        if level == 0 and name in _SOURCE_PACKAGES and set(fromlist or ()) <= _SOURCE_PACKAGES[name]:
            pool = {**_SOURCE_TYPES, **_aggregator_helpers()}
            return type("_ns", (), {n: pool[n] for n in fromlist})()
        raise ImportError(f"import of {name!r} is not allowed")

    builtins["__import__"] = restricted_import
    namespace: Dict[str, Any] = {"__builtins__": builtins, "__name__": "candidate"}
    namespace.update(_SOURCE_MODULES)
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
    tasks = [Task(id=f"{name}:{seed}", prompt=f"{name} (seed {seed})",
                  meta={"problem": name, "seed": seed})
             for name in named for seed in seeds]
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
    if slot not in SLOTS:
        raise ValueError(f"{slot!r} is not an evolvable slot; choose one of {SLOTS}")
    for taken in ("strategy", "run", "reward", "agent"):
        if taken in evolve_kwargs:
            raise TypeError(f"meta_evolve() sets {taken}= itself")
    if propose is None:
        if model is None:
            raise ValueError("meta_evolve() needs propose= or model=")
        propose = slot_reflector(model, spec)
    tasks, named = _outer_tasks(problems, seeds)
    score = meta_reward or auc
    spec.compile(spec.render(spec.initial()))      # the seed must pass its own gate

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

    evolve_kwargs.setdefault("self_verify", False)
    evolve_kwargs.setdefault("solved_threshold", 1.1)
    return evolve(tasks, reward, run=run, propose=propose, strategy=spec,
                  blast_radius=blast_radius, artifact_id=artifact_id,
                  initial_state=spec.initial(), **evolve_kwargs)


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
