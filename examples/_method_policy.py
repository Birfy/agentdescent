"""The declarative contract between a candidate method and the runner.

A method is a :class:`MethodPolicy`: an artifact representation (``strategy``),
frozen datasets, pure ``solve``/``propose``/``reward`` functions, and the engine
decision-plane seams its mechanism plugs into (``engine=Policies(...)``). The
definition never sees the recorder, phase strings, modes, budgets, or
thresholds -- :mod:`examples._method_runner` owns all of that, for the same
reason ``agentdescent/policies.py`` gives for the engine: the algorithm is not
allowed to know about the execution plane.

Validation lives in exactly one place: the strategy's ``to_diff``. ``propose``
returns the model's raw text (or ``None``); an unparseable proposal costs its
share of the candidate budget, increments ``invalid_proposals``, and produces no
diff. No fallback substitution happens anywhere.

The shared strategies here all hold **plain-text values per key**, which is what
makes :class:`~agentdescent.fusion.ReflectiveFusion` safe to install on top:
disjoint keys union without any model call, and contested keys are merged as
text. Strategies whose values are code or strict JSON opt out via
``MethodPolicy(reflective=False)``.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from agentdescent.evolution import Task
from agentdescent.evolvable import Diff
from agentdescent.policies import Policies
from agentdescent.strategies import SingleSlot, rule_id

from ._measure import PhasedLLM


Validator = Callable[[str], str]
Solve = Callable[[PhasedLLM, str, Task], str]
Propose = Callable[[PhasedLLM, str, Task, str, float], Optional[str]]
Reward = Callable[[Task, str], float]


def clip_text(value, *, fallback: str = "", max_len: int = 900) -> str:
    """Normalise whitespace and **truncate** to `max_len`.

    It used to discard: `if not cleaned or len(cleaned) > max_len: return
    fallback`. A function named `clip_text` that drops a 901-character answer
    and keeps a 900-character one is not a validation rule anyone chose, and the
    cost lands twice -- the proposal is lost, and it is counted as *invalid*,
    which reads in the metrics as the model producing junk.

    Measured on R-Zero, whose two update prompts ask for a policy statement and
    get one: four of six replies ran 977-1632 characters, so both its fields
    came back empty and the run reported `invalid=41/80`, `44/80`, `48/80`
    against 2-11 for the ports whose prompts ask for a single sentence. Nothing
    was wrong with those proposals except their length.

    Truncation is on a word boundary where one is near the limit, so a clipped
    instruction ends mid-sentence rather than mid-word.
    """
    if not isinstance(value, str):
        return fallback
    cleaned = " ".join(value.split()).strip()
    if not cleaned:
        return fallback
    if len(cleaned) <= max_len:
        return cleaned
    head = cleaned[:max_len]
    space = head.rfind(" ")
    return head[:space] if space > max_len * 0.8 else head


@dataclass
class ValidatedSlot(SingleSlot):
    """`SingleSlot` plus a validator and an invalid-proposal counter.

    The fourth reimplementation of `SingleSlot` in the examples tree is hereby
    retired: this subclasses it and adds only what the candidate matrix needs --
    a `ValueError`-raising validator applied in ``to_diff`` (the single
    validation point) and a thread-safe count of proposals it rejected.
    """

    validator: Validator = staticmethod(lambda text: text.strip())
    invalid_proposals: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _rejected(self) -> None:
        with self._lock:
            self.invalid_proposals += 1

    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]:
        try:
            value = self.validator(proposal or "")
        except (KeyError, TypeError, ValueError):
            self._rejected()
            return None
        if not value:
            self._rejected()
            return None
        return super().to_diff(state, value, author, base_version, target)


FIELD_HEADER = re.compile(r"^\[(?P<name>[\w\-]+)\]$", re.MULTILINE)


def read_fields(rendered: str) -> Dict[str, str]:
    """Invert :meth:`FieldSlots.render` -- tolerant of merged/edited values."""
    fields: Dict[str, str] = {}
    matches = list(FIELD_HEADER.finditer(rendered))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(rendered)
        fields[match.group("name")] = rendered[match.end():end].strip()
    return fields


@dataclass
class FieldSlots:
    """A genome of named plain-text fields; each field is its own ledger key.

    Proposals are model text carrying a JSON object; every declared field that
    parses non-empty and differs from the current state becomes one op. Because
    each field is a separate key holding plain text, two workers editing
    different fields **union-merge** without a model call, and edits to the same
    field are text that :class:`~agentdescent.fusion.ReflectiveFusion` can
    synthesise -- no ranking evaluation either way.
    """

    fields: Mapping[str, str]
    parse: Callable[[str], Dict[str, object]] = None  # type: ignore[assignment]
    max_len: int = 900
    invalid_proposals: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def keys(self) -> Sequence[str]:
        return list(self.fields)

    def initial(self) -> Dict[str, str]:
        return dict(self.fields)

    def render(self, state: Dict[str, str]) -> str:
        return "\n\n".join(
            f"[{name}]\n{state.get(name, default)}"
            for name, default in self.fields.items()
        )

    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]:
        try:
            payload = self.parse(proposal or "")
        except (KeyError, TypeError, ValueError):
            with self._lock:
                self.invalid_proposals += 1
            return None
        ops = {}
        for name in self.fields:
            value = clip_text(payload.get(name), max_len=self.max_len)
            if value and value != state.get(name):
                ops[name] = value
        if not ops:
            with self._lock:
                self.invalid_proposals += 1
            return None
        digest = rule_id(repr(sorted(ops.items())))
        return Diff(diff_id=f"{author}:{digest}:{base_version}", target=target,
                    ops=ops, author=author)


@dataclass
class WindowedMemory:
    """Reflexion's episodic memory: append-only, rendered as the last Ω entries.

    Every accepted reflection gets a fresh content-addressed key prefixed with
    the base version, so entries sort in commit order and **never contradict**:
    parallel workers' reflections merge as a plain union, with no ranking
    evaluation and no model call. ``render`` shows only the last ``window``
    entries -- the paper's Ω=1-3 sliding bound.
    """

    seed_text: str
    window: int = 3
    title: str = "# Episodic memory (most recent last)"
    max_len: int = 900
    invalid_proposals: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def keys(self) -> Sequence[str]:
        return []

    def initial(self) -> Dict[str, str]:
        return {}

    def render(self, state: Dict[str, str]) -> str:
        entries = [state[key] for key in sorted(state)][-self.window:]
        lines = [self.seed_text, self.title]
        lines += [f"- {entry}" for entry in entries] if entries else ["(empty)"]
        return "\n".join(lines)

    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]:
        value = clip_text(proposal or "", max_len=self.max_len)
        if not value or value in state.values():
            with self._lock:
                self.invalid_proposals += 1
            return None
        key = f"{base_version:06d}-{rule_id(value)}"
        return Diff(diff_id=f"{author}:{key}", target=target,
                    ops={key: value}, author=author)


@dataclass
class SkillLibrary:
    """Voyager/SkillWeaver's growing library: one validated entry per goal key.

    Proposals are ``"<key>: <value>"``; the value must pass ``value_validator``.
    Skills for different goals accumulate and union-merge; a second proposal for
    an existing goal contradicts and is resolved by the aggregator. Keys outside
    ``categories`` are rejected (counted), not appended.
    """

    categories: Sequence[str]
    value_validator: Validator
    initial_entries: Mapping[str, str] = field(default_factory=dict)
    title: str = "# Skill library"
    invalid_proposals: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def keys(self) -> Sequence[str]:
        return [c.strip().lower() for c in self.categories]

    def initial(self) -> Dict[str, str]:
        return {k.strip().lower(): v for k, v in self.initial_entries.items()}

    def render(self, state: Dict[str, str]) -> str:
        if not state:
            return f"{self.title}\n(empty)"
        return "\n".join([self.title] + [f"## {k}\n{state[k]}" for k in sorted(state)])

    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]:
        match = re.match(r"\s*([\w\-. /]+?)\s*:\s*(.+)", proposal or "", re.DOTALL)
        try:
            if not match:
                raise ValueError("proposal is not '<key>: <value>'")
            key = match.group(1).strip().lower()
            if key not in set(self.keys()):
                raise ValueError(f"unknown skill key {key!r}")
            value = self.value_validator(match.group(2).strip())
        except (KeyError, TypeError, ValueError):
            with self._lock:
                self.invalid_proposals += 1
            return None
        if not value or state.get(key) == value:
            with self._lock:
                self.invalid_proposals += 1
            return None
        return Diff(diff_id=f"{author}:{key}:{base_version}", target=target,
                    ops={key: value}, author=author)


@dataclass(frozen=True)
class PopulationContext:
    """What a method's own population layer gets from the runner.

    :class:`~examples._population.PopulationAggregator` covers the common case --
    keep every committed head, ask a `SelectionPolicy` who the next parent is.
    A method whose published algorithm says more than that needs more than that:
    PromptBreeder's Algorithm 1 keeps a **fixed-size** population, evaluates the
    two sampled units on a **training batch**, and **replaces the loser**, none
    of which the shared aggregator does or should.

    ``fitness(state, batch)`` scores a state on a **random batch** of the
    *training* split, which is where several papers put their selection signal;
    the shared aggregator's held-out score is the acceptance gate's, a different
    question. The batch is the caller's: PromptBreeder's Algorithm 1 evaluates
    "on a random batch of training data", and scoring the whole split instead is
    both a departure -- the sampling noise is what keeps the tournament a
    comparison rather than a verdict -- and, at one model call per item per unit
    per tournament, the dominant cost of a run.
    """

    selection: object
    artifact_id: str
    conflict: object
    fusion: object
    acceptance: object
    llm: Callable[..., str]
    fitness: Callable[[Dict[str, str], int], float]
    train_size: int
    seed: int


@dataclass(frozen=True)
class MethodPolicy:
    """One candidate method, declaratively.

    ``engine`` carries the decision-plane seams the method's distinctive
    mechanism plugs into (selection / task_sampler / acceptance / ...); the
    runner injects :func:`~agentdescent.fusion.reflective_merge` on top when
    ``reflective`` is true. ``self_verify`` asks the engine for its worker
    self-check rollout per proposal (the environment-graded critic analogue).
    """

    name: str
    fidelity: str
    notes: Tuple[str, ...]
    strategy: object
    train_tasks: Tuple[Task, ...]
    held_out_tasks: Tuple[Task, ...]
    test_tasks: Tuple[Task, ...]
    solve: Solve
    propose: Propose
    reward: Reward
    proposal_calls_per_candidate: int
    engine: Policies = field(default_factory=Policies)
    reflective: bool = True
    self_verify: bool = False
    #: Optional replacement for the shared population layer, for a method whose
    #: paper specifies population dynamics the generic archive does not have.
    #: Takes a :class:`PopulationContext`, returns an ``aggregator_factory``.
    population: Optional[Callable[["PopulationContext"], object]] = None
    #: Optional lines about what this method's own mechanism actually did, printed
    #: after the run summary. For anything a run *chooses* rather than is told:
    #: PromptBreeder samples its operator uniformly, so the realised mix is an
    #: observation, and claiming it is reported without printing it anywhere is
    #: how a documented property turns out to have no code behind it.
    report: Optional[Callable[[], str]] = None

    @property
    def engine_tasks(self) -> List[Task]:
        return list(self.train_tasks) + list(self.held_out_tasks)

    @property
    def held_out_frac(self) -> float:
        return len(self.held_out_tasks) / len(self.engine_tasks)

    @property
    def invalid_proposals(self) -> int:
        return int(getattr(self.strategy, "invalid_proposals", 0))
