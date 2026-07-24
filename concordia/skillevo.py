"""Skill self-evolution -- ONE application built on the Concordia framework.

This module is a convenience layer, not the framework itself. It wires the
general pieces (ledger, aggregator, staleness, governance) into a simple loop
for the common case: evolve a text "skill" by accumulating lessons. Everything
about it is customizable:

* **the agent** -- bring any object with ``solve`` / ``propose`` (see
  :class:`Agent`), or wrap a completion from :mod:`concordia.agents` with
  :class:`LLMAgent`. Provider adapters (Claude, etc.) live in
  :mod:`concordia.agents`, NOT here.
* **the reward** -- any ``(task, output) -> [0, 1]`` function.
* **the evolution rule/logic** -- a :class:`SkillStrategy` decides how the skill
  is represented, how it renders into a prompt, and how an agent's proposal
  becomes a :class:`~concordia.evolvable.Diff`. :class:`AppendRules` (the
  default) accumulates deduped lessons; :class:`KeyedRules` keeps one lesson per
  category so competing proposals *contradict* and are resolved by the
  aggregator. Write your own to evolve anything.

For full control, skip this module and drive :class:`~concordia.ledger.Ledger` +
:class:`~concordia.aggregator.Aggregator` directly.
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from .agents import Completion, claude
from .aggregator import Aggregator, AggregatorConfig
from .evolvable import Contract, Diff, EvidenceCard
from .governance import assert_mutable
from .ledger import Ledger
from .scheduler import AuditScheduler
from .staleness import StalenessPolicy


# ---------------------------------------------------------------------------
# Task + Agent contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    """One unit of work the skill is evaluated on."""

    id: str
    prompt: str
    meta: Dict[str, Any] = field(default_factory=dict)


# reward(task, agent_output) -> score in [0, 1]
Reward = Callable[[Task, str], float]


@runtime_checkable
class Agent(Protocol):
    """The task interface an agent implements. Two methods."""

    def solve(self, skill_text: str, task: Task) -> str:
        """Run ``task`` using the skill; return the agent's output."""
        ...

    def propose(self, skill_text: str, task: Task, output: str, reward: float) -> Optional[str]:
        """Reflect on a failure and propose ONE improvement (or None)."""
        ...


_SOLVE_TMPL = (
    "You are executing a skill defined below.\n\n{skill}\n\n"
    "Apply it to this input and output ONLY the result, nothing else.\n\nInput:\n{prompt}"
)
_PROPOSE_TMPL = (
    "A skill just failed a task (score {reward:.2f} out of 1.0).\n\n"
    "Skill so far:\n{skill}\n\nTask input:\n{prompt}\n\n"
    "The skill produced:\n{output}\n\n"
    "Propose exactly ONE concise, general rule (a single imperative sentence) to "
    "improve the skill for this and similar cases. Output only the rule text, or "
    "NONE if no rule would help."
)


@dataclass
class LLMAgent:
    """Adapt a ``Completion`` (from :mod:`concordia.agents`) into an :class:`Agent`.

    Example: ``LLMAgent(claude(model="claude-haiku-4-5"))``."""

    complete: Completion
    solve_template: str = _SOLVE_TMPL
    propose_template: str = _PROPOSE_TMPL

    def solve(self, skill_text: str, task: Task) -> str:
        return self.complete(
            self.solve_template.format(skill=skill_text, prompt=task.prompt)
        ).strip()

    def propose(self, skill_text: str, task: Task, output: str, reward: float) -> Optional[str]:
        rule = self.complete(self.propose_template.format(
            skill=skill_text, prompt=task.prompt, output=output, reward=reward)).strip()
        return None if (not rule or rule.upper().startswith("NONE")) else rule


def claude_agent(model: str = "claude-opus-4-8", max_tokens: int = 1024) -> LLMAgent:
    """Convenience: ``LLMAgent(claude(model))``. The provider code lives in
    :mod:`concordia.agents`; this just wires it to the skillevo task interface."""
    return LLMAgent(claude(model=model, max_tokens=max_tokens))


# ---------------------------------------------------------------------------
# Evolution strategy: how a proposal becomes a change (user-customizable)
# ---------------------------------------------------------------------------


def rule_id(text: str) -> str:
    """Content-address a rule so identical proposals dedupe automatically."""
    return "r" + hashlib.sha1(text.strip().lower().encode()).hexdigest()[:10]


@runtime_checkable
class SkillStrategy(Protocol):
    """Defines *what evolves and how* -- the customizable core of a skill.

    A skill's state is a flat ``{key: value}`` dict (the diff op-space the
    aggregator resolves conflicts and fusion over). A strategy decides the
    initial state, how it renders into a prompt, and how an agent proposal turns
    into a :class:`Diff`."""

    def initial(self) -> Dict[str, str]: ...

    def render(self, state: Dict[str, str]) -> str: ...

    def to_diff(self, state: Dict[str, str], proposal: str, author: str,
                base_version: int, target: str) -> Optional[Diff]: ...


@dataclass
class AppendRules:
    """Default strategy: accumulate a deduped playbook of lessons.

    Each proposal becomes a new content-addressed rule (identical proposals from
    different workers collapse to one; complementary rules from parallel workers
    are *fused* by the aggregator)."""

    title: str = "# Skill Playbook"

    def initial(self) -> Dict[str, str]:
        return {}

    def render(self, state: Dict[str, str]) -> str:
        if not state:
            return f"{self.title}\n(no rules yet)"
        return "\n".join([self.title] + [f"- {state[k]}" for k in sorted(state)])

    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]:
        rid = rule_id(proposal)
        if rid in state:
            return None  # already known
        return Diff(diff_id=f"{author}:{rid}:{base_version}", target=target,
                    ops={rid: proposal}, author=author)


@dataclass
class KeyedRules:
    """Strategy with one lesson per *category*: competing proposals collide.

    Proposals are expected to look like ``"category: text"``. A new proposal for
    an existing category **overwrites** it -- i.e. two workers proposing
    different text for the same category produce a *contradiction* the
    aggregator must resolve (keeping the one that scores better). Unrecognized
    categories fall back to append behaviour. Demonstrates user-defined logic
    that exercises conflict resolution."""

    categories: Sequence[str]
    title: str = "# Skill (by category)"

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


# ---------------------------------------------------------------------------
# Evaluation cache + the evolving artifact
# ---------------------------------------------------------------------------


class _EvalCache:
    def __init__(self) -> None:
        self._d: Dict[Any, float] = {}
        self._lock = threading.Lock()

    def get_or_eval(self, key: Any, fn: Callable[[], float]) -> float:
        with self._lock:
            if key in self._d:
                return self._d[key]
        value = fn()
        with self._lock:
            self._d[key] = value
        return value


class SkillArtifact:
    """An :class:`~concordia.evolvable.Evolvable` skill: flat state + a strategy.

    The strategy handles representation (``render``) and proposal-to-diff; this
    class handles the Evolvable plumbing and agent-based scoring."""

    def __init__(self, id: str, state: Optional[Dict[str, str]] = None,
                 version: int = 1, blast_radius: float = 0.2,
                 runtime: Optional["_SkillRuntime"] = None,
                 strategy: Optional[SkillStrategy] = None) -> None:
        self.id = id
        self.state: Dict[str, str] = dict(state or {})
        self.version = version
        self.blast_radius = blast_radius
        self.contract = Contract(input_schema="task", output_schema="text", major=1)
        self._rt = runtime
        self._strategy = strategy or AppendRules()

    def render(self) -> str:
        return self._strategy.render(self.state)

    def diff(self, other: "SkillArtifact") -> Diff:
        ops = {k: v for k, v in other.state.items() if self.state.get(k) != v}
        return Diff(diff_id=f"{self.id}:diff", target=self.id, ops=ops)

    def apply(self, diff: Diff) -> "SkillArtifact":
        new_state = dict(self.state)
        new_state.update(diff.ops)
        return SkillArtifact(self.id, new_state, self.version + 1, self.blast_radius,
                             self._rt, self._strategy)

    def _signature(self):
        return tuple(sorted(self.state.items()))

    def score(self, tasks: Sequence[Task]) -> float:
        if not tasks or self._rt is None:
            return 0.0
        return sum(self._rt.eval_one(self, t) for t in tasks) / len(tasks)

    def cheap_eval(self, evidence: EvidenceCard) -> float:
        return self.score([t for t in evidence.trajectory_refs if isinstance(t, Task)])

    def full_eval(self, task_set: Sequence[Task]) -> Dict[str, float]:
        return {"reward": self.score(task_set)}


# backwards-compatible alias (the append-rules artifact was formerly RuleSkill)
RuleSkill = SkillArtifact


@dataclass
class _SkillRuntime:
    agent: Agent
    reward: Reward
    cache: _EvalCache

    def eval_one(self, skill: SkillArtifact, task: Task) -> float:
        key = (skill._signature(), task.id)
        return self.cache.get_or_eval(
            key, lambda: self.reward(task, self.agent.solve(skill.render(), task)))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class RoundInfo:
    round: int
    held_out_reward: float
    n_rules: int
    committed: int
    rejected: int


@dataclass
class SkillEvoResult:
    state: Dict[str, str]
    playbook: str
    final_reward: float
    history: List[RoundInfo]
    ledger_log: List[str]

    @property
    def rules(self) -> Dict[str, str]:  # back-compat
        return self.state


def evolve_skill(
    agent: Agent,
    tasks: Sequence[Task],
    reward: Reward,
    *,
    strategy: Optional[SkillStrategy] = None,
    initial_state: Optional[Dict[str, str]] = None,
    rounds: int = 15,
    n_workers: int = 4,
    held_out_frac: float = 0.4,
    skill_id: str = "skill",
    blast_radius: float = 0.2,
    repo_path: Optional[str] = None,
    agg_config: Optional[AggregatorConfig] = None,
    staleness_policy: Optional[StalenessPolicy] = None,
    oracle_budget: int = 200,
    verbose: bool = False,
) -> SkillEvoResult:
    """Evolve a skill with ``agent`` over ``tasks``, scored by ``reward``.

    ``strategy`` (default :class:`AppendRules`) decides how the skill is
    represented and how proposals become diffs. The aggregator dedupes, resolves
    contradictions, fuses complementary changes, and commits a change only if it
    improves held-out reward.

    ``blast_radius`` controls governance: the default ``0.2`` is an L2 (fast,
    local) skill; raise it (e.g. ``0.6``) to evolve an **L1** artifact -- a
    harness module, context policy, tool router, or learned verifier -- which the
    aggregator treats more conservatively (every merge is forced through the
    oracle, and staleness tolerance widens). See the design doc, section 6."""
    import tempfile

    strategy = strategy or AppendRules()
    tasks = list(tasks)
    if len(tasks) < 4:
        raise ValueError("need at least 4 tasks to split train/held-out")
    cut = max(1, int(len(tasks) * (1 - held_out_frac)))
    train, held_out = tasks[:cut], tasks[cut:]
    if not held_out:
        train, held_out = tasks[:-1], tasks[-1:]

    runtime = _SkillRuntime(agent=agent, reward=reward, cache=_EvalCache())

    def serialize(skill: SkillArtifact) -> dict:
        return {"state": skill.state, "blast_radius": skill.blast_radius}

    def deserialize(artifact_id: str, version: int, state: dict) -> SkillArtifact:
        return SkillArtifact(artifact_id, state.get("state", {}), version,
                             state.get("blast_radius", blast_radius), runtime, strategy)

    repo = repo_path or tempfile.mkdtemp(prefix="concordia-skill-")
    ledger = Ledger(repo, serialize, deserialize)
    ledger.register(SkillArtifact(skill_id, initial_state or strategy.initial(),
                                  blast_radius=blast_radius, runtime=runtime,
                                  strategy=strategy))

    from .verifier import ThreeLayerVerifier, VerifierBudget
    verifier = ThreeLayerVerifier(
        eval_fn=lambda skill, ts: skill.score(ts), held_out=held_out,
        rule_subset=min(8, len(held_out)),
        budget=VerifierBudget(oracle_calls_remaining=oracle_budget))
    aggregator = Aggregator(
        ledger, verifier, AuditScheduler(),
        agg_config or AggregatorConfig(batch_trigger=2, max_wait_rounds=1),
        staleness_policy=staleness_policy)

    history: List[RoundInfo] = []
    for r in range(rounds):
        snap = ledger.snapshot(Ledger.DEV)
        skill = snap.get(skill_id)
        base_v = snap.version.get(skill_id, 0)
        assert_mutable(skill)

        for w in range(n_workers):
            task = train[(r * n_workers + w) % len(train)]
            output = agent.solve(skill.render(), task)
            r_score = reward(task, output)
            if r_score >= 0.999:
                continue
            proposal = agent.propose(skill.render(), task, output, r_score)
            if not proposal:
                continue
            diff = strategy.to_diff(skill.state, proposal, f"w{w}", base_v, skill_id)
            if diff is None:
                continue
            after = reward(task, agent.solve(skill.apply(diff).render(), task))
            aggregator.ingest(EvidenceCard(
                diff=diff, base_version={skill_id: base_v}, touched=[skill_id],
                before_after_delta=after - r_score, trajectory_refs=[task]))

        reports = aggregator.step()
        committed = sum(1 for x in reports if x.committed_version is not None)
        rejected = sum(1 for x in reports if x.committed_version is None)
        dev = ledger.snapshot(Ledger.DEV).get(skill_id)
        info = RoundInfo(r, dev.score(held_out), len(dev.state), committed, rejected)
        history.append(info)
        if verbose:
            print(f"round {r:>3}  reward={info.held_out_reward:.3f}  "
                  f"items={info.n_rules}  +{committed}/-{rejected}")

    final = ledger.snapshot(Ledger.DEV).get(skill_id)
    return SkillEvoResult(state=dict(final.state), playbook=final.render(),
                          final_reward=final.score(held_out), history=history,
                          ledger_log=ledger.log(Ledger.DEV, limit=40))
