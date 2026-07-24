"""Agent-driven skill self-evolution -- the convenient front door.

Everything else in Concordia is a *mechanism* (ledger, aggregator, staleness,
governance). This module is the **ergonomic API** that lets you evolve a real
skill with *any* agent in a few lines:

    from concordia.skillevo import evolve_skill

    result = evolve_skill(agent, tasks, reward, rounds=15, n_workers=4)
    print(result.playbook)      # the evolved skill text
    print(result.final_reward)  # held-out reward

An **agent** is anything that implements the tiny :class:`Agent` protocol -- two
methods, ``solve`` and ``propose``. Bring your own (an LLM, a tool-using loop, a
rule engine), or wrap a completion function with :class:`LLMAgent` /
:func:`claude_agent`.

The evolving artifact is a **playbook**: an accumulating set of rules/lessons
(the ExpeL / "lessons learned" pattern). Each round, workers run tasks through
the current playbook, and on failure ask the agent to *propose* a new rule. The
aggregator then does the hard part -- deduping, resolving contradictions, fusing
complementary rules, and (crucially) **only committing a rule if it improves
held-out reward** under the Beta-posterior test. Bad rules are rejected
automatically; good rules from parallel workers are merged into one playbook.

> Cost note: evaluation runs the agent on held-out tasks, so a real LLM agent
> makes many calls. Keep ``held_out_frac`` / task counts modest, and rely on the
> built-in memoization (identical (playbook, task) pairs are cached within a run).
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from .aggregator import Aggregator, AggregatorConfig
from .evolvable import Contract, Diff, EvidenceCard
from .governance import assert_mutable
from .ledger import Ledger
from .scheduler import AuditScheduler
from .staleness import StalenessPolicy, get_policy
from .verifier import ThreeLayerVerifier, VerifierBudget


# ---------------------------------------------------------------------------
# Task + Agent contract (what a user brings)
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
    """The whole interface an agent must implement. Two methods."""

    def solve(self, skill_text: str, task: Task) -> str:
        """Run ``task`` using the skill playbook; return the agent's output."""
        ...

    def propose(self, skill_text: str, task: Task, output: str, reward: float) -> Optional[str]:
        """Reflect on a failure and propose ONE new rule (or None)."""
        ...


# ---------------------------------------------------------------------------
# LLM adapter: turn any completion function into an Agent
# ---------------------------------------------------------------------------

Completion = Callable[[str], str]

_SOLVE_TMPL = (
    "You are executing a skill defined by the playbook below.\n\n"
    "{skill}\n\n"
    "Apply the playbook to this input and output ONLY the result, nothing else.\n\n"
    "Input:\n{prompt}"
)

_PROPOSE_TMPL = (
    "A skill just failed a task (score {reward:.2f} out of 1.0).\n\n"
    "Playbook so far:\n{skill}\n\n"
    "Task input:\n{prompt}\n\n"
    "The skill produced:\n{output}\n\n"
    "Propose exactly ONE concise, general rule (a single imperative sentence) to "
    "add to the playbook so it handles this and similar cases. Output only the "
    "rule text, or the word NONE if no rule would help."
)


@dataclass
class LLMAgent:
    """Wrap a ``prompt -> completion`` function as an :class:`Agent`.

    Works with any model. See :func:`claude_agent` for a ready-made Claude
    adapter."""

    complete: Completion
    solve_template: str = _SOLVE_TMPL
    propose_template: str = _PROPOSE_TMPL

    def solve(self, skill_text: str, task: Task) -> str:
        return self.complete(
            self.solve_template.format(skill=skill_text, prompt=task.prompt)
        ).strip()

    def propose(self, skill_text: str, task: Task, output: str, reward: float) -> Optional[str]:
        rule = self.complete(
            self.propose_template.format(
                skill=skill_text, prompt=task.prompt, output=output, reward=reward
            )
        ).strip()
        if not rule or rule.strip().upper().startswith("NONE"):
            return None
        return rule


def claude_agent(model: str = "claude-opus-4-8", max_tokens: int = 1024) -> LLMAgent:
    """A Claude-backed :class:`Agent` (requires ``pip install anthropic`` and
    credentials in the environment).

    Swap ``model`` for a cheaper tier (e.g. ``"claude-haiku-4-5"``) when running
    many evolution rounds, since evaluation calls the agent repeatedly."""
    from anthropic import Anthropic  # lazy import; optional dependency

    client = Anthropic()

    def complete(prompt: str) -> str:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    return LLMAgent(complete)


# ---------------------------------------------------------------------------
# Evaluation cache (memoize identical (playbook, task) evaluations in a run)
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


# ---------------------------------------------------------------------------
# RuleSkill: the evolving playbook artifact (an Evolvable)
# ---------------------------------------------------------------------------


def rule_id(text: str) -> str:
    """Content-address a rule so identical proposals from different workers
    dedupe automatically."""
    return "r" + hashlib.sha1(text.strip().lower().encode()).hexdigest()[:10]


class RuleSkill:
    """An :class:`~concordia.evolvable.Evolvable` playbook of rules.

    State is ``{rule_id: rule_text}``. Evaluation (running the agent + scoring)
    is attached via a shared runtime so materialized skills can score
    themselves; only the rules are serialized to the ledger."""

    def __init__(
        self,
        id: str,
        rules: Optional[Dict[str, str]] = None,
        version: int = 1,
        blast_radius: float = 0.2,
        runtime: Optional["_SkillRuntime"] = None,
    ) -> None:
        self.id = id
        self.rules: Dict[str, str] = dict(rules or {})
        self.version = version
        self.blast_radius = blast_radius
        self.contract = Contract(input_schema="task", output_schema="text", major=1)
        self._rt = runtime

    # -- rendering -----------------------------------------------------------

    def render(self) -> str:
        if not self.rules:
            return "# Skill Playbook\n(no rules yet)"
        lines = ["# Skill Playbook"]
        for rid in sorted(self.rules):
            lines.append(f"- {self.rules[rid]}")
        return "\n".join(lines)

    # -- Evolvable protocol --------------------------------------------------

    def diff(self, other: "RuleSkill") -> Diff:
        ops = {k: v for k, v in other.rules.items() if self.rules.get(k) != v}
        return Diff(diff_id=f"{self.id}:diff", target=self.id, ops=ops)

    def apply(self, diff: Diff) -> "RuleSkill":
        new_rules = dict(self.rules)
        new_rules.update(diff.ops)
        return RuleSkill(self.id, new_rules, self.version + 1, self.blast_radius, self._rt)

    def _signature(self):
        return tuple(sorted(self.rules))

    def score(self, tasks: Sequence[Task]) -> float:
        """Mean reward over ``tasks`` (runs the agent; memoized)."""
        if not tasks or self._rt is None:
            return 0.0
        total = 0.0
        for t in tasks:
            total += self._rt.eval_one(self, t)
        return total / len(tasks)

    def cheap_eval(self, evidence: EvidenceCard) -> float:
        tasks = [t for t in evidence.trajectory_refs if isinstance(t, Task)]
        return self.score(tasks)

    def full_eval(self, task_set: Sequence[Task]) -> Dict[str, float]:
        return {"reward": self.score(task_set)}


@dataclass
class _SkillRuntime:
    """Binds an agent + reward + cache to skills materialized from the ledger."""

    agent: Agent
    reward: Reward
    cache: _EvalCache

    def eval_one(self, skill: RuleSkill, task: Task) -> float:
        key = (skill._signature(), task.id)
        return self.cache.get_or_eval(
            key, lambda: self.reward(task, self.agent.solve(skill.render(), task))
        )


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
    rules: Dict[str, str]
    playbook: str
    final_reward: float
    history: List[RoundInfo]
    ledger_log: List[str]


def evolve_skill(
    agent: Agent,
    tasks: Sequence[Task],
    reward: Reward,
    *,
    initial_rules: Optional[Dict[str, str]] = None,
    rounds: int = 15,
    n_workers: int = 4,
    held_out_frac: float = 0.4,
    skill_id: str = "playbook",
    repo_path: Optional[str] = None,
    agg_config: Optional[AggregatorConfig] = None,
    staleness_policy: Optional[StalenessPolicy] = None,
    oracle_budget: int = 200,
    verbose: bool = False,
) -> SkillEvoResult:
    """Evolve a skill playbook using ``agent`` over ``tasks``.

    ``reward(task, output) -> [0,1]`` scores an agent's output. Returns the
    evolved playbook plus a per-round history. All the heavy lifting -- dedup,
    contradiction resolution, fusion, and held-out statistical acceptance --
    happens in the aggregator; a proposed rule is committed only if it actually
    improves held-out reward.
    """
    import tempfile

    tasks = list(tasks)
    if len(tasks) < 4:
        raise ValueError("need at least 4 tasks to split train/held-out")
    cut = max(1, int(len(tasks) * (1 - held_out_frac)))
    train, held_out = tasks[:cut], tasks[cut:]
    if not held_out:
        train, held_out = tasks[:-1], tasks[-1:]

    cache = _EvalCache()
    runtime = _SkillRuntime(agent=agent, reward=reward, cache=cache)

    def serialize(skill: RuleSkill) -> dict:
        return {"rules": skill.rules, "blast_radius": skill.blast_radius}

    def deserialize(artifact_id: str, version: int, state: dict) -> RuleSkill:
        return RuleSkill(artifact_id, state.get("rules", {}), version,
                         state.get("blast_radius", 0.2), runtime)

    repo = repo_path or tempfile.mkdtemp(prefix="concordia-skill-")
    ledger = Ledger(repo, serialize, deserialize)
    ledger.register(RuleSkill(skill_id, initial_rules or {}, runtime=runtime))

    verifier = ThreeLayerVerifier(
        eval_fn=lambda skill, ts: skill.score(ts),
        held_out=held_out,
        rule_subset=min(8, len(held_out)),
        budget=VerifierBudget(oracle_calls_remaining=oracle_budget),
    )
    audit = AuditScheduler()
    aggregator = Aggregator(
        ledger, verifier, audit,
        agg_config or AggregatorConfig(batch_trigger=2, max_wait_rounds=1),
        staleness_policy=staleness_policy,
    )

    history: List[RoundInfo] = []
    for r in range(rounds):
        snap = ledger.snapshot(Ledger.DEV)
        skill = snap.get(skill_id)
        base_version = snap.version
        assert_mutable(skill)

        for w in range(n_workers):
            task = train[(r * n_workers + w) % len(train)]
            output = agent.solve(skill.render(), task)
            r_score = reward(task, output)
            if r_score >= 0.999:
                continue  # already handled
            rule = agent.propose(skill.render(), task, output, r_score)
            if not rule:
                continue
            rid = rule_id(rule)
            if rid in skill.rules:
                continue  # already known
            diff = Diff(diff_id=f"w{w}:{rid}:{base_version.get(skill_id, 0)}",
                        target=skill_id, ops={rid: rule}, author=f"w{w}")
            # local before/after delta on this task (the "gradient").
            after = reward(task, agent.solve(skill.apply(diff).render(), task))
            card = EvidenceCard(
                diff=diff,
                base_version={skill_id: base_version.get(skill_id, 0)},
                touched=[skill_id],
                before_after_delta=after - r_score,
                trajectory_refs=[task],
            )
            aggregator.ingest(card)

        reports = aggregator.step()
        committed = sum(1 for x in reports if x.committed_version is not None)
        rejected = sum(1 for x in reports if x.committed_version is None)
        dev = ledger.snapshot(Ledger.DEV).get(skill_id)
        info = RoundInfo(r, dev.score(held_out), len(dev.rules), committed, rejected)
        history.append(info)
        if verbose:
            print(f"round {r:>3}  reward={info.held_out_reward:.3f}  "
                  f"rules={info.n_rules}  +{committed}/-{rejected}")

    final = ledger.snapshot(Ledger.DEV).get(skill_id)
    return SkillEvoResult(
        rules=dict(final.rules),
        playbook=final.render(),
        final_reward=final.score(held_out),
        history=history,
        ledger_log=ledger.log(Ledger.DEV, limit=40),
    )
