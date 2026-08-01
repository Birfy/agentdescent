"""One call from a directory on disk to an evolved directory on disk.

Three wrappers over :func:`~agentdescent.evolution.evolve`, one per thing people
actually point at:

    evolve_skill_dir(path, ...)    a skill directory   -> L2, merged freely
    evolve_agent_dir(path, ...)    an agent directory  -> L1, oracle-gated
    evolve_agent_code(path, ...)   a small codebase    -> L1 + a test gate

They are wrappers, not a second system: same engine, same ``EvolutionResult``,
and any extra keyword argument passes straight through. What they contribute is
the wiring that is identical every time (load the tree, build the runner, build
the reflector, install the governance layer) and **two defaults that the plain
engine gets wrong for this workload**:

* ``self_verify=False`` -- the default re-runs each proposal's trajectory, which
  on a real coding agent doubles the cost of every proposal;
* ``cheap_eval_tasks=4`` -- otherwise the aggregator's *ranking* passes score
  every candidate on the whole held-out set, and with ``eval_fn`` running an
  agent that is the dominant cost of the run (see the note in
  ``evolution._build_engine``).

Both are overridable; they are defaults here because the cost of getting them
wrong is measured in dollars rather than in a warning.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

from .agents import Completion
from .aggregator import AggregatorConfig
from .evolution import EvolutionResult, Task, evolve, tasks_from
from .filetree import TreeSpec, load_tree
from .runners import TEST_FAILURE_MARKER, code_runner, tree_runner
from .skill import SCORERS
from .treestrategy import FileTree, tree_reflector

__all__ = ["evolve_skill_dir", "evolve_agent_dir", "evolve_agent_code"]

#: L2: a skill is local, so its changes merge on held-out reward alone.
SKILL_BLAST_RADIUS = 0.2
#: L1: an agent definition or its code is a harness -- every merge is forced
#: through the oracle by :meth:`~agentdescent.scheduler.AuditScheduler.force_oracle`.
#: Free, as it happens: the oracle scores the same artifact on the same held-out
#: set that the acceptance test just scored, and the evaluation cache serves it.
HARNESS_BLAST_RADIUS = 0.6


def _artifact_id(path: str, name: Optional[str]) -> str:
    import os

    raw = name or os.path.basename(os.path.abspath(os.path.expanduser(path)))
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-") or "artifact"
    return cleaned


def _as_tasks(data: Sequence[Any], prompt: str, gold: str) -> Sequence[Task]:
    if not data:
        raise ValueError("no tasks given")
    return (list(data) if isinstance(data[0], Task)
            else tasks_from(data, prompt=prompt, gold=gold))


def _as_reward(score: Union[str, Callable]) -> Callable:
    if callable(score):
        return score
    if score in SCORERS:
        return SCORERS[score]()
    raise ValueError(
        f"unknown score={score!r}; use one of {sorted(SCORERS)} or pass a "
        "callable (task, output) -> float")


def _build(path, *, spec, editable, frozen, max_files_per_diff, agg_config):
    spec = spec or TreeSpec()
    cfg = agg_config or AggregatorConfig(batch_trigger=2, max_wait_rounds=1)
    spec.validate_against(cfg.trust_region_chars)
    tree = load_tree(path, spec)
    strategy = FileTree(tree, editable=editable, frozen=frozen,
                        max_files_per_diff=max_files_per_diff,
                        max_file_bytes=spec.max_file_bytes)
    return tree, strategy, cfg


def _defaults(kwargs: Dict[str, Any], n_tasks: int) -> None:
    held_out_frac = kwargs.get("held_out_frac", 0.3)
    workers = max(1, min(4, max(1, int(n_tasks * (1 - held_out_frac)))))
    for key, value in (("rounds", 6), ("n_workers", workers),
                       ("target_reward", 0.98), ("patience", 3),
                       ("held_out_frac", held_out_frac),
                       ("self_verify", False), ("cheap_eval_tasks", 4)):
        kwargs.setdefault(key, value)
    if not kwargs.get("asynchronous"):
        kwargs.setdefault("max_concurrency", kwargs["n_workers"])


def evolve_skill_dir(
    path: str,
    data: Sequence[Any],
    *,
    agent: Completion,
    score: Union[str, Callable] = "contains",
    reflect_with: Optional[Completion] = None,
    prompt: str = "prompt",
    gold: str = "gold",
    name: Optional[str] = None,
    layout: str = "claude_skill",
    spec: Optional[TreeSpec] = None,
    editable: Sequence[str] = ("**",),
    frozen: Sequence[str] = (),
    max_files_per_diff: int = 2,
    prompt_template: Optional[str] = None,
    fixtures: Optional[Callable[[Task], Mapping[str, str]]] = None,
    answer_file: Optional[str] = None,
    blast_radius: float = SKILL_BLAST_RADIUS,
    **evolve_kwargs: Any,
) -> EvolutionResult:
    """Evolve a **skill directory**, executed by a real agent that reads it.

    ``path`` is loaded with :func:`~agentdescent.filetree.load_tree`; every
    rollout materialises the current candidate into a fresh workspace under
    ``layout`` (default ``.claude/skills/<name>/``) and runs ``agent`` there.

    ::

        result = evolve_skill_dir(
            "~/.claude/skills/pdf-audit", rows,
            agent=claude_code(extra_args=["--permission-mode", "acceptEdits"]),
            reflect_with=openai_compatible(model="deepseek-v4-flash"),
            prompt="question", gold="answer", score="contains")
        result.write_to("~/.claude/skills/pdf-audit")     # opt in, backs up first

    ``agent`` must be a :class:`~agentdescent.agents.WorkspaceAgent`; a plain API
    completion cannot read files, so the directory would not participate.
    ``reflect_with`` defaults to ``agent`` -- a cheap model there and an expensive
    agent for the rollouts is usually the right trade.
    """
    tasks = _as_tasks(data, prompt, gold)
    aid = _artifact_id(path, name)
    tree, strategy, cfg = _build(path, spec=spec, editable=editable, frozen=frozen,
                                 max_files_per_diff=max_files_per_diff,
                                 agg_config=evolve_kwargs.pop("agg_config", None))
    runner_kwargs: Dict[str, Any] = {}
    if prompt_template:
        runner_kwargs["prompt_template"] = prompt_template
    run = tree_runner(agent, layout=layout, name=aid,
                      overlay=strategy.frozen_files(tree), fixtures=fixtures,
                      answer_file=answer_file, **runner_kwargs)
    _defaults(evolve_kwargs, len(tasks))
    evolve_kwargs.setdefault("propose", tree_reflector(reflect_with or agent,
                                                       strategy=strategy))
    return evolve(tasks, _as_reward(score), run=run, strategy=strategy,
                  artifact_id=aid, blast_radius=blast_radius,
                  agg_config=cfg, **evolve_kwargs)


def evolve_agent_dir(
    path: str,
    data: Sequence[Any],
    *,
    agent: Completion,
    score: Union[str, Callable] = "contains",
    layout: str = "claude_agent",
    frozen: Sequence[str] = (),
    **kwargs: Any,
) -> EvolutionResult:
    """Evolve an **agent directory** (subagent definitions, tool config, harness).

    Identical to :func:`evolve_skill_dir` except for the governance layer: an
    agent definition is a harness, so it runs at ``blast_radius=0.6`` -- L1, where
    every merge is additionally gated by the oracle."""
    kwargs.setdefault("blast_radius", HARNESS_BLAST_RADIUS)
    return evolve_skill_dir(path, data, agent=agent, score=score, layout=layout,
                            frozen=frozen, **kwargs)


def evolve_agent_code(
    path: str,
    data: Sequence[Any],
    *,
    entrypoint: Sequence[str],
    score: Union[str, Callable] = "contains",
    reflect_with: Completion,
    prompt: str = "prompt",
    gold: str = "gold",
    name: Optional[str] = None,
    spec: Optional[TreeSpec] = None,
    editable: Sequence[str] = ("**",),
    frozen: Sequence[str] = ("tests/**", "conftest.py"),
    max_files_per_diff: int = 2,
    setup_cmd: Optional[Sequence[str]] = None,
    test_cmd: Optional[Sequence[str]] = ("python", "-m", "pytest", "-q"),
    fixtures: Optional[Callable[[Task], Mapping[str, str]]] = None,
    timeout: float = 120.0,
    **evolve_kwargs: Any,
) -> EvolutionResult:
    """Evolve **agent code**: the tree is executed, and a test gate guards it.

    Each rollout materialises the candidate, overwrites every ``frozen`` path with
    its pristine content, runs ``setup_cmd`` then ``test_cmd``, and only then
    executes ``entrypoint + [task.prompt]``. A failing gate scores 0 and the
    failure text is what the reflector sees, so "you broke the tests" is a
    learning signal rather than a crashed run.

    ``frozen`` defaults to the test suite for a reason: without it the shortest
    path to a high score is to weaken the thing measuring it. The proposal filter
    stops the reflector from editing those files and the runner's overlay stops
    the *candidate* from rewriting them at run time -- both are needed.

    This is isolation, not a sandbox (trimmed environment, ``HOME`` inside the
    workspace, hard timeout). Model-written code still runs as your user.
    """
    tasks = _as_tasks(data, prompt, gold)
    aid = _artifact_id(path, name)
    tree, strategy, cfg = _build(path, spec=spec, editable=editable, frozen=frozen,
                                 max_files_per_diff=max_files_per_diff,
                                 agg_config=evolve_kwargs.pop("agg_config", None))
    run = code_runner(entrypoint, layout="root", name=aid, setup_cmd=setup_cmd,
                      test_cmd=test_cmd, overlay=strategy.frozen_files(tree),
                      fixtures=fixtures, timeout=timeout)
    base_reward = _as_reward(score)

    def reward(task: Task, output: str) -> float:
        # The gate speaks in-band so the reflector can read it; scoring it as a
        # normal answer would let a candidate "pass" by echoing the marker.
        if isinstance(output, str) and output.startswith(TEST_FAILURE_MARKER):
            return 0.0
        return base_reward(task, output)

    _defaults(evolve_kwargs, len(tasks))
    evolve_kwargs.setdefault("propose", tree_reflector(
        reflect_with, strategy=strategy, context_files=("**/*.py",)))
    return evolve(tasks, reward, run=run, strategy=strategy, artifact_id=aid,
                  blast_radius=HARNESS_BLAST_RADIUS, agg_config=cfg, **evolve_kwargs)
