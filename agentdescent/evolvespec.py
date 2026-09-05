"""An ``evolve()`` call as data, so a caller that is not Python can make one.

``workspec.RolloutSpec`` describes *one rollout* completely enough to run it in
another process. This module does the same for a *whole run*: which artifact,
against which data, scored how, executed by which agent, under which policies.
An :class:`EvolveSpec` is JSON; a host agent (Claude Code, DeepSeek Harness,
Codex) writes one from a user's request, the CLI and the MCP server read it, and
:func:`compose` turns it into exactly the ``evolve()`` call the quickstarts show.

Two things it deliberately is not.

**It is not a second Python entry point.** The one-call wrappers
(``evolve_skill_dir`` and friends) were removed because each was another
signature to keep in step with ``evolve()``. Python callers should keep writing
the ``evolve()`` call; this module exists for callers who cannot. It is tested
against the quickstarts so the two cannot drift.

**It carries no code and no secrets.** Agents, scorers and policies are named
(:class:`~agentdescent.workspec.Ref`) and resolved on the machine that runs the
spec, inside an import allowlist. Provider keys are read from the environment
by the adapters, never written into the spec -- a spec is saved beside the run,
shown to the user and logged, so anything in it is public.

The composition table -- what each ``kind`` assembles -- is
:data:`KIND_ROWS` and is meant to be read::

    kind        strategy       run                 propose          reward         layer
    text        SingleSlot     model(template)     reflector        scorer         L2
    skill_dir   FileTree       tree_runner         tree_reflector   scorer         L2
    agent_dir   FileTree       tree_runner         tree_reflector   scorer         L1
    agent_code  FileTree       code_runner         tree_reflector   gated_reward   L1
    plugin      FileTree       plugin_runner       tree_reflector   gated_reward   L1
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .agents import Completion, Usage, WorkspaceAgent
from .aggregator import AggregatorConfig
from .evolution import EvolutionResult, SingleSlot, Task, evolve, reflector, tasks_from
from .filetree import TreeSpec, load_tree
from .governance import HARNESS_BLAST_RADIUS, SKILL_BLAST_RADIUS
from .policies import Policies
from .rewards import command_scorer, scorer
from .runners import code_runner, gated_reward, tree_runner
from .staleness import get_policy
from .treestrategy import FileTree, tree_reflector
from .workspec import DEFAULT_ALLOWED_PREFIXES, Ref, RefError

__all__ = [
    "KINDS",
    "KIND_ROWS",
    "SHORT_REFS",
    "Composition",
    "EvolveSpec",
    "SpecError",
    "compose",
    "estimate",
    "load_spec",
    "run_spec",
    "to_ref",
]

#: The artifact shapes a spec can name. ``plugin`` is a host plugin (a DSH Cordis
#: package, a Claude Code plugin directory); see :func:`~agentdescent.runners.plugin_runner`.
KINDS: Tuple[str, ...] = ("text", "skill_dir", "agent_dir", "agent_code", "plugin")

#: The composition table, for readers and for the test that checks it against
#: the quickstarts. Values are the *names* of what is assembled; :func:`compose`
#: is the code.
KIND_ROWS: Dict[str, Dict[str, str]] = {
    "text":       {"strategy": "SingleSlot", "run": "model(template)", "propose": "reflector",
                   "reward": "scorer", "blast_radius": "SKILL_BLAST_RADIUS"},
    "skill_dir":  {"strategy": "FileTree", "run": "tree_runner", "propose": "tree_reflector",
                   "reward": "scorer", "blast_radius": "SKILL_BLAST_RADIUS"},
    "agent_dir":  {"strategy": "FileTree", "run": "tree_runner", "propose": "tree_reflector",
                   "reward": "scorer", "blast_radius": "HARNESS_BLAST_RADIUS"},
    "agent_code": {"strategy": "FileTree", "run": "code_runner", "propose": "tree_reflector",
                   "reward": "gated_reward(scorer)", "blast_radius": "HARNESS_BLAST_RADIUS"},
    "plugin":     {"strategy": "FileTree", "run": "plugin_runner", "propose": "tree_reflector",
                   "reward": "gated_reward(scorer)", "blast_radius": "HARNESS_BLAST_RADIUS"},
}

#: Short names a spec may use instead of ``module:attribute``. Every entry is a
#: public factory of this package; a name outside the table must be spelled out
#: in full and its module must be inside the allowlist.
SHORT_REFS: Dict[str, str] = {
    # agents and models
    "claude_code": "agentdescent.agents:claude_code",
    "codex": "agentdescent.agents:codex",
    "dsh": "agentdescent.agents:dsh",
    "cli_agent": "agentdescent.agents:cli_agent",
    "openai_compatible": "agentdescent.agents:openai_compatible",
    "claude": "agentdescent.agents:claude",
    "echo": "agentdescent.agents:echo",
    # selection
    "SingleHead": "agentdescent.selection:SingleHead",
    "Beam": "agentdescent.selection:Beam",
    "ParetoFrontier": "agentdescent.selection:ParetoFrontier",
    "Archive": "agentdescent.selection:Archive",
    "MCTS": "agentdescent.selection:MCTS",
    # task sampling
    "RoundRobin": "agentdescent.sampling:RoundRobin",
    "DifficultyWeighted": "agentdescent.sampling:DifficultyWeighted",
    # merge-side rules and wrappers (installed by the aggregator via bind/configure)
    "AdvantageAcceptance": "agentdescent.advantage:AdvantageAcceptance",
    "StableDistanceAcceptance": "agentdescent.advantage:StableDistanceAcceptance",
    "AdvantageConflict": "agentdescent.advantage:AdvantageConflict",
    "DefaultConflict": "agentdescent.defaults:DefaultConflict",
    "DefaultFusion": "agentdescent.defaults:DefaultFusion",
    "DefaultAcceptance": "agentdescent.defaults:DefaultAcceptance",
    "DefaultPromotion": "agentdescent.defaults:DefaultPromotion",
    "KeepContradictions": "agentdescent.fusion:KeepContradictions",
    "ReflectiveFusion": "agentdescent.fusion:ReflectiveFusion",
    # the pair, as one name -- see compose()
    "reflective_merge": "agentdescent.fusion:reflective_merge",
}

_POLICY_SLOTS = ("selection", "task_sampler", "proposal", "conflict", "fusion",
                 "acceptance", "promotion", "staleness")

_TEXT_TEMPLATE = "{skill}\n\n{prompt}"

#: What a run against a real agent should default to, and why the plain engine
#: defaults are wrong for it: a rollout is an agent invocation, so re-running
#: each proposal's trajectory (`self_verify`) doubles the cost and ranking on
#: the whole held-out set is the dominant expense. The removed wrappers set
#: these; the quickstarts pass them explicitly; the spec sets them because its
#: author is a model that has not read the cost model.
_AGENT_RUN_DEFAULTS = {"self_verify": False, "cheap_eval_tasks": 4}


class SpecError(ValueError):
    """A spec that cannot be composed, and the field that is wrong."""


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------


@dataclass
class EvolveSpec:
    """What to evolve, against what, scored how, by whom -- as data.

    Only ``kind``, ``target``, ``data`` and ``score`` are always required;
    ``agent`` is required for every kind (it is the model for ``text``).
    Everything under ``evolve`` passes straight through to :func:`evolve` and
    overrides the defaults :func:`compose` picks for the kind.
    """

    kind: str
    target: str
    data: Dict[str, Any]
    score: Union[str, Dict[str, Any]] = "contains"
    agent: Optional[Union[str, Dict[str, Any]]] = None
    reflect: Optional[Union[str, Dict[str, Any]]] = None
    name: Optional[str] = None
    #: ``text`` only: how the skill meets the question. Must contain ``{skill}``
    #: and ``{prompt}``.
    template: str = _TEXT_TEMPLATE
    #: directory kinds: where the tree lands in each workspace
    #: (:data:`~agentdescent.runners.LAYOUTS` key or a literal prefix).
    layout: Optional[str] = None
    #: directory kinds: how the task reaches the agent. May use ``{prompt}``,
    #: ``{tree_dir}`` and ``{name}``; ``None`` is the runner's default, which
    #: tells the agent where the files are and to reply with only the answer.
    prompt_template: Optional[str] = None
    editable: Sequence[str] = ("**",)
    frozen: Sequence[str] = ()
    max_files_per_diff: int = 2
    #: ``agent_code`` only.
    entrypoint: Sequence[str] = ()
    setup_cmd: Sequence[str] = ()
    test_cmd: Sequence[str] = ("python", "-m", "pytest", "-q")
    timeout: float = 120.0
    #: ``plugin`` only: which host loads the plugin (``dsh`` / ``claude_code`` / ``codex``).
    host: Optional[str] = None
    #: Names of environment variables the worker may see. Names, never values.
    env_passthrough: Sequence[str] = ()
    #: One ref per slot; ``"staleness"`` may be a policy name. ``reflective_merge``
    #: fills ``conflict`` and ``fusion`` together.
    policies: Dict[str, Any] = field(default_factory=dict)
    agg_config: Dict[str, Any] = field(default_factory=dict)
    #: Passed straight to ``evolve()``.
    evolve: Dict[str, Any] = field(default_factory=dict)
    #: Extra import prefixes a ``module:attribute`` ref may resolve into. The
    #: package's own modules are always allowed; widening this is the moment to
    #: think about who can write the spec.
    allow: Sequence[str] = ()
    version: int = 1

    # -- (de)serialisation ------------------------------------------------

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EvolveSpec":
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(d) - known)
        if unknown:
            raise SpecError(f"unknown spec field(s) {unknown}; known: {sorted(known)}")
        for req in ("kind", "target", "data"):
            if req not in d:
                raise SpecError(f"spec needs {req!r}")
        return cls(**dict(d))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # tuples are lists on the wire
        for key in ("editable", "frozen", "entrypoint", "setup_cmd", "test_cmd",
                    "env_passthrough", "allow"):
            d[key] = list(d[key])
        return d

    def artifact_id(self) -> str:
        import re

        raw = self.name or os.path.basename(
            os.path.abspath(os.path.expanduser(self.target)).rstrip(os.sep))
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-") or "artifact"

    def allowed_prefixes(self) -> Tuple[str, ...]:
        return tuple(DEFAULT_ALLOWED_PREFIXES) + tuple(self.allow)


def load_spec(path: str) -> EvolveSpec:
    """Read a spec from a JSON file."""
    with open(os.path.expanduser(path), encoding="utf-8") as fh:
        return EvolveSpec.from_dict(json.load(fh))


# ---------------------------------------------------------------------------
# Refs
# ---------------------------------------------------------------------------


def to_ref(value: Any, *, where: str) -> Ref:
    """``"claude_code"`` / ``"pkg.mod:fn"`` / ``{"ref": ..., **config}`` -> :class:`Ref`.

    ``"call": false`` says the attribute *is* the callable rather than a factory
    for one (a module-level scorer, an agent object). Nested ``{"ref": ...}``
    dicts inside the config become nested refs, so
    ``{"ref": "reflective_merge", "complete": {"ref": "openai_compatible", "model": "x"}}``
    composes the way ``reflective_merge(openai_compatible(model="x"))`` does.
    """
    if isinstance(value, str):
        name, config = value, {}
    elif isinstance(value, Mapping) and "ref" in value:
        name = value["ref"]
        config = {k: v for k, v in value.items() if k != "ref"}
    else:
        raise SpecError(
            f"{where}: expected a short name, 'module:attribute', or "
            f"{{'ref': ..., **config}}; got {value!r}")
    if not isinstance(name, str):
        raise SpecError(f"{where}: 'ref' must be a string, got {name!r}")
    target = SHORT_REFS.get(name, name)
    if ":" not in target:
        raise SpecError(
            f"{where}: {name!r} is not a known short name {sorted(SHORT_REFS)} "
            "and not of the form 'module:attribute'")
    call = config.pop("call", True)
    if not isinstance(call, bool):
        raise SpecError(f"{where}: 'call' must be true or false")
    nested = {k: (to_ref(v, where=f"{where}.{k}")
                  if isinstance(v, Mapping) and "ref" in v else v)
              for k, v in config.items()}
    try:
        return Ref(target, nested, call=call)
    except RefError as e:
        raise SpecError(f"{where}: {e}") from None


def _resolve(value: Any, spec: EvolveSpec, *, where: str) -> Any:
    try:
        return to_ref(value, where=where).resolve(spec.allowed_prefixes())
    except RefError as e:
        raise SpecError(f"{where}: {e}") from None


# ---------------------------------------------------------------------------
# Data and reward
# ---------------------------------------------------------------------------


def _read_rows_file(path: str) -> List[Dict[str, Any]]:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise SpecError(f"data.path {path!r} does not exist")
    ext = os.path.splitext(path)[1].lower()
    with open(path, encoding="utf-8") as fh:
        if ext == ".jsonl":
            return [json.loads(line) for line in fh if line.strip()]
        if ext == ".json":
            rows = json.load(fh)
            if isinstance(rows, Mapping):
                rows = rows.get("rows") or rows.get("data") or rows.get("tasks")
            if not isinstance(rows, list):
                raise SpecError(f"data.path {path!r}: expected a JSON list of rows")
            return rows
        if ext in (".csv", ".tsv"):
            return list(csv.DictReader(fh, delimiter="\t" if ext == ".tsv" else ","))
    raise SpecError(f"data.path {path!r}: use .json, .jsonl, .csv or .tsv")


def load_rows(data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The rows a spec's ``data`` names: a local file, a HF dataset, or inline."""
    sources = [k for k in ("path", "hf", "inline") if k in data]
    if len(sources) != 1:
        raise SpecError(f"data needs exactly one of 'path', 'hf', 'inline'; got {sources}")
    src = sources[0]
    if src == "path":
        return _read_rows_file(str(data["path"]))
    if src == "inline":
        rows = data["inline"]
        if not isinstance(rows, list) or not rows:
            raise SpecError("data.inline must be a non-empty list of rows")
        return list(rows)
    from .dataloader import hf_rows

    hf = dict(data["hf"])
    for req in ("dataset", "split"):
        if req not in hf:
            raise SpecError(f"data.hf needs {req!r}")
    return hf_rows(hf["dataset"], hf["split"], config=hf.get("config", "default"),
                   limit=int(hf.get("limit", 100)))


def build_tasks(spec: EvolveSpec) -> List[Task]:
    rows = load_rows(spec.data)
    if rows and isinstance(rows[0], Task):
        return list(rows)
    prompt = str(spec.data.get("prompt", "prompt"))
    gold = str(spec.data.get("gold", "gold"))
    extra: Dict[str, str] = {}
    if rows and isinstance(rows[0], Mapping) and "fixtures" in rows[0]:
        extra["fixtures"] = "fixtures"       # staged into the workspace by the runners
    try:
        return tasks_from(rows, prompt=prompt, gold=gold,
                          id=spec.data.get("id"), **extra)
    except KeyError as e:
        raise SpecError(f"data: {e}") from None


def build_reward(spec: EvolveSpec) -> Callable:
    score = spec.score
    if isinstance(score, str):
        try:
            return scorer(score)
        except ValueError as e:
            raise SpecError(f"score: {e}") from None
    if isinstance(score, Mapping):
        if "cmd" in score:
            return command_scorer(score["cmd"], timeout=float(score.get("timeout", 60.0)))
        if "ref" in score:
            return _resolve(score, spec, where="score")
    raise SpecError("score must be a scorer name, {'cmd': ...} or {'ref': ...}")


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


def build_policies(spec: EvolveSpec) -> Optional[Policies]:
    """The ``Policies`` bundle a spec asks for, or ``None`` for the shipped run.

    Every merge-side rule is installed by the aggregator through
    ``bind``/``configure``, so nothing here needs a verifier or a threshold.
    ``reflective_merge`` is the one name that fills two slots, and it can only
    be asked for as the pair -- the half-installed version the policy guide
    warns about cannot be written.
    """
    asked = dict(spec.policies)
    if not asked:
        return None
    unknown = sorted(set(asked) - set(_POLICY_SLOTS) - {"reflective_merge"})
    if unknown:
        raise SpecError(f"policies: unknown slot(s) {unknown}; slots are {_POLICY_SLOTS} "
                        "plus 'reflective_merge'")
    fields: Dict[str, Any] = {}
    if "reflective_merge" in asked:
        if "conflict" in asked or "fusion" in asked:
            raise SpecError("policies: 'reflective_merge' fills conflict and fusion; "
                            "do not also set them")
        pair = _resolve(asked.pop("reflective_merge"), spec, where="policies.reflective_merge")
        if not isinstance(pair, Mapping) or set(pair) != {"fusion", "conflict"}:
            raise SpecError("policies.reflective_merge must resolve to the pair "
                            "reflective_merge() returns")
        fields.update(pair)
    for slot, value in asked.items():
        if slot == "staleness" and isinstance(value, str) and ":" not in value \
                and value not in SHORT_REFS:
            try:
                fields[slot] = get_policy(value)
            except ValueError as e:
                raise SpecError(f"policies.staleness: {e}") from None
            continue
        fields[slot] = _resolve(value, spec, where=f"policies.{slot}")
    return Policies(**fields)


def build_agg_config(spec: EvolveSpec, **defaults: Any) -> Optional[AggregatorConfig]:
    merged = {**defaults, **spec.agg_config}
    if not merged:
        return None
    known = set(AggregatorConfig.__dataclass_fields__)
    unknown = sorted(set(merged) - known)
    if unknown:
        raise SpecError(f"agg_config: unknown field(s) {unknown}; known: {sorted(known)}")
    return AggregatorConfig(**merged)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


@dataclass
class Composition:
    """What :func:`compose` produced: the ``evolve()`` call, taken apart.

    ``kwargs`` is everything after ``tasks, reward``; ``tree`` is the original
    directory for the directory kinds (what ``show --diff`` diffs against);
    ``notes`` are things the caller should be told before running.
    """

    tasks: List[Task]
    reward: Callable
    kwargs: Dict[str, Any]
    tree: Optional[Dict[str, str]] = None
    notes: List[str] = field(default_factory=list)

    def run(self) -> EvolutionResult:
        return evolve(self.tasks, self.reward, **self.kwargs)


def _train_count(n_tasks: int, held_out_frac: float) -> int:
    return max(1, n_tasks - int(round(n_tasks * held_out_frac)))


def _read_text_target(target: str) -> str:
    path = os.path.expanduser(target)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    return target


def compose(spec: EvolveSpec, *, usage: Optional[Usage] = None,
            on_round: Optional[Callable] = None, repo_path: Optional[str] = None,
            workspace_root: Optional[str] = None, sandbox_pool: Any = None,
            **overrides: Any) -> Composition:
    """Turn a spec into the ``evolve()`` call the quickstarts would write.

    Resolves every ref, loads the data and the tree, builds the strategy, the
    runner, the reflector, the reward and the ``Policies`` bundle, and picks the
    defaults the kind wants. Raises :class:`SpecError` for anything the spec
    gets wrong, *before* a single rollout runs -- which is what ``plan`` is for.
    """
    if spec.kind not in KINDS:
        raise SpecError(f"kind must be one of {KINDS}, got {spec.kind!r}")
    if spec.agent is None:
        raise SpecError("agent is required (for kind='text' it is the model)")
    if spec.version != 1:
        raise SpecError(f"unsupported spec version {spec.version}")

    tasks = build_tasks(spec)
    reward = build_reward(spec)
    notes: List[str] = []
    agent = _resolve(spec.agent, spec, where="agent")
    reflect = _resolve(spec.reflect, spec, where="reflect") if spec.reflect else agent
    aid = spec.artifact_id()

    kwargs: Dict[str, Any] = {"artifact_id": aid}
    tree: Optional[Dict[str, str]] = None

    if spec.kind == "text":
        if "{skill}" not in spec.template or "{prompt}" not in spec.template:
            raise SpecError("template must contain {skill} and {prompt}")
        template = spec.template
        model: Completion = agent

        def run(skill: str, task: Task) -> str:
            return model(template.format(skill=skill, prompt=task.prompt))

        kwargs.update(
            run=run, propose=reflector(reflect),
            strategy=SingleSlot(initial_value=_read_text_target(spec.target)),
            blast_radius=SKILL_BLAST_RADIUS)
        defaults = {"rounds": 8, "held_out_frac": 0.3, "patience": 3,
                    "target_reward": 0.98}
        default_workers = 8
    else:
        path = os.path.expanduser(spec.target)
        if not os.path.isdir(path):
            raise SpecError(f"target {spec.target!r} is not a directory")
        tspec = TreeSpec()
        agg_defaults = {"batch_trigger": 2, "max_wait_rounds": 1}
        cfg = build_agg_config(spec, **agg_defaults)
        tspec.validate_against(cfg.trust_region_chars)
        tree = load_tree(path, tspec)
        frozen = list(spec.frozen)
        if spec.kind == "agent_code" and not frozen:
            frozen = ["tests/**", "conftest.py"]
        if spec.kind == "plugin":
            from .runners import PLUGIN_FROZEN

            if spec.host not in PLUGIN_FROZEN:
                raise SpecError(f"kind='plugin' needs host in {sorted(PLUGIN_FROZEN)}, "
                                f"got {spec.host!r}")
            frozen = list(PLUGIN_FROZEN[spec.host]) + frozen
        strategy = FileTree(tree, editable=list(spec.editable), frozen=frozen,
                            max_files_per_diff=spec.max_files_per_diff,
                            max_file_bytes=tspec.max_file_bytes)
        overlay = strategy.frozen_files(tree)
        runner_common = dict(name=aid, overlay=overlay, workspace_root=workspace_root,
                             sandbox_pool=sandbox_pool)
        context_files: Sequence[str] = ("**/SKILL.md", "**/AGENT.md", "*.md")

        if spec.kind in ("skill_dir", "agent_dir"):
            if not isinstance(agent, WorkspaceAgent):
                raise SpecError(
                    f"kind={spec.kind!r} needs a WorkspaceAgent (claude_code, codex, dsh, "
                    f"cli_agent) so the directory is put in front of it; got "
                    f"{type(agent).__name__}")
            layout = spec.layout or ("claude_skill" if spec.kind == "skill_dir" else "claude_agent")
            extra = {"prompt_template": spec.prompt_template} if spec.prompt_template else {}
            kwargs["run"] = tree_runner(agent, layout=layout, **extra, **runner_common)
            kwargs["blast_radius"] = (SKILL_BLAST_RADIUS if spec.kind == "skill_dir"
                                      else HARNESS_BLAST_RADIUS)
        elif spec.kind == "agent_code":
            if not spec.entrypoint:
                raise SpecError("kind='agent_code' needs entrypoint (argv the task prompt is appended to)")
            kwargs["run"] = code_runner(
                list(spec.entrypoint), layout=spec.layout or "root",
                setup_cmd=list(spec.setup_cmd) or None,
                test_cmd=list(spec.test_cmd) or None, timeout=spec.timeout,
                env={k: os.environ[k] for k in spec.env_passthrough if k in os.environ},
                **runner_common)
            reward = gated_reward(reward)
            kwargs["blast_radius"] = HARNESS_BLAST_RADIUS
            context_files = ("**/*.py",)
        else:  # plugin
            from .runners import PLUGIN_CONTEXT, plugin_runner

            kwargs["run"] = plugin_runner(
                spec.host, agent_args=list(spec.entrypoint),
                env_passthrough=list(spec.env_passthrough), timeout=spec.timeout,
                **runner_common)
            reward = gated_reward(reward)
            kwargs["blast_radius"] = HARNESS_BLAST_RADIUS
            context_files = PLUGIN_CONTEXT[spec.host]
            notes.append("kind='plugin': candidate plugin code runs inside the host "
                         "process; prefer a container sandbox (see docs/plugins.md).")

        kwargs.update(strategy=strategy, agg_config=cfg,
                      propose=tree_reflector(reflect, strategy=strategy,
                                             context_files=context_files))
        defaults = {"rounds": 6, "held_out_frac": 0.3, "patience": 3,
                    "target_reward": 0.98, **_AGENT_RUN_DEFAULTS}
        default_workers = 4

    # -- knobs: kind defaults < spec.evolve < caller overrides ------------------
    knobs: Dict[str, Any] = {**defaults, **spec.evolve, **overrides}
    held_out = float(knobs.get("held_out_frac", 0.3))
    train = _train_count(len(tasks), held_out)
    knobs.setdefault("n_workers", max(1, min(default_workers, train)))
    if not knobs.get("asynchronous"):
        knobs.setdefault("max_concurrency", knobs["n_workers"])
    elif "max_seconds" not in knobs:
        knobs["max_seconds"] = 600.0
        notes.append("asynchronous=True needs max_seconds; defaulted to 600.")
    if "agg_config" in knobs and isinstance(knobs["agg_config"], Mapping):
        knobs["agg_config"] = build_agg_config(spec, **knobs["agg_config"])
    elif spec.kind == "text" and spec.agg_config:
        knobs["agg_config"] = build_agg_config(spec)
    policies = build_policies(spec)
    if policies is not None:
        knobs["policies"] = policies
    if usage is not None:
        knobs["usage"] = usage
    if on_round is not None:
        knobs["on_round"] = on_round
    if repo_path is not None:
        knobs["repo_path"] = repo_path
    kwargs.update(knobs)

    if len(tasks) < 4:
        notes.append(f"only {len(tasks)} tasks: the held-out split will be tiny and the "
                     "acceptance test weak; 8-20 is a workable minimum.")
    return Composition(tasks=tasks, reward=reward, kwargs=kwargs, tree=tree, notes=notes)


def run_spec(spec: EvolveSpec, **hooks: Any) -> EvolutionResult:
    """Compose and run. ``hooks`` are :func:`compose`'s keyword arguments."""
    return compose(spec, **hooks).run()


# ---------------------------------------------------------------------------
# Cost estimate
# ---------------------------------------------------------------------------


def estimate(comp: Composition, *, usd_per_call: Optional[float] = None) -> Dict[str, Any]:
    """How much a composed run will call the agent, before it runs.

    Counts, not dollars, unless ``usd_per_call`` is given -- the per-call price of
    a tool-using agent is not known in advance and the estimate says so rather
    than inventing one. Assumptions are listed in the result so a host can show
    them next to the number.
    """
    k = comp.kwargs
    rounds = int(k.get("rounds", 15))
    n_workers = int(k.get("n_workers", 4))
    n_tasks = len(comp.tasks)
    held_out = int(round(n_tasks * float(k.get("held_out_frac", 0.4))))
    cheap = k.get("cheap_eval_tasks") or held_out
    self_verify = bool(k.get("self_verify", True))
    per_round_rollouts = n_workers
    per_round_proposals = n_workers
    per_round_verify = n_workers if self_verify else 0
    per_round_rank = int(cheap) * 2                 # base and candidate, cheap layer
    per_round_gate = held_out                       # one held-out sweep per candidate
    per_round = (per_round_rollouts + per_round_proposals + per_round_verify
                 + per_round_rank + per_round_gate)
    total = per_round * rounds
    out: Dict[str, Any] = {
        "rounds": rounds, "n_workers": n_workers, "tasks": n_tasks,
        "held_out_tasks": held_out,
        "calls_per_round": {"rollouts": per_round_rollouts,
                            "proposals": per_round_proposals,
                            "self_verify": per_round_verify,
                            "ranking": per_round_rank, "gate": per_round_gate},
        "agent_calls_upper_bound": total,
        "assumptions": [
            "every round runs to completion (target_reward / patience may stop it earlier)",
            "one candidate reaches the gate per round",
            "the evaluation cache serves repeated (candidate, task) pairs; counted as misses here",
        ],
    }
    if usd_per_call is not None:
        out["usd_upper_bound"] = round(total * usd_per_call, 2)
    else:
        out["usd_upper_bound"] = None
        out["assumptions"].append(
            "no per-call price known for this agent; pass usd_per_call to price it")
    return out
