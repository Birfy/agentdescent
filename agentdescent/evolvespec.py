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
    policy_slot SlotSpec       one inner search    slot_reflector   meta_reward    L1

``policy_slot`` is the odd one and the reason the table is worth reading: its
artifact is not a file but a *decision rule of the optimiser itself*, and one
rollout is a whole inner ``evolve()`` or tree search rather than one model call.
Because it is still an ``evolve()`` call, plan, cost, detach, status and watch
need nothing from it. ``show`` and ``apply`` do: the rendered value is source
text like ``kind: "text"``, but ``target`` names a *slot* rather than a file, so
there is nothing to diff against and ``apply`` has to be told a destination.
"""

from __future__ import annotations

import csv
import json
import os
import shlex
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .agents import Completion, Usage, WorkspaceAgent
from .aggregator import AggregatorConfig
from .evolution import EvolutionResult, SingleSlot, Task, evolve, reflector, tasks_from
from .filetree import TreeSpec, load_tree
from .fusion import reflective_merge
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
KINDS: Tuple[str, ...] = ("text", "skill_dir", "agent_dir", "agent_code", "plugin",
                          "policy_slot")

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
    # The odd row: the artifact is a decision rule of the optimiser rather than
    # a file, and `run` is a whole inner search rather than one model call.
    "policy_slot": {"strategy": "SlotSpec", "run": "one inner search",
                    "propose": "slot_reflector", "reward": "meta_reward(auc)",
                    "blast_radius": "HARNESS_BLAST_RADIUS"},
}

#: Short names a spec may use instead of ``module:attribute``. Every entry is a
#: public factory of this package; a name outside the table must be spelled out
#: in full and its module must be inside the allowlist.
SHORT_REFS: Dict[str, str] = {
    # agents and models
    "claude_code": "agentdescent.agents:claude_code",
    "codex": "agentdescent.agents:codex",
    "dsh": "agentdescent.agents:dsh",
    "opencode": "agentdescent.agents:opencode",
    "cli_agent": "agentdescent.agents:cli_agent",
    "openai_compatible": "agentdescent.agents:openai_compatible",
    # The model of the agent session that started the run, over MCP sampling.
    # Only resolvable inside a run the MCP server launched; see agentdescent.host_sampling.
    "host_model": "agentdescent.host_sampling:host_model",
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
    #: Pair the reward against ground truth. ``{"oracle": "mod:fn"}`` is the
    #: whole minimum. Only worth setting when ``score`` is a model judging an
    #: output rather than a fact about it -- then the loop is optimising a proxy
    #: and nothing in it can tell. See :func:`build_audit` for the keys, and
    #: note that ``enabled`` defaults to **false**: the audit collects without
    #: changing what commits until you say otherwise.
    audit: Optional[Dict[str, Any]] = None
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

    def absolutise(self, base: Optional[str] = None) -> "EvolveSpec":
        """A copy whose file paths are absolute, resolved against ``base`` (cwd).

        A spec is written by one process and run by another: the CLI validates it
        where you typed the command, then a detached child runs it from the run
        directory, and an MCP server's cwd is wherever the host happened to
        launch it. A relative ``data.path`` therefore names a different file in
        each of them -- which showed up as a run that planned cleanly and then
        failed on its first line with "cases.jsonl does not exist".

        So relative paths are resolved **once, where the spec was read**, and the
        copy stored beside the run is unambiguous: it is also what makes the
        stored spec re-runnable from anywhere, which is the point of keeping it.
        Only paths are touched -- ``target``, ``data.path`` and a ``score.cmd``
        that names a file (``./grade.sh``, ``tools/grade.py``) rather than a
        program on ``PATH``.
        """
        base = os.path.abspath(base or os.getcwd())

        def resolve(p: str) -> str:
            expanded = os.path.expanduser(p)
            return expanded if os.path.isabs(expanded) else os.path.normpath(
                os.path.join(base, expanded))

        # `text` is the one kind whose target may be the instruction itself
        # rather than a path, so it is resolved only when it names a real file.
        # `policy_slot` names a *slot*, never a path, so it is never resolved --
        # absolutising it would turn `selection` into `$PWD/selection`.
        target = self.target
        if self.kind == "policy_slot":
            pass
        elif self.kind != "text" or os.path.isfile(resolve(self.target)):
            target = resolve(self.target)
        out = replace(self, target=target)
        if isinstance(self.data, Mapping) and "path" in self.data:
            out.data = {**self.data, "path": resolve(str(self.data["path"]))}
        if isinstance(self.audit, Mapping) and self.audit.get("store"):
            out.audit = {**self.audit, "store": resolve(str(self.audit["store"]))}
        if isinstance(self.score, Mapping) and "cmd" in self.score:
            cmd = self.score["cmd"]
            argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
            if argv and (os.sep in argv[0] or argv[0].startswith(".")):
                out.score = {**self.score, "cmd": [resolve(argv[0]), *argv[1:]]}
        return out


def load_spec(path: str, *, absolutise: bool = True) -> EvolveSpec:
    """Read a spec from a JSON file.

    Relative paths in it are resolved against the current directory (see
    :meth:`EvolveSpec.absolutise`), because the process that *runs* the spec is
    usually not this one. Pass ``absolutise=False`` to read it verbatim."""
    with open(os.path.expanduser(path), encoding="utf-8") as fh:
        spec = EvolveSpec.from_dict(json.load(fh))
    return spec.absolutise() if absolutise else spec


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


#: Keys :func:`build_audit` accepts. Spelled out so a typo is an error with a
#: list beside it rather than a setting that silently did nothing -- which is
#: how `sample_rate` becomes `sample-rate` and an audit runs at the default.
_AUDIT_KEYS = ("oracle", "store", "sample_rate", "calibration_fraction",
               "draw_by", "enabled", "seed", "watch_ids", "watch_globs", "verifier_version", "version_extra",
               "stratify")


def build_audit(spec: EvolveSpec, *, reward: Callable, run: Optional[Callable],
                repo_path: Optional[str] = None) -> Optional[Any]:
    """The :class:`~agentdescent.audit.wiring.Audit` a spec asks for, or ``None``.

    ``oracle`` and ``stratify`` are refs, resolved through the spec's own
    allowlist -- the audit is not a reason to widen the trust boundary.

    ``store`` defaults to ``audit.jsonl`` **beside the run's ledger**, so the
    ``audit_*`` MCP tools can find it from the run id without being told where
    it went. A run that writes its audit somewhere only the caller knows is a
    run whose audit nobody reads.

    ``enabled`` defaults to **false**. Collecting records is free and changes
    nothing; correcting the acceptance gate changes what commits, and a spec
    that merely names an oracle has not asked for that. Set it once you have
    looked at what the first run measured.
    """
    if not spec.audit:
        return None
    from .audit import attach

    unknown = sorted(set(spec.audit) - set(_AUDIT_KEYS))
    if unknown:
        raise SpecError(f"audit: unknown key(s) {unknown}; keys are "
                        f"{list(_AUDIT_KEYS)}")
    cfg = dict(spec.audit)
    oracle = cfg.pop("oracle", None)
    if oracle is not None:
        oracle = _resolve(oracle, spec, where="audit.oracle")
    stratify = cfg.pop("stratify", None)
    if stratify is not None:
        stratify = _resolve(stratify, spec, where="audit.stratify")
    store = cfg.pop("store", None)
    if store is None and repo_path:
        store = os.path.join(os.path.dirname(os.path.abspath(repo_path)),
                             "audit.jsonl")
    cfg.setdefault("enabled", False)
    try:
        return attach(reward, oracle=oracle, store=store, run=run,
                      stratify=stratify, **cfg)
    except (TypeError, ValueError) as e:
        raise SpecError(f"audit: {e}") from None


def build_policies(spec: EvolveSpec, *, merger: Optional[Any] = None) -> Optional[Policies]:
    """The ``Policies`` bundle a spec asks for, or the default merge pair.

    Every merge-side rule is installed by the aggregator through
    ``bind``/``configure``, so nothing here needs a verifier or a threshold.
    ``reflective_merge`` is the one name that fills two slots, and it can only
    be asked for as the pair -- the half-installed version the policy guide
    warns about cannot be written.
    """
    asked = dict(spec.policies)
    # The merge pair is the default, not an opt-in. Without it a run whose
    # artifact is one key -- every `kind: "text"` target -- has its worker
    # proposals contradict by construction, so conflict resolution collapses
    # them to one candidate and `n_workers` buys per-round best-of-N *selection*
    # rather than the merge this project is about. `merger` is the reflector's
    # model, which the spec already names.
    #
    # Keyed off the merge slots alone, not off `asked` being empty: a spec that
    # names an unrelated slot -- `staleness`, say -- is not asking to stop
    # merging, and silently dropping the pair there would make the default
    # depend on a field that has nothing to do with it.
    default_merge = (merger is not None
                     and not {"reflective_merge", "conflict", "fusion"} & set(asked))
    if not asked:
        return Policies(**reflective_merge(merger)) if default_merge else None
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
    if default_merge:
        fields.update(reflective_merge(merger))
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
    #: The sparse audit, when the spec asked for one. ``reward`` and
    #: ``kwargs["run"]`` are already its wrapped versions; this is the handle for
    #: reading what it measured.
    audit: Optional[Any] = None

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


#: The default :class:`~agentdescent.meta.SlotSpec` factory per slot. Only
#: ``selection`` has a purpose-built one; every other slot falls back to the
#: generic source gate, which is why naming ``data.slot_spec`` is the honest
#: thing to do for them.
_SLOT_SPEC_DEFAULTS = {"selection": "agentdescent.meta:priority_selection"}

#: ``score`` values a ``policy_slot`` accepts by short name, and whether the
#: attribute *is* the reward or a factory for one. `auc` and `final_reward` are
#: rewards; `rollouts_to(target)` builds one and has no default target, so it
#: can only be named in the configured form and says so rather than scoring
#: every outcome 0.0. Anything else is a ``module:attribute`` ref.
_META_REWARDS = {"auc": ("agentdescent.meta:auc", False),
                 "final_reward": ("agentdescent.meta:final_reward", False),
                 "rollouts_to": ("agentdescent.meta:rollouts_to", True)}


def _compose_policy_slot(spec: EvolveSpec, *, usage: Optional[Usage] = None,
                         on_round: Optional[Callable] = None,
                         **overrides: Any) -> Composition:
    """``kind: "policy_slot"`` -- evolve one decision rule of the optimiser.

    The artifact is a slot of :class:`~agentdescent.policies.Policies`, and one
    rollout is a **whole inner search**, so the cost model is unlike every other
    kind: rounds are cheap to ask for and expensive to get. The assembly itself
    is :func:`~agentdescent.meta.meta_parts`, shared with
    :func:`~agentdescent.meta.meta_evolve` so the two cannot drift.

    ``data`` holds refs rather than rows, because inner problems are callables
    and no row format can express one::

        data:
          problems: "mypkg.landscape:source_problems"   # -> Mapping[str, Problem]
          seeds: [0]                                    # optional
          slot_spec: "agentdescent.meta:priority_selection"   # optional
    """
    from .meta import SLOTS, meta_parts, slot_reflector

    if not isinstance(spec.data, Mapping) or "problems" not in spec.data:
        raise SpecError("kind='policy_slot' needs data.problems, a "
                        "'module:attribute' ref to the inner problems")
    slot = spec.target
    if slot not in SLOTS:
        raise SpecError(f"target must be an evolvable slot, one of {SLOTS}; got {slot!r}")

    problems = _resolve(spec.data["problems"], spec, where="data.problems")
    if callable(problems) and not isinstance(problems, Mapping):
        problems = problems()          # a zero-arg builder is the usual shape
    if not problems:
        raise SpecError("data.problems resolved to nothing to evolve against")

    seeds = spec.data.get("seeds", [0])
    if not isinstance(seeds, Sequence) or isinstance(seeds, str) or not seeds:
        raise SpecError("data.seeds must be a non-empty list of integers")
    seeds = [int(x) for x in seeds]

    slot_spec_ref = spec.data.get("slot_spec") or _SLOT_SPEC_DEFAULTS.get(slot)
    if slot_spec_ref is None:
        raise SpecError(f"slot {slot!r} has no default SlotSpec; name one in "
                        "data.slot_spec (see agentdescent.meta.policy_source)")
    slot_spec = _resolve(slot_spec_ref, spec, where="data.slot_spec")

    # `EvolveSpec.score` defaults to "contains", a *text* scorer: it compares an
    # answer to a gold string and cannot read a MetaOutcome. Treat the unset
    # default as this kind's default rather than failing on a field the author
    # never wrote.
    score = spec.score
    if score in (None, "contains"):
        score = "auc"
    if isinstance(score, str):
        known = _META_REWARDS.get(score)
        if known is None:
            if ":" not in score:
                raise SpecError(f"score: {score!r} is not one of "
                                f"{sorted(_META_REWARDS)} and not 'module:attribute'")
            # An unknown ref: `auc`-shaped is by far the common case, so a bare
            # string means the attribute is the reward. `{"ref": ..., "call": true,
            # ...}` still says otherwise, and `_check_meta_reward` catches either
            # way round before a single rollout runs.
            score = {"ref": score, "call": False}
        else:
            target, is_factory = known
            if is_factory:
                raise SpecError(
                    f"score: {spec.score!r} builds a reward and needs its argument -- "
                    f'write {{"ref": "{spec.score}", "target": 0.9}}. Naming it bare '
                    "would score every outcome 0.0.")
            score = {"ref": target, "call": False}
    elif isinstance(score, Mapping) and "ref" in score:
        named = _META_REWARDS.get(score["ref"])
        if named is not None:
            score = {**score, "ref": named[0]}
    meta_reward = _resolve(score, spec, where="score")

    model = _resolve(spec.reflect or spec.agent, spec, where="reflect")
    tasks, reward, kwargs = meta_parts(
        problems, slot=slot, spec=slot_spec,
        propose=slot_reflector(model, slot_spec), meta_reward=meta_reward,
        seeds=seeds, artifact_id=spec.artifact_id())

    # `rounds` deliberately low: a round here is `n_workers` inner searches, and
    # the recorded run asked for 8 and got 2 inside a 90-minute budget.
    defaults: Dict[str, Any] = {"rounds": 4, "n_workers": 2, "held_out_frac": 0.4}
    kwargs.update({k: v for k, v in defaults.items() if k not in spec.evolve})
    kwargs.update(spec.evolve)
    kwargs.update(overrides)

    # `policies` and `agg_config` are generic and were being dropped on the
    # floor here -- written in the spec, no error, no effect. They configure the
    # *outer* loop, which for this kind is the meta-search; naming
    # `policies.selection` while evolving `selection` is legal and means the two
    # different things it says.
    if spec.agg_config:
        kwargs.setdefault("agg_config", build_agg_config(spec))
    # `merger=None`, so the reflective-merge pair is **opt-in** here where it is
    # the default for every other kind. `build_policies`' reasoning for that
    # default -- an artifact that is one key makes worker proposals contradict
    # by construction -- applies, but `meta_evolve` does not install it, and the
    # recorded real-data result was produced without it. Measured on the
    # landscape with two workers proposing two valid rules: without the pair a
    # better rule commits; with it, both rounds came back `oracle-rejected` and
    # the artifact stayed at the seed. A spec asking for `reflective_merge` gets
    # it; the CLI silently diverging from the library and from the published run
    # is the worse default.
    policies = build_policies(spec, merger=None)
    if policies is not None:
        kwargs.setdefault("policies", policies)
    if usage is not None:
        kwargs.setdefault("usage", usage)
    if on_round is not None:
        kwargs.setdefault("on_round", on_round)

    n_train = _train_count(len(tasks), kwargs.get("held_out_frac", 0.4))
    n_gate = len(tasks) - n_train
    notes = [f"one rollout is a whole inner search on one of {len(problems)} problem(s); "
             f"{len(tasks)} outer task(s), {n_train} for training and {n_gate} for the gate"]
    # The recurring defect in this line of work, at four different levels: a
    # gate too small to resolve the effect it is judging. It has failed closed
    # (nothing commits) and open (a rule that loses on its own family was
    # committed on 9 held-out instances). Say it before the run, not after.
    if n_gate == 0:
        notes.append(
            "NO held-out task: every acceptance decision would be made on data the search "
            "trained on. Add problems, or raise held_out_frac -- this run cannot tell an "
            "improvement from a fit.")
    elif n_gate < 4:
        notes.append(
            f"only {n_gate} held-out task(s): the gate is being asked to resolve a small "
            "effect from very few samples, which has failed both ways here -- silently "
            "committing a worse rule, and committing nothing at all. More problems is the fix.")
    if len(seeds) > 1:
        # Not an assertion that these seeds are useless -- on a landscape the
        # seed picks the instance and they are real replicates. It is that the
        # question has to be asked, because on a domain whose inner run is a
        # function of the value (deterministic evaluator, cached completions)
        # the seed changes nothing, and five seeds are then one comparison
        # counted five times. That went unnoticed through a whole run here.
        notes.append(
            f"seeds={seeds}: a seed is a replicate only if it changes the inner run. Run one "
            "problem at two seeds and compare the curves before trusting the count -- where "
            "the evaluator is deterministic and completions are cached they will be identical, "
            "and the extra seeds buy nothing that more problems would not buy better.")
    return Composition(tasks=tasks, reward=reward, kwargs=kwargs, notes=notes)


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

    if spec.kind == "policy_slot":
        return _compose_policy_slot(spec, usage=usage, on_round=on_round, **overrides)

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
        # A host plugin is code, and the loader's default extensions are not:
        # `.md .txt .py .json .yaml .yml .toml .sh .cfg .ini`. Without this a
        # `kind: "plugin"` tree is the manifests, the patch and the README, and
        # never the behaviour -- `PLUGIN_CONTEXT["dsh"]` even names
        # `src/**/*.ts`, which the loader could not produce.
        #
        # Widened here rather than in `TreeSpec`, because the default is shared
        # with every other kind and `load_tree` *raises* on a file it matches
        # but cannot represent. A skill directory that happens to carry a
        # minified bundle works today and would start failing outright.
        tspec = TreeSpec()
        if spec.kind == "plugin":
            tspec = replace(tspec, include=tuple(tspec.include) + (
                "**/*.js", "**/*.mjs", "**/*.cjs", "**/*.ts", "**/*.jsx", "**/*.tsx"))
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
    # A CLI agent can reflect, but paying a whole agent session to merge two
    # diffs is not a default anyone would choose; those specs name a cheap model
    # in `reflect`, and that is what merges.
    merger = reflect if not hasattr(reflect, "in_workspace") else None
    policies = build_policies(spec, merger=merger)

    # The audit wraps the reward and the run, so it has to be built after both
    # exist and before the policies bundle is finalised -- its acceptance rule
    # wraps whatever the spec asked for rather than replacing it.
    audit = build_audit(spec, reward=reward, run=kwargs.get("run"),
                        repo_path=repo_path)
    if audit is not None:
        reward = audit.reward
        if audit.run is not None:
            kwargs["run"] = audit.run
        if policies is None:
            policies = Policies()
        audit.acceptance.inner = (policies.acceptance if policies.acceptance
                                  is not None else audit.acceptance.inner)
        policies = replace(policies, acceptance=audit.acceptance)
        notes.append(
            f"audit on: records go to {audit.store.path or 'memory'}"
            + ("" if audit.enabled else
               "; enabled=false, so it collects and corrects nothing -- read "
               "`audit status` before turning it on"))
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
    return Composition(tasks=tasks, reward=reward, kwargs=kwargs, tree=tree,
                       notes=notes, audit=audit)


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
