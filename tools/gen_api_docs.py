"""Generate ``docs/api.md`` from the package's own signatures and docstrings.

A hand-written API reference is wrong the moment a signature changes, and nobody
notices until a reader copies a call that no longer exists. This generates the
page from :data:`agentdescent.__all__` -- so the reference cannot list a symbol
the package does not export, cannot miss one it does, and cannot disagree with a
signature.

    python -m tools.gen_api_docs            # rewrite docs/api.md
    python -m tools.gen_api_docs --check    # exit 1 if it is out of date (CI)

``tests/test_api_reference.py`` runs the ``--check`` path, so a signature change
that is not regenerated fails the suite rather than shipping a stale page.
"""

from __future__ import annotations

import argparse
import enum
import inspect
import os
import re
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agentdescent  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "api.md")

#: Module -> (title, one-line role, doc page). Order here is the order on the
#: page: the things you touch first come first.
SECTIONS: List[Tuple[str, str, str, str]] = [
    ("agentdescent.evolution", "The loop",
     "`evolve()`, the artifact, the actor, and what a run returns.", "evolution.md"),
    ("agentdescent.skill", "One-call skill evolution",
     "The shortest path from a dataset to an evolved instruction.", "quickstart-skill.md"),
    ("agentdescent.skilldir", "One-call directory evolution",
     "The same, for a skill folder, an agent folder, or its code.",
     "directory-evolution.md"),
    ("agentdescent.agents", "Agents and models",
     "Any `prompt -> text` is a completion; a `WorkspaceAgent` also has a directory.",
     "agents.md"),

    ("agentdescent.filetree", "Directories as state",
     "Load a directory into state, materialise it back, serialise it losslessly.",
     "directory-evolution.md"),
    ("agentdescent.treestrategy", "The file-tree strategy",
     "One state key per file, plus the multi-file proposal protocol.",
     "directory-evolution.md"),
    ("agentdescent.runners", "Runners",
     "Give a real agent the candidate directory, one workspace per rollout.",
     "directory-evolution.md"),
    ("agentdescent.evolvable", "The data model",
     "What a unit of evolution is, and what a gradient looks like here.",
     "data-model.md"),
    ("agentdescent.aggregator", "The aggregator (the optimizer)",
     "Staleness filter, conflict resolution, fusion, acceptance, commit.",
     "aggregator.md"),
    ("agentdescent.ledger", "The ledger",
     "The git-backed, compare-and-swap artifact store.", "ledger.md"),
    ("agentdescent.verifier", "The verifier",
     "Rule / learned / oracle, and the budget that bounds the expensive one.",
     "verifier.md"),
    ("agentdescent.governance", "Governance",
     "L0 frozen / L1 slow / L2 fast, assigned by blast radius.", "governance.md"),
    ("agentdescent.staleness", "Staleness policies",
     "What to do with a diff proposed against a version that has moved.",
     "staleness.md"),
    ("agentdescent.parallel", "Parallelism methods",
     "How a round's work is split across workers: DP / TP / PP.", "parallelism.md"),
    ("agentdescent.sampling", "Task sampling",
     "Which task a worker rolls out next.", "sampling.md"),
    ("agentdescent.selection", "Candidate selection",
     "Which candidate the next batch of workers starts from.", "selection.md"),
    ("agentdescent.advantage", "Borrowed RL decision rules",
     "Group-relative advantage, an adaptive trust region, distance from stable.",
     "concepts.md"),
    ("agentdescent.scheduler", "Scheduling and audits",
     "Duration-aware dispatch, straggler handling, and the oracle audit queue.",
     "duration-scheduling.md"),
    ("agentdescent.dataloader", "The data layer",
     "Datasets, splits, and cached fetches from HuggingFace or raw URLs.",
     "dataloader.md"),

    ("agentdescent.async_evolve", "Barrier-free evolution",
     "`evolve()` without the round barrier.", "async.md"),
    ("agentdescent.async_runtime", "The async orchestrator",
     "The reference barrier-free runtime and its statistics.", "async.md"),
    ("agentdescent.orchestrator", "The reference orchestrator",
     "The round loop the research results were measured with.", "orchestrator.md"),
    ("agentdescent.worker", "The worker", "One worker's rollout and proposal.",
     "orchestrator.md"),
]

#: Symbols exported as plain values (not classes or functions) need a written
#: line each -- ``inspect`` has nothing useful to say about an int or a frozenset.
CONSTANTS: Dict[str, str] = {
    "SOLVED": "Reward at or above which a task counts as solved (`0.999`). "
              "Lower it for a graded scorer, or every rollout asks the reflector "
              "to fix an answer that was already good.",
    "FAST_MAX": "The L2/L1 blast-radius boundary (`0.30`).",
    "FROZEN_IDS": "Artifact ids the loop may read but never mutate (L0).",
    "LAYOUTS": "Where a runner writes the evolving tree inside a workspace "
               "(`claude_skill`, `skill_library`, `claude_agent`, `root`).",
    "TEST_FAILURE_MARKER": "Prefix of the output `code_runner` produces when the "
                           "frozen gate fails, so the failure scores 0 and the "
                           "reflector can read it.",
    "EDIT_PROTOCOL": "The multi-file proposal format a `FileTree` reflector is "
                     "told to emit.",
    "Completion": "`Callable[[str], str]` — the one contract every model and "
                  "agent satisfies.",
    "VersionVector": "`Dict[str, int]` — artifact id to version.",
    "AggregatorFactory": "`(ledger, verifier, audit, config, policy) -> "
                         "AggregatorProtocol` — how a custom optimizer is installed.",
    "LedgerFailure": "The exception tuple a caller catches to treat any ledger "
                     "problem as recoverable.",
}


#: Sphinx roles the source uses for cross-references. They are correct in the
#: docstrings and meaningless on a Markdown page, so they are rewritten to the
#: thing a reader wants to see rather than left as `:class:`~a.b.C``.
_ROLE = re.compile(r":(?:class|func|meth|mod|data|attr|obj|exc):`~?([^`]+)`")


def _clean(text: str) -> str:
    """Docstring prose -> Markdown: RST roles and double-backticks resolved."""
    text = _ROLE.sub(lambda m: f"`{m.group(1).rsplit('.', 1)[-1]}`", text)
    text = text.replace("``", "`")
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("|", r"\|")          # tables


def _is_enum(obj: Any) -> bool:
    return inspect.isclass(obj) and issubclass(obj, enum.Enum)


def _summary(obj: Any) -> str:
    """The first paragraph of a docstring, collapsed to one line."""
    doc = inspect.getdoc(obj) or ""
    # A dataclass with no docstring inherits an auto-generated one that is just
    # its own repr -- noise, and it duplicates the signature directly above it.
    if inspect.isclass(obj) and doc.startswith(f"{obj.__name__}("):
        return ""
    # `getdoc` walks the MRO, so a class with no docstring of its own reports its
    # parent's -- which says nothing about *this* class, and for enums is
    # version-dependent boilerplate ("An enumeration." on 3.9, "Enum where
    # members are also (and must be) ints" on 3.12). Either would make the page's
    # contents depend on the interpreter that generated it, which the sync test
    # then reports as the page being stale. `vars()` sees only what the class
    # itself defines.
    if inspect.isclass(obj) and vars(obj).get("__doc__") is None:
        return ""
    # Python 3.9's EnumMeta goes further and *writes* a docstring onto an enum
    # that has none, so `vars()` cannot see the difference there. It is the same
    # non-information, so drop it by name.
    if _is_enum(obj) and doc.strip() == "An enumeration.":
        return ""
    para: List[str] = []
    for line in doc.splitlines():
        if not line.strip():
            break
        para.append(line.strip())
    return _clean(" ".join(para))


def _signature(name: str, obj: Any) -> str:
    # A Protocol's __init__ is `(*args, **kwargs)`: true, and useless. What a
    # reader needs is the method set, which is listed underneath.
    if inspect.isclass(obj) and getattr(obj, "_is_protocol", False):
        return name
    # An enum's "constructor" is `EnumMeta.__call__`, whose signature changes
    # between Python versions (`(value, names=None, *, module=None, ...)` on 3.9,
    # `(*values)` on 3.12) -- so rendering it makes the page differ by
    # interpreter, which the sync test then reports as the page being stale. It
    # is also not what anyone wants to read: the members are, and they are listed
    # underneath.
    if _is_enum(obj):
        return name
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return name
    # Every module uses `from __future__ import annotations`, so annotations reach
    # us as strings and `str(signature)` renders them quoted -- `task: 'Task'`.
    # Rebuild instead, unquoting annotations while leaving genuine string
    # *defaults* quoted, which is the distinction str() cannot make.
    parts: List[str] = []
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        prefix = {param.VAR_POSITIONAL: "*", param.VAR_KEYWORD: "**"}.get(param.kind, "")
        if param.kind is param.KEYWORD_ONLY and "*" not in "".join(parts[-1:]):
            parts.append("*")
        text = prefix + pname
        if param.annotation is not param.empty:
            ann = param.annotation
            text += f": {ann if isinstance(ann, str) else getattr(ann, '__name__', ann)}"
        if param.default is not param.empty:
            text += f" = {param.default!r}" if param.annotation is param.empty \
                else f" = {param.default!r}"
        parts.append(text)
    ret = ""
    if sig.return_annotation is not sig.empty:
        r = sig.return_annotation
        ret = f" -> {r if isinstance(r, str) else getattr(r, '__name__', r)}"
    rendered = f"({', '.join(parts)}){ret}"
    # Long signatures are unreadable on one line and mkdocs will not wrap them;
    # the full text stays available in the linked source.
    if len(name) + len(rendered) > 96:
        rendered = "(...)"
    return f"{name}{rendered}"


def _kind(obj: Any) -> str:
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return "function"
    return "value"


def _methods(cls: type) -> List[Tuple[str, str]]:
    """Public methods worth listing: defined here, documented, not dunder."""
    out = []
    for name, member in sorted(vars(cls).items()):
        if name.startswith("_") or not callable(member):
            continue
        if isinstance(member, (staticmethod, classmethod)):
            member = member.__func__
        summary = _summary(member)
        if not summary:
            continue
        out.append((_signature(name, member), summary))
    return out


#: Modules reached as `agentdescent.<module>.<name>` rather than re-exported at
#: the top level. They have doc pages and a public API, so leaving them out of
#: the reference because of an import style would be a hole in it.
SUBMODULES: List[Tuple[str, str, str, str]] = [
    ("agentdescent.backends", "Document backends",
     "A tool-using agent over a document that is too big for a prompt.",
     "backends.md"),
    ("agentdescent.rewards", "Ready-made scorers",
     "The reward functions everyone writes, with the details right.", "rewards.md"),
    ("agentdescent.baselines", "Equal-budget baselines",
     "merge-of-N against best-of-N fork and serial, on one rollout budget.",
     "results.md"),
]


def _public_names(module) -> Dict[str, Any]:
    """A submodule's public API: its `__all__`, or every public name it defines."""
    names = getattr(module, "__all__", None)
    if names is None:
        names = [n for n, o in vars(module).items()
                 if not n.startswith("_")
                 and getattr(o, "__module__", None) == module.__name__]
    return {n: getattr(module, n) for n in names}


def render() -> str:
    import importlib

    exported = {name: getattr(agentdescent, name) for name in agentdescent.__all__}
    for module_name, _, _, _ in SUBMODULES:
        exported.update(_public_names(importlib.import_module(module_name)))
    by_module: Dict[str, List[str]] = {}
    for name, obj in exported.items():
        by_module.setdefault(getattr(obj, "__module__", "") or "", []).append(name)

    lines: List[str] = [
        "# API reference",
        "",
        "Every name `agentdescent` exports, grouped by the module it comes from.",
        "**Generated** from the package's own signatures and docstrings by",
        "`python -m tools.gen_api_docs` — `tests/test_api_reference.py` fails if this",
        "page and the code disagree, so a signature here is the signature you get.",
        "",
        "Each section links to the page that explains *why* the module is shaped the",
        "way it is; this page is the *what*.",
        "",
        f"{len(exported)} public names across "
        f"{len([m for m in by_module if m.startswith('agentdescent')])} modules.",
        "",
        "---",
        "",
    ]

    covered = set()
    for module, title, role, page in SECTIONS + SUBMODULES:
        names = sorted(by_module.get(module, []))
        if not names:
            continue
        covered.update(names)
        lines += [f"## {title}", "",
                  f"{role} &nbsp;·&nbsp; `{module}` &nbsp;·&nbsp; [guide]({page})", ""]
        for name in names:
            obj = exported[name]
            kind = _kind(obj)
            if kind == "value":
                lines += [f"### `{name}`", "",
                          CONSTANTS.get(name, _summary(obj) or "—"), ""]
                continue
            lines += [f"### `{_signature(name, obj)}`", ""]
            summary = _summary(obj)
            if summary:
                lines += [summary, ""]
            if _is_enum(obj):
                lines += ["| member | value |", "|---|---|"]
                lines += [f"| `{m.name}` | `{m.value!r}` |" for m in obj]
                lines += [""]
                continue
            if kind == "class":
                methods = _methods(obj)
                if methods:
                    lines += ["| method | what it does |", "|---|---|"]
                    lines += [f"| `{sig}` | {doc} |" for sig, doc in methods]
                    lines += [""]
        lines += ["---", ""]

    leftovers = sorted(set(exported) - covered)
    if leftovers:
        lines += ["## Type aliases and constants", "",
                  "Values rather than classes or functions.", ""]
        for name in leftovers:
            lines += [f"### `{name}`", "",
                      CONSTANTS.get(name, _summary(exported[name]) or "—"), ""]
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if docs/api.md is out of date")
    args = ap.parse_args()
    generated = render()
    if args.check:
        current = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
        if current != generated:
            print("docs/api.md is out of date; run: python -m tools.gen_api_docs",
                  file=sys.stderr)
            return 1
        print("docs/api.md is up to date")
        return 0
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(generated)
    print(f"wrote {OUT} ({len(generated.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
