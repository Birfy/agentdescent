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
    ("agentdescent.population", "The population layer",
     "What makes a selection policy take effect on a one-branch ledger.",
     "selection.md"),
    ("agentdescent.fusion", "Model-assisted fusion",
     "Combine competing values for the same key, when a dict update cannot.",
     "aggregator.md"),
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
    parts, ret = _signature_parts(sig)
    rendered = f"({', '.join(parts)}){ret}"
    # Long signatures are unreadable on one line and mkdocs will not wrap them.
    # The heading collapses; `_signature_block` prints the whole thing below it,
    # one parameter per line, so nothing is actually lost.
    if len(name) + len(rendered) > 96:
        rendered = "(...)"
    return f"{name}{rendered}"


def _signature_parts(sig: inspect.Signature) -> Tuple[List[str], str]:
    """The rendered parameters and return annotation of a signature.

    Every module uses `from __future__ import annotations`, so annotations reach
    us as strings and `str(signature)` renders them quoted -- `task: 'Task'`.
    Rebuild instead, unquoting annotations while leaving genuine string
    *defaults* quoted, which is the distinction `str()` cannot make.
    """
    parts: List[str] = []
    # The keyword-only marker is emitted **once**, before the first keyword-only
    # parameter. The check used to read `"*" not in parts[-1:]`, which is true
    # again as soon as one keyword-only parameter has been appended -- so
    # `evolve`'s thirty-odd of them each got their own `*`. It was invisible
    # while every long signature collapsed to `(...)`.
    marked = False
    for pname, param in sig.parameters.items():
        # `cls` for the same reason as `self`: a caller writes
        # `EvolutionResult.load(path)`, and the bound first argument is noise in
        # a one-line signature. It only started mattering once `_methods` began
        # unwrapping classmethods at all.
        if pname in ("self", "cls"):
            continue
        prefix = {param.VAR_POSITIONAL: "*", param.VAR_KEYWORD: "**"}.get(param.kind, "")
        if param.kind is param.KEYWORD_ONLY and not marked:
            parts.append("*")
            marked = True
        elif param.kind is param.VAR_POSITIONAL:
            marked = True
        text = prefix + pname
        if param.annotation is not param.empty:
            text += f": {_annotation(param.annotation)}"
        if param.default is not param.empty:
            text += f" = {param.default!r}"
        parts.append(text)
    ret = ""
    if sig.return_annotation is not sig.empty:
        ret = f" -> {_annotation(sig.return_annotation)}"
    return parts, ret


def _is_collapsed(name: str, obj: Any) -> bool:
    return _signature(name, obj).endswith("(...)")


def _signature_block(name: str, obj: Any) -> List[str]:
    """The full signature, wrapped, for a heading that had to collapse."""
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return []
    parts, ret = _signature_parts(sig)
    if not parts:
        return []
    body = ",\n".join(f"    {part}" for part in parts)
    return ["```python", f"{name}(", body, f"){ret}", "```", ""]


def _doc_params(obj: Any) -> Dict[str, str]:
    """Per-parameter prose from a NumPy-style ``Parameters`` section.

    The signature says a parameter's *type* and *default*; only the docstring
    says what passing it does, and until this existed the page dropped every
    word of it. ``evolve``'s ``solved_threshold`` paragraph -- "lower it for a
    graded scorer, or every rollout asks the reflector to fix an answer that was
    already good" -- was written, tested, and invisible to a reader of the
    reference.

    Grouped entries (``shuffle, seed:``) document several parameters with one
    paragraph, which is how the sources already write them; each name gets the
    shared text rather than the group being dropped for not matching a name.
    """
    doc = inspect.getdoc(obj) or ""
    lines = doc.splitlines()
    start = None
    for index in range(len(lines) - 1):
        if lines[index].strip() == "Parameters" and set(lines[index + 1].strip()) == {"-"}:
            start = index + 2
            break
    if start is None:
        return {}
    out: Dict[str, str] = {}
    names: List[str] = []
    body: List[str] = []

    def flush() -> None:
        text = _clean(" ".join(body))
        if not text:
            return
        # A grouped entry (`shuffle, seed:`) documents several parameters with
        # one paragraph. Printing that paragraph in every row repeats it
        # verbatim -- `max_rollouts` and `max_calls` share 800 words -- so the
        # first name carries it and the rest point at it.
        out[names[0]] = text
        for name in names[1:]:
            out[name] = f"As `{names[0]}`."

    for index in range(start, len(lines)):
        line = lines[index]
        # Another underlined section header ends the block.
        if (index + 1 < len(lines) and line.strip()
                and set(lines[index + 1].strip()) == {"-"} and not line[:1].isspace()):
            break
        if line[:1] not in ("", " ", "\t") and line.rstrip().endswith(":"):
            flush()
            names = [n.strip() for n in line.rstrip()[:-1].split(",") if n.strip()]
            body = []
        elif names:
            body.append(line.strip())
    flush()
    return out


def _annotation(value: Any) -> str:
    if isinstance(value, str):
        return value
    return getattr(value, "__name__", None) or str(value)


def _parameters(obj: Any) -> List[Tuple[str, str, str, str]]:
    """``(name, type, default, prose)`` for every parameter a caller can pass."""
    if inspect.isclass(obj) and getattr(obj, "_is_protocol", False):
        return []
    if _is_enum(obj):
        return []
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return []
    documented = _doc_params(obj)
    rows: List[Tuple[str, str, str, str]] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        prefix = {param.VAR_POSITIONAL: "*", param.VAR_KEYWORD: "**"}.get(param.kind, "")
        annotation = "" if param.annotation is param.empty else _annotation(param.annotation)
        if param.default is param.empty:
            default = "" if prefix else "*required*"
        else:
            rendered = repr(param.default)
            # A default factory or a long literal reprs to something no reader
            # wants in a table cell; the linked source has the whole of it.
            default = f"`{rendered}`" if len(rendered) <= 40 else "*see source*"
        # Sources write `**evolve_kwargs:` in the Parameters block, stars and
        # all, so look the starred form up too rather than dropping the one
        # entry that explains where every unlisted keyword goes.
        prose = documented.get(name) or documented.get(f"{prefix}{name}", "")
        rows.append((f"`{prefix}{name}`",
                     f"`{_clean(annotation)}`" if annotation else "",
                     default,
                     prose))
    return rows


def _param_section(name: str, obj: Any) -> List[str]:
    """The full signature and a parameter table, where a reader needs them.

    The **signature block** is printed when the one-line heading had to collapse
    to ``(...)`` -- 67 of them did, including `evolve`, `evolve_skill` and every
    `evolve_*_dir`, so the page named the entry points and showed none of their
    parameters.

    The **table** is printed only when the docstring actually documents a
    parameter. A table whose prose column is empty on every row says nothing the
    signature above it did not, and a dataclass of eighteen counters produces
    exactly that.
    """
    rows = _parameters(obj)
    if not rows:
        return []
    collapsed = _is_collapsed(name, obj)
    documented = any(prose for *_, prose in rows)
    if not collapsed and not documented:
        return []
    lines = _signature_block(name, obj) if collapsed else []
    if documented:
        lines += ["| parameter | type | default | what it is |", "|---|---|---|---|"]
        lines += [f"| {pname} | {ann} | {default} | {prose} |"
                  for pname, ann, default, prose in rows]
        lines += [""]
    return lines


def _kind(obj: Any) -> str:
    if inspect.isclass(obj):
        return "class"
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return "function"
    return "value"


def _methods(cls: type) -> List[Tuple[str, str]]:
    """Public methods worth listing: defined here, documented, not dunder.

    Unwrap **before** the callable test. A ``staticmethod`` object only became
    callable in 3.10 (bpo-43682), so testing first meant this page had a
    different set of rows on 3.9 than on 3.10+ -- the generated file cannot be
    committed from one interpreter and checked on another, which is how the
    sync test came to pass on 3.9 and fail on 3.11 for the same tree. Six
    `@staticmethod`/`@classmethod` members were missing from the reference on
    the version that wrote it, and the unwrap below was unreachable there.
    """
    out = []
    for name, member in sorted(vars(cls).items()):
        if name.startswith("_"):
            continue
        if isinstance(member, (staticmethod, classmethod)):
            member = member.__func__
        if not callable(member):
            continue
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
        "A signature too long for its heading is printed in full below it, one",
        "parameter per line, followed by a table of what each one does — type,",
        "default, and the docstring's own prose. `*required*` in the default column",
        "means the parameter has none.",
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
            lines += _param_section(name, obj)
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
