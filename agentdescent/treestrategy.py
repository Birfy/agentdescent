"""The :class:`FileTree` strategy: a directory as the thing being evolved.

``AppendRules`` accumulates lessons, ``SingleSlot`` holds one instruction,
``KeyedRules`` holds one entry per category -- and :class:`FileTree` holds one
entry per **file**. Nothing else changes: the aggregator still dedupes, resolves
contradictions, fuses complements and gates on held-out reward, except that now
"two workers touched the same key" means "two workers rewrote the same file".

Two things live here that a text strategy never needs:

* :func:`parse_edits` -- a multi-file proposal protocol, because one string can
  no longer mean "the new artifact";
* ``frozen`` -- paths the loop may read but never write. This is L0 in the file
  world (:mod:`agentdescent.governance` freezes *artifacts* by id, which cannot
  express "this artifact may evolve, but not its test suite"). Note that the
  filter here only stops a *proposal*; when the frozen files are what score the
  candidate, the enforcement that matters is the pristine overlay in
  :func:`agentdescent.runners.tree_runner`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .agents import Completion
from .evolvable import Diff, stable_hash
from .filetree import (
    DEFAULT_MAX_FILE_BYTES,
    TreeError,
    canonical,
    match_any,
    parse_tree,
    safe_relpath,
    tree_summary,
)

__all__ = ["FileTree", "parse_edits", "tree_reflector", "EDIT_PROTOCOL"]


# ---------------------------------------------------------------------------
# The proposal protocol
# ---------------------------------------------------------------------------

#: What a reflector is told to emit. Whole-file replacement rather than a unified
#: diff: a model-written patch that does not apply is a *silent* failure mode
#: (wrong context lines, drifted offsets), whereas a whole file either parses or
#: does not. The cost is tokens, which `max_file_bytes` and "split the skill into
#: several files" both bound.
EDIT_PROTOCOL = """Reply with ONE block in exactly this format and nothing else:

<EDITS>
{{"rationale": "<one sentence: what was wrong and why this fixes it>",
 "edits": [{{"path": "<relative path>", "content": "<the COMPLETE new file>"}}]}}
</EDITS>

Rules:
- `content` is the whole file, not a patch or an excerpt.
- Edit at most {max_files} file(s) per reply; prefer the smallest change that fixes
  the observed failure and generalises beyond it.
- Only these paths are editable: {editable}
- Never edit: {frozen}
- To delete a file use {{"path": "...", "delete": true}}."""

_BLOCK_RE = re.compile(r"<EDITS>\s*(.*?)\s*</EDITS>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _first_json_object(text: str) -> Optional[str]:
    """Extract the first balanced ``{...}`` run, ignoring braces inside strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_edits(proposal: str) -> Dict[str, Optional[str]]:
    """Parse a reflector reply into ``{path: new_content}`` (``None`` = delete).

    Lenient about the wrapper -- ``<EDITS>`` block, fenced JSON, or a bare object
    -- and strict about the payload. Returns ``{}`` when nothing parses, because
    a reflector that ignores the protocol is a *quality* problem the run should
    absorb and count (``to_diff`` returns ``None``), not a crash.
    """
    if not isinstance(proposal, str) or not proposal.strip():
        return {}
    m = _BLOCK_RE.search(proposal)
    raw = m.group(1) if m else None
    if raw is None:
        f = _FENCE_RE.search(proposal)
        raw = f.group(1) if f else _first_json_object(proposal)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception:  # noqa: BLE001 - malformed model output, not a bug
        return {}
    if not isinstance(data, dict):
        return {}
    items: Sequence[Any]
    if isinstance(data.get("edits"), list):
        items = data["edits"]
    elif "path" in data:
        items = [data]
    else:
        return {}
    out: Dict[str, Optional[str]] = {}
    for item in items:
        if not isinstance(item, dict) or "path" not in item:
            continue
        try:
            path = safe_relpath(str(item["path"]))
        except TreeError:
            continue                       # a hostile or malformed path: drop it
        if item.get("delete") is True:
            out[path] = None
        elif isinstance(item.get("content"), str):
            out[path] = item["content"]
    return out


# ---------------------------------------------------------------------------
# The strategy
# ---------------------------------------------------------------------------


@dataclass
class FileTree:
    """The artifact **is a directory**; each state key is a relative file path.

    ``editable`` / ``frozen`` are glob lists (``"**"``, ``"tests/**"``,
    ``"*.md"``). ``frozen`` wins over ``editable``. ``max_files_per_diff`` is a
    trust region expressed in the unit that matters here -- a reflector that
    rewrites the whole tree in one proposal is almost never right, and the
    aggregator's own ``trust_region_ops`` (6) is the backstop above it.

        FileTree(load_tree("~/.claude/skills/pdf-audit"),
                 frozen=["tests/**"], max_files_per_diff=2)
    """

    initial_files: Mapping[str, str] = field(default_factory=dict)
    editable: Sequence[str] = ("**",)
    frozen: Sequence[str] = ()
    max_files_per_diff: int = 2
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    #: Paths that do not exist yet but may be created. Only consulted by
    #: :meth:`keys` -- i.e. only under tensor parallelism, where the section map
    #: is fixed before the first round and an unknown path is a section
    #: violation by construction. Under DataParallel (the default) any editable
    #: path can be created without declaring it here.
    planned_paths: Sequence[str] = ()

    def __post_init__(self) -> None:
        self.initial_files = {safe_relpath(k): v for k, v in dict(self.initial_files).items()}

    # -- the Strategy protocol ---------------------------------------------

    def keys(self) -> Sequence[str]:
        """The declared key space, for :class:`~agentdescent.parallel.TensorParallel`.

        **New files cannot be created under TP.** The section map is built once
        from this list (``evolve._resolve_sections``) and a path that is not in it
        maps to ``None``, which never equals a worker's section -- so every
        creation is counted as ``section-violation``. Declare the paths up front
        via ``planned_paths``, or use DataParallel, which has no section map."""
        declared = [p for p in self.initial_files if self.writable(p)]
        declared += [safe_relpath(p) for p in self.planned_paths]
        return sorted(set(declared))

    def initial(self) -> Dict[str, str]:
        return dict(self.initial_files)

    def render(self, state: Mapping[str, str]) -> str:
        return canonical(state)

    def to_diff(self, state, proposal, author, base_version, target) -> Optional[Diff]:
        edits = parse_edits(proposal)
        ops: Dict[str, Optional[str]] = {}
        for path, content in edits.items():
            if not self.writable(path):
                continue
            if content is None:
                if path in state:
                    ops[path] = None       # delete (EvolvingArtifact.apply pops it)
                continue
            if len(content) > self.max_file_bytes:
                continue                   # would be rejected as oversized anyway
            if state.get(path) != content:
                ops[path] = content
        if not ops or len(ops) > self.max_files_per_diff:
            return None
        fingerprint = stable_hash(tuple(sorted((k, v) for k, v in ops.items())))
        return Diff(diff_id=f"{author}:{fingerprint & 0xFFFFFFFF:08x}:{base_version}",
                    target=target, ops=dict(ops), author=author)

    # -- helpers ------------------------------------------------------------

    def writable(self, path: str) -> bool:
        """May the loop write this path? ``frozen`` beats ``editable``."""
        return match_any(path, self.editable) and not match_any(path, self.frozen)

    def frozen_files(self, source: Mapping[str, str]) -> Dict[str, str]:
        """The pristine content of every frozen path, for the runner's overlay."""
        return {p: c for p, c in source.items() if match_any(p, self.frozen)}


# ---------------------------------------------------------------------------
# The reflector
# ---------------------------------------------------------------------------

_TREE_PROPOSE_TMPL = """You maintain the files below. An agent used them to do a task and did \
poorly (reward {reward:.2f} out of 1.00). Improve the files so this class of failure stops \
happening -- generalise, do not hard-code this one case.

FILES IN THE ARTIFACT:
{listing}

{contents}
TASK THE AGENT WAS GIVEN:
{prompt}

WHAT THE AGENT PRODUCED:
{output}
{expected}
{protocol}"""


def tree_reflector(complete: Completion, *, strategy: "FileTree",
                   context_files: Sequence[str] = ("**/SKILL.md", "**/AGENT.md", "*.md"),
                   max_context_chars: int = 12_000,
                   max_output_chars: int = 2_000,
                   template: str = _TREE_PROPOSE_TMPL) -> Any:
    """A ``propose`` callable that asks ``complete`` for multi-file edits.

    It deliberately does **not** put the whole tree in the prompt.
    ``render()`` is the full serialisation because the evaluation cache needs it
    to be lossless -- but a reflection only needs to *see* the files it might
    change, so this shows the full text of ``context_files`` (within
    ``max_context_chars``) and a path+size listing for the rest. On a tree of any
    size that is the difference between a workable prompt and an unaffordable one.
    """
    def propose(rendered: str, task, output: str, reward: float) -> Optional[str]:
        try:
            state = parse_tree(rendered)
        except TreeError:
            return None
        shown, budget = [], max_context_chars
        for path in sorted(state):
            if not strategy.writable(path) or not match_any(path, context_files):
                continue
            body = state[path]
            if len(body) > budget:
                continue
            budget -= len(body)
            shown.append(f"--- {path} ---\n{body}\n")
        gold = (task.meta or {}).get("gold") or (task.meta or {}).get("answer")
        prompt = template.format(
            reward=reward,
            listing=tree_summary(state),
            contents=("CURRENT CONTENT OF THE FILES YOU MAY EDIT:\n"
                      + "\n".join(shown) + "\n") if shown else "",
            prompt=task.prompt,
            output=(output or "")[:max_output_chars],
            expected=f"\nEXPECTED:\n{gold}\n" if gold else "",
            protocol=EDIT_PROTOCOL.format(
                max_files=strategy.max_files_per_diff,
                editable=", ".join(strategy.editable) or "(none)",
                frozen=", ".join(strategy.frozen) or "(nothing)"),
        )
        reply = complete(prompt)
        return reply if isinstance(reply, str) and reply.strip() else None

    return propose
