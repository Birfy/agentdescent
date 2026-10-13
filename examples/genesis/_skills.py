"""The write half of the skill loop: distilling an accepted contribution into a skill.

``LocalWorld.skills()`` already collected ``.agents/skills/`` along the node chain and
put the *names* in the brief -- the read half, and nothing else: a skill a person wrote,
inherited, never written back. This module is the other half. It ports the
``skill_extractor`` role (``agents/skill_extractor.ex``): it checks the skills that
already exist so it does not duplicate one, places each at the node where it is
relevant, and is allowed to find nothing.

**Placement is the one difference that matters, and it is deliberate.** The extractor
runs *inside the episode that produced the contribution*, so the skill it writes is part
of the same proposal and the next agent in the run inherits it. The reference role runs
after the PR merges; on this engine that is a version no later episode of the same run
ever sees, so a literal trigger would produce the write half and still leave the loop
open. The root episode boundary is the cheapest honest reading of "extract from a
completed contribution".

**The tools are not ported, and the node field is why they do not have to be.**
``skill_add``/``skill_edit`` write a file, ``skill_enable`` places it at the Context Tree
node "where it's most relevant", and ``skill_read`` fetches a body on demand. The
extractor here returns the file contents and the node, the spatial contract enforces
that the node is one the root may write, and the brief carries the bodies because the
port's agents have no read tool to fetch them with -- see :meth:`LocalWorld.situate`.

The frontmatter is rendered here rather than by the model: the format is the tool's
(``name``, ``description``, ``parameters``), and a model that has to emit YAML is a
model that can emit YAML that will not parse.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agentdescent.filetree import match_any

from ._world import SKILLS_DIR, normalise

if TYPE_CHECKING:                       # `_delegation` imports this module
    from ._delegation import Brief, Edit

__all__ = ["SKILL_PROMPT", "SkillExtractor", "is_skill_path", "skill_name", "skill_path"]

#: Skill names are kebab-case and become a path component, so the rule is enforced
#: rather than trusted to the model.
_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


SKILL_PROMPT = """You are the skill-extraction agent of a persistent software world. \
A contribution was just accepted; your job is to distil the part of it that a later \
agent would otherwise have to re-discover into a **skill** -- a markdown file under \
`.agents/skills/` that a future agent working here will be handed.

{context}

The contribution was made against this objective:

{objective}

These are the files it wrote, in full:

{work}

The skills that already exist in this world:

{existing}

Answer only from what is in front of you. A skill earns its place only if it is \
**complex, important and reusable** -- a procedure, a convention, a gotcha, an \
invariant that a later agent working here would get wrong without being told. Do \
**not** write one for:

- anything a competent developer would already know;
- a one-off fix with no reusable pattern;
- something the files above already state plainly;
- a duplicate of an existing skill -- do not rewrite it, skip it.

It is a correct and expected answer to find nothing worth extracting.

For each skill, choose the **node** -- the directory a later agent would be standing \
in when the advice is relevant; `""` is the repository root -- and give it a \
`kebab-case` name, a one-line `description`, and the `body` in markdown. List inputs \
under `parameters` only if the skill really takes any.

Reply with ONE JSON object and nothing else:
{{"skills": [{{"node": "<directory, or \\"\\" for the repository root>",
              "name": "<kebab-case>",
              "description": "<one line>",
              "body": "<the skill, in markdown>",
              "parameters": [{{"name": "<arg>", "type": "<type>",
                               "description": "<what it is>",
                               "required": true}}]}}],
 "reason": "<one sentence: why these, or why nothing>"}}"""


def skill_path(node: str, name: str) -> str:
    """``<node>/.agents/skills/<name>.md``, at the root when ``node`` is empty."""
    node = normalise(node)
    return f"{node}/{SKILLS_DIR}/{name}.md" if node else f"{SKILLS_DIR}/{name}.md"


def is_skill_path(path: str) -> bool:
    """Is this a skill file rather than work? The same test ``_suite.cold_start`` uses."""
    return f"/{SKILLS_DIR}/" in f"/{normalise(path)}"


def skill_name(path: str) -> Optional[str]:
    """The skill's name -- its basename without ``.md`` -- or ``None``."""
    if not is_skill_path(path):
        return None
    tail = path.rsplit("/", 1)[-1]
    return tail[:-3] if tail.endswith(".md") else tail


class SkillExtractor:
    """``(parent, work) -> [Edit]`` -- skills distilled from an accepted contribution.

    Installs on :class:`~examples.genesis._delegation.RecursiveDelegation` as
    ``skills=``. ``None`` (the default there) is the old behaviour: the world inherits
    skills and never writes one.

    The counters are the point of the counters: a run whose extractor was never called,
    a run that was called and always found nothing, and a run that tried to write a
    duplicate look identical in the artifact and need opposite fixes.
    """

    def __init__(self, complete, *, contracts: Sequence[str] = (),
                 frozen: Sequence[str] = (), max_skills: int = 2,
                 max_chars: int = 8_000, per_file_chars: int = 4_000,
                 max_work_files: int = 8):
        self._complete = complete
        self._contracts = tuple(contracts)
        #: Globs a skill may never be written under -- the frozen specification is a
        #: contract, not a place the world is allowed to grow a skill into.
        self._frozen = tuple(frozen)
        self._max_skills = max_skills
        self._max_chars = max_chars
        self._per_file_chars = per_file_chars
        self._max_work_files = max_work_files
        #: Model calls made, skills written, and the three ways one was refused.
        self.calls = 0
        self.extracted = 0
        self.deduped = 0
        self.refused = 0
        self.unparsed = 0

    def __call__(self, parent: "Brief", work: Sequence["Edit"]) -> List["Edit"]:
        from ._delegation import Edit        # circular at import time, not at call time

        contributed = [e for e in work if e.kind == "work" and e.content]
        if not contributed:
            # Nothing but bookkeeping came back: there is no contribution to learn
            # from, and a call spent on the routing table would be a call spent on
            # nothing. Not counted -- this is the common case, not a failure.
            return []
        existing = [k for k in parent.state if is_skill_path(k)]
        self.calls += 1
        prompt = SKILL_PROMPT.format(
            context=parent.context, objective=parent.objective,
            work=_render(contributed[:self._max_work_files], self._per_file_chars,
                         self._max_chars),
            existing="\n".join(f"  {k}" for k in sorted(existing)) or "  (none yet)")
        try:
            reply = self._complete(prompt) or ""
        except Exception:  # noqa: BLE001 - a dead call must not cost the contribution
            self.unparsed += 1
            return []
        items, usable = _parse(reply)
        if not usable:
            self.unparsed += 1
            return []

        dirs = _directories(parent.state)
        taken = {n for n in (skill_name(k) for k in existing) if n}
        out: List["Edit"] = []
        for item in items:
            if len(out) >= self._max_skills:
                break
            node = normalise(str(item.get("node", "")))
            name = str(item.get("name", "")).strip()
            body = str(item.get("body", "")).strip()
            path = skill_path(node, name)
            if not name or not body or not _NAME_RE.match(name) or len(name) > 64:
                self.refused += 1
                continue
            if not _is_node(node, parent.state, dirs):
                self.refused += 1
                continue
            if self._frozen and match_any(path, self._frozen):
                self.refused += 1
                continue
            if path in parent.state or name in taken:
                # The extractor is required to look before it writes: a skill that
                # already exists is not improved by a second copy of it.
                self.deduped += 1
                continue
            taken.add(name)
            out.append(Edit(
                owner=parent.world.path, path=path, kind="skill",
                content=skill_file(name, str(item.get("description", "")),
                                   item.get("parameters"), body)))
        self.extracted += len(out)
        return out

    def stats(self) -> str:
        return (f"skills={self.extracted} written/{self.deduped} dupe/"
                f"{self.refused} refused/{self.unparsed} unparsed "
                f"over {self.calls} calls")


def skill_file(name: str, description: str, parameters: Any, body: str) -> str:
    """One skill file: the frontmatter, then the body.

    Rendered rather than model-emitted for the reason the module docstring gives: the
    format is the tool's, not the model's.
    """
    description = " ".join(str(description or "").split())[:160] or name
    lines = ["---", f"name: {name}",
             f'description: "{description.replace(chr(34), chr(39))}"']
    params = parameters if isinstance(parameters, list) else []
    if params:
        lines.append("parameters:")
        for p in params:
            if not isinstance(p, Mapping):
                continue
            lines.append(f"  - name: {str(p.get('name', '')).strip()}")
            lines.append(f"    type: {str(p.get('type', 'string')).strip() or 'string'}")
            lines.append("    description: \""
                         + " ".join(str(p.get("description", "")).split())[:160]
                         .replace(chr(34), chr(39)) + "\"")
            lines.append("    required: "
                         + str(bool(p.get("required", False))).lower())
    lines.append("---")
    return "\n".join(lines) + "\n\n" + body.rstrip() + "\n"


def _render(work: Sequence["Edit"], per_file: int, budget: int) -> str:
    """Whole files, clipped per file, then trimmed from the end within ``budget``."""
    out: List[str] = []
    for edit in work:
        body = edit.content or ""
        if per_file and len(body) > per_file:
            body = body[:per_file] + "\n... [truncated] ..."
        out.append(f"# {edit.path}\n```\n{body}\n```")
    text = "\n\n".join(out)
    return text if len(text) <= budget else text[:budget] + "\n... [truncated] ..."


def _directories(state: Mapping[str, str]) -> set:
    """Every directory the state implies, root included."""
    out = {""}
    for key in state:
        parts = normalise(key).split("/")[:-1]
        acc = ""
        for part in parts:
            acc = f"{acc}/{part}" if acc else part
            out.add(acc)
    return out


def _is_node(node: str, state: Mapping[str, str], dirs: Mapping[str, Any]) -> bool:
    """Is ``node`` a directory this world has, and not a file of the same name?"""
    if node == "":
        return True
    if node in state:                        # a file, not a node
        return False
    return node in dirs


def _parse(reply: str) -> Tuple[List[Dict[str, Any]], bool]:
    """``(items, usable)``. ``usable`` is False when the reply is not a JSON object.

    A reply that parses but carries an empty ``skills`` list is **usable**: finding
    nothing is an explicitly valid answer, and counting it as a parse failure would make
    "the model looked and found nothing" indistinguishable from "the model broke", which
    are opposite problems.
    """
    start = reply.find("{")
    if start < 0:
        return [], False
    try:
        data = json.loads(reply[start:reply.rfind("}") + 1])
    except Exception:  # noqa: BLE001 - malformed model output, not a bug
        return [], False
    if not isinstance(data, Mapping):
        return [], False
    if "skills" not in data or data["skills"] is None:
        # The key absent is the model saying "nothing" without spelling it out; the
        # key present but the wrong type is a broken protocol. Different problems.
        return [], True
    items = data["skills"]
    if not isinstance(items, list):
        return [], False
    return [i for i in items if isinstance(i, Mapping)], True
