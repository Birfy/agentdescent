"""Structural checks for an Agent Skills directory (``kind="skill_dir"``).

What makes a directory a *skill* in every host AgentDescent ships to (Claude
Code, Codex, OpenCode, DeepSeek Harness) is a root ``SKILL.md`` whose YAML
frontmatter carries a valid ``name`` and ``description``. The rollout cannot
see that invariant: :func:`~agentdescent.runners.tree_runner` points the agent
at the files directly and skips description-based discovery, so a candidate
that deletes ``SKILL.md`` or empties its ``description`` can score as well as
its parent under evaluation and then never trigger once installed.

:func:`validate_skill_tree` is the cheapest gate for that: it costs zero
rollouts, and :func:`skill_tree_validator` adapts it to
:class:`~agentdescent.treestrategy.FileTree`'s ``validators`` hook so a
structurally broken candidate is rejected before it is ever measured.

Hand-parsed, like ``integrations._skill_frontmatter``: the core has no PyYAML
dependency. Only hard errors are reported; the specification's "keep
``SKILL.md`` under 500 lines" is guidance and is deliberately not enforced, so
skills users already have are not rejected.

Constraints follow https://agentskills.io/specification.
"""

from __future__ import annotations

import re
from typing import Dict, List, Mapping, Optional, Set, Tuple

__all__ = [
    "SKILL_FILE",
    "SKILL_RULES",
    "SkillTreeValidator",
    "parse_skill_frontmatter",
    "skill_tree_validator",
    "validate_skill_tree",
]

SKILL_FILE = "SKILL.md"

#: Reason code -> human-readable rule. Codes are what
#: ``FileTree.structural_rejections`` is keyed by (prefixed with ``skill-``).
SKILL_RULES: Dict[str, str] = {
    "missing-skill-md": "SKILL.md must exist at the root of the skill directory",
    "no-frontmatter": "SKILL.md must start with a closed '---' YAML frontmatter block",
    "frontmatter-malformed": "'name' / 'description' must be plain YAML strings: no "
                             "unterminated quotes, no [list] or {mapping} values, "
                             "and no unquoted ': ' inside a value",
    "name-missing": "frontmatter needs a non-empty 'name'",
    "name-invalid": "'name' must be 1-64 chars of lowercase letters, digits and "
                    "single hyphens, not starting or ending with a hyphen",
    "name-mismatch": "'name' must match the skill directory name",
    "description-missing": "frontmatter needs a non-empty 'description'",
    "description-too-long": "'description' must be at most 1024 characters",
    "compatibility-too-long": "'compatibility' must be at most 500 characters",
}

_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_BLOCK_SCALAR = re.compile(r"^[|>][+-]?\d*$")


_INLINE_COMMENT = re.compile(r"\s+#.*$")


def _closed_quote(value: str) -> Optional[str]:
    """The quoted scalar ``value`` without a trailing comment, if it is closed."""
    for v in (value, _INLINE_COMMENT.sub("", value)):
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            return v
    return None


def _unquote(value: str) -> str:
    quoted = _closed_quote(value)
    if quoted is not None:
        return quoted[1:-1]
    if value[:1] not in "'\"":
        # A plain scalar ends at " #": `name: pdf-audit  # main` is `pdf-audit`.
        value = _INLINE_COMMENT.sub("", value)
    return value


def _malformed(value: str) -> bool:
    """Would a YAML parser refuse this first-line value, or read a non-string?

    Not a YAML parser: it catches the cheap, common ways a model-written header
    stops loading in the host -- an unterminated quote, a flow ``[...]`` /
    ``{...}`` value, and a plain scalar with ``": "`` in it (YAML reads that as
    a nested mapping and errors)."""
    if not value:
        return False
    if value[0] in "'\"":
        return _closed_quote(value) is None
    if value[0] in "[{":
        return True
    return ": " in _INLINE_COMMENT.sub("", value) or value.endswith(":")


def parse_skill_frontmatter(text: str) -> Optional[Dict[str, str]]:
    """The top-level ``key: value`` pairs of a ``SKILL.md`` frontmatter block.

    Returns ``None`` when the text does not open with a closed ``---`` block.
    Indented continuation lines (plain multi-line or ``|`` / ``>`` block
    scalars) are folded into the preceding key with single spaces; surrounding
    quotes are stripped. This is enough for the flat header the spec defines,
    not a YAML parser.
    """
    parsed = _parse(text)
    return None if parsed is None else parsed[0]


def _parse(text: str) -> Optional[Tuple[Dict[str, str], Set[str]]]:
    """``(pairs, keys whose first-line value is malformed)``; see
    :func:`parse_skill_frontmatter`."""
    if not isinstance(text, str):
        return None
    if text.startswith("\ufeff"):         # a UTF-8 BOM is not content
        text = text[1:]
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines or lines[0].strip() != "---":
        return None
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        return None
    out: Dict[str, str] = {}
    malformed: Set[str] = set()
    key: Optional[str] = None
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[:1].isspace():
            if key is not None:
                out[key] = (out[key] + " " + line.strip()).strip()
            continue
        if ":" not in line:
            key = None
            continue
        k, _, v = line.partition(":")
        key = k.strip()
        v = v.strip()
        if _BLOCK_SCALAR.match(v):
            out[key] = ""
            continue
        if _malformed(v):
            malformed.add(key)
        out[key] = _unquote(v)
    return out, malformed


def validate_skill_tree(state: Mapping[str, Optional[str]], *,
                        root_name: Optional[str] = None) -> List[str]:
    """Reason codes (keys of :data:`SKILL_RULES`) for every hard violation.

    ``state`` is a :class:`~agentdescent.treestrategy.FileTree` state: relative
    path -> file text. ``root_name``, when given, is the name the skill must
    keep (normally the ``name`` of the *initial* tree, since the spec requires
    it to equal the parent directory name). An empty list means valid.
    """
    text = state.get(SKILL_FILE)
    if text is None:
        return ["missing-skill-md"]
    parsed = _parse(text)
    if parsed is None:
        return ["no-frontmatter"]
    fm, malformed = parsed
    errors: List[str] = []
    if malformed & {"name", "description"}:
        errors.append("frontmatter-malformed")
    name = fm.get("name", "").strip()
    if not name:
        errors.append("name-missing")
    elif len(name) > 64 or not _NAME_RE.match(name):
        errors.append("name-invalid")
    elif root_name is not None and name != root_name:
        errors.append("name-mismatch")
    desc = fm.get("description", "").strip()
    if not desc:
        errors.append("description-missing")
    elif len(desc) > 1024:
        errors.append("description-too-long")
    if len(fm.get("compatibility", "").strip()) > 500:
        errors.append("compatibility-too-long")
    return errors


class SkillTreeValidator:
    """A ``FileTree(validators=...)`` callable: ``None`` when the candidate tree
    is a valid skill, else the first violation as ``"skill-<code>"``.

    A class rather than a closure so a :class:`FileTree` carrying it still
    pickles (specs and strategies cross process boundaries)."""

    __slots__ = ("root_name",)

    def __init__(self, root_name: Optional[str] = None) -> None:
        self.root_name = root_name

    def __call__(self, state: Mapping[str, Optional[str]]) -> Optional[str]:
        errors = validate_skill_tree(state, root_name=self.root_name)
        return f"skill-{errors[0]}" if errors else None

    def __getstate__(self):
        return {"root_name": self.root_name}

    def __setstate__(self, state):
        self.root_name = state["root_name"]

    def __repr__(self) -> str:
        return f"SkillTreeValidator(root_name={self.root_name!r})"


def skill_tree_validator(*, root_name: Optional[str] = None) -> SkillTreeValidator:
    """Shorthand for :class:`SkillTreeValidator`."""
    return SkillTreeValidator(root_name)
