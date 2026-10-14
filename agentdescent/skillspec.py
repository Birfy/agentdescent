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
    "frontmatter-malformed": "'name' / 'description' must be YAML strings: no "
                             "unterminated quotes or invalid escapes, no [list] or "
                             "{mapping} values, no unquoted ': ' inside a value, and "
                             "no unquoted null / true / number / date values",
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

#: Plain scalars YAML 1.1 (PyYAML's ``safe_load``, which most hosts use)
#: resolves to something other than a string: null, booleans, ints, floats and
#: dates. ``description: null`` loads as ``None``; ``name: 123`` as an int.
_NON_STRING_PLAIN = re.compile(r"""^(?:
    ~|null|Null|NULL
  | yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF
  | [-+]?0b[01_]+ | [-+]?0x[0-9a-fA-F_]+ | [-+]?0[0-7_]+ | [-+]?(?:0|[1-9][0-9_]*)
  | [-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+
  | [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
  | [-+]?\.[0-9_]+(?:[eE][-+]?[0-9]+)?
  | [-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
  | [-+]?\.(?:inf|Inf|INF) | \.(?:nan|NaN|NAN)
  | [0-9]{4}-[0-9]{2}-[0-9]{2}
  | [0-9]{4}-[0-9]{1,2}-[0-9]{1,2}(?:[Tt]|[ \t]+)[0-9]{1,2}:[0-9]{2}:[0-9]{2}
        (?:\.[0-9]*)?(?:[ \t]*(?:Z|[-+][0-9]{1,2}(?::[0-9]{2})?))?
)$""", re.X)

#: Characters that cannot start a YAML plain scalar (reserved, alias, tag,
#: directive), or that start a nested block collection when followed by space.
_BAD_PLAIN_START = ("@", "`", "*", "%", "!", "- ", "? ", ": ")

#: Single-character escapes a YAML double-quoted scalar allows.
_DQ_ESCAPES = {"0": "\0", "a": "\a", "b": "\b", "t": "\t", "\t": "\t", "n": "\n",
               "v": "\v", "f": "\f", "r": "\r", "e": "\x1b", " ": " ", '"': '"',
               "/": "/", "\\": "\\", "N": "\x85", "_": "\xa0", "L": "\u2028",
               "P": "\u2029"}
_DQ_HEX = {"x": 2, "u": 4, "U": 8}


def _scan_quoted(value: str) -> Optional[str]:
    """Decode a one-line quoted scalar, or ``None`` if YAML would refuse it.

    The closing quote must be followed by nothing or a ``# comment``. Inside
    single quotes ``''`` is the only escape; inside double quotes only YAML's
    escape set is valid (``"bad\\q"`` is a scanner error)."""
    q = value[0]
    out: List[str] = []
    i = 1
    while i < len(value):
        c = value[i]
        if q == "'" and c == "'":
            if value[i + 1:i + 2] == "'":
                out.append("'")
                i += 2
                continue
            break
        if q == '"' and c == '"':
            break
        if q == '"' and c == "\\":
            e = value[i + 1:i + 2]
            if e in _DQ_ESCAPES:
                out.append(_DQ_ESCAPES[e])
                i += 2
                continue
            if e in _DQ_HEX:
                n = _DQ_HEX[e]
                digits = value[i + 2:i + 2 + n]
                if len(digits) != n or not re.fullmatch(r"[0-9a-fA-F]+", digits):
                    return None
                out.append(chr(int(digits, 16)) if int(digits, 16) <= 0x10FFFF else "")
                i += 2 + n
                continue
            return None
        out.append(c)
        i += 1
    else:
        return None                        # never closed on this line
    rest = value[i + 1:]
    if rest and not re.fullmatch(r"\s+#.*", rest) and rest.strip():
        return None                        # text after the closing quote
    return "".join(out)


def _plain(value: str) -> str:
    """A plain scalar without its trailing comment (``# x`` alone is empty)."""
    if value.startswith("#"):
        return ""
    return _INLINE_COMMENT.sub("", value).rstrip()


def _unquote(value: str) -> str:
    if value and value[0] in "'\"":
        decoded = _scan_quoted(value)
        if decoded is not None:
            return decoded
        return value[1:]
    # A plain scalar ends at " #": `name: pdf-audit  # main` is `pdf-audit`.
    return _plain(value)


def _malformed(value: str) -> bool:
    """Would a YAML parser refuse this first-line value, or read a non-string?

    Not a YAML parser: it catches the cheap, common ways a model-written header
    stops loading in the host, or loads as something other than text -- an
    unterminated quote or invalid ``\\`` escape, text after a closing quote, a
    flow ``[...]`` / ``{...}`` value, a plain scalar with ``": "`` in it or a
    trailing ``:`` (YAML reads that as a nested mapping and errors), a
    reserved/alias/directive first character, and plain values YAML resolves to
    null / bool / number / date. Colons and type checks look at the value with
    its ``# comment`` removed, so ``Shout things. # Usage:`` is fine."""
    if not value:
        return False
    if value[0] in "'\"":
        return _scan_quoted(value) is None
    if value[0] in "[{":
        return True
    plain = _plain(value)
    if not plain:
        return False
    if plain.startswith(_BAD_PLAIN_START) or plain in ("-", "?"):
        return True
    if ": " in plain or plain.endswith(":"):
        return True
    return bool(_NON_STRING_PLAIN.match(plain))


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
