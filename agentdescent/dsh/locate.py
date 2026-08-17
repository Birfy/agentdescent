"""Where a DeepSeek Harness installation keeps the skills this package evolves.

M0 of the plugin design evolves a skill directory that a *real* `dsh` already
loads. Which directory that is is not a guess: `dsh-skill-filesystem` resolves
five roots in a fixed rank order, and a lower rank wins a duplicate name.

    100  project-dsh      <projectRoot>/.dsh/skills
    200  project-agents   <projectRoot>/.agents/skills
    300  custom           whatever the provider was configured with
    400  user-dsh         <dshHome>/skills          (skipping its .system child)
    500  user-agents      <agentsHome>/skills

This module mirrors that resolution so ``evolve_skill_dir`` can be pointed at
the directory the harness will actually load, rather than at one the user
*thinks* it loads. Getting this wrong is not a crash -- it is an evolution run
that improves a shadowed copy while the harness keeps serving the winner, which
looks like a method that does not work.

**The project root is the nearest ancestor holding ``.git``**, matching the
provider. Note ``.git`` is a *file* in a worktree and a directory in an ordinary
clone, so both count.

**What this module does not do.** It reads a skill's name from YAML frontmatter
with a regex, not a YAML parser -- this package's core has no dependencies. A
name that needs real YAML (block scalars, anchors) falls back to the path stem,
which is what the name almost always is anyway. It also does not read
frontmatter's invocation policy: `dsh` decides whether a skill is model- or
user-invocable, and nothing here should second-guess that.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

__all__ = [
    "SkillRoot",
    "Skill",
    "dsh_home",
    "agents_home",
    "project_root",
    "skill_roots",
    "list_skills",
    "find_skill",
]

#: The child of the user DSH root that holds system-owned directories. The
#: provider skips it so they are not served as ordinary user skills.
SYSTEM_CHILD = ".system"

#: The file that makes a *directory* a skill. A skill may also be one flat
#: Markdown file, which is why discovery looks for both.
SKILL_FILE = "SKILL.md"

_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)
_NAME_LINE = re.compile(r"^name:[ \t]*(.+?)[ \t]*$", re.M)


@dataclass(frozen=True)
class SkillRoot:
    """One directory the filesystem provider scans, with the rank it scans at."""

    rank: int
    source: str
    path: str

    @property
    def exists(self) -> bool:
        return os.path.isdir(self.path)


@dataclass(frozen=True)
class Skill:
    """One discovered skill: its name, where it is, and which root won it."""

    name: str
    #: The directory to evolve. For a flat skill this is the file's parent,
    #: which is the root itself -- see :attr:`is_directory` before writing.
    path: str
    root: SkillRoot
    #: True when the skill is ``<dir>/SKILL.md``; False when it is one flat
    #: ``<name>.md``. Only a directory skill can grow ``references/`` or
    #: ``scripts/``, so only a directory skill is a file *tree* to evolve.
    is_directory: bool

    @property
    def entrypoint(self) -> str:
        """The Markdown file holding the skill body."""
        return os.path.join(self.path, SKILL_FILE) if self.is_directory else self.path


def _env(name: str, environ: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Read ``name``, treating an empty value as absent.

    The harness does the same for its own variables (an empty `DSH_CORDIS_CONFIG`
    is "not set", not "set to nothing"), and a shell exporting an unset variable
    is common enough that the distinction is load-bearing.
    """
    raw = (environ if environ is not None else os.environ).get(name)
    return raw if raw else None


def dsh_home(environ: Optional[Dict[str, str]] = None) -> str:
    """``$DSH_HOME``, else ``~/.dsh``."""
    return os.path.abspath(os.path.expanduser(_env("DSH_HOME", environ) or "~/.dsh"))


def agents_home(environ: Optional[Dict[str, str]] = None) -> str:
    """``$DSH_AGENTS_HOME``, else ``~/.agents`` -- the shared agent config root."""
    return os.path.abspath(
        os.path.expanduser(_env("DSH_AGENTS_HOME", environ) or "~/.agents"))


def project_root(start: Optional[str] = None) -> str:
    """The nearest ancestor of ``start`` containing ``.git``; ``start`` if none.

    ``.git`` is a directory in an ordinary clone and a *file* in a worktree or
    submodule, so both are accepted -- testing only for a directory would send a
    worktree's skills to the wrong root.
    """
    here = os.path.abspath(os.path.expanduser(start or os.getcwd()))
    current = here
    while True:
        if os.path.exists(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:            # filesystem root, no repo above
            return here
        current = parent


def skill_roots(cwd: Optional[str] = None, *,
                custom_dirs: Sequence[str] = (),
                include_default_roots: bool = True,
                environ: Optional[Dict[str, str]] = None) -> List[SkillRoot]:
    """The roots the filesystem provider scans, in rank order.

    Missing directories are included: a confirmed missing path is valid empty
    state for the provider, and a caller deciding *where to create* a skill
    needs the path whether or not it exists yet. Filter on
    :attr:`SkillRoot.exists` when only populated roots matter.
    """
    roots: List[SkillRoot] = []
    if include_default_roots:
        project = project_root(cwd)
        roots.append(SkillRoot(100, "project-dsh", os.path.join(project, ".dsh", "skills")))
        roots.append(SkillRoot(200, "project-agents", os.path.join(project, ".agents", "skills")))
    for directory in custom_dirs:
        roots.append(SkillRoot(300, "custom",
                               os.path.abspath(os.path.expanduser(directory))))
    if include_default_roots:
        roots.append(SkillRoot(400, "user-dsh", os.path.join(dsh_home(environ), "skills")))
        roots.append(SkillRoot(500, "user-agents",
                               os.path.join(agents_home(environ), "skills")))
    return roots


def _declared_name(markdown_path: str) -> Optional[str]:
    """The ``name:`` from YAML frontmatter, if there is one this regex can read."""
    try:
        with open(markdown_path, "r", encoding="utf-8", errors="replace") as fh:
            head = fh.read(4096)
    except OSError:
        return None
    block = _FRONTMATTER.match(head)
    if not block:
        return None
    found = _NAME_LINE.search(block.group(1))
    if not found:
        return None
    value = found.group(1).strip()
    if value[:1] in ("|", ">"):
        # A block scalar: the name is on the *following* lines, and this regex
        # does not read those. Taking the indicator literally would name the
        # skill ">", so the honest answer is "no name here" and the caller
        # falls back to the path.
        return None
    return value.strip("'\"") or None


def _scan(root: SkillRoot) -> List[Skill]:
    """Every skill in one root, directory form and flat form."""
    try:
        entries = sorted(os.listdir(root.path))
    except OSError:                       # missing or unreadable -> empty state
        return []
    found: List[Skill] = []
    for entry in entries:
        full = os.path.join(root.path, entry)
        if root.source == "user-dsh" and entry == SYSTEM_CHILD:
            continue                      # system-owned, not a user skill
        if os.path.isdir(full):
            body = os.path.join(full, SKILL_FILE)
            if os.path.isfile(body):
                found.append(Skill(_declared_name(body) or entry, full, root, True))
        elif entry.lower().endswith(".md") and entry != SKILL_FILE:
            stem = entry[: -len(".md")]
            found.append(Skill(_declared_name(full) or stem, full, root, False))
    return found


def list_skills(cwd: Optional[str] = None, *,
                custom_dirs: Sequence[str] = (),
                include_default_roots: bool = True,
                environ: Optional[Dict[str, str]] = None) -> List[Skill]:
    """Every skill the harness would serve here, nearest root winning a name.

    Sorted by name, the way the registry sorts its summaries. A name found in
    two roots appears once, from the lower-ranked (nearer) root -- the shadowed
    copy is not returned, because the harness will never load it.
    """
    winners: Dict[str, Skill] = {}
    for root in skill_roots(cwd, custom_dirs=custom_dirs,
                            include_default_roots=include_default_roots,
                            environ=environ):
        for skill in _scan(root):
            winners.setdefault(skill.name, skill)   # rank order -> first wins
    return sorted(winners.values(), key=lambda s: s.name)


def find_skill(name: str, cwd: Optional[str] = None, *,
               custom_dirs: Sequence[str] = (),
               include_default_roots: bool = True,
               environ: Optional[Dict[str, str]] = None) -> Optional[Skill]:
    """The skill the harness would load for ``name``, or ``None``."""
    for skill in list_skills(cwd, custom_dirs=custom_dirs,
                             include_default_roots=include_default_roots,
                             environ=environ):
        if skill.name == name:
            return skill
    return None
