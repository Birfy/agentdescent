"""Evolving a skill **library**, not one skill.

`evolve_skill_dir` refines a skill someone already decided to have. This is the
other half: the artifact is the whole root, so a run can **add a skill nobody
wrote**, drop one that never earned its place, and move a lesson from one skill
into another. Discovery rather than refinement -- the shape EvoSkill and Voyager
are after, on the harness's real skill root.

Nothing new is needed in the engine for that, and it is worth saying why: an
artifact's state is a flat ``{path: content}`` map, and the aggregator only asks
whether two diffs touch the same key. With the root as the artifact, "two
workers wrote different new skills" is two complementary diffs that fuse, and
"two workers rewrote the same skill" is a contradiction resolved on held-out
score. Adding a file is just a key that was not there before.

What *is* different is where a rollout stages it. A single skill goes to
``.dsh/skills/<name>``; a library is the root itself, ``.dsh/skills`` -- which
is `LAYOUTS["dsh_skill_library"]`.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Sequence, Union

from ..agents import Completion
from ..evolution import EvolutionResult
from ..skilldir import SKILL_BLAST_RADIUS, evolve_skill_dir
from .locate import Skill, skill_roots

__all__ = ["evolve_skill_library", "library_root"]

#: Where a rollout stages the whole library inside its workspace. The harness
#: resolves its project root to the workspace (no `.git` there), so this is the
#: rank-100 `project-dsh` root it scans.
LIBRARY_LAYOUT = "dsh_skill_library"

#: A library's `SKILL.md`s are the catalogue, and the catalogue is what a
#: reflector needs to see to decide a *new* skill is missing. Bodies of nested
#: resources stay out of the prompt; the listing still shows they exist.
LIBRARY_CONTEXT_FILES: Sequence[str] = ("*/SKILL.md",)


def library_root(source: str = "user-dsh", cwd: Optional[str] = None, *,
                 environ: Optional[dict] = None) -> Optional[str]:
    """The path of one of the harness's skill roots, by its provider source.

    ``"project-dsh"`` is this project's own library; ``"user-dsh"`` is the one
    that follows the user everywhere. Returns ``None`` when that root does not
    exist yet -- a library with no directory is not an artifact, and creating one
    silently would put skills somewhere the user never chose.
    """
    for root in skill_roots(cwd, environ=environ):
        if root.source == source:
            return root.path if root.exists else None
    return None


def evolve_skill_library(
    path: str,
    data: Sequence[Any],
    *,
    agent: Completion,
    score: Union[str, Callable] = "contains",
    reflect_with: Optional[Completion] = None,
    frozen: Sequence[str] = (),
    max_files_per_diff: int = 3,
    **evolve_kwargs: Any,
) -> EvolutionResult:
    """Evolve a whole skill root: add, edit, retire.

    ``path`` is the root that *holds* skills (``~/.dsh/skills``), not one skill's
    directory. Every rollout materialises the candidate library into a workspace
    at ``.dsh/skills/`` and the agent reads whichever skills it needs, which is
    the point of a library over an inlined prompt.

    ``max_files_per_diff`` is 3 rather than the single-skill default of 2,
    because the smallest interesting proposal here -- a brand new skill with one
    reference file, alongside a mention in an existing one -- does not fit in
    two. It is still a trust region: a reflector rewriting the whole library in
    one proposal is almost never right.

    Everything else is the ordinary engine. ``frozen`` accepts globs relative to
    the root, so ``frozen=("private/**",)`` keeps a subtree out of both the
    proposal filter and the candidate's reach.
    """
    return evolve_skill_dir(
        path, data,
        agent=agent,
        score=score,
        reflect_with=reflect_with,
        name=os.path.basename(os.path.abspath(os.path.expanduser(path))) or "skills",
        layout=LIBRARY_LAYOUT,
        frozen=tuple(frozen),
        max_files_per_diff=max_files_per_diff,
        blast_radius=SKILL_BLAST_RADIUS,
        **evolve_kwargs,
    )
