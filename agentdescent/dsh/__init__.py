"""Talking to a DeepSeek Harness installation from Python.

`dsh` makes every layer of the agent a swappable plugin, which is the first
time an agent's *parameters* -- its skills, prompt sections, tool set, subagent
presets -- have been addressable rather than buried in a fixed stack. Evolving
them is what this package does; this subpackage is the part that knows where
they live and how to reach them.

Today it holds :mod:`agentdescent.dsh.locate` (where a real installation keeps
its skills). The agent itself is :func:`agentdescent.backends.dsh`, which sits
beside ``openhands()`` and ``claude_code()`` because it is the same contract
they are, not a special case.

The plan is ``docs/design-dsh-plugin.md``.
"""

from __future__ import annotations

from .locate import (
    Skill,
    SkillRoot,
    agents_home,
    dsh_home,
    find_skill,
    list_skills,
    project_root,
    skill_roots,
)

__all__ = [
    "Skill",
    "SkillRoot",
    "agents_home",
    "dsh_home",
    "find_skill",
    "list_skills",
    "project_root",
    "skill_roots",
]
