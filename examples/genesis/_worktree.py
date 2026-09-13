"""Worktree isolation, a commit per episode, and the worktree deleted after.

Upstream this is not an implementation detail, it is the scheduling model
(``agents/manager.ex``):

    When you spawn a subagent, you must yield: commit your changes, release your
    worktree, and wait. The subagent gets its own worktree. Once it completes, you
    are re-queued. This is why "commit before delegating" is not just a rule -- it's
    a fundamental requirement of the worktree scheduling model. If you don't commit,
    your changes are invisible to subagents.

and the architect's Phase 1 spells out the order: create the child directory and its
``CONTEXT.md`` with ``make_dir`` (**auto-commits**), *then* spawn the subagent there.

What this class makes real, and what it does not. The episode tree in this port is a
pure function over a ``state`` dict, so isolation is not what keeps two siblings from
corrupting each other -- they already branch from the same base by construction, and
a test holds that. What a real repository adds is threefold, and none of it is
ceremony:

* **The phylogenetic graph becomes git history.** Every episode commits, and a parent
  that merged three children leaves a three-parent merge commit. ``git log --graph``
  over a rollout is the tree the paper draws, and ``phylo_graph_node.ex``'s
  ``find_merge_base/2`` has something to find.
* **"Commit before delegating" becomes checkable.** A child's base is a SHA, so
  "the parent's uncommitted work is invisible to its children" is a property with a
  witness rather than an assertion about a dict.
* **The lifecycle is accounted.** Every worktree created is removed -- ``git worktree
  remove --force``, which is the deletion the scheduling model requires -- and the
  counters say so. A leak is a bug with a number attached.

Costs about four git invocations per episode and is off unless ``--worktrees`` is
given.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Mapping, Optional, Sequence

from agentdescent.filetree import materialize

from ._delegation import apply_edits as edited

__all__ = ["Rollout", "WorktreeLedger", "edited", "git_worktrees_available"]


def _git(*args: str, cwd: str) -> str:
    out = subprocess.run(("git",) + args, cwd=cwd, check=True,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out.stdout.decode("utf-8", "replace").strip()


def git_worktrees_available() -> bool:
    """``git worktree`` exists and works here. Checked once, cheaply."""
    try:
        probe = tempfile.mkdtemp(prefix="genesis-probe-")
        try:
            _git("init", "-q", probe, cwd=os.path.dirname(probe) or ".")
            _git("worktree", "list", cwd=probe)
            return True
        finally:
            shutil.rmtree(probe, ignore_errors=True)
    except Exception:  # noqa: BLE001 - no git, or a git too old
        return False


class WorktreeLedger:
    """What the lifecycle did, in numbers. Shared by every rollout of a run."""

    def __init__(self) -> None:
        self.created = 0
        self.removed = 0
        self.commits = 0
        self.merge_commits = 0
        self.max_parents = 0
        self.failures = 0

    @property
    def leaked(self) -> int:
        return self.created - self.removed

    def summary(self) -> str:
        return (f"worktrees={self.created} created/{self.removed} removed  "
                f"commits={self.commits} (+{self.merge_commits} merges, "
                f"max parents {self.max_parents})"
                + (f"  LEAKED={self.leaked}" if self.leaked else "")
                + (f"  git_failures={self.failures}" if self.failures else ""))


class Rollout:
    """One repository for one rollout: a commit per episode, a worktree per commit.

    Used as a context manager so the repository is gone when the rollout is, which
    is the whole point of a transient agent's workspace.
    """

    def __init__(self, ledger: WorktreeLedger, *, root: Optional[str] = None,
                 branch_prefix: str = "genesis") -> None:
        self._ledger = ledger
        self._branch = branch_prefix
        self._repo = tempfile.mkdtemp(prefix="genesis-world-", dir=root)
        self._trees = os.path.join(self._repo, ".worktrees")
        _git("init", "-q", "-b", "main", self._repo, cwd=os.path.dirname(self._repo))
        _git("config", "user.email", "genesis@example.invalid", cwd=self._repo)
        _git("config", "user.name", "Genesis", cwd=self._repo)
        # A root commit over the empty tree: `git worktree add` needs a commit to
        # start from, and the first episode of a formation run starts from nothing.
        self._empty = _git("commit-tree", _git("mktree", cwd=self._repo),
                           "-m", "empty world", cwd=self._repo)
        _git("update-ref", "refs/heads/main", self._empty, cwd=self._repo)

    # -- the lifecycle -----------------------------------------------------

    def commit(self, agent_id: str, state: Mapping[str, str], message: str,
               parents: Sequence[str] = ()) -> str:
        """Stage ``state`` in a worktree of this agent's own, commit it, remove it.

        The three steps the scheduling model names, in order, every time: a worktree
        to work in, a commit so the work is visible to anyone branching from it, and
        the worktree released so the next agent can be scheduled.
        """
        path = self._add(agent_id, parents[0] if parents else None)
        try:
            for name in os.listdir(path):
                if name != ".git":
                    target = os.path.join(path, name)
                    shutil.rmtree(target) if os.path.isdir(target) else os.remove(target)
            materialize(state, path)
            _git("add", "-A", cwd=path)
            tree = _git("write-tree", cwd=path)
            args = ["commit-tree", tree]
            for parent in parents:
                args += ["-p", parent]
            sha = _git(*args, "-m", message, cwd=path)
            # A branch per agent, as upstream commits to `evogit-agent-*`: without a
            # ref the commit is unreachable and the graph it belongs to is not there.
            _git("update-ref", f"refs/heads/{self._branch}/{agent_id}", sha,
                 cwd=self._repo)
            self._ledger.commits += 1
            if len(parents) > 1:
                self._ledger.merge_commits += 1
            self._ledger.max_parents = max(self._ledger.max_parents, len(parents))
            return sha
        finally:
            self._release(path)

    def history(self) -> List[str]:
        """``sha parents subject`` per commit, newest first -- the graph, as text."""
        try:
            out = _git("log", "--all", "--format=%h %p %s", cwd=self._repo)
        except Exception:  # noqa: BLE001 - nothing committed yet
            return []
        return [line for line in out.splitlines() if line.strip()]

    def close(self) -> None:
        shutil.rmtree(self._repo, ignore_errors=True)

    def __enter__(self) -> "Rollout":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _add(self, agent_id: str, base: Optional[str]) -> str:
        """``git worktree add`` off ``base``, or off an empty tree for the first one."""
        path = os.path.join(self._trees, agent_id)
        _git("worktree", "add", "--detach", "-f", path, base or self._empty,
             cwd=self._repo)
        self._ledger.created += 1
        return path

    def _release(self, path: str) -> None:
        try:
            _git("worktree", "remove", "--force", path, cwd=self._repo)
        except Exception:  # noqa: BLE001 - fall back to the filesystem
            shutil.rmtree(path, ignore_errors=True)
            try:
                _git("worktree", "prune", cwd=self._repo)
            except Exception:  # noqa: BLE001
                self._ledger.failures += 1
        self._ledger.removed += 1
