"""Git-backed versioned Ledger (design doc, sections 3.1 and 4.5).

The Ledger is the "parameter server" of the framework, with one crucial extra
property that a classic parameter server lacks: a *commit history*.  That extra
semantics is exactly what makes rebase / staleness handling possible (design
doc, section 1, precision boundary #1).

Concretely:

* Each artifact is stored as a JSON blob under ``artifacts/<id>.json`` in a real
  git repository.
* A per-artifact integer version is kept in ``versions.json`` and bumped on every
  commit that touches the artifact.  The collection of these integers is the
  *version vector* that diffs are stamped against.
* Two branches exist: ``dev`` (fast branch, absorbs everything that passes the
  statistical test) and ``stable`` (slow branch, an EMA-style confirmation of
  dev -- design doc, section 4.5).
* Commits use compare-and-swap: a writer declares the base version it read, and
  the commit is rejected if the head has moved past it.  Multi-artifact
  contract-breaking diffs commit atomically (2PC-style: stage all, then a single
  git commit).

Because AgentDescent's reference implementation runs the workers in-process, the
git operations are serialized through a single lock; the CAS check is what makes
the *logical* concurrency safe.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from .evolvable import Diff, Evolvable, VersionVector


class CASConflict(Exception):
    """Raised when a commit's declared base version is stale."""


class ContractRejected(Exception):
    """Raised when the ledger refuses a diff bound to a superseded major."""


# A serializer turns an Evolvable into a JSON-friendly dict; the deserializer
# reconstructs it.  The ledger is otherwise domain-agnostic.
Serializer = Callable[[Evolvable], dict]
Deserializer = Callable[[str, int, dict], Evolvable]


@dataclass
class Snapshot:
    """An immutable view of one branch at one point in time."""

    artifacts: Dict[str, Evolvable]
    version: VersionVector

    def get(self, artifact_id: str) -> Optional[Evolvable]:
        return self.artifacts.get(artifact_id)


class GitError(RuntimeError):
    """A git command failed; the message carries git's own stderr."""


#: What a ledger read or write can fail with -- a third category alongside
#: ``ContractError`` (a caller bug: propagate) and a backend failure (absorb and
#: report). The ledger is *infrastructure*: a held ``index.lock``, a full
#: ``$TMPDIR``, a killed ``git``. Its failure ends the run but must still leave a
#: partial result behind, so the drivers catch this tuple rather than letting it
#: escape (which used to discard a completed run -- even when the failing call was
#: only fetching the cosmetic ``ledger_log``).
LedgerFailure = (GitError, OSError, json.JSONDecodeError)


# Config the ledger imposes on every invocation, overriding whatever the user has
# in ~/.gitconfig. These commits are the ledger's bookkeeping, not the user's, and
# a scratch repo in $TMPDIR has no business honouring personal git preferences:
# `commit.gpgsign = true` -- a common setting -- failed the genesis commit, so
# `evolve()` raised before running a single task, from a call that mentions git
# nowhere. `core.hooksPath` would likewise run the user's pre-commit hook against
# a temp directory it knows nothing about.
_GIT_OVERRIDES = (
    "-c", "commit.gpgsign=false",
    "-c", "tag.gpgsign=false",
    "-c", "core.hooksPath=",
    "-c", "gc.auto=0",
)
# 120s is enormous for the operations here (all local, all tiny) and exists only
# so a wedged git -- an NFS stall, a held index.lock -- surfaces as a GitError
# instead of hanging the process. Every call holds the ledger lock, so one
# blocked git would otherwise stall every worker.
_GIT_TIMEOUT = 120.0


def _git(repo: str, *args: str) -> str:
    """Run a git command, surfacing *why* it failed.

    ``capture_output=True`` swallows stderr, so a failure used to read only
    "returned non-zero exit status 128" -- the actual cause (a missing repo, an
    index lock held by another process) was discarded.

    The environment is isolated (see ``_GIT_OVERRIDES``): system and user config
    must not decide whether the ledger can write, and ``GIT_TERMINAL_PROMPT=0``
    keeps a credential prompt from wedging a run that has no terminal.
    """
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0"}
    try:
        out = subprocess.run(
            ["git", "-C", repo, *_GIT_OVERRIDES, *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            env=env,
        )
    except FileNotFoundError as e:
        # Otherwise a machine without git fails with a bare OSError traceback,
        # while every other missing-executable path in the package names it.
        raise GitError(
            "'git' is not installed or not on PATH; the Ledger stores artifacts "
            "in a git repository, so it is required") from e
    except subprocess.TimeoutExpired as e:
        raise GitError(
            f"git {' '.join(args)} in {repo} exceeded {_GIT_TIMEOUT}s and was "
            "abandoned (a held index.lock or a stalled filesystem)") from e
    if out.returncode != 0:
        detail = (out.stderr or out.stdout or "").strip() or "no output"
        raise GitError(f"git {' '.join(args)} failed in {repo}: {detail}")
    return out.stdout.strip()


class Ledger:
    """A git-backed, version-vectored artifact store with dual branches."""

    STABLE = "stable"
    DEV = "dev"

    def __init__(
        self,
        repo_path: str,
        serialize: Serializer,
        deserialize: Deserializer,
        author: str = "agentdescent <bot@agentdescent.local>",
    ) -> None:
        self.repo_path = repo_path
        self._serialize = serialize
        self._deserialize = deserialize
        self._author = author
        self._lock = threading.RLock()
        self._current_branch: Optional[str] = None   # see _checkout()
        self._closed = False
        self._init_repo()

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Refuse further use of this ledger. Idempotent.

        A driver that owns a throwaway repo removes it when it returns, but a
        round abandoned on ``round_timeout`` leaves worker threads running that
        Python cannot cancel -- and one of those, finishing later, would commit
        into the deleted directory and *recreate* it (``_dump_artifact`` calls
        ``makedirs(exist_ok=True)``). Closing first turns that late write into an
        ordinary error inside the straggler's own thread, where the driver already
        absorbs it."""
        with self._lock:
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise GitError(
                f"this Ledger has been closed and its repository at "
                f"{self.repo_path} removed; a rollout abandoned by round_timeout "
                "finished after the run returned")

    # -- repository bootstrap -------------------------------------------------

    def _init_repo(self) -> None:
        os.makedirs(self.repo_path, exist_ok=True)
        if not os.path.isdir(os.path.join(self.repo_path, ".git")):
            _git(self.repo_path, "init", "-q")
            _git(self.repo_path, "config", "user.name", "agentdescent")
            _git(self.repo_path, "config", "user.email", "bot@agentdescent.local")
            os.makedirs(os.path.join(self.repo_path, "artifacts"), exist_ok=True)
            self._write_versions({})
            _git(self.repo_path, "add", "-A")
            _git(self.repo_path, "commit", "-q", "-m", "genesis")
            # create dev branch off the genesis commit
            _git(self.repo_path, "branch", "-f", self.DEV)
            _git(self.repo_path, "branch", "-m", self.STABLE)

    # -- low-level version bookkeeping ---------------------------------------

    def _versions_path(self) -> str:
        return os.path.join(self.repo_path, "versions.json")

    def _read_versions(self) -> VersionVector:
        try:
            with open(self._versions_path()) as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def _write_versions(self, vv: VersionVector) -> None:
        with open(self._versions_path(), "w") as f:
            json.dump(vv, f, indent=2, sort_keys=True)

    def _artifact_path(self, artifact_id: str) -> str:
        return os.path.join(self.repo_path, "artifacts", f"{artifact_id}.json")

    def _checkout(self, branch: str) -> None:
        """Switch branches, skipping the subprocess when already there.

        Every read (`snapshot`, `head_version`) used to fork a ``git checkout``
        even when the branch was already current -- ~19 ms each, serialised on
        ``self._lock``, which capped the whole pipeline at ~50 ledger ops/sec no
        matter how many workers were running. All callers hold the lock, so this
        instance is the only writer of the working tree and can track it.

        The cached branch assumes this ``Ledger`` owns ``repo_path``; two
        instances sharing one path already race over the working tree (use one
        ``Ledger`` per repo).
        """
        if self._current_branch == branch:
            return
        _git(self.repo_path, "checkout", "-q", branch)
        self._current_branch = branch

    # -- public API -----------------------------------------------------------

    def register(self, artifact: Evolvable, branch: str = DEV) -> None:
        """Add a brand-new artifact at version 1 on both branches."""
        with self._lock:
            self._ensure_open()
            for br in (self.STABLE, self.DEV):
                self._checkout(br)
                vv = self._read_versions()
                if artifact.id not in vv:
                    self._dump_artifact(artifact, version=1)
                    vv[artifact.id] = 1
                    self._write_versions(vv)
                    self._commit(f"register {artifact.id}")

    def _dump_artifact(self, artifact: Evolvable, version: int) -> None:
        payload = {"version": version, "state": self._serialize(artifact)}
        # switching branches can drop an empty (untracked) artifacts/ dir.
        os.makedirs(os.path.join(self.repo_path, "artifacts"), exist_ok=True)
        with open(self._artifact_path(artifact.id), "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _commit(self, message: str) -> str:
        _git(self.repo_path, "add", "-A")
        # allow empty so idempotent registrations do not explode
        _git(self.repo_path, "commit", "-q", "--allow-empty", "-m", message)
        return _git(self.repo_path, "rev-parse", "HEAD")

    def snapshot(self, branch: str = DEV) -> Snapshot:
        """Materialize every artifact on ``branch`` into live Evolvables."""
        with self._lock:
            self._ensure_open()
            self._checkout(branch)
            vv = self._read_versions()
            artifacts: Dict[str, Evolvable] = {}
            for artifact_id, version in vv.items():
                with open(self._artifact_path(artifact_id)) as f:
                    payload = json.load(f)
                artifacts[artifact_id] = self._deserialize(
                    artifact_id, payload["version"], payload["state"]
                )
            return Snapshot(artifacts=artifacts, version=dict(vv))

    def head_version(self, branch: str = DEV) -> VersionVector:
        with self._lock:
            self._ensure_open()
            self._checkout(branch)
            return self._read_versions()

    def commit(
        self,
        new_state: Evolvable,
        base_version: VersionVector,
        branch: str = DEV,
        message: str = "",
    ) -> Tuple[str, int]:
        """Compare-and-swap commit of a single artifact.

        ``base_version`` is the version the writer read for this artifact.  If
        the head has advanced past it, :class:`CASConflict` is raised and the
        caller must rebase (design doc, section 4.2).

        The vector **must** mention this artifact: defaulting a missing entry to
        the current head would make the check unfalsifiable, so a writer that
        declared no base would always win -- precisely the lost update CAS
        exists to prevent."""
        with self._lock:
            self._ensure_open()
            self._checkout(branch)
            vv = self._read_versions()
            aid = new_state.id
            head = vv.get(aid, 0)
            if aid not in base_version:
                raise CASConflict(
                    f"{aid}: base_version must declare the version this write was "
                    f"based on (got {dict(base_version)!r}); head={head}")
            expected = base_version[aid]
            if head != expected:
                raise CASConflict(
                    f"{aid}: head={head} but diff based on {expected}"
                )
            new_version = head + 1
            self._dump_artifact(new_state, new_version)
            vv[aid] = new_version
            self._write_versions(vv)
            commit_hash = self._commit(message or f"update {aid} -> v{new_version}")
            return commit_hash, new_version

    def commit_atomic(
        self,
        new_states: List[Evolvable],
        base_version: VersionVector,
        branch: str = DEV,
        message: str = "",
    ) -> Tuple[str, VersionVector]:
        """Two-phase, all-or-nothing commit of several artifacts.

        Used for contract-breaking diffs that must land together with their
        adapter diffs (design doc, section 6, "atomic adaptation transaction").
        Phase 1 validates every CAS precondition; phase 2 writes and commits in
        a single git commit."""
        with self._lock:
            self._ensure_open()
            self._checkout(branch)
            vv = self._read_versions()
            # phase 1: validate all (every id must declare its base -- see commit())
            for st in new_states:
                head = vv.get(st.id, 0)
                if st.id not in base_version:
                    raise CASConflict(
                        f"{st.id}: base_version must declare the version this write "
                        f"was based on (got {dict(base_version)!r}); head={head}")
                expected = base_version[st.id]
                if head != expected:
                    raise CASConflict(
                        f"{st.id}: head={head} but diff based on {expected}"
                    )
            # phase 2: write all, then one commit
            for st in new_states:
                nv = vv.get(st.id, 0) + 1
                self._dump_artifact(st, nv)
                vv[st.id] = nv
            self._write_versions(vv)
            ids = ",".join(s.id for s in new_states)
            commit_hash = self._commit(message or f"atomic update {ids}")
            return commit_hash, dict(vv)

    def promote_to_stable(self, artifact_id: str) -> Optional[int]:
        """EMA-style confirmation: copy dev's current artifact onto stable.

        Called by the aggregator's dual-branch logic after an artifact has
        survived K rounds on dev without a regression report (design doc,
        section 4.5)."""
        with self._lock:
            self._ensure_open()
            self._checkout(self.DEV)
            dev_vv = self._read_versions()
            if artifact_id not in dev_vv:
                return None
            with open(self._artifact_path(artifact_id)) as f:
                payload = json.load(f)
            dev_state = self._deserialize(
                artifact_id, payload["version"], payload["state"]
            )
            self._checkout(self.STABLE)
            svv = self._read_versions()
            nv = svv.get(artifact_id, 0) + 1
            self._dump_artifact(dev_state, nv)
            svv[artifact_id] = nv
            self._write_versions(svv)
            self._commit(f"promote {artifact_id} to stable v{nv}")
            return nv

    def log(self, branch: str = DEV, limit: int = 20) -> List[str]:
        with self._lock:
            self._ensure_open()
            self._checkout(branch)
            out = _git(
                self.repo_path, "log", f"-{limit}", "--pretty=format:%h %s"
            )
            return out.splitlines() if out else []
