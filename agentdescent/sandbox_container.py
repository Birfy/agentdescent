"""A sandbox that is actually a boundary.

`sandbox.py` manages *lifetime*: how many workspaces exist, who owns one, who
cleans up when an owner dies. It does not manage *isolation*, and its only
provider does not provide any: `HOME` and `TMPDIR` point inside the workspace,
which redirects a lookup but not a path. Model-authored code that opens
`/Users/you/.ssh/id_rsa` gets it, and can post it anywhere, because nothing about
a trimmed environment restricts the filesystem or the network.

This provider runs the candidate inside a container instead. What changes:

    filesystem   only the workspace is visible; the host's is not
    network      off unless the spec asks for it
    privileges   no capabilities, no new privileges, not root
    resources    memory and CPU ceilings on every platform, not just POSIX

What does not change: containers share the host kernel, and escapes exist. This
is much stronger than a trimmed environment and it is **not** a licence to run
deliberately hostile code.

**Why the CLI rather than a Python SDK.** The core of this package has no
dependencies, and an SDK would make container support a Python dependency
instead of a machine one. The CLI also gets `podman` for free -- its interface is
docker-compatible -- and the only output that has to be parsed is a container id
and one column of `ps`.

**Why the command runner is injectable.** Nearly all of the logic here *is* the
command: which flags, in what order, with what defaults. That can be asserted
exactly, on a machine with no container engine at all, which is the difference
between "this provider was tested" and "this provider was run once". The parts
that need a real engine are tested separately and skipped when there is none.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .policies import SandboxSpec

__all__ = [
    "CONTAINER_WORKDIR",
    "ENGINES",
    "ContainerProvider",
    "ContainerSandbox",
    "EngineError",
    "detect_engine",
    "engine_available",
]

#: Where the workspace is mounted inside the container. Fixed rather than
#: configurable: it appears in every exec, and a moving target would have to be
#: threaded through the runner for no benefit.
CONTAINER_WORKDIR = "/work"

_LABEL = "agentdescent.lease"
_WS_PREFIX = "agentdescent-ws-"


class EngineError(RuntimeError):
    """The container engine failed, with its own stderr attached."""


#: Engines tried, in order, when none is named. Docker first because it is what
#: most machines and CI images have; podman is a drop-in with the same CLI.
ENGINES: Tuple[str, ...] = ("docker", "podman")


def detect_engine(preferred: Optional[str] = None) -> Optional[str]:
    """The first engine that is installed *and* answering, or ``None``.

    Both halves matter. `docker` is on the PATH of plenty of machines where
    nothing is listening -- Docker Desktop quit, a rootless daemon never
    started -- and a provider built against a binary that cannot talk to a
    daemon fails at the first rollout rather than at construction, by which time
    a run is underway.
    """
    for engine in ((preferred,) if preferred else ENGINES):
        if engine and engine_available(engine):
            return engine
    return None


def engine_available(engine: str = "docker") -> bool:
    """Is this engine installed *and* talking to a daemon?

    Both halves matter: the binary exists on plenty of machines where nothing is
    listening, and a provider that constructs perfect commands for a daemon that
    is not there fails at the first rollout rather than at startup."""
    if shutil.which(engine) is None:
        return False
    try:
        # `info` with no template on purpose. The two engines answer the same
        # questions with different Go structures -- `{{.Host.Arch}}` is podman's
        # and renders empty on docker -- so a probe that formats anything is a
        # probe that reports whichever engine it was written against. The exit
        # code means the same thing on both.
        out = subprocess.run([engine, "info"], capture_output=True, timeout=30)
        return out.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@dataclass
class ContainerSandbox:
    """A running container, plus the host directory bind-mounted into it."""

    id: str
    root: str                  # host path -- staging still happens here
    fingerprint: str
    lease_id: str
    container_id: str
    engine: str = "docker"
    acquired_at: float = 0.0

    #: Set by the provider that made it, so a timeout can stop the container
    #: without the runner needing to know which provider it came from.
    _killer: Optional[Callable[[], None]] = None

    def exec_prefix(self) -> List[str]:
        """What to put in front of a command so it runs inside this sandbox.

        The candidate's environment is set here rather than inherited: `HOME` and
        `TMPDIR` have to name paths that exist *inside* the container, and the
        host's values name paths that do not."""
        return [self.engine, "exec", "-w", CONTAINER_WORKDIR,
                "-e", f"HOME={CONTAINER_WORKDIR}", "-e", "TMPDIR=/tmp",
                self.container_id]

    def kill(self) -> None:
        """Stop everything inside. Called on timeout, where killing the local
        process group would only have killed the `exec`."""
        if self._killer is not None:
            self._killer()


def _run(cmd: Sequence[str], timeout: float = 120.0) -> str:
    proc = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise EngineError(
            f"{' '.join(cmd[:3])} failed ({proc.returncode}): "
            f"{(proc.stderr or proc.stdout or '').strip()[:400]}")
    return proc.stdout.strip()


class ContainerProvider:
    """Provisions containers as sandboxes, one per lease.

    Implements the same three methods as `WorkspaceProvider` and nothing more.
    If this class had needed a fourth, the protocol in `policies.py` would have
    been the wrong shape -- a contract with one implementation is a guess, and
    this is the check on it.
    """

    def __init__(self, image: str, *, engine: Optional[str] = None,
                 ttl: float = 300.0, pids_limit: int = 512,
                 runner: Optional[Callable[..., str]] = None) -> None:
        if not image:
            raise ValueError("a container sandbox needs an image to run")
        self.image = image
        # `None` means "whichever is here". Naming one is for the case where both
        # are installed and the choice matters; being made to name one when only
        # one exists is friction with no payoff.
        if engine is None and runner is None:
            engine = detect_engine()
            if engine is None:
                raise EngineError(
                    "no container engine is available: tried "
                    f"{', '.join(ENGINES)}. Install one, or start the daemon -- "
                    "the binary being on PATH is not enough. Without one, the "
                    "default local provider still works, but it is not a "
                    "boundary (see docs/directory-evolution.md).")
        self.engine = engine or "docker"
        self.ttl = ttl
        self.pids_limit = pids_limit
        self._run = runner or _run
        self._n = 0

    # -- command construction -------------------------------------------------

    def run_command(self, spec: SandboxSpec, ws: str, lease_id: str) -> List[str]:
        """The full ``run`` argv for one sandbox. Public because it is the logic.

        Defaults are closed and opened per field, rather than open and closed per
        field: a spec that says nothing gets a container with no network, no
        capabilities, a read-only root and the host's own uid.
        """
        cmd = [self.engine, "run", "--detach", "--init",
               "--label", f"{_LABEL}={lease_id}",
               "--label", f"agentdescent.fingerprint={spec.fingerprint()}",
               # Our own clock, not the engine's. `docker inspect` reports
               # RFC3339 and `podman inspect` reports Go's default layout, and a
               # reaper that misparses a timestamp either spares an orphan
               # forever or deletes a container that started a second ago.
               "--label", f"agentdescent.started={int(time.time())}",
               "--volume", f"{ws}:{CONTAINER_WORKDIR}:rw",
               "--workdir", CONTAINER_WORKDIR,
               # Read-only root with writable /work and /tmp: the candidate can
               # write where its work belongs and nowhere else.
               "--read-only",
               "--tmpfs", "/tmp:rw,exec,size=256m",
               "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges:true",
               f"--pids-limit={self.pids_limit}"]

        # Network is off unless asked for. A candidate that can reach the network
        # can send anything it managed to read, so "off" is the only safe default
        # -- and `inherit` is the single word that turns it on.
        if spec.network != "inherit":
            # `None` means "provider decides", and this provider decides no.
            cmd += ["--network", "none"]

        # Run as the host user, so files written into the bind mount belong to
        # whoever started the run. As root, the host cannot even delete them.
        if os.name != "nt":
            cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]

        if spec.memory_mb:
            cmd += ["--memory", f"{int(spec.memory_mb)}m"]
        if spec.cpu:
            cmd += ["--cpus", str(spec.cpu)]

        # Names only ever leave the host; values are read here and passed one at a
        # time. Nothing bulk (`--env-file`, `--env` without a value) is used, so a
        # variable that is not on the list cannot arrive by accident.
        for name in spec.env_allowlist:
            value = os.environ.get(name)
            if value is not None:
                cmd += ["--env", f"{name}={value}"]

        image = spec.image or self.image
        # Idle forever; the work arrives via `exec`. `--init` above reaps whatever
        # the candidate leaves behind.
        cmd += [image, "sleep", "infinity"]
        return cmd

    # -- SandboxProvider ------------------------------------------------------

    def acquire(self, spec: SandboxSpec) -> ContainerSandbox:
        ws = tempfile.mkdtemp(prefix=_WS_PREFIX, dir=spec.workspace_root)
        # A file to look for from the inside. Staging has not happened yet, so
        # without one the workspace and a failed mount are both empty.
        with open(os.path.join(ws, ".agentdescent-mount"), "w") as fh:
            fh.write("ok")
        lease_id = uuid.uuid4().hex
        try:
            container_id = self._run(self.run_command(spec, ws, lease_id))
        except Exception:
            shutil.rmtree(ws, ignore_errors=True)
            raise
        self._verify_mount(container_id, ws)
        self._n += 1
        sandbox = ContainerSandbox(
            id=f"{self.engine}-{os.getpid()}-{self._n}", root=ws,
            fingerprint=spec.fingerprint(), lease_id=lease_id,
            container_id=container_id, engine=self.engine,
            acquired_at=time.time())
        sandbox._killer = lambda: self.kill(sandbox)
        return sandbox

    def _verify_mount(self, container_id: str, ws: str) -> None:
        """Fail now, with an explanation, if the workspace did not arrive.

        On Linux any host path can be bind-mounted. On macOS and Windows the
        engine runs in a VM that shares only some of the host, so a workspace
        under `$TMPDIR` -- which is where one goes by default -- is frequently
        outside it. The container starts happily and `/work` is simply empty, so
        the first symptom is the candidate's own `FileNotFoundError`, blamed on
        the candidate.
        """
        try:
            self._run([self.engine, "exec", container_id,
                       "test", "-f", f"{CONTAINER_WORKDIR}/.agentdescent-mount"],
                      timeout=30.0)
            return
        except Exception:
            pass
        try:
            self._run([self.engine, "rm", "--force", container_id], timeout=60.0)
        except Exception:
            pass
        raise EngineError(
            f"the workspace {ws} did not appear inside the container. On macOS "
            "and Windows the engine runs in a VM that shares only part of the "
            "host filesystem, and the system temporary directory is usually "
            "outside it. Point the run at a shared location -- "
            "`SandboxSpec(workspace_root=...)` under your home directory is the "
            "usual answer -- or add the path to the VM's mounts "
            "(`colima start --mount <path>:w`, or Docker Desktop's File Sharing).")

    def release(self, sandbox: ContainerSandbox) -> None:
        """Stop the container and remove the workspace. Never raises.

        Release runs from a `finally`, including on the path where the rollout
        already failed. An exception here would replace the real error with a
        cleanup error."""
        try:
            self._run([self.engine, "rm", "--force", "--volumes",
                       sandbox.container_id], timeout=60.0)
        except Exception:
            pass
        shutil.rmtree(sandbox.root, ignore_errors=True)

    def reset(self, sandbox: ContainerSandbox) -> None:
        """Empty the workspace for reuse. The container itself is stateless
        enough to keep -- its root is read-only and `/tmp` is a tmpfs."""
        for name in os.listdir(sandbox.root):
            path = os.path.join(sandbox.root, name)
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                except OSError:
                    pass

    def renew(self, sandbox: ContainerSandbox) -> None:
        """No-op: a container's liveness is the engine's to track, not a file's.

        `WorkspaceProvider` needs a renewed lease file because a bare directory
        cannot say whether anyone still wants it. A container can: it is either
        running or it is not, and `reap` asks the engine."""

    def kill(self, sandbox: ContainerSandbox) -> None:
        """Stop everything inside, now. Used on timeout.

        Killing an `exec` does not kill what it started -- the process keeps
        running inside the container. Since a container is per-lease and
        disposable, killing the container is both simpler and more complete than
        hunting a process group inside it."""
        try:
            self._run([self.engine, "kill", sandbox.container_id], timeout=30.0)
        except Exception:
            pass

    def reap(self, older_than: Optional[float] = None) -> int:
        """Remove containers whose owner is gone, found by label.

        The same idea as the workspace lease file, asked of the engine instead:
        anything carrying our label and older than the TTL, whose lease nobody
        renewed, is an orphan of a killed run."""
        ttl = self.ttl if older_than is None else older_than
        try:
            raw = self._run([self.engine, "ps", "--all", "--filter",
                             f"label={_LABEL}", "--format",
                             "{{.ID}}\t{{.CreatedAt}}"], timeout=60.0)
        except Exception:
            return 0
        removed = 0
        for line in (raw or "").splitlines():
            cid = line.split("\t")[0].strip()
            if not cid:
                continue
            try:
                age = self._age(cid)
                if age is None or age <= ttl:
                    # An age we cannot read is not evidence of abandonment.
                    # Leaking a container costs disk; deleting a live one costs
                    # somebody's run, so the unreadable case declines to act.
                    continue
                self._run([self.engine, "rm", "--force", "--volumes", cid],
                          timeout=60.0)
                removed += 1
            except Exception:
                continue
        return removed

    def _age(self, container_id: str) -> Optional[float]:
        """Seconds since we started it, or ``None`` when that cannot be read."""
        raw = self._run([self.engine, "inspect", "--format",
                         "{{index .Config.Labels \"agentdescent.started\"}}",
                         container_id], timeout=30.0)
        try:
            return max(0.0, time.time() - float((raw or "").strip().strip('"')))
        except (TypeError, ValueError):
            return None
