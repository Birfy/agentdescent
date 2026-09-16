"""Where a session runs, and what it can see from there.

The port's three fences all bound what a session's work *becomes*: an edit outside the
node is a request, a write to a frozen path is dropped, and the worktree is a throwaway
copy so nothing survives except through the diff. None of them bounds what a session can
*read*, and a session with a shell reads whatever the process can. Measured, in one fly
run: four of twelve episodes ran ``find /`` and opened a previous run's output from
``/tmp``, and one of those read the very ``_cli.py`` that answered the acceptance failure
it had been handed to reproduce. The blind property was a rule in a prompt.

The engine already owns the fix. :mod:`agentdescent.sandbox_container` is titled "A
sandbox that is actually a boundary" and says what it changes: only the workspace is
visible, the network is off unless the spec asks, no capabilities, no new privileges,
resource ceilings. This module uses it, and adds the one thing an *agent* session needs
that a candidate's test run does not -- **the toolchain that runs the agent**.

Three read-only mounts and one writable one, and each is there for a reason:

* the ``claude`` binary's own install, because the image has no agent in it;
* the proxy's CA bundle when there is one, because a session behind a TLS-inspecting
  proxy that cannot verify it spends its turns on certificate errors rather than on the
  objective (measured: a probe session lost two turns to `pip`'s
  ``CERTIFICATE_VERIFY_FAILED`` before working around it);
* nothing else of the host, which is the point;
* and a per-session CLI state directory, so the transcript outlives the container.
  Reading 52 of those transcripts is how the wall was found to be what ends an episode,
  and a boundary that also destroys the evidence is a bad trade.

**The network is on, and that is a real weakening.** ``SandboxSpec.network="inherit"``
is required: the session's whole job is to talk to a model endpoint. So this is not a
boundary against hostile code -- a session that can reach the network can send whatever
it read. It is a boundary against *contamination*, which is the failure that actually
happened, and against that it is complete: the filesystem the session sees contains the
work and nothing else.

**When there is no engine** the run says so once and falls back to a plain directory.
A fallback that pretended to isolate would be worse than none.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import List, Mapping, Optional, Sequence

from agentdescent.policies import SandboxSpec
from agentdescent.sandbox import LEASE_FILE
from agentdescent.sandbox_container import (CONTAINER_WORKDIR, ContainerProvider,
                                            detect_engine)

__all__ = ["CONTAINER_HOME", "PROVIDER_FILES", "LocalSandbox", "SessionSandbox",
           "Workspace", "sandbox_engine", "toolchain_root"]


#: What the provider itself writes into a workspace, and which is therefore not the
#: session's work. `.agentdescent-mount` is the file `_verify_mount` looks for from the
#: inside to tell a working bind mount from an empty one; without this the first diff of
#: every sandboxed episode reported it as an edit the node never made.
PROVIDER_FILES = frozenset({".agentdescent-mount", LEASE_FILE})


#: Where the CLI's own state goes inside the container. Not `/work`: `HOME` has to name
#: somewhere writable, and the CLI drops `.claude.json` beside it -- in the workspace
#: that is a file the diff would report as the session's work.
CONTAINER_HOME = "/clihome"

#: Passed through to the container by name, and nothing else is. The values are read on
#: the host, one at a time, the way `SandboxSpec.env_allowlist` does it: a variable that
#: is not on this list cannot arrive by accident, which is the same reason the host's
#: session identity is dropped in `_session.session_env`.
CONTAINER_ENV: Sequence[str] = (
    "ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "no_proxy",
    "MAX_THINKING_TOKENS",
)

#: Where a TLS-inspecting proxy keeps the certificate everything inside has to trust.
_CA_CANDIDATES = ("/root/.ccr/ca-bundle.crt", "/etc/ssl/certs/ca-certificates.crt")

#: Small, has a Python, and is a name rather than a build: an image the run has to build
#: first is an image that fails at the first episode on a machine with no build context.
DEFAULT_IMAGE = "python:3.12-slim"


def signed_in_only(env: Optional[Mapping[str, str]] = None) -> bool:
    """True when the CLI authenticates as a signed-in user rather than from a key.

    A key is a string in the environment and crosses into a container with it. A
    sign-in is not: the CLI reaches the endpoint through the host's session ingress,
    which the container does not have and which this module does not put there. So the
    two arrangements differ in whether an isolated session can authenticate at all, and
    a run needs to know which one it is *before* it spends episodes finding out.
    """
    out = os.environ if env is None else env
    if out.get("ANTHROPIC_API_KEY") or out.get("ANTHROPIC_AUTH_TOKEN"):
        return False
    return any(out.get(name) for name in
               ("CLAUDE_CODE_REMOTE", "CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST"))


def sandbox_engine(preferred: str = "") -> Optional[str]:
    """The container engine this machine can actually use, or ``None``.

    :func:`agentdescent.sandbox_container.detect_engine` already asks both halves of the
    question -- installed *and* answering -- which is the difference that matters here:
    `dockerd` sitting on the PATH with no daemon behind it took one investigation's
    first look and produced the wrong answer.
    """
    return detect_engine(preferred or None)


def toolchain_root(binary: str = "claude") -> Optional[str]:
    """The directory to mount so `binary` exists inside the container.

    A self-contained install puts the executable at ``<root>/bin/<name>``, and the rest
    of ``<root>`` is what it needs; anything else, mount the directory the executable is
    in. Returns ``None`` when the binary is somewhere shared like ``/usr/bin``, where
    mounting the parent would mean mounting the host's system directories -- the thing
    this module exists to avoid.
    """
    found = shutil.which(binary)
    if not found:
        return None
    real = os.path.realpath(found)
    holder = os.path.dirname(real)
    root = os.path.dirname(holder) if os.path.basename(holder) == "bin" else holder
    if root in ("", "/", "/usr", "/usr/local", "/bin", "/sbin", "/opt"):
        return None
    return root


class Workspace:
    """A directory to materialise into, and how to run a command against it.

    The same two questions whichever way the session runs: *where do I put the state and
    read the answer back from*, and *what argv actually runs*. A local workspace answers
    the second with the command itself; a container answers it with an ``exec`` into the
    boundary, and the directory is the bind mount on the host side.
    """

    def __init__(self, path: str, *, prefix: Sequence[str] = (),
                 env: Optional[Mapping[str, str]] = None,
                 close=None, kill=None, binary: str = "",
                 binary_names: Sequence[str] = ()):
        self.path = path
        self._prefix = list(prefix)
        self._env = dict(env or {})
        self._close = close
        self._kill = kill
        #: The session binary as it exists *under the mount*, and the spellings of it a
        #: caller might use. Only those are rewritten: a command is not always the agent
        #: -- the tests run `sh` through here -- and rewriting every argv[0] turns a
        #: shell probe into a confusing "Please run /login".
        self._binary = binary
        self._names = {n for n in binary_names if n}

    def command(self, argv: Sequence[str], env: Mapping[str, str]) -> List[str]:
        """The argv to run on *this* host for a session command."""
        argv = list(argv)
        if not self._prefix:
            return argv
        if self._binary and argv and argv[0] in self._names:
            argv[0] = self._binary       # the mounted path, not the host's PATH
        passed: List[str] = []
        for name in CONTAINER_ENV:
            value = env.get(name)
            if value is not None:
                passed += ["--env", f"{name}={value}"]
        for name, value in self._env.items():
            passed += ["--env", f"{name}={value}"]
        return self._prefix[:-1] + passed + [self._prefix[-1]] + argv

    def kill(self) -> None:
        """Stop whatever is still running inside. A no-op for a local workspace."""
        if self._kill is not None:
            try:
                self._kill()
            except Exception:  # noqa: BLE001 - killing is best effort by definition
                pass

    def close(self) -> None:
        if self._close is not None:
            self._close()
        else:
            shutil.rmtree(self.path, ignore_errors=True)


class LocalSandbox:
    """No boundary: a throwaway directory, which is where this port started.

    Kept as a class rather than a `None` check so the two paths have one shape, and so a
    run that is not isolated says so in the same place a run that is says the opposite.
    """

    available = False
    reason = "not asked for"

    def __init__(self, root: str = ""):
        self._root = root or None
        self.opened = 0

    def open(self, prefix: str = "genesis-") -> Workspace:
        self.opened += 1
        return Workspace(tempfile.mkdtemp(prefix=prefix, dir=self._root))

    def summary(self) -> str:
        return "off (a plain directory; a session can read this machine)"


class SessionSandbox:
    """Hands out workspaces: containers when there is an engine, directories otherwise.

    One of these per run. `available` is decided once at construction, because a run
    that cannot isolate should say so once rather than at every episode -- and because
    the answer cannot change halfway through without the run being a different
    experiment in the middle.
    """

    def __init__(self, *, engine: str = "", image: str = "", root: str = "",
                 binary: str = "claude", home: str = "",
                 memory_mb: int = 4096, cpu: float = 0.0):
        self.engine = sandbox_engine(engine)
        self.toolchain = toolchain_root(binary)
        # The *resolved* path, because that is the one under the mount: `claude` is a
        # symlink from a node install's `bin` here, and exec'ing the symlink's own path
        # inside the container is `stat: no such file or directory`.
        found = shutil.which(binary)
        self.binary = os.path.realpath(found) if found else binary
        #: Every spelling of the session binary a caller might hand us.
        self.binary_names = tuple({binary, found or binary, self.binary})
        self._image = image or DEFAULT_IMAGE
        self._root = root or None
        self._home = home
        self._memory_mb = memory_mb
        self._cpu = cpu
        self._ca = next((p for p in _CA_CANDIDATES if os.path.exists(p)), "")
        #: Why the run is not isolated, when it is not. Empty when it is.
        self.reason = ""
        if self.engine is None:
            self.reason = ("no container engine is answering (docker or podman); "
                           "sessions run in a plain directory and can read this machine")
        elif self.toolchain is None:
            self.reason = (f"the `{binary}` binary is not in a self-contained install, "
                           "so there is nothing to mount into a container")
        elif signed_in_only():
            # The third way a container cannot work, and the one that used to be
            # silent. A signed-in CLI does not authenticate from an environment key:
            # it goes through the host's session ingress, which `CONTAINER_ENV` does
            # not carry and is not this module's to carry. Inside a container the CLI
            # therefore answers `Not logged in`, and the run finds out one episode at
            # a time -- measured, `sessions=4 failed=4`, a whole run for a condition
            # that was knowable before the first one started.
            self.reason = ("this CLI is signed in rather than keyed, and a signed-in "
                           "session cannot authenticate from inside a container; "
                           "sessions run in a plain directory and can read this "
                           "machine. Give the run an API key for an isolated one")
        self.provider = None
        #: Containers left behind by a run that was killed, removed at construction.
        #: A session's container outlives its `release` when the process holding it
        #: dies -- a session container restart took one run with eight episodes in
        #: flight, and all eight were still up and idling afterwards. The engine
        #: already knows how to find them: they carry its label and a start time, and
        #: `reap` removes the ones past the TTL, which by construction are nobody's.
        #: (The host workspace directories they were mounted from are a separate
        #: lease, and `agentdescent.sandbox`'s own reaper is what clears those.)
        self.reaped = 0
        if not self.reason:
            self.provider = _Provider(self, engine=self.engine, image=self._image)
            try:
                self.reaped = self.provider.reap()
            except Exception:  # noqa: BLE001 - a run should not fail on housekeeping
                self.reaped = 0
        #: Containers handed out, for the run's own report.
        self.opened = 0

    @property
    def available(self) -> bool:
        return self.provider is not None

    def spec(self) -> SandboxSpec:
        # `network="inherit"` is the one place this weakens the engine's defaults, and
        # it is not optional: the session exists to reach a model endpoint. Everything
        # else stays as `run_command` writes it -- read-only root, no capabilities, no
        # new privileges, a pids ceiling, and the host uid so the diff is readable.
        return SandboxSpec(image=self._image, network="inherit",
                           workspace_root=self._root,
                           memory_mb=self._memory_mb or None,
                           cpu=self._cpu or None)

    def open(self, prefix: str = "genesis-") -> Workspace:
        """A workspace for one session."""
        if self.provider is None:
            return Workspace(tempfile.mkdtemp(prefix=prefix, dir=self._root))
        sandbox = self.provider.acquire(self.spec())
        self.opened += 1
        env = {"HOME": CONTAINER_HOME, "TMPDIR": "/tmp",
               "CLAUDE_CONFIG_DIR": CONTAINER_HOME}
        if self._ca:
            # Four names for one file because four different stacks look for it: node
            # for the CLI, and requests/openssl/pip for whatever the session installs.
            env.update({"NODE_EXTRA_CA_CERTS": self._ca, "REQUESTS_CA_BUNDLE": self._ca,
                        "SSL_CERT_FILE": self._ca, "PIP_CERT": self._ca})
        return Workspace(
            sandbox.root,
            prefix=[self.engine, "exec", "-w", CONTAINER_WORKDIR, sandbox.container_id],
            env=env, binary=self.binary, binary_names=self.binary_names,
            kill=sandbox.kill,
            close=lambda: self.provider.release(sandbox))

    def summary(self) -> str:
        if not self.available:
            return f"off ({self.reason})"
        reaped = f", {self.reaped} orphan(s) reaped" if self.reaped else ""
        return (f"{self.engine} {self._image}, {self.opened} container(s){reaped}; "
                f"only the workspace visible, network inherited")


class _Provider(ContainerProvider):
    """The engine's provider, plus the mounts an *agent* session needs.

    Everything the boundary is made of stays where it is: this adds volumes to the
    command the base class builds and changes nothing else. A session's own CLI state
    directory is created per container so eight of them do not write one `.claude.json`
    between them, and it lives under the run's state directory rather than inside the
    workspace, where it would show up in the diff as work.
    """

    def __init__(self, sandbox: SessionSandbox, **kwargs):
        super().__init__(**kwargs)
        self._owner = sandbox

    def run_command(self, spec: SandboxSpec, ws: str, lease_id: str) -> List[str]:
        command = super().run_command(spec, ws, lease_id)
        mounts: List[str] = []
        if spec.network == "inherit":
            # The base class reads `inherit` as "do not switch the network off", which
            # leaves the engine's default bridge. That is not the host's network, and
            # the difference is the whole thing here: the endpoint is reached through a
            # proxy on the host's loopback, and on a bridge `127.0.0.1` is the container.
            mounts += ["--network", "host"]
        if self._owner.toolchain:
            mounts += ["--volume", f"{self._owner.toolchain}:{self._owner.toolchain}:ro"]
        if self._owner._ca:
            mounts += ["--volume", f"{self._owner._ca}:{self._owner._ca}:ro"]
        home = os.path.join(self._owner._home or ws, f"cli-{lease_id[:8]}")
        os.makedirs(home, exist_ok=True)
        mounts += ["--volume", f"{home}:{CONTAINER_HOME}:rw"]
        # After `run` and its flags, before the image name and the idle command: the
        # base class puts the image last but one, so inserting at the front of the
        # flag block is the only placement that does not depend on its tail.
        return command[:2] + mounts + command[2:]
