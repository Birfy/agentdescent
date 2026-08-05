"""A container sandbox is mostly its command line, so that is what is asserted.

Two halves, deliberately separated:

* **construction** -- which flags, which defaults, which spec field maps where.
  This is nearly all of the logic and it needs no container engine, so it runs
  everywhere and it runs exhaustively.
* **behaviour** -- does the candidate actually lose sight of the host. That needs
  a real engine and is skipped without one.

Only the second half can prove isolation, so the first half is written to make
the second half's failures obvious rather than to stand in for it.
"""

import os
import shutil
import subprocess
import tempfile

import pytest

from agentdescent.policies import SandboxSpec
from agentdescent.sandbox_container import (
    CONTAINER_WORKDIR, ContainerProvider, ContainerSandbox, engine_available,
)

IMAGE = os.environ.get("AGENTDESCENT_TEST_IMAGE", "docker.io/library/python:3.11-slim")

#: Every engine actually present. The behaviour tests run against each of them
#: rather than against "whichever was found first": the two CLIs are compatible
#: in the flags used here, and the only way that claim stays true is if both are
#: exercised. Machines with neither skip the lot.
LIVE_ENGINES = [e for e in ("docker", "podman") if engine_available(e)]
ENGINE = LIVE_ENGINES[0] if LIVE_ENGINES else None
needs_engine = pytest.mark.skipif(
    not LIVE_ENGINES, reason="no container engine with a live daemon")
every_engine = pytest.mark.parametrize("engine", LIVE_ENGINES or [None])


@pytest.fixture
def shared_root(tmp_path_factory):
    """A workspace root the engine's VM can actually see.

    On macOS the engine runs in a VM that shares only part of the host, and the
    system temporary directory -- where `tmp_path` lives -- is usually outside
    it. Home is shared by every common setup, so the behaviour tests stage there
    rather than exercising the VM's mount configuration by accident."""
    root = os.path.join(os.path.expanduser("~"), ".agentdescent-tests")
    os.makedirs(root, exist_ok=True)
    d = tempfile.mkdtemp(prefix="ws-", dir=root)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(autouse=True, scope="module")
def _neutral_engine_config(tmp_path_factory):
    """Give the engine CLI a config directory with no credential helper in it.

    Not a workaround for the code under test. A developer machine often carries a
    `credsStore` for a helper that is not installed -- a removed Docker Desktop
    leaves one behind -- and the engine then fails to pull *public* images that
    need no credentials at all. The sandbox has no opinion about registry auth and
    the tests should not inherit one.

    The endpoint is carried over explicitly, because contexts live inside
    `DOCKER_CONFIG`: replacing that directory silently loses the daemon along
    with the credential helper, and the symptom is "cannot connect to the docker
    API", which points nowhere near the cause.
    """
    endpoint = _current_endpoint()
    cfg = tmp_path_factory.mktemp("engine-config")
    saved = {k: os.environ.get(k) for k in ("DOCKER_CONFIG", "DOCKER_HOST")}
    os.environ["DOCKER_CONFIG"] = str(cfg)
    if endpoint:
        os.environ["DOCKER_HOST"] = endpoint
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _current_endpoint():
    """Where docker is listening right now, asked before the config is swapped."""
    if "docker" not in LIVE_ENGINES:
        return None
    if os.environ.get("DOCKER_HOST"):
        return os.environ["DOCKER_HOST"]
    try:
        out = subprocess.run(
            ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
            capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def provider(**kw):
    """A provider whose commands are captured instead of run."""
    seen = []

    def runner(cmd, timeout=None):
        seen.append(list(cmd))
        return "deadbeef"

    p = ContainerProvider(IMAGE, runner=runner, **kw)
    p.seen = seen                       # type: ignore[attr-defined]
    return p


def command_for(spec=None, **kw):
    kw.setdefault("rootless", False)      # command construction, not this machine
    p = provider(**kw)
    return p.run_command(spec or SandboxSpec(), "/host/ws", "lease-1")


def flag_value(cmd, flag):
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


# ---------------------------------------------------------------------------
# Defaults are closed
# ---------------------------------------------------------------------------


def test_a_spec_that_asks_for_nothing_gets_everything_locked_down():
    """Closed by default and opened per field, not the other way round.

    A spec is written by whoever is evolving something, and the failure mode of
    an open default is silent: the candidate gets network or capabilities and
    nothing in the run says so."""
    cmd = command_for()
    assert ["--network", "none"] == cmd[cmd.index("--network"):cmd.index("--network") + 2]
    assert "--read-only" in cmd
    assert ["--cap-drop", "ALL"] == cmd[cmd.index("--cap-drop"):cmd.index("--cap-drop") + 2]
    # The `:true` form is what docker documents; podman accepts it too, and one
    # spelling that both take beats branching on the engine.
    assert flag_value(cmd, "--security-opt") == "no-new-privileges:true"
    assert any(a.startswith("--pids-limit") for a in cmd), "a fork bomb is cheap"


def test_the_workspace_is_bind_mounted_and_is_the_working_directory():
    """Staging still happens on the host, so `materialize` and the frozen-file
    overlay need no changes at all -- the container is where work *runs*, not
    where it is laid out."""
    cmd = command_for()
    assert flag_value(cmd, "--volume") == f"/host/ws:{CONTAINER_WORKDIR}:rw"
    assert flag_value(cmd, "--workdir") == CONTAINER_WORKDIR


@pytest.mark.skipif(os.name == "nt", reason="uid/gid are POSIX")
def test_a_rootful_engine_runs_as_the_host_user():
    """Root inside the container writes root-owned files into the bind mount,
    and then the host cannot even delete its own workspace."""
    assert flag_value(command_for(), "--user") == f"{os.getuid()}:{os.getgid()}"


@pytest.mark.skipif(os.name == "nt", reason="uid/gid are POSIX")
def test_a_rootless_engine_gets_no_user_flag():
    """It already maps the host user to root inside. Naming a uid there maps it
    to a *subuid*, which cannot read files owned by the host user -- so the bind
    mount reads as empty, indistinguishable from a mount that never happened.

    Found on CI: every podman case failed and every docker case passed, because
    the runner's podman is rootless and its docker is not."""
    assert "--user" not in command_for(rootless=True)


def test_the_lease_is_a_label_so_orphans_can_be_found_later():
    cmd = command_for()
    assert "agentdescent.lease=lease-1" in cmd
    p = provider()
    p.reap()
    ps = next(c for c in p.seen if "ps" in c)
    assert any("label=agentdescent.lease" in a for a in ps), (
        "reap must look for the same label acquire writes")


# ---------------------------------------------------------------------------
# Each field opens exactly one thing
# ---------------------------------------------------------------------------


def test_network_is_off_until_the_spec_says_inherit():
    assert "--network" in command_for(SandboxSpec(network="none"))
    cmd = command_for(SandboxSpec(network="inherit"))
    assert "--network" not in cmd, "only `inherit` may connect the container"


def test_limits_appear_only_when_asked_for():
    """An unset field must produce no flag, not a default value: a default
    memory ceiling nobody chose is a run that dies for a reason nobody wrote."""
    bare = command_for()
    assert "--memory" not in bare and "--cpus" not in bare

    cmd = command_for(SandboxSpec(memory_mb=512, cpu=1.5))
    assert flag_value(cmd, "--memory") == "512m"
    assert flag_value(cmd, "--cpus") == "1.5"


def test_the_image_can_be_overridden_per_spec():
    assert command_for()[-3] == IMAGE
    assert command_for(SandboxSpec(image="alpine:3.19"))[-3] == "alpine:3.19"


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


def test_only_allowlisted_variables_cross(monkeypatch):
    monkeypatch.setenv("AD_ALLOWED", "yes")
    monkeypatch.setenv("AD_SECRET", "sk-do-not-copy-me")
    cmd = command_for(SandboxSpec(env_allowlist=("AD_ALLOWED",)))
    assert "AD_ALLOWED=yes" in cmd
    joined = " ".join(cmd)
    assert "AD_SECRET" not in joined and "sk-do-not-copy-me" not in joined


def test_nothing_bulk_is_used_to_pass_the_environment(monkeypatch):
    """`--env-file`, or `--env NAME` without a value, would let a variable arrive
    without anyone having named it. The allowlist is only an allowlist if it is
    the sole route in."""
    monkeypatch.setenv("AD_SECRET", "sk-nope")
    cmd = command_for(SandboxSpec(env_allowlist=("AD_SECRET",)))
    assert "--env-file" not in cmd
    for i, arg in enumerate(cmd):
        if arg == "--env":
            assert "=" in cmd[i + 1], "a valueless --env copies from the host"


def test_an_allowlisted_variable_the_host_does_not_have_is_skipped(monkeypatch):
    monkeypatch.delenv("AD_MISSING", raising=False)
    cmd = command_for(SandboxSpec(env_allowlist=("AD_MISSING",)))
    assert not any(a.startswith("AD_MISSING") for a in cmd)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_a_failed_start_does_not_leave_a_workspace_behind(tmp_path):
    def boom(cmd, timeout=None):
        raise RuntimeError("no such image")

    p = ContainerProvider(IMAGE, runner=boom)
    before = set(os.listdir(tmp_path))
    with pytest.raises(RuntimeError):
        p.acquire(SandboxSpec(workspace_root=str(tmp_path)))
    assert set(os.listdir(tmp_path)) == before, "a failed acquire kept its directory"


def test_release_never_raises(tmp_path):
    """It runs from a `finally`, often on a path that is already failing. An
    exception here replaces the real error with a cleanup error."""
    def boom(cmd, timeout=None):
        raise RuntimeError("engine went away")

    p = ContainerProvider(IMAGE, runner=boom)
    ws = tmp_path / "ws"
    ws.mkdir()
    p.release(ContainerSandbox(id="s", root=str(ws), fingerprint="f",
                               lease_id="l", container_id="c"))
    assert not ws.exists(), "the workspace survived a failing release"


def test_the_exec_prefix_lands_in_the_container_with_a_usable_environment():
    """`HOME` and `TMPDIR` are set to paths that exist *inside* the container.

    Inheriting the host's would name a directory that is not mounted -- and the
    host's own values are needed by the engine CLI in the prefix, which runs
    outside. The two environments are not the same environment."""
    sb = ContainerSandbox(id="s", root="/host/ws", fingerprint="f", lease_id="l",
                          container_id="abc123", engine="podman")
    prefix = sb.exec_prefix()
    assert prefix[:4] == ["podman", "exec", "-w", CONTAINER_WORKDIR]
    assert prefix[-1] == "abc123"
    assert f"HOME={CONTAINER_WORKDIR}" in prefix and "TMPDIR=/tmp" in prefix


def test_a_timeout_kills_the_container_not_the_exec():
    """Killing an `exec` leaves the process running inside. The container is
    per-lease and disposable, so killing it is both simpler and complete."""
    p = provider()
    p.kill(ContainerSandbox(id="s", root="/w", fingerprint="f", lease_id="l",
                            container_id="abc123"))
    assert ["docker", "kill", "abc123"] in p.seen


def test_the_provider_satisfies_the_protocol_without_extra_methods():
    """The check on #60's contract. A second implementation needing a fourth
    method would mean the protocol was shaped around its only user."""
    from agentdescent.policies import SandboxProvider
    assert isinstance(ContainerProvider(IMAGE, runner=lambda c, timeout=None: "x"), SandboxProvider)


# ---------------------------------------------------------------------------
# Behaviour -- only a real engine can answer these
# ---------------------------------------------------------------------------


@needs_engine
@every_engine
def test_the_host_home_directory_is_not_visible(engine, shared_root):
    """The reason this provider exists.

    The local provider fails this: `HOME` is redirected but absolute paths are
    not, so candidate code reading `~/.ssh/id_rsa` by full path gets it."""
    p = ContainerProvider(IMAGE, engine=engine)
    sb = p.acquire(SandboxSpec(workspace_root=shared_root))
    try:
        probe = f"import os; print(os.path.exists({os.path.expanduser('~')!r}))"
        out = subprocess.run([*sb.exec_prefix(), "python", "-c", probe],
                             capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "False", (
            "the host home directory was visible inside the container")
    finally:
        p.release(sb)


@needs_engine
@every_engine
def test_the_local_provider_does_not_stop_what_this_one_does(engine, shared_root):
    """The contrast, run side by side, because it is the whole justification.

    `HOME` redirection changes where a *lookup* goes; it does not change what a
    *path* reaches. Candidate code that names the host's home directory outright
    gets it under the local provider and does not under this one."""
    from agentdescent.sandbox import WorkspaceProvider

    home = os.path.expanduser("~")
    probe = f"import os; print(os.path.exists({home!r}))"

    local = WorkspaceProvider()
    sb = local.acquire(SandboxSpec(workspace_root=shared_root))
    try:
        env = dict(os.environ, HOME=sb.root, TMPDIR=sb.root)
        out = subprocess.run(["python3", "-c", probe], cwd=sb.root, env=env,
                             capture_output=True, text=True, timeout=60)
        assert out.stdout.strip() == "True", (
            "the premise has changed: the local workspace now hides the host "
            "home directory, and this provider needs a new justification")
    finally:
        local.release(sb)

    contained = ContainerProvider(IMAGE, engine=engine)
    csb = contained.acquire(SandboxSpec(workspace_root=shared_root))
    try:
        out = subprocess.run([*csb.exec_prefix(), "python", "-c", probe],
                             capture_output=True, text=True, timeout=120)
        assert out.stdout.strip() == "False"
    finally:
        contained.release(csb)


@needs_engine
@every_engine
def test_the_network_is_off_by_default_and_on_when_asked(engine, shared_root):
    """Asserted on the network namespace, not by dialling out.

    An earlier version connected to 1.1.1.1. That made the test depend on the
    machine having working egress, so it went red when this laptop's DNS dropped
    and would go red in any CI without outbound access -- reporting "isolation is
    broken" when the isolation was fine and the internet was not.

    `--network none` leaves a container with loopback and nothing else. That is
    the property being claimed, and it is observable from inside without a single
    packet leaving."""
    p = ContainerProvider(IMAGE, engine=engine)
    probe = ("import socket;"
             "print(sorted(i[1] for i in socket.if_nameindex()))")

    closed = p.acquire(SandboxSpec(workspace_root=shared_root))
    try:
        out = subprocess.run([*closed.exec_prefix(), "python", "-c", probe],
                             capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "['lo']", (
            f"a container with --network none had more than loopback: {out.stdout}")
    finally:
        p.release(closed)

    opened = p.acquire(SandboxSpec(workspace_root=shared_root, network="inherit"))
    try:
        out = subprocess.run([*opened.exec_prefix(), "python", "-c", probe],
                             capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() != "['lo']", (
            "network=inherit gave the container no interface but loopback")
    finally:
        p.release(opened)


@needs_engine
@every_engine
def test_files_written_inside_belong_to_the_host_user(engine, shared_root):
    p = ContainerProvider(IMAGE, engine=engine)
    sb = p.acquire(SandboxSpec(workspace_root=shared_root))
    try:
        subprocess.run([*sb.exec_prefix(), "python", "-c",
                "open('made-inside.txt','w').write('hi')"],
               capture_output=True, text=True, timeout=120, check=True)
        made = os.path.join(sb.root, "made-inside.txt")
        assert os.path.exists(made), "the bind mount did not reach the host"
        assert os.stat(made).st_uid == os.getuid(), (
            "a root-owned file the host cannot delete")
    finally:
        p.release(sb)


@needs_engine
@every_engine
def test_the_root_filesystem_is_read_only(engine, shared_root):
    p = ContainerProvider(IMAGE, engine=engine)
    sb = p.acquire(SandboxSpec(workspace_root=shared_root))
    try:
        out = subprocess.run([*sb.exec_prefix(), "python", "-c",
                      "open('/etc/passwd','a').write('x')"],
                     capture_output=True, text=True, timeout=120)
        assert out.returncode != 0, "the container root was writable"
        out = subprocess.run([*sb.exec_prefix(), "python", "-c",
                      "open('/tmp/ok','w').write('x'); print('wrote')"],
                     capture_output=True, text=True, timeout=120)
        assert "wrote" in out.stdout, "/tmp must stay writable for real toolchains"
    finally:
        p.release(sb)


@needs_engine
@every_engine
def test_reap_collects_orphans_and_leaves_the_living(engine, shared_root):
    p = ContainerProvider(IMAGE, engine=engine, ttl=3600.0)
    alive = p.acquire(SandboxSpec(workspace_root=shared_root))
    try:
        assert p.reap() == 0, "reap removed a container that had just started"
        p.ttl = 0.0                     # everything is now older than the TTL
        assert p.reap() >= 1
    finally:
        p.release(alive)


# ---------------------------------------------------------------------------
# End to end: a real rollout, inside a container
# ---------------------------------------------------------------------------


@needs_engine
@every_engine
def test_a_rollout_runs_inside_the_container(engine, shared_root):
    """The whole path: pool -> lease -> stage on the host -> execute inside.

    Staging is deliberately still a host operation, so `materialize` and the
    frozen-file overlay are untouched; only execution moves."""
    from agentdescent.evolution import Task
    from agentdescent.filetree import canonical
    from agentdescent.runners import code_runner
    from agentdescent.sandbox import SandboxPool

    pool = SandboxPool(ContainerProvider(IMAGE, engine=engine), max_sandboxes=2)
    run = code_runner(["python", "main.py"], sandbox_pool=pool,
                      workspace_root=shared_root)
    try:
        tree = canonical({"main.py": "import sys; print(sys.argv[1].upper())"})
        assert run(tree, Task("t", "hello")) == "HELLO"
    finally:
        pool.close(strict=True)


@needs_engine
@every_engine
def test_a_candidate_cannot_read_the_host_from_inside_a_rollout(engine, shared_root):
    """The same guarantee, reached through the runner rather than the provider --
    because that is the path a real run takes."""
    from agentdescent.evolution import Task
    from agentdescent.filetree import canonical
    from agentdescent.runners import code_runner
    from agentdescent.sandbox import SandboxPool

    home = os.path.expanduser("~")
    pool = SandboxPool(ContainerProvider(IMAGE, engine=engine), max_sandboxes=1)
    run = code_runner(["python", "main.py"], sandbox_pool=pool,
                      workspace_root=shared_root)
    try:
        tree = canonical({"main.py": f"import os; print(os.path.exists({home!r}))"})
        assert run(tree, Task("t", "x")) == "False"
    finally:
        pool.close(strict=True)


@needs_engine
@every_engine
def test_the_frozen_test_gate_still_runs_and_still_gates(engine, shared_root):
    """`frozen` is the reason a candidate cannot weaken its own tests, and it has
    to keep working when the tests run in a container."""
    from agentdescent.evolution import Task
    from agentdescent.filetree import canonical
    from agentdescent.runners import TEST_FAILURE_MARKER, code_runner
    from agentdescent.sandbox import SandboxPool

    pool = SandboxPool(ContainerProvider(IMAGE, engine=engine), max_sandboxes=1)
    run = code_runner(["python", "main.py"], test_cmd=["python", "check.py"],
                      overlay={"check.py": "raise SystemExit(1)"},
                      sandbox_pool=pool, workspace_root=shared_root)
    try:
        # the candidate ships a permissive check.py; the overlay restores the real one
        tree = canonical({"main.py": "print('x')", "check.py": "raise SystemExit(0)"})
        assert run(tree, Task("t", "x")).startswith(TEST_FAILURE_MARKER)
    finally:
        pool.close(strict=True)


@needs_engine
@every_engine
def test_an_unmounted_workspace_says_so_instead_of_looking_like_a_missing_file(
        engine, tmp_path):
    """The macOS trap, as a test.

    A workspace the VM cannot see produces an empty `/work`, so the first
    symptom is the candidate failing to open its own files -- which reads as the
    candidate's bug. Only reproducible where the engine is in a VM, so it accepts
    either outcome and asserts the *message* when it does fail."""
    p = ContainerProvider(IMAGE, engine=engine)
    try:
        sb = p.acquire(SandboxSpec(workspace_root=str(tmp_path)))
    except Exception as e:
        assert "did not appear inside the container" in str(e)
        assert "workspace_root" in str(e), "the error must say what to do"
        return
    p.release(sb)          # this host shares the whole filesystem; nothing to test


# ---------------------------------------------------------------------------
# The whole chain, against a real engine
# ---------------------------------------------------------------------------


@needs_engine
def test_candidate_code_is_judged_by_real_tests_inside_a_container():
    """The execution plane, end to end: materialise a candidate, run its frozen
    test suite **inside the container**, then run the entrypoint there.

    Every piece of this was unit-tested and the chain never was -- the ten live
    tests above were `skipif`-ed on every machine that has run this suite, so
    `sandbox_container.py` shipped without once being exercised against an
    engine. What that hid is not in any one component: it is that
    `runners._sh` has to notice `sandbox.exec_prefix()`, the workspace has to
    appear at `CONTAINER_WORKDIR`, and a missing tool has to come back as a
    scored failure rather than an exception.

    The workspace goes under `$HOME` rather than `$TMPDIR` on purpose. On macOS
    and Windows the engine runs in a VM that shares only part of the host
    filesystem, and the system temp directory is usually outside it -- the
    provider says so, with the fix, and this is the fix.
    """
    import os
    import tempfile

    from agentdescent.evolution import Task
    from agentdescent.filetree import canonical, load_tree
    from agentdescent.runners import TEST_FAILURE_MARKER, code_runner
    from agentdescent.sandbox import SandboxPool

    source = tempfile.mkdtemp(dir=os.path.expanduser("~"))
    workspaces = tempfile.mkdtemp(dir=os.path.expanduser("~"))
    with open(os.path.join(source, "solver.py"), "w") as fh:
        fh.write("import sys\n\n"
                 "def solve(expr):\n"
                 "    return sum(int(x) for x in expr.split('+'))\n\n"
                 "if __name__ == '__main__':\n"
                 "    print(solve(sys.argv[1]))\n")
    rendered = canonical(load_tree(source))

    pool = SandboxPool(ContainerProvider("python:3.11-slim"), max_sandboxes=1)
    try:
        # No pytest in the base image, so the gate is a plain interpreter check --
        # the point is that it runs *there*, not what runs it.
        run = code_runner(
            ["python", "solver.py"], layout="root", name="solver",
            test_cmd=["python", "-c", "from solver import solve; assert solve('2+3') == 5"],
            sandbox_pool=pool, workspace_root=workspaces)
        out = run(rendered, Task(id="t0", prompt="2+3", meta={"gold": "5"}))
        assert out == "5", f"the entrypoint did not run in the container: {out!r}"

        # ...and a gate that fails comes back in-band, scored, not raised: "you
        # broke the tests" is a learning signal, not a crashed run.
        broken = code_runner(
            ["python", "solver.py"], layout="root", name="solver",
            test_cmd=["python", "-c", "raise SystemExit(1)"],
            sandbox_pool=pool, workspace_root=workspaces)
        assert broken(rendered, Task(id="t1", prompt="2+3")).startswith(
            TEST_FAILURE_MARKER)
    finally:
        pool.close()
        shutil.rmtree(source, ignore_errors=True)
        shutil.rmtree(workspaces, ignore_errors=True)
