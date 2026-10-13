"""The setup script must stop before it can touch any host after pip fails."""

import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "setup-hosts.sh"


@pytest.mark.parametrize("from_checkout", [True, False], ids=["checkout", "pypi"])
def test_failed_python_install_exits_before_host_configuration(tmp_path, from_checkout):
    workdir = tmp_path / "work"
    bindir = tmp_path / "bin"
    home = tmp_path / "home"
    config = home / ".config" / "claude"
    workdir.mkdir()
    bindir.mkdir()
    config.mkdir(parents=True)
    sentinel = config / "settings.json"
    sentinel.write_text('{"keep": true}\n', encoding="utf-8")
    if from_checkout:
        (workdir / "pyproject.toml").write_text(
            '[project]\nname = "agentdescent"\n', encoding="utf-8"
        )

    log = tmp_path / "stub.log"
    python = bindir / "python3"
    python.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = --version ]; then echo 'Python 3.11.0'; exit 0; fi\n"
        "if [ \"$1\" = -c ]; then echo 1; exit 0; fi\n"
        f"printf 'python3 %s\\n' \"$*\" >> '{log}'\n"
        "echo 'ERROR: simulated pip dependency resolution failure' >&2\n"
        "exit 23\n",
        encoding="utf-8",
    )
    python.chmod(0o755)

    # The host stubs make any attempted configuration observable. Their only
    # read-only invocation is `--version`, which the installer uses to report
    # what it found before attempting the package install.
    for name in ("claude", "agentdescent"):
        stub = bindir / name
        stub.write_text(
            "#!/bin/sh\n"
            f"printf '{name} %s\\n' \"$*\" >> '{log}'\n"
            "if [ \"$1\" = --version ]; then echo 'stub version'; exit 0; fi\n"
            "exit 0\n",
            encoding="utf-8",
        )
        stub.chmod(0o755)

    env = os.environ.copy()
    env.update({
        "HOME": str(home),
        "PATH": f"{bindir}:/usr/bin:/bin",
        "XDG_CONFIG_HOME": str(home / ".config"),
    })
    result = subprocess.run(
        ["bash", str(SCRIPT)], cwd=workdir, env=env,
        text=True, capture_output=True, timeout=30,
    )

    assert result.returncode != 0, result.stdout + result.stderr
    output = result.stdout + result.stderr
    assert "simulated pip dependency resolution failure" in output
    assert "could not be installed" in output
    assert "No host configuration was changed" in output
    assert "pip install" in output
    assert sentinel.read_text(encoding="utf-8") == '{"keep": true}\n'
    calls = log.read_text(encoding="utf-8")
    assert "python3 -m pip install" in calls
    assert "agentdescent install" not in calls
    assert "agentdescent doctor" not in calls
