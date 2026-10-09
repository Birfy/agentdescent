"""Guard the built-wheel smoke wrapper's isolation and session timeout."""

import asyncio
import importlib.util
import os
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def smoke_script():
    script = ROOT / "scripts" / "smoke-installed-wheel.py"
    spec = importlib.util.spec_from_file_location("smoke_installed_wheel", script)
    assert spec and spec.loader
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    return smoke


def test_wheel_smoke_removes_inherited_python_overrides(tmp_path):
    """Stub Python executables so the test cannot install or touch host config."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python3 = fake_bin / "python3"
    fake_python3.write_text(
        """#!/bin/sh
set -eu
[ "$1" = "-m" ] && [ "$2" = "venv" ]
mkdir -p "$3/bin"
cat > "$3/bin/python" <<'PYTHON_STUB'
#!/bin/sh
set -eu
if [ "${PYTHONPATH+x}" ] || [ "${PYTHONHOME+x}" ]; then
  echo "Python override leaked into smoke venv command" >&2
  exit 1
fi
printf '%s\\n' "$*" >> "$SMOKE_CALL_LOG"
PYTHON_STUB
chmod +x "$3/bin/python"
""",
        encoding="utf-8",
    )
    fake_python3.chmod(fake_python3.stat().st_mode | stat.S_IXUSR)
    wheel = tmp_path / "agentdescent-test.whl"
    wheel.write_bytes(b"stub wheel; pip is replaced by the fake venv python")
    call_log = tmp_path / "calls.txt"

    env = os.environ.copy()
    env.update({
        "PATH": str(fake_bin) + os.pathsep + env.get("PATH", ""),
        "PYTHONPATH": "inherited-path-must-not-leak",
        "PYTHONHOME": "inherited-home-must-not-leak",
        "SMOKE_CALL_LOG": str(call_log),
    })
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "smoke-wheel.sh"), str(wheel)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2, calls
    assert calls[0].startswith("-m pip install ")
    assert calls[1].endswith("scripts/smoke-installed-wheel.py")


def test_wheel_smoke_mcp_protocol_check_times_out(smoke_script):
    """A stuck initialize/list/call sequence cannot hang the CI job."""
    async def hang_forever():
        await asyncio.Event().wait()

    with pytest.raises(TimeoutError, match="MCP stdio smoke session exceeded"):
        asyncio.run(smoke_script.run_mcp_check(hang_forever, timeout=0.01))


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (SimpleNamespace(isError=False), False),
        (SimpleNamespace(isError=True), True),
        (SimpleNamespace(is_error=False), False),
        (SimpleNamespace(is_error=True), True),
    ],
)
def test_wheel_smoke_reads_both_supported_mcp_error_field_names(
    smoke_script, result, expected
):
    """MCP v1 uses `isError`; v2 uses `is_error`."""
    assert smoke_script.mcp_tool_result_is_error(result) is expected


def test_wheel_smoke_rejects_unknown_mcp_tool_result_shape(smoke_script):
    with pytest.raises(AssertionError, match="neither `is_error` nor `isError`"):
        smoke_script.mcp_tool_result_is_error(SimpleNamespace())
