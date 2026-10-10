"""Local test helpers must win over unrelated installed packages."""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_repository_test_and_script_packages_win_over_installed_packages(tmp_path):
    fake_site = tmp_path / "site-packages"
    for name in ("tests", "scripts"):
        conflicting = fake_site / name
        conflicting.mkdir(parents=True)
        (conflicting / "__init__.py").write_text(
            f"# simulated installed {name} package\n"
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(fake_site), str(ROOT / "tests")))
    code = (
        "import tests.test_evolvespec, scripts.audit_workloads, faults, audit_ppi_cases; "
        "import tests, scripts; print(tests.__file__); print(scripts.__file__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert str(ROOT / "tests" / "__init__.py") in result.stdout
    assert str(ROOT / "scripts" / "__init__.py") in result.stdout
