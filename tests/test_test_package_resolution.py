"""Detached test workers must import this repo's helpers ahead of installed packages."""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_local_tests_package_wins_over_an_installed_tests_package(tmp_path):
    fake_site = tmp_path / "site-packages"
    conflicting = fake_site / "tests"
    conflicting.mkdir(parents=True)
    (conflicting / "__init__.py").write_text("# simulated installed package\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_site)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import tests.test_evolvespec; import tests; print(tests.__file__)",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert str(ROOT / "tests" / "__init__.py") in result.stdout
