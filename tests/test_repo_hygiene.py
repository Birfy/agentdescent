"""What the repository must not track."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args):
    if shutil.which("git") is None or not os.path.isdir(os.path.join(ROOT, ".git")):
        pytest.skip("not a git checkout")
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def test_no_installed_dependencies_are_tracked():
    """``integrations/dsh-agentdescent/node_modules`` (210 files, pnpm symlinks
    included) was committed by accident (#165). The package's ``files`` list never
    shipped it and nothing reads it: the tests build their own tmp ``node_modules``."""
    done = _git("ls-files", "-z")
    assert done.returncode == 0, done.stderr
    tracked = [p for p in done.stdout.split("\0")
               if p and ("node_modules/" in p or p.startswith("node_modules"))]
    assert tracked == [], f"{len(tracked)} tracked file(s) under node_modules, e.g. {tracked[:3]}"


def test_node_modules_is_ignored_wherever_it_appears():
    # Paths that do not exist on disk: a pnpm symlink under the leftover
    # on-disk ``node_modules`` makes ``git check-ignore`` refuse with
    # "beyond a symbolic link", which is not what we are testing.
    for path in ("node_modules/a/index.js",
                 "integrations/dsh-agentdescent/node_modules/fake-pkg/package.json"):
        done = _git("check-ignore", "-q", path)
        assert done.returncode == 0, f"{path} is not ignored"
