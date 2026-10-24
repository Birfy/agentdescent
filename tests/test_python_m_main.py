"""``python -m agentdescent`` runs the CLI (#510)."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-m", "agentdescent", *args],
                          capture_output=True, text=True, env=env, timeout=120)


def test_python_m_agentdescent_help():
    r = _run("--help")
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("usage: agentdescent")


def test_python_m_agentdescent_propagates_exit_code():
    r = _run("definitely-not-a-command")
    assert r.returncode == 2
    assert "usage: agentdescent" in r.stderr
