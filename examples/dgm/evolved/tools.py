"""What the agent can do. Editing this file is how the agent improves."""
import pathlib
import subprocess
import sys


def read_source(task_dir):
    return (pathlib.Path(task_dir) / "lib.py").read_text()


def write_source(task_dir, text):
    (pathlib.Path(task_dir) / "lib.py").write_text(text)


def run_tests(task_dir, timeout=60):
    """Run the task's tests. Returns (passed, failed, output)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "test_lib.py"],
        cwd=str(task_dir), capture_output=True, text=True, timeout=timeout)
    out = proc.stdout + proc.stderr
    passed = out.count(" passed")
    failed = out.count(" failed")
    return passed, failed, out
