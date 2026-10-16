"""The vendored surrogate for SWE-smith: one tiny repository, twelve injected bugs.

gskill's tasks come from SWE-smith: take a repository, inject a bug into a clean
checkout, and keep the instance if some tests flip from pass to fail. Each
instance has a problem statement, ``FAIL_TO_PASS`` tests (fail on the bugged
tree, pass on the clean one) and ``PASS_TO_PASS`` tests (the regression check).
That structure is kept here, offline:

* ``fixtures/kvtools/`` is the **clean** repository -- the thing the skill is
  *about*. Four house conventions hold across it and none is stated in code.
* ``fixtures/bugs.json`` is the SWE-smith part: per bug, the mutation (``old`` ->
  ``new``) that injects it, the problem statement the agent is shown, and its
  ``FAIL_TO_PASS`` test ids. ``PASS_TO_PASS`` is every other test.
* ``fixtures/tests/`` and ``fixtures/runner.py`` are the verifier. The agent
  never sees them.

The gold patch is the inverse of the mutation. It lives in ``bugs.json`` and is
used only by :func:`verify_fixtures` and the tests -- :class:`Instance.prompt`
(what the agent and the reflector read) is built from the bugged source and the
problem statement and nothing else.

Boundary, stated rather than hidden: this is **not** SWE-smith and there is no
Docker. Candidate code is executed in a throw-away directory under a plain
subprocess with a wall-clock timeout (and POSIX CPU / memory / file-size limits
applied by ``runner.py`` itself, before it imports any candidate code -- not via
``preexec_fn``, which is unsafe in a multi-threaded parent),
which is containment against a runaway loop and not a sandbox against hostile
code. Real gskill runs each rollout in a container for that reason.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parent / "fixtures"
REPO_PACKAGE = "kvtools"
TEST_TIMEOUT_S = 15.0

#: Verdicts, after the harness in ``gskill/swe_fitness_fn.py``: no patch, the
#: FAIL_TO_PASS tests still fail, a PASS_TO_PASS test regressed, or all passed.
NO_PATCH = "no_patch"
F2P_FAILED = "f2p_failed"
P2P_REGRESSION = "p2p_regression"
ALL_PASSED = "all_passed"


@dataclass(frozen=True)
class Instance:
    """One bug-fixing task: a SWE-smith instance in miniature."""

    id: str
    convention: str
    file: str
    old: str
    new: str
    issue: str
    fail_to_pass: Tuple[str, ...]

    def clean_source(self) -> str:
        return _read(self.file)

    def buggy_source(self) -> str:
        clean = self.clean_source()
        if clean.count(self.old) != 1:
            raise ValueError(f"{self.id}: mutation must match exactly once in {self.file}")
        return clean.replace(self.old, self.new)

    def gold_source(self) -> str:
        """The reference fix. Never goes into a prompt."""
        return self.clean_source()

    @property
    def prompt(self) -> str:
        """The problem statement and the bugged file -- what the agent is given."""
        return (f"Problem statement:\n{self.issue}\n\n"
                f"The bug is in `{self.file}`. Current contents:\n"
                f"```python\n{self.buggy_source()}```")


@dataclass(frozen=True)
class Outcome:
    """The verdict on one attempt, plus the evidence a reflector reads."""

    status: str
    f2p_failed: Tuple[str, ...]
    p2p_failed: Tuple[str, ...]
    details: Mapping[str, str]

    @property
    def passed(self) -> bool:
        return self.status == ALL_PASSED

    @property
    def score(self) -> float:
        """Binary, as ``create_swe_fitness_fn`` scores it: 1.0 or 0.0."""
        return 1.0 if self.passed else 0.0

    def report(self) -> str:
        lines = [f"verdict: {self.status}"]
        for key in self.f2p_failed + self.p2p_failed:
            kind = "FAIL_TO_PASS" if key in self.f2p_failed else "PASS_TO_PASS"
            lines.append(f"{kind} {key} -> {self.details.get(key, '')}")
        return "\n".join(lines)


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def load_instances() -> Tuple[Instance, ...]:
    rows = json.loads((ROOT / "bugs.json").read_text(encoding="utf-8"))
    return tuple(Instance(
        id=row["id"], convention=row["convention"], file=row["file"],
        old=row["old"], new=row["new"], issue=row["issue"],
        fail_to_pass=tuple(row["fail_to_pass"])) for row in rows)


def instance(instance_id: str) -> Instance:
    for item in load_instances():
        if item.id == instance_id:
            return item
    raise KeyError(instance_id)


def repo_files() -> Dict[str, str]:
    """Every file of the clean workspace (package, tests, runner), by path."""
    files: Dict[str, str] = {}
    for sub in (REPO_PACKAGE, "tests"):
        for path in sorted((ROOT / sub).rglob("*.py")):
            files[path.relative_to(ROOT).as_posix()] = path.read_text(encoding="utf-8")
    files["runner.py"] = _read("runner.py")
    return files


def run_tests(files: Mapping[str, str], *,
              timeout: float = TEST_TIMEOUT_S) -> Dict[str, str]:
    """Run the fixture tests on a workspace; test id -> failure text ('' = pass).

    A crash, a timeout, or unreadable output is reported as one pseudo-test,
    ``"<workspace>"``, whose failure text says why -- never as a pass.
    """
    with tempfile.TemporaryDirectory(prefix="gskill-") as tmp:
        for rel, text in files.items():
            target = Path(tmp, rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        env = {k: v for k, v in os.environ.items()
               if k in ("PATH", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP")}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            done = subprocess.run(
                [sys.executable, "-I", "runner.py", tmp], cwd=tmp, env=env,
                capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"<workspace>": f"timed out after {timeout:.0f}s"}
        try:
            parsed = json.loads(done.stdout.strip().splitlines()[-1])
            if not isinstance(parsed, dict):
                raise ValueError
            return {str(k): str(v) for k, v in parsed.items()}
        except (IndexError, ValueError):
            return {"<workspace>": f"test runner crashed: {done.stderr.strip()[-300:]}"}


def attempt_files(item: Instance, source: Optional[str]) -> Dict[str, str]:
    """The workspace after the agent's edit: the bugged tree, target replaced."""
    files = repo_files()
    files[item.file] = item.buggy_source() if source is None else source
    return files


def judge(item: Instance, source: Optional[str]) -> Outcome:
    """Score one attempt the way gskill's fitness function does.

    ``source`` is the agent's complete replacement for ``item.file``; ``None`` or
    a file identical to the bugged one is **no patch**. Otherwise the FAIL_TO_PASS
    tests must now pass and every other test must still pass (the regression
    check): any miss scores 0, all-pass scores 1.
    """
    if source is None or not source.strip() or source == item.buggy_source():
        return Outcome(NO_PATCH, (), (), {})
    results = run_tests(attempt_files(item, source))
    broken = {k: v for k, v in results.items() if v}
    f2p = set(item.fail_to_pass)
    # A pseudo-test or an import failure means the tree did not run at all.
    wholesale = any(k == "<workspace>" or k.endswith("::<import>") for k in broken)
    f2p_failed = tuple(sorted(f2p if wholesale else f2p & set(broken)))
    if f2p_failed:
        return Outcome(F2P_FAILED, f2p_failed, (), broken)
    p2p_failed = tuple(sorted(set(broken) - f2p))
    if p2p_failed:
        return Outcome(P2P_REGRESSION, (), p2p_failed, broken)
    # A test the runner never reported cannot count as a pass.
    if not f2p <= set(results):
        return Outcome(F2P_FAILED, tuple(sorted(f2p - set(results))), (), broken)
    return Outcome(ALL_PASSED, (), (), {})


def verify_fixtures() -> List[str]:
    """Self-check, for the tests: problems with the vendored fixtures, or ``[]``.

    Every instance's mutation must apply once; on the bugged tree exactly its
    declared FAIL_TO_PASS tests fail; on the clean tree nothing fails.
    """
    problems: List[str] = []
    clean = run_tests(repo_files())
    if any(clean.values()):
        problems.append(f"clean repo has failing tests: {sorted(k for k, v in clean.items() if v)}")
    for item in load_instances():
        try:
            buggy = item.buggy_source()
        except ValueError as error:
            problems.append(str(error))
            continue
        failing = sorted(k for k, v in run_tests(attempt_files(item, buggy)).items() if v)
        if failing != sorted(item.fail_to_pass):
            problems.append(f"{item.id}: failing {failing} != fail_to_pass "
                            f"{sorted(item.fail_to_pass)}")
    return problems
