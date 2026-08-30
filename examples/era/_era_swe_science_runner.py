"""The release's own grader, with a test selection added, run inside the verifier.

Executed **inside the task's verifier container** as
``python /era/runner.py /era/config.json``. Standard library only: the verifier
image carries the task's scientific stack and nothing of this repository, and a
runner that needed an import from here could not be bind-mounted into it.

Why it exists at all. The release ships one grader per task,
``tasks/task_NNN/tests/grader.py``, and *all 119 are byte-identical once the
task id is normalised* -- ``tests/test_era_swe_science.py`` pins that against
the published bundles. It runs the public reproduction and then the whole
private suite, and reports one binary reward. The ERA tree needs two things it
does not offer:

* a **graded** score, because a flat 0/1 gives PUCT nothing to rank on until
  something already works, and
* a **selection**, because the search must be scored on tests it is allowed to
  see while others are held back for the report.

So this is that grader with a node-id selection added and the two counts kept
apart. Everything else -- the working directory, ``PYTHONPATH``, the
``SCI_BENCH_*`` variables, running ``reproduce.py`` first, reading the counts
out of pytest's JUnit XML -- is copied from it, so a full-suite run here and the
release's own grader answer the same question in the same way.

The candidate patch is applied here rather than on the host: the verifier image
holds its own copy of the baseline tree, and applying it anywhere else would
score a different checkout than the one the tests import.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


#: Printed on stdout so the host can find the payload among whatever the task's
#: own build system, compiler or test suite wrote there.
SENTINEL = "___ERA_SWE_SCIENCE_RESULT___"


def _run(command, cwd, env, timeout):
    started = time.time()
    try:
        proc = subprocess.run(
            command, cwd=cwd, env=env, text=True, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        return proc.returncode, proc.stdout or "", time.time() - started
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        return 124, output + f"\n[timed out after {timeout}s]", time.time() - started


def _counts(junit):
    """pytest's own totals, read the way the release's grader reads them.

    ``passed = tests - failures - errors`` is the grader's arithmetic verbatim,
    skips included -- a skipped test counts as passed there, and a runner that
    quietly counted it otherwise would report a different number than the
    benchmark does for the same run.
    """
    if not os.path.isfile(junit):
        return 0, 0, 0
    root = ET.parse(junit).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall(".//testsuite")
    total = sum(int(s.attrib.get("tests", "0")) for s in suites)
    failed = sum(int(s.attrib.get("failures", "0")) + int(s.attrib.get("errors", "0"))
                 for s in suites)
    return total, max(total - failed, 0), failed


def _collect(config, env):
    """The private suite's node ids, listed from the verifier's own pytest.

    Run from the task directory and against the absolute test root, so the ids
    that come back are exactly the ids an evaluation may pass straight back to
    pytest from the same working directory.
    """
    code, output, _ = _run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "--collect-only", config.get("tests_root", "/tests/private_tests")],
        config["workdir"], env, float(config.get("private_timeout", 1800)))
    root = config.get("tests_root", "/tests/private_tests")
    ids = []
    for line in output.splitlines():
        line = line.strip()
        # `-q --collect-only` prints one node id per line; everything else pytest
        # writes (warnings, the summary, a plugin's banner) has no `::` in it.
        if "::" not in line or line.startswith(("E ", "warning", "-")):
            continue
        # pytest prints ids relative to the rootdir *it* computed, which for a
        # private suite under /tests is not the task directory the grader runs
        # from. Re-anchor them here, once, so an evaluation can hand them
        # straight back from that directory.
        path = line.split("::", 1)[0]
        if not os.path.isabs(path) and not os.path.exists(
                os.path.join(config["workdir"], path)):
            line = os.path.join(root, line)
        ids.append(line)
    return {"mode": "collect", "return_code": code, "tests": ids,
            "output": output[-int(config.get("tail_chars", 4000)):]}


def main(argv):
    with open(argv[1], "r", encoding="utf-8") as handle:
        config = json.load(handle)

    workdir = config["workdir"]
    task_id = config["task_id"]
    patch_path = config.get("patch") or ""
    selection = list(config.get("tests") or [])
    junit = config.get("junit") or "/tmp/era-junit.xml"
    result_path = config.get("result") or ""
    tail = int(config.get("tail_chars", 4000))

    env = os.environ.copy()
    env.update({
        "SCI_BENCH_TASK_ID": task_id,
        "SCI_BENCH_TASK_DIR": workdir,
        "PYTHONPATH": "%s:%s" % (workdir, os.path.join(workdir, "source")),
    })

    if config.get("mode") == "collect":
        _emit(_collect(config, env), result_path)
        return 0

    payload = {
        "applied": True,
        "apply_error": "",
        "public": {"passed": 0, "return_code": None, "output": "", "seconds": 0.0},
        "private": {"collected": 0, "passed": 0, "failed": 0, "return_code": None,
                    "output": "", "seconds": 0.0, "selected": len(selection)},
    }

    # 1. The candidate patch, applied to the verifier's own checkout.
    if patch_path and os.path.getsize(patch_path) > 0:
        code, output, _ = _run(
            ["git", "apply", "--binary", "-p1", patch_path], workdir, env, 300)
        if code != 0:
            # A patch that will not apply is not a failing program, it is a
            # candidate that never ran -- the caller turns this into an invalid
            # node scoring -inf rather than into a score of zero.
            payload["applied"] = False
            payload["apply_error"] = output.strip()[-tail:] or "git apply failed"
            _emit(payload, result_path)
            return 0

    # 2. The public reproduction, exactly as the release's grader runs it.
    if config.get("public", True):
        code, output, seconds = _run(
            [sys.executable, "reproduce.py"], workdir, env,
            float(config.get("public_timeout", 900)))
        payload["public"] = {"passed": int(code == 0), "return_code": code,
                             "output": output[-tail:], "seconds": round(seconds, 3)}

    # 3. The private suite, restricted to the node ids this evaluation may see.
    if selection:
        for stale in (junit,):
            if os.path.exists(stale):
                os.remove(stale)
        code, output, seconds = _run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
             "--junitxml=%s" % junit, *selection],
            workdir, env, float(config.get("private_timeout", 1800)))
        total, passed, failed = _counts(junit)
        payload["private"] = {"collected": total, "passed": passed, "failed": failed,
                              "return_code": code, "output": output[-tail:],
                              "seconds": round(seconds, 3), "selected": len(selection)}

    _emit(payload, result_path)
    return 0


def _emit(payload, result_path):
    text = json.dumps(payload)
    if result_path:
        try:
            with open(result_path, "w", encoding="utf-8") as handle:
                handle.write(text + "\n")
        except OSError:
            pass
    sys.stdout.write("\n" + SENTINEL + text + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
