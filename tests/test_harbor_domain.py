"""examples/metasearch/_harbor: a Harbor task as an ERA Domain, offline.

A real `git`, a real `tests/test.sh` and a real ERA tree search over patches;
only the agent is scripted. The Docker runner is the stated boundary and is
covered for its refusal paths only.
"""

import json
import subprocess
from pathlib import Path

import pytest

from agentdescent.meta import PrioritySelection

from examples.era.era_empirical_software import run_agentdescent_era
from examples.metasearch import _harbor as hb


TEST_SH = """#!/bin/bash
# Two metrics, each a property of x.txt in the working directory.
mkdir -p /logs/verifier
a=0; b=0
grep -q alpha x.txt && a=1
grep -q beta x.txt && b=1
echo "{\\"alpha\\": $a, \\"beta\\": $b}" > /logs/verifier/reward.json
"""

TASK_TOML = """schema_version = "1.4"

[task]
name = "science-org/fixture-task"
description = "make x.txt say the right things"

[verifier]
timeout_sec = 30

[agent]
timeout_sec = 60

[environment]
cpus = 1
"""


@pytest.fixture
def task_dir(tmp_path):
    root = tmp_path / "task"
    (root / "tests").mkdir(parents=True)
    (root / "environment").mkdir()
    (root / "task.toml").write_text(TASK_TOML)
    (root / "instruction.md").write_text("Edit x.txt so the verifier passes.\n")
    (root / "tests" / "test.sh").write_text(TEST_SH)
    (root / "environment" / "Dockerfile").write_text("FROM scratch\n")
    source = tmp_path / "source"
    source.mkdir()
    (source / "x.txt").write_text("nothing here\n")
    return root, source


def _git_ok():
    return subprocess.run(["git", "--version"], capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(not _git_ok(), reason="needs git")


def test_load_task_reads_the_fields_this_module_needs(task_dir):
    root, _ = task_dir
    task = hb.load_task(root)
    assert task.name == "science-org/fixture-task"
    assert task.verifier_timeout == 30.0 and task.agent_timeout == 60.0
    assert task.dockerfile == root / "environment" / "Dockerfile" and task.docker_image is None
    assert "x.txt" in task.instruction and task.workdir == "/app"
    with pytest.raises(FileNotFoundError):
        hb.load_task(root.parent)


def test_parse_reward_prefers_json_and_falls_back_to_txt(tmp_path):
    (tmp_path / "reward.txt").write_text("1\n")
    assert hb.parse_reward(tmp_path) == {"reward": 1.0}
    (tmp_path / "reward.json").write_text(json.dumps({"a": 0.5, "b": 1, "note": "x", "ok": True}))
    assert hb.parse_reward(tmp_path) == {"a": 0.5, "b": 1.0}
    (tmp_path / "reward.json").write_text("[]")
    with pytest.raises(ValueError):
        hb.parse_reward(tmp_path)
    with pytest.raises(FileNotFoundError):
        hb.parse_reward(tmp_path / "nope")


def _patch_setting(runner, task, text):
    """A patch, made the way an agent's work becomes one: edit, then diff."""
    ws = runner.materialize(task, "")
    (ws / "x.txt").write_text(text)
    return runner.diff(ws)


def test_local_runner_applies_a_patch_and_runs_the_verifier(task_dir):
    root, source = task_dir
    task, runner = hb.load_task(root), hb.LocalRunner(source)
    assert runner.verify(task, "") == {"alpha": 0.0, "beta": 0.0}
    patch = _patch_setting(runner, task, "alpha\n")
    assert "+alpha" in patch
    assert runner.verify(task, patch) == {"alpha": 1.0, "beta": 0.0}
    with pytest.raises(RuntimeError, match="did not apply"):
        runner.verify(task, "--- a/x.txt\n+++ b/x.txt\n@@ -1 +1 @@\n-not there\n+x\n")


def test_harbor_domain_splits_metrics_into_shards_and_caches_the_verifier(task_dir):
    root, source = task_dir
    task, runner = hb.load_task(root), hb.LocalRunner(source)
    calls = []
    real_verify = runner.verify

    def counted(t, patch):
        calls.append(patch)
        return real_verify(t, patch)

    runner.verify = counted
    domain = hb.harbor_domain(task, runner, scoring=["alpha"], held_back=["beta"], shards=4)
    assert domain.test_shards == (4,)
    assert domain.data_summary["shards"] == 4
    valid, metrics, error = domain.evaluate("", (0, 1, 2, 3))
    assert valid and metrics["score"] == 0.0 and error == ""
    both = _patch_setting(runner, task, "alpha beta\n")
    assert domain.evaluate(both, (0,)) [1]["score"] == 1.0
    assert domain.evaluate(both, (4,))[1]["metrics"] == {"beta": 1.0}
    assert len(calls) == 2, "one verifier run per distinct patch"
    valid, metrics, error = domain.evaluate("garbage patch", (0,))
    assert not valid and metrics["score"] == float("-inf") and "did not apply" in error
    assert domain.reward({"score": float("-inf")}) == 0.0 and domain.reward({"score": 0.5}) == 0.5
    with pytest.raises(ValueError, match="both scoring and held back"):
        hb.harbor_domain(task, runner, scoring=["alpha"], held_back=["alpha"])


class _ScriptedAgent:
    """A WorkspaceAgent that edits x.txt where it is told to work."""

    def __init__(self):
        self.seen = []

    def __call__(self, prompt):
        return ""

    def in_workspace(self, path):
        def run(prompt):
            ws = Path(path)
            self.seen.append((ws / "x.txt").read_text())
            current = (ws / "x.txt").read_text()
            (ws / "x.txt").write_text("alpha beta\n" if "alpha" in current else "alpha\n")
            return "done"
        return run


def test_harbor_completion_materialises_the_parent_and_returns_a_fenced_patch(task_dir):
    root, source = task_dir
    task, runner, agent = hb.load_task(root), hb.LocalRunner(source), _ScriptedAgent()
    complete = hb.harbor_completion(task, runner, agent)
    domain = hb.harbor_domain(task, runner, scoring=["alpha"], held_back=["beta"], shards=4)
    from examples.era._era_support import Program

    root_node = Program("root", 0, None, "", "", {"metrics": {}}, True)
    reply = complete(domain.prompt(root_node))
    assert reply.startswith("```\n") and "+alpha" in reply
    assert agent.seen[-1] == "nothing here\n"
    first = reply.strip("`\n")
    child = Program("c", 1, "root", first, "", {"metrics": {"alpha": 1.0}}, True)
    reply2 = complete(domain.prompt(child))
    assert agent.seen[-1] == "alpha\n", "the parent patch was applied before the agent ran"
    assert "+alpha beta" in reply2 and "-nothing here" in reply2, "cumulative against the baseline"


def test_the_era_tree_search_runs_over_harbor_patches_end_to_end(task_dir):
    root, source = task_dir
    task, runner, agent = hb.load_task(root), hb.LocalRunner(source), _ScriptedAgent()
    domain = hb.harbor_domain(task, runner, scoring=["alpha"], held_back=["beta"], shards=4)
    run = run_agentdescent_era(
        hb.harbor_completion(task, runner, agent), mode="serial", iterations=2, workers=1,
        shards=4, test_shards=1, held_out_frac=0.5, domain=domain,
        selection=PrioritySelection(), max_seconds=120.0)
    assert run.result.error is None
    # The first expansion sets `alpha`, which solves every scoring shard, so the
    # search stops there (solved_threshold=1.0): two nodes, not three.
    assert len(run.tree.nodes) == 2 and run.result.outcomes() == {"committed": 1}
    assert run.tree.best().score == 1.0
    assert "+alpha" in run.tree.best().program.code
    # ...and the held-back metric says the task is not done: the split is doing
    # its job -- the reported figure is not the figure the search optimised.
    assert run.baseline_test_metrics["metrics"] == {"beta": 0.0}
    assert run.best_test_metrics["metrics"] == {"beta": 0.0}
    assert run.tree.summary()["selection"] == "PrioritySelection"


def test_docker_runner_refuses_a_task_with_no_image_and_no_dockerfile(task_dir):
    root, _ = task_dir
    (root / "environment" / "Dockerfile").unlink()
    task = hb.load_task(root)
    with pytest.raises(RuntimeError, match="no docker_image"):
        hb.DockerRunner().image_for(task)
    assert hb.DockerRunner(docker="/nonexistent-docker").image_for(
        hb.HarborTask(root, "t", "", "python:3.11-slim", None, root / "tests", 1, 1)
    ) == "python:3.11-slim"
