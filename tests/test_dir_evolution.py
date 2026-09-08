"""End to end: a directory on disk, evolved by a real (sub)process agent.

No API key and no network -- the "agent" is a tiny Python program invoked through
`cli_agent`, so it is a genuine `WorkspaceAgent`: it runs in a directory, reads
the candidate skill from disk, and behaves differently depending on what the
skill says. That is the property under test. If it merely received the skill in a
prompt, none of the staging, layout, overlay or isolation logic would be exercised.
"""

import json
import os
import re
import sys
import tempfile
import threading

import pytest

from agentdescent import (
    AggregatorConfig, evolve, gated_reward, load_tree, scorer, tasks_from,
    tree_reflector,
)
from agentdescent.agents import cli_agent, echo
from agentdescent.evolution import EvolutionResult, Task
from agentdescent.filetree import TreeError, canonical
from agentdescent.governance import HARNESS_BLAST_RADIUS, SKILL_BLAST_RADIUS
from agentdescent.runners import TEST_FAILURE_MARKER, code_runner, tree_runner
from agentdescent.treestrategy import FileTree

# ---------------------------------------------------------------------------
# a real workspace agent: it reads the skill it was given and obeys it
# ---------------------------------------------------------------------------

_SKILL_NAME = "shout-skill"

_AGENT_SRC = (
    "import os,sys;"
    "q=sys.argv[1];"
    f"p=os.path.join('.claude','skills','{_SKILL_NAME}','rules.md');"
    "t=open(p).read() if os.path.exists(p) else '';"
    "print(q[::-1] if 'MODE: reverse' in t else q)"
)

_WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
          "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
          "oscar", "papa"]


def _agent():
    return cli_agent([sys.executable, "-c", _AGENT_SRC])


def _skill_dir(rules="MODE: forward\n"):
    root = tempfile.mkdtemp()
    path = os.path.join(root, _SKILL_NAME)
    os.makedirs(path)
    with open(os.path.join(path, "rules.md"), "w") as fh:
        fh.write(rules)
    return path


def _rows():
    return [{"prompt": w, "gold": w[::-1]} for w in _WORDS]


def _reflector():
    """A stub that reads the reflection prompt and proposes the fix.

    Deliberately not stateful: it derives the new file from the *current* content
    shown to it, exactly as a model would."""
    def complete(prompt: str) -> str:
        m = re.search(r"--- rules\.md ---\n(.*?)\nTASK THE AGENT WAS GIVEN:",
                      prompt, re.DOTALL)
        body = (m.group(1) if m else "").strip()
        new = body.replace("MODE: forward", "MODE: reverse")
        if new == body:
            new = body + "\nMODE: reverse"
        return ('<EDITS>' + json.dumps(
            {"rationale": "answers must be reversed",
             "edits": [{"path": "rules.md", "content": new + "\n"}]}) + '</EDITS>')
    return complete


# ---------------------------------------------------------------------------
# the wiring -- what a directory run hands evolve()
# ---------------------------------------------------------------------------


def _dir_defaults(kw, n_tasks):
    """The engine defaults a directory workload wants: no self-verify re-run, a
    small cheap layer, a merge every two cards. Explicit, so nothing is hidden."""
    for key, value in (("self_verify", False), ("cheap_eval_tasks", 4),
                       ("held_out_frac", 0.3), ("rounds", 6), ("patience", 3),
                       ("target_reward", 0.98)):
        kw.setdefault(key, value)
    kw.setdefault("n_workers", max(1, min(4, int(n_tasks * (1 - kw["held_out_frac"])))))
    kw.setdefault("max_concurrency", kw["n_workers"])
    kw.setdefault("agg_config", AggregatorConfig(batch_trigger=2, max_wait_rounds=1))


def _evolve_dir(path, rows, *, agent, reflect_with, score="exact", layout="claude_skill",
                frozen=(), blast_radius=SKILL_BLAST_RADIUS, prompt_template="{prompt}",
                **kw):
    """A skill or agent directory, run by a real workspace agent."""
    tasks = tasks_from(rows)
    name = os.path.basename(path)
    tree = load_tree(path)
    strategy = FileTree(tree, frozen=frozen, max_files_per_diff=2)
    run = tree_runner(agent, layout=layout, name=name,
                      overlay=strategy.frozen_files(tree),
                      prompt_template=prompt_template)
    _dir_defaults(kw, len(tasks))
    return evolve(tasks, scorer(score), run=run, strategy=strategy, artifact_id=name,
                  propose=tree_reflector(reflect_with, strategy=strategy),
                  blast_radius=blast_radius, **kw)


def _evolve_code(path, rows, *, entrypoint, reflect_with, score="exact",
                 frozen=("tests/**", "conftest.py"), test_cmd=None, **kw):
    """Agent code: the tree is executed behind a frozen test gate."""
    tasks = tasks_from(rows)
    name = os.path.basename(path)
    tree = load_tree(path)
    strategy = FileTree(tree, frozen=frozen, max_files_per_diff=2)
    run = code_runner(entrypoint, layout="root", name=name, test_cmd=test_cmd,
                      overlay=strategy.frozen_files(tree))
    _dir_defaults(kw, len(tasks))
    return evolve(tasks, gated_reward(scorer(score)), run=run, strategy=strategy,
                  artifact_id=name, blast_radius=HARNESS_BLAST_RADIUS,
                  propose=tree_reflector(reflect_with, strategy=strategy,
                                         context_files=("**/*.py",)),
                  **kw)


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------


def test_a_skill_directory_evolves_and_the_agent_really_reads_it():
    path = _skill_dir()
    result = _evolve_dir(
        path, _rows(), agent=_agent(), reflect_with=_reflector(),
        rounds=4, n_workers=2, max_concurrency=1, seed=0)

    assert result.error is None, result.error
    assert result.final_reward == 1.0
    assert result.outcomes().get("committed", 0) >= 1
    assert "MODE: reverse" in result.state["rules.md"]
    # the source directory is never touched by the run itself
    assert load_tree(path)["rules.md"] == "MODE: forward\n"


def test_the_starting_directory_is_the_starting_artifact():
    path = _skill_dir(rules="MODE: reverse\n")   # already correct
    result = _evolve_dir(
        path, _rows(), agent=_agent(), reflect_with=_reflector(),
        rounds=2, n_workers=1, max_concurrency=1)
    assert result.final_reward == 1.0
    assert result.history[0].held_out_reward == 1.0


def test_frozen_files_are_never_proposed_even_when_the_reflector_targets_them():
    path = _skill_dir()
    with open(os.path.join(path, "GRADING.md"), "w") as fh:
        fh.write("gold answers: alpha=ALPHA\n")

    def sneaky(prompt: str) -> str:
        return ('<EDITS>' + json.dumps({"edits": [
            {"path": "GRADING.md", "content": "everything scores 1.0"}]}) + '</EDITS>')

    result = _evolve_dir(
        path, _rows(), agent=_agent(), reflect_with=sneaky, frozen=["GRADING.md"],
        rounds=2, n_workers=1, max_concurrency=1)
    assert result.state["GRADING.md"] == "gold answers: alpha=ALPHA\n"
    assert result.outcomes().get("committed", 0) == 0


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------


def test_tree_runner_refuses_a_completion_that_cannot_see_files():
    with pytest.raises(TypeError) as e:
        tree_runner(echo(), layout="claude_skill", name="x")
    assert "WorkspaceAgent" in str(e.value)


def test_the_overlay_restores_frozen_files_over_a_tampered_candidate():
    # the enforcement that `frozen` in the strategy cannot provide: a candidate
    # that arrives with the file already rewritten (a resumed ledger, a custom
    # aggregator, a run-time write) still sees the pristine copy.
    cat = cli_agent([sys.executable, "-c",
                     "print(open('tests/t.txt').read().strip())"])
    run = tree_runner(cat, layout="root", name="x",
                      overlay={"tests/t.txt": "PRISTINE"},
                      prompt_template="{prompt}")
    tampered = canonical({"tests/t.txt": "TAMPERED", "a.md": "x"})
    assert run(tampered, Task("t", "go")) == "PRISTINE"


def test_every_rollout_gets_its_own_workspace():
    pwd = cli_agent([sys.executable, "-c", "import os;print(os.getcwd())"])
    run = tree_runner(pwd, layout="root", name="x", prompt_template="{prompt}")
    rendered = canonical({"a.md": "x"})
    seen, lock = [], threading.Lock()

    def once(i):
        out = run(rendered, Task(f"t{i}", "go"))
        with lock:
            seen.append(out)

    threads = [threading.Thread(target=once, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(seen)) == 8            # no two rollouts shared a directory
    assert not any(os.path.exists(p) for p in seen)      # and all cleaned up


def test_fixtures_are_staged_beside_the_tree():
    cat = cli_agent([sys.executable, "-c", "print(open('input.txt').read().strip())"])
    run = tree_runner(cat, layout="claude_skill", name="s",
                      prompt_template="{prompt}")
    task = Task("t", "go", {"fixtures": {"input.txt": "hello from the fixture"}})
    assert run(canonical({"SKILL.md": "x"}), task) == "hello from the fixture"


def test_the_layout_decides_where_the_tree_lands():
    ls = cli_agent([sys.executable, "-c",
                    "import os;print(os.path.exists('.claude/skills/demo/SKILL.md'))"])
    run = tree_runner(ls, layout="claude_skill", name="demo", prompt_template="{prompt}")
    assert run(canonical({"SKILL.md": "x"}), Task("t", "go")) == "True"


# ---------------------------------------------------------------------------
# code_runner: the tree is executed, and the gate runs first
# ---------------------------------------------------------------------------


def test_code_runner_executes_the_candidate():
    run = code_runner([sys.executable, "main.py"])
    tree = canonical({"main.py": "import sys;print(sys.argv[1].upper())"})
    assert run(tree, Task("t", "hello")) == "HELLO"


def test_a_failing_gate_scores_zero_in_band_instead_of_raising():
    run = code_runner([sys.executable, "main.py"],
                      test_cmd=[sys.executable, "-c", "raise SystemExit(1)"])
    out = run(canonical({"main.py": "print('x')"}), Task("t", "hello"))
    assert out.startswith(TEST_FAILURE_MARKER)


def test_the_gate_runs_against_the_pristine_tests_not_the_candidates():
    # the candidate ships a weakened test file; the overlay puts the real one back.
    run = code_runner([sys.executable, "main.py"],
                      test_cmd=[sys.executable, "check.py"],
                      overlay={"check.py": "raise SystemExit(1)"})
    tree = canonical({"main.py": "print('x')", "check.py": "raise SystemExit(0)"})
    assert run(tree, Task("t", "hi")).startswith(TEST_FAILURE_MARKER)


def test_candidate_code_does_not_inherit_the_ambient_environment():
    os.environ["AGENTDESCENT_SECRET_FOR_TEST"] = "leak-me"
    try:
        run = code_runner([sys.executable, "main.py"])
        tree = canonical({"main.py":
                          "import os;print(os.environ.get('AGENTDESCENT_SECRET_FOR_TEST','none'))"})
        assert run(tree, Task("t", "x")) == "none"
    finally:
        del os.environ["AGENTDESCENT_SECRET_FOR_TEST"]


def test_a_hung_candidate_is_killed_by_the_timeout():
    run = code_runner([sys.executable, "main.py"], timeout=1.0)
    out = run(canonical({"main.py": "import time;time.sleep(30)"}), Task("t", "x"))
    assert out.startswith(TEST_FAILURE_MARKER) and "timed out" in out


# ---------------------------------------------------------------------------
# installing the result back
# ---------------------------------------------------------------------------


def _result(state):
    return EvolutionResult(state=state, rendered=canonical(state), final_reward=1.0,
                           history=[], ledger_log=[])


def test_write_to_dry_run_reports_the_plan_and_touches_nothing():
    dest = _skill_dir()
    plan = _result({"rules.md": "MODE: reverse\n", "new.md": "x"}).write_to(
        dest, dry_run=True)
    assert plan["written"] == ["new.md", "rules.md"]
    assert not os.path.exists(os.path.join(dest, "new.md"))


def test_write_to_backs_up_before_overwriting():
    dest = _skill_dir()
    plan = _result({"rules.md": "MODE: reverse\n"}).write_to(dest)
    assert plan["backup"] and os.path.isdir(plan["backup"][0])
    assert load_tree(dest)["rules.md"] == "MODE: reverse\n"
    assert load_tree(plan["backup"][0])["rules.md"] == "MODE: forward\n"


def test_write_to_leaves_unknown_files_alone_unless_asked():
    dest = _skill_dir()
    with open(os.path.join(dest, "notes.md"), "w") as fh:
        fh.write("mine")
    plan = _result({"rules.md": "x"}).write_to(dest, backup=False)
    assert plan["extra"] == ["notes.md"] and os.path.exists(os.path.join(dest, "notes.md"))

    plan = _result({"rules.md": "x"}).write_to(dest, backup=False, prune=True)
    assert plan["deleted"] == ["notes.md"]
    assert not os.path.exists(os.path.join(dest, "notes.md"))


def test_write_to_refuses_a_result_that_is_not_a_file_tree():
    # an AppendRules result: its keys are rule hashes, which are *syntactically*
    # valid paths -- so the guard is the rendered form, not the keys.
    playbook = EvolutionResult(state={"a3f2c1": "always check units"},
                               rendered="# Playbook\n- always check units",
                               final_reward=1.0, history=[], ledger_log=[])
    with pytest.raises(TreeError) as e:
        playbook.write_to(tempfile.mkdtemp())
    assert "FileTree" in str(e.value)


# ---------------------------------------------------------------------------
# an agent directory, and agent code
# ---------------------------------------------------------------------------


def test_an_agent_directory_runs_at_the_harness_layer():
    from agentdescent.evolution import EvolvingArtifact
    from agentdescent.governance import Layer, classify

    path = _skill_dir()
    result = _evolve_dir(
        path, _rows(), agent=_agent(), reflect_with=_reflector(),
        layout="claude_agent", blast_radius=HARNESS_BLAST_RADIUS,
        rounds=2, n_workers=1, max_concurrency=1)
    assert result.error is None, result.error
    # an agent definition is a harness: L1, so every merge also passes the oracle.
    art = EvolvingArtifact("a", {}, blast_radius=HARNESS_BLAST_RADIUS)
    assert classify(art) is Layer.L1_SLOW


def test_agent_code_evolves_behind_a_test_gate_it_cannot_rewrite():
    root = tempfile.mkdtemp()
    src = os.path.join(root, "tiny-agent")
    os.makedirs(os.path.join(src, "tests"))
    # `main.py` answers by echoing; the fix is to reverse, as the tests demand.
    with open(os.path.join(src, "main.py"), "w") as fh:
        fh.write("import sys\nprint(sys.argv[1])\n")
    with open(os.path.join(src, "tests", "test_main.py"), "w") as fh:
        fh.write("def test_placeholder():\n    assert True\n")

    def reflect(prompt):
        return "<EDITS>" + json.dumps({"edits": [
            {"path": "main.py", "content": "import sys\nprint(sys.argv[1][::-1])\n"},
            # a second edit aimed at the frozen tests: must be dropped
            {"path": "tests/test_main.py", "content": "assert False"}]}) + "</EDITS>"

    result = _evolve_code(
        src, _rows(), entrypoint=[sys.executable, "main.py"], reflect_with=reflect,
        test_cmd=[sys.executable, "-c", "import ast;ast.parse(open('main.py').read())"],
        rounds=2, n_workers=1, max_concurrency=1)

    assert result.error is None, result.error
    assert result.final_reward == 1.0
    assert "[::-1]" in result.state["main.py"]
    assert result.state["tests/test_main.py"] == (
        "def test_placeholder():\n    assert True\n")     # frozen, never rewritten


def test_a_candidate_that_breaks_the_gate_scores_zero_and_is_not_committed():
    root = tempfile.mkdtemp()
    src = os.path.join(root, "tiny-agent-2")
    os.makedirs(src)
    with open(os.path.join(src, "main.py"), "w") as fh:
        fh.write("import sys\nprint(sys.argv[1][::-1])\n")

    def wrecker(prompt):
        return "<EDITS>" + json.dumps({"edits": [
            {"path": "main.py", "content": "this is not python("}]}) + "</EDITS>"

    result = _evolve_code(
        src, _rows(), entrypoint=[sys.executable, "main.py"], reflect_with=wrecker,
        test_cmd=[sys.executable, "-c", "import ast;ast.parse(open('main.py').read())"],
        rounds=2, n_workers=1, max_concurrency=1)

    assert result.state["main.py"] == "import sys\nprint(sys.argv[1][::-1])\n"
    assert result.outcomes().get("committed", 0) == 0
