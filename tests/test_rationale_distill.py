"""Rationale distillation has to be right where the artifact learns to carry
its own reasoning.

A rollout yields a rationale -- a success says how it answered, a near-miss
says what it needed -- and `distill` dedupes them into the skill the artifact
carries. The fixtures pin the surfaces that make this honest: rationales are
extracted from success and near-miss alike, distillation dedupes, and
`rationale_evolve` runs the cap domain end to end and returns the skill now
carrying its own notes.
"""

import pytest

from agentdescent import (
    RationaleReport,
    Task,
    distill,
    extract_rationale,
    rationale_evolve,
)


def _task(difficulty, seed=0):
    return Task(id=f"t{difficulty}", prompt=f"item {difficulty}",
                meta={"difficulty": difficulty, "gold": str(2 * difficulty)})


def test_rationales_capture_success_and_near_miss():
    assert extract_rationale(_task(3), "6", True) == "answered '6' on item 3"
    assert extract_rationale(_task(3), "?", False) == "failed with '?' on item 3"
    # domain-neutral: no cap/difficulty assumption, so no off-by-one
    assert extract_rationale(_task(3), "x", False) != extract_rationale(_task(3), "X", False)


def test_distill_dedupes_by_content():
    records = [(_task(2), "4", True), (_task(2), "4", True),
               (_task(5), "?", False)]
    rules = distill(records)
    assert len(rules) == 2
    assert any("failed" in r for r in rules)


def test_distill_preserves_answer_content():
    # '-1' vs '1' and 'A' vs 'a' are distinct outputs: normalising case or
    # punctuation would silently drop a distinct rule
    records = [(_task(1), "-1", True), (_task(1), "1", True),
               (_task(3), "A", True), (_task(3), "a", True)]
    rules = distill(records)
    assert len(rules) == 4
    assert any("'-1'" in r for r in rules) and any("'1'" in r for r in rules)
    assert any("'A'" in r for r in rules) and any("'a'" in r for r in rules)


def test_rationale_evolve_distills_the_skill_s_own_notes():
    def run(rendered, task):
        import re
        m = re.search(r"cap:(\d+)", rendered)
        cap = int(m.group(1)) if m else 0
        return task.meta["gold"] if task.meta["difficulty"] <= cap else "?"

    def reward(task, output):
        return 1.0 if str(output).strip() == str(task.meta["gold"]).strip() else 0.0

    def propose(rendered, task, output, reward):
        if reward >= 0.999:
            return rendered
        return f"cap:{task.meta['difficulty']}"     # jump to the failing task

    problems = {f"p{d}": _task(d) for d in (2, 4, 6, 8)}
    report, result = rationale_evolve(
        problems, run=run, reward=reward, propose=propose,
        rounds=3, n_workers=1, repo_path=None)
    assert isinstance(report, RationaleReport)
    assert report.n_rationales >= 1
    assert "What I learned" in report.skill     # carries its near-miss notes
    assert report.successes >= 1
    assert result.meta["n_rationales"] == report.n_rationales
    # the artifact itself carries the distilled reasoning, not just the report
    assert "What I learned" in result.rendered
    assert "What I learned" in result.state["value"]
    assert result.rendered.count("# What I learned") == 1


def test_reseeding_does_not_duplicate_the_notes_block():
    def run(rendered, task):
        import re
        m = re.search(r"cap:(\d+)", rendered)
        cap = int(m.group(1)) if m else 0
        return task.meta["gold"] if task.meta["difficulty"] <= cap else "?"

    def reward(task, output):
        return 1.0 if str(output).strip() == str(task.meta["gold"]).strip() else 0.0

    def propose(rendered, task, output, reward):
        if reward >= 0.999:
            return rendered
        return f"cap:{task.meta['difficulty']}"

    problems = {f"p{d}": _task(d) for d in (2, 4, 6, 8)}
    first, first_result = rationale_evolve(
        problems, run=run, reward=reward, propose=propose,
        rounds=3, n_workers=1, repo_path=None)
    second, second_result = rationale_evolve(
        problems, run=run, reward=reward, propose=propose,
        seed_skill=first_result.rendered, rounds=3, n_workers=1,
        repo_path=None)
    # seeding from a previous result replaces, not appends, the notes block
    assert second_result.rendered.count("# What I learned") == 1
