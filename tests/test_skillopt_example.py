"""Offline tests for the SkillOpt (ReflACT) example.

Exercise the four faithful invariants -- bounded edit ops, strict accept gate,
learning-rate budget, rejected-edit buffer -- plus SearchQA scoring. No network.
"""

from examples.skillopt_skill_training import (
    LRScheduler,
    apply_patch,
    em_score,
    f1_score,
    propose_patch,
    run_skillopt,
    valid_edit,
)


def test_apply_patch_append_and_insert_after():
    doc = "# Skill\nrule A\n"
    assert apply_patch(doc, [{"op": "append", "content": "rule B"}]).endswith("rule B")
    out = apply_patch(doc, [{"op": "insert_after", "target": "rule A", "content": "rule A2"}])
    assert "rule A\nrule A2" in out


def test_apply_patch_replace_and_delete_skip_missing_targets():
    doc = "# Skill\nrule A\n"
    assert apply_patch(doc, [{"op": "replace", "target": "rule A", "content": "Z"}]).count("Z") == 1
    assert "rule A" not in apply_patch(doc, [{"op": "delete", "target": "rule A"}])
    # missing target -> op is a no-op (faithful to optimizer/skill.py)
    assert apply_patch(doc, [{"op": "replace", "target": "nope", "content": "x"}]) == doc
    assert apply_patch(doc, [{"op": "delete", "target": "nope"}]) == doc


def test_edit_validation():
    assert valid_edit({"op": "append", "content": "x"})
    assert valid_edit({"op": "delete", "target": "t"})
    assert not valid_edit({"op": "append"})               # append needs content
    assert not valid_edit({"op": "replace", "content": "x"})  # replace needs target
    assert not valid_edit({"op": "rewrite", "content": "x"})  # unknown op


def test_lr_scheduler_shapes():
    cos = LRScheduler(max_lr=4, min_lr=2, total_steps=6, mode="cosine")
    seq = [cos.step() for _ in range(6)]
    assert seq[0] == 4 and seq[-1] == 2 and all(2 <= x <= 4 for x in seq)
    const = LRScheduler(max_lr=4, min_lr=2, total_steps=6, mode="constant")
    assert all(x == 4 for x in [const.step() for _ in range(6)])


def test_propose_patch_caps_at_budget():
    stub = lambda prompt: ('{"reasoning":"r","edits":['
                           '{"op":"append","content":"a"},'
                           '{"op":"append","content":"b"},'
                           '{"op":"append","content":"c"}]}')
    edits = propose_patch(stub, "skill", [("q", "p", "g")], [], budget=2)
    assert len(edits) == 2


def test_searchqa_em_and_f1():
    assert em_score("The Answer.", ["answer"]) == 1.0
    assert em_score("wrong", ["answer"]) == 0.0
    assert f1_score("new york", ["new york"]) == 1.0
    assert 0.0 < f1_score("new york city", ["new york"]) < 1.0


def test_strict_gate_accepts_only_improvements_and_buffers_rejects():
    """The loop accepts a candidate only when it strictly improves held-out EM,
    and remembers rejected edits."""
    class Stub:
        def __call__(self, prompt):
            if "## Skill" in prompt:                     # rollout
                return "<answer>42</answer>" if "MAGIC" in prompt else "<answer>0</answer>"
            return '{"reasoning":"r","edits":[{"op":"append","content":"MAGIC rule"}]}'

    train = [{"question": f"q{i}", "context": "c", "answers": ["42"]} for i in range(8)]
    val = [{"question": f"v{i}", "context": "c", "answers": ["42"]} for i in range(6)]
    res = run_skillopt(Stub(), train, val, steps=4, lr=3, minibatch=3, seed=0)
    assert res.seed_em == 0.0
    assert res.best_em == 1.0        # the improving edit was accepted
    assert res.accepted >= 1


def test_useless_edits_are_rejected_and_buffered():
    """An edit that never helps is rejected every step and pushed to the buffer."""
    class Stub:
        def __call__(self, prompt):
            if "## Skill" in prompt:
                return "<answer>0</answer>"              # always wrong
            return '{"reasoning":"r","edits":[{"op":"append","content":"noise"}]}'

    train = [{"question": f"q{i}", "context": "c", "answers": ["42"]} for i in range(8)]
    val = [{"question": f"v{i}", "context": "c", "answers": ["42"]} for i in range(6)]
    res = run_skillopt(Stub(), train, val, steps=3, lr=2, minibatch=3, seed=0)
    assert res.best_em == res.seed_em == 0.0
    assert res.rejected >= 1
