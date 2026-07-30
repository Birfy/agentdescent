"""Tensor parallelism must actually be tensor-parallel.

TP's promise is that each worker owns a *disjoint section* of the artifact, which
is what makes the merge a conflict-free union. `evolve()` used only
`WorkUnit.keys` and `WorkUnit.worker` and ignored `WorkUnit.section`, so every
worker could edit the same hot key: passing `TensorParallel` changed which tasks a
worker sampled and nothing else, i.e. it was differently-sharded DP.
"""

from agentdescent.evolution import KeyedRules, Task, evolve
from agentdescent.parallel import DataParallel, TensorParallel, section_of

CATS = ["alpha", "beta", "gamma", "delta"]


class _AlwaysEditsAlpha:
    """Every worker tries to edit the same hot category, and it genuinely helps.

    A real improvement signal matters: with a flat reward the Beta acceptance test
    is marginal and commits are luck, which would make these assertions flaky.
    """

    def solve(self, rendered, task):
        return "yes" if "a rule" in rendered else "no"

    def propose(self, rendered, task, output, reward):
        return "alpha: a rule"


REWARD = lambda t, o: 1.0 if o == "yes" else 0.0


def _tasks(n=16):
    return [Task(id=f"t{i}", prompt="q") for i in range(n)]


def _run(parallel, rounds=6, n_sections=4):
    return evolve(_tasks(), REWARD, agent=_AlwaysEditsAlpha(),
                  strategy=KeyedRules(categories=CATS), parallel=parallel,
                  rounds=rounds, n_workers=4, max_concurrency=4)


def test_only_the_owning_worker_may_edit_a_section():
    res = _run(TensorParallel(n_sections=4))
    owner = f"w{section_of('alpha', 4)}"
    authors = [line for line in res.ledger_log if "alpha" in line]
    assert authors, "the owning worker's edit should still land"
    for line in authors:
        assert owner in line, f"an out-of-section worker committed: {line}"


def test_data_parallel_is_unrestricted():
    """The section check must apply only to TP -- DP has no sections."""
    res = _run(DataParallel())
    assert res.state.get("alpha"), "DP should still let any worker edit any key"


def test_section_assignment_partitions_the_key_space():
    secs = {c: section_of(c, 4) for c in CATS}
    assert all(0 <= v < 4 for v in secs.values())
    # a key belongs to exactly one section, and that is stable
    assert secs == {c: section_of(c, 4) for c in CATS}


def test_tensor_parallel_still_makes_progress_when_workers_own_what_they_edit():
    """Enforcement must not deadlock progress: give each worker its own category."""
    class _EditsOwnSection:
        def solve(self, rendered, task):
            return "yes" if "rule" in rendered else "no"

        def propose(self, rendered, task, output, reward):
            # propose for whichever category this task maps to
            return f"{CATS[int(task.id[1:]) % len(CATS)]}: rule"

    res = evolve(_tasks(24), REWARD, agent=_EditsOwnSection(),
                 strategy=KeyedRules(categories=CATS),
                 parallel=TensorParallel(n_sections=4), rounds=8,
                 n_workers=4, max_concurrency=4)
    assert res.error is None
    assert res.state, "with sections respected, edits must still land"
