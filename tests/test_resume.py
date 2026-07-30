"""Resuming a run.

The ledger is a git repo, so re-using `repo_path` continues an interrupted run --
what you want when a multi-hour LLM run dies partway. That behaviour existed but
was undocumented and untested, and it silently discarded `initial_state`.
"""

import warnings

from agentdescent.evolution import AppendRules, Task, evolve


class _Agent:
    def solve(self, rendered, task):
        return "yes" if task.meta["h"] in rendered else "no"

    def propose(self, rendered, task, output, reward):
        return task.meta["h"]


def _tasks(n=12):
    return [Task(id=f"t{i}", prompt="q", meta={"h": f"H{i % 3}"}) for i in range(n)]


REWARD = lambda t, o: 1.0 if o == "yes" else 0.0


def test_same_repo_path_resumes_the_artifact(tmp_path):
    repo = str(tmp_path / "ledger")
    first = evolve(_tasks(), REWARD, agent=_Agent(), strategy=AppendRules(),
                   rounds=3, repo_path=repo)
    assert first.state, "first run should have learned something to resume from"

    second = evolve(_tasks(), REWARD, agent=_Agent(), strategy=AppendRules(),
                    rounds=2, repo_path=repo)
    assert set(first.state) <= set(second.state), "resume lost the earlier artifact"


def test_fresh_repo_path_starts_over(tmp_path):
    a = evolve(_tasks(), REWARD, agent=_Agent(), strategy=AppendRules(),
               rounds=3, repo_path=str(tmp_path / "one"))
    b = evolve(_tasks(), REWARD, agent=_Agent(), strategy=AppendRules(),
               rounds=1, repo_path=str(tmp_path / "two"))
    assert len(b.history) == 1
    assert not (set(a.state) and set(a.state) < set(b.state))


def test_initial_state_on_resume_warns_instead_of_vanishing(tmp_path):
    repo = str(tmp_path / "ledger")
    evolve(_tasks(), REWARD, agent=_Agent(), strategy=AppendRules(),
           rounds=2, repo_path=repo)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = evolve(_tasks(), REWARD, agent=_Agent(), strategy=AppendRules(),
                     rounds=1, repo_path=repo, initial_state={"ZZZ": "ignored"})
    assert "ZZZ" not in res.state                      # still ignored ...
    assert any("initial_state is ignored" in str(x.message) for x in w)   # ... but said so


def test_initial_state_applies_on_a_fresh_repo(tmp_path):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = evolve(_tasks(), REWARD, agent=_Agent(), strategy=AppendRules(),
                     rounds=1, repo_path=str(tmp_path / "new"),
                     initial_state={"seeded": "a starting rule"})
    assert res.state.get("seeded") == "a starting rule"
    assert not any("initial_state is ignored" in str(x.message) for x in w)
