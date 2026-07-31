"""The ledger must not be able to take a run down with it.

The ledger is the third failure category, alongside a caller bug (`ContractError`,
propagate) and a backend blip (absorb and report): it is *infrastructure*. Three
things used to go wrong and each is pinned here.

1. It refused to start at all under an ordinary `~/.gitconfig` -- `commit.gpgsign
   = true` failed the genesis commit, so `evolve()` raised before running a single
   task, from a call that mentions git nowhere.
2. A failure mid-run escaped as an exception, discarding the artifact evolved so
   far -- directly against the documented contract that a run that died still
   returns a partial result. Worst case, the failing call was `ledger.log()`,
   fetched inside the `return` expression purely for the cosmetic `ledger_log`.
3. Scratch repos were reclaimed only at interpreter exit, so a killed process
   leaked one and a long-lived process (a notebook, a sweep) held every repo it
   had ever created.
"""

import glob
import os
import tempfile
import time
import warnings

import pytest

import agentdescent.ledger as ledger_mod
from agentdescent.evolution import (
    AppendRules, SingleSlot, Task, _reap_stale_scratch_repos, evolve,
)
from agentdescent.ledger import GitError, Ledger

TASKS = [Task(id=str(i), prompt=str(i), meta={"gold": str(i)}) for i in range(8)]


def _reward(task, output):
    return 1.0 if "rule" in (output or "") else 0.0


def _run(rendered, task):
    return "rule ok" if "rule" in rendered else "no"


def _propose(rendered, task, output, score):
    return "rule-A"


def _evolve(**kw):
    kw.setdefault("rounds", 8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evolve(TASKS, _reward, run=_run, propose=_propose,
                      strategy=SingleSlot(initial_value="v"),
                      n_workers=2, max_concurrency=1, held_out_frac=0.5, **kw)


# -- 1. the user's git config must not decide whether the ledger can write ------


def test_a_global_gitconfig_with_commit_signing_does_not_break_a_run(tmp_path,
                                                                    monkeypatch):
    """`commit.gpgsign = true` is a common setting and used to be fatal.

    These are the ledger's own bookkeeping commits in a throwaway directory --
    they have no business honouring personal git preferences.
    """
    cfg = tmp_path / "gitconfig"
    cfg.write_text("[commit]\n\tgpgsign = true\n[user]\n\tsigningkey = DEADBEEF\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    res = _evolve()
    assert res.error is None, f"a user gitconfig broke the run: {res.error}"
    assert res.rendered == "rule-A", "the run did not actually evolve anything"


def test_a_global_hooks_path_does_not_run_against_the_scratch_repo(tmp_path,
                                                                  monkeypatch):
    """A user pre-commit hook must not fire on the ledger's internal commits."""
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    hook = hooks / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n")      # a hook that rejects every commit
    hook.chmod(0o755)
    cfg = tmp_path / "gitconfig"
    cfg.write_text(f"[core]\n\thooksPath = {hooks}\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    res = _evolve()
    assert res.error is None, f"a user pre-commit hook broke the run: {res.error}"


def test_a_missing_git_binary_names_git(tmp_path, monkeypatch):
    """Otherwise it is a bare OSError traceback, unlike every other exec path."""
    monkeypatch.setenv("PATH", str(tmp_path))     # no git on it
    with pytest.raises(GitError) as excinfo:
        Ledger(str(tmp_path / "repo"), lambda a: {}, lambda i, v, s: None)
    assert "git" in str(excinfo.value).lower()
    assert "path" in str(excinfo.value).lower()


# -- 2. a ledger failure ends the run but still hands back what it learned ------


def _git_dies_after(n):
    """Patch `_git` to fail after `n` calls -- a held index.lock, a full disk."""
    original = ledger_mod._git
    calls = {"n": 0}

    def failing(repo, *args):
        calls["n"] += 1
        if calls["n"] > n:
            raise GitError(f"git {args[0]} failed in {repo}: index.lock exists")
        return original(repo, *args)
    return failing, original, calls


def _count_git_calls(fn):
    original = ledger_mod._git
    calls = {"n": 0}

    def counting(repo, *args):
        calls["n"] += 1
        return original(repo, *args)
    ledger_mod._git = counting
    try:
        fn()
    finally:
        ledger_mod._git = original
    return calls["n"]


def test_a_ledger_failure_mid_run_returns_a_partial_result_rather_than_raising():
    """The contract: "a run that died still returns a (partial) result"."""
    n = [0]

    def many_proposals(rendered, task, output, score):
        n[0] += 1
        return f"rule number {n[0]}"

    def kw():
        return dict(strategy=AppendRules(), rounds=10, n_workers=3,
                    max_concurrency=1, held_out_frac=0.5)

    def clean():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            evolve(TASKS, lambda t, o: 0.5, run=lambda r, t: r,
                   propose=many_proposals, **kw())

    n[0] = 0
    total = _count_git_calls(clean)
    # Fail somewhere inside the loop, past construction (which legitimately raises:
    # nothing has been produced yet, so there is no partial result to preserve).
    failing, original, _ = _git_dies_after(total - 4)
    ledger_mod._git = failing
    n[0] = 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = evolve(TASKS, lambda t, o: 0.5, run=lambda r, t: r,
                         propose=many_proposals, **kw())
    finally:
        ledger_mod._git = original
    assert res.error is not None, "a dead ledger ended the run silently"
    assert "git" in res.error.lower()
    assert res.rendered, "the artifact evolved before the failure was discarded"


def test_a_failing_ledger_log_does_not_discard_a_completed_run():
    """`ledger_log` is a diagnostic. Losing it must not lose the run.

    It used to be fetched inside the `return` expression, so a git failure there
    threw away a run that had already finished every round and scored itself.
    """
    original = ledger_mod._git

    def no_log(repo, *args):
        if args and args[0] == "log":
            raise GitError("git log failed: index.lock exists")
        return original(repo, *args)

    ledger_mod._git = no_log
    try:
        res = _evolve()
    finally:
        ledger_mod._git = original
    assert res.rendered == "rule-A", "a completed run was discarded"
    assert res.final_reward == 1.0
    assert res.ledger_log == [], "the log should degrade to empty, not explode"
    assert res.error is None, "losing a diagnostic is not a failed run"


# -- 3. scratch ledgers are reclaimed, not held for the process lifetime --------


def _live_scratch_repos():
    return set(glob.glob(os.path.join(tempfile.gettempdir(),
                                      "agentdescent-evolve-*")))


def test_a_scratch_ledger_is_reclaimed_when_evolve_returns():
    """atexit alone meant a sweep held one git repo per run until the process died."""
    before = _live_scratch_repos()
    for _ in range(3):
        _evolve(rounds=2)
    assert _live_scratch_repos() - before == set(), \
        "evolve() left its scratch ledger behind"


def test_a_caller_supplied_repo_path_is_never_reclaimed(tmp_path):
    """`repo_path` is caller-owned -- and how a run is resumed."""
    repo = tmp_path / "mine"
    _evolve(repo_path=str(repo), rounds=2)
    assert repo.is_dir() and (repo / ".git").is_dir(), \
        "a caller-owned repo was deleted; resuming would be impossible"


def test_the_reaper_collects_orphans_but_spares_live_repos():
    """A killed process (SIGKILL, OOM) skips atexit and leaks its repo."""
    orphan = tempfile.mkdtemp(prefix="agentdescent-evolve-")
    fresh = tempfile.mkdtemp(prefix="agentdescent-evolve-")
    old = time.time() - 48 * 3600
    os.utime(orphan, (old, old))
    try:
        _reap_stale_scratch_repos()
        assert not os.path.exists(orphan), "an orphaned scratch ledger was not reaped"
        assert os.path.exists(fresh), "a live scratch ledger was reaped -- data loss"
    finally:
        for d in (orphan, fresh):
            if os.path.exists(d):
                os.rmdir(d)
