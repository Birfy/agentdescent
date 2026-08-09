"""A real objective for the DGM port: a self-editing agent, run against pytest.

The shipped objective is a *surrogate* -- each SWE instance is assigned a latent
capability set by hashing its id, and an agent "resolves" it when its own
capability set covers that. The DGM loop is faithful on top of it, but the
objective it optimises has three properties the real one does not:

* it is **monotone** -- adding a capability can never un-resolve a task, so a
  self-modification can never regress;
* capabilities never **interact** -- no edit can help one task and break another;
* the required set is a **hash**, unrelated to the task's actual difficulty.

Monotonicity is the one that matters, because it removes the reason DGM keeps an
archive at all. Open-ended search retains stepping stones because a worse
intermediate can lead somewhere better; under a monotone objective there are no
worse intermediates, and `keep-all` plus the novelty-discounted parent draw are
machinery with nothing to do.

So this module replaces the objective rather than the algorithm. The agent is a
**real directory of Python source** that edits itself; a task is a **real
package with a failing pytest**; a score is what pytest actually reports. Every
DGM mechanism above it -- keep-all archive, `sigmoid(10*(s-0.5)) / (1+children)`
parent selection, staged small->medium escalation -- is untouched.

**This is not SWE-bench.** Six vendored bugs in `examples/dgm/tasks/` are not 500
real repository issues, and a number from here cannot be compared with the paper.
What it does reproduce is the *shape*: real code, real execution, real pass/fail,
and self-edits that can genuinely make the agent worse -- including edits that
leave it unable to run at all.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from agentdescent.filetree import parse_tree
from agentdescent.treestrategy import FileTree as _FileTree

TASKS_DIR = pathlib.Path(__file__).resolve().parent / "tasks"

#: Empty replies seen by :func:`propose_self_edit`, for the warning below.
_EMPTY = {"n": 0}

#: The agent, as source. This is the artifact DGM evolves: a `FileTree` whose
#: keys are real relative paths, so a candidate can be written to a workspace and
#: imported. The seed is deliberately weak -- it edits nothing and scores zero --
#: because a seed that already solved the tasks would leave the search nothing to
#: find, which is the saturation trap every other row in this matrix hit.
SEED_AGENT: Dict[str, str] = {
    "agent/solve.py": '''"""The agent: read a failing task, edit `lib.py` to fix it."""
from tools import read_source, run_tests, write_source


def solve(task_dir, llm):
    """Return nothing; edit files in `task_dir` in place.

    `llm(prompt) -> str` is the only model access. `run_tests(task_dir)`
    returns (passed, failed, output).
    """
    source = read_source(task_dir)
    _, _, failure = run_tests(task_dir)
    reply = llm(
        "Fix the bug so every test passes. Reply with the complete new "
        "contents of lib.py and nothing else -- no fences, no commentary.\\n\\n"
        f"lib.py:\\n{source}\\n\\npytest says:\\n{failure}"
    )
    write_source(task_dir, reply)
''',
    "agent/tools.py": '''"""What the agent can do. Editing this file is how the agent improves."""
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
''',
}


@dataclass
class TaskCase:
    """One vendored bug: a package with a failing test."""

    id: str
    path: pathlib.Path
    why: str


def load_tasks(limit: Optional[int] = None) -> List[TaskCase]:
    cases = []
    for d in sorted(TASKS_DIR.iterdir()):
        meta = d / "task.json"
        if not meta.is_file():
            continue
        info = json.loads(meta.read_text())
        cases.append(TaskCase(id=info["id"], path=d, why=info["why"]))
    return cases[:limit] if limit else cases


def _materialise(agent_files: Dict[str, str], root: pathlib.Path) -> None:
    """Write the candidate agent's source into a workspace.

    Paths are checked rather than trusted: a self-edit is model output, and this
    writes to a real filesystem. `..` or an absolute path would escape the
    workspace, which is the one failure mode here that is not just a bad score.
    """
    for rel, body in agent_files.items():
        target = (root / rel).resolve()
        if not str(target).startswith(str(root.resolve()) + os.sep):
            raise ValueError(f"agent file escapes the workspace: {rel!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)


def run_agent_on_task(agent_files: Dict[str, str], case: TaskCase,
                      llm: Callable[[str], str], timeout: float = 180.0,
                      max_calls: int = 6) -> Tuple[bool, str]:
    """Run one candidate agent against one task in a scratch copy.

    Returns ``(resolved, detail)``. **A broken agent scores zero rather than
    raising**: a self-modification that leaves a syntax error, imports a module
    that is not there, or loops until the timeout is exactly the regression this
    objective exists to expose, and turning it into an exception would stop the
    search instead of scoring it.
    """
    work = pathlib.Path(tempfile.mkdtemp(prefix="dgm-real-"))
    try:
        task_dir = work / "task"
        shutil.copytree(case.path, task_dir)
        _materialise(agent_files, work)

        driver = work / "_run.py"
        driver.write_text(
            "import sys\n"
            "sys.path.insert(0, 'agent')\n"
            "from solve import solve\n"
            "import _bridge\n"
            "solve(sys.argv[1], _bridge.llm)\n")
        # A line-delimited request/reply channel on the child's stdio, because
        # the agent runs in its own interpreter (a self-edit can and does break
        # imports) and the model lives in this one.
        #
        # This replaced a pre-fetch scheme: run the agent once to capture its
        # prompt, answer it here, then re-run with that single reply served from
        # a file. That made "one model call per task" a property of the *harness*
        # rather than of the agent -- and the first self-modification this
        # objective ever produced was a retry loop, which is exactly the
        # improvement the pre-fetch made impossible. It scored 0.875 -> 0.500,
        # and the regression was the harness's, not the agent's.
        (work / "_bridge.py").write_text(
            "import json, sys\n"
            "def llm(prompt):\n"
            "    sys.stdout.write(json.dumps({'prompt': prompt}) + '\\n')\n"
            "    sys.stdout.flush()\n"
            "    line = sys.stdin.readline()\n"
            "    if not line:\n"
            "        raise RuntimeError('the harness closed the channel')\n"
            "    return json.loads(line)['reply']\n")

        proc = subprocess.Popen(
            [sys.executable, "-u", str(driver), str(task_dir)],
            cwd=str(work), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True)
        calls, deadline = 0, time.monotonic() + timeout
        try:
            while True:
                if time.monotonic() > deadline:
                    proc.kill()
                    return False, f"timed out after {calls} model call(s)"
                line = proc.stdout.readline()
                if not line:                       # the agent finished or died
                    break
                try:
                    request = json.loads(line)
                except json.JSONDecodeError:
                    continue                       # the agent printed something else
                if calls >= max_calls:
                    # A budget, not a correctness rule: an agent is free to
                    # learn a retry loop, and not free to loop forever on the
                    # matrix's money. Closing the channel surfaces as an error
                    # inside the agent, which is scoreable.
                    proc.kill()
                    return False, f"exceeded {max_calls} model calls"
                calls += 1
                reply = llm(request.get("prompt", ""))
                proc.stdin.write(json.dumps({"reply": reply}) + "\n")
                proc.stdin.flush()
            proc.wait(timeout=max(1.0, deadline - time.monotonic()))
        finally:
            if proc.poll() is None:
                proc.kill()
        if calls == 0:
            return False, "agent never reached a model call"
        if proc.returncode != 0:
            tail = (proc.stderr.read() or "").strip().splitlines()
            return False, "agent crashed: " + (tail[-1][:160] if tail else "?")

        check = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", "test_lib.py"],
            cwd=str(task_dir), capture_output=True, text=True, timeout=timeout)
        passed = check.returncode == 0
        last = (check.stdout + check.stderr).strip().splitlines()
        return passed, (last[-1][:160] if last else "")
    except subprocess.TimeoutExpired:
        return False, "timed out"
    except Exception as exc:                       # noqa: BLE001 - scored, not raised
        return False, f"{type(exc).__name__}: {exc}"[:160]
    finally:
        shutil.rmtree(work, ignore_errors=True)


def make_real_evaluator(llm: Callable[[str], str], concurrency: int = 4):
    """An ``evaluate_fn(agent_files, cases) -> float`` that runs pytest.

    Signature-compatible with the surrogate's, so `run_dgm` takes either.
    """
    def evaluate_fn(agent_files: Dict[str, str], cases: List[TaskCase]) -> float:
        if not cases:
            return 0.0
        from concurrent.futures import ThreadPoolExecutor
        n = max(1, min(concurrency, len(cases)))
        with ThreadPoolExecutor(n) as pool:
            results = list(pool.map(
                lambda c: run_agent_on_task(agent_files, c, llm)[0], cases))
        return sum(results) / len(results)
    return evaluate_fn


# ===========================================================================
# The self-modification: an agent edits its own source
# ===========================================================================

#: Which of its own files the agent may rewrite. `solve.py` is its workflow and
#: `tools.py` is what it can do; both are fair game upstream, where DGM edits
#: "tools, prompts, workflow". Nothing else is writable, so a self-edit cannot
#: reach the harness that scores it -- upstream has the same boundary, enforced
#: by running the candidate in its own container rather than by a list.
EDITABLE = ("agent/solve.py", "agent/tools.py")

#: Token budget for this module's model calls. Both of them ask for a **whole
#: file**, and on a reasoning model the budget is spent on hidden reasoning first
#: with the visible content being whatever is left. Measured here on
#: `deepseek-v4-flash` at the library default of 4096: `propose_self_edit`
#: returned **zero characters**, every rollout, so no evidence card ever reached
#: the merger and a 13-rollout run finished byte-identical to its seed -- which
#: reads as "the search found nothing" rather than "the proposer never spoke".
#: ADAS hit exactly this and documents the same number; it did not travel here
#: with the code that needed it.
MAX_TOKENS = 16384

_PROPOSE_TMPL = """You are a self-improving coding agent. This is your own source.

{sources}

You were run on {n} tasks and resolved {resolved}. Here is what happened:

{log}

Diagnose the single biggest weakness in YOUR OWN code above -- not in the tasks --
and rewrite exactly one file to fix it. Reply with:

    PATH: <one of {editable}>
    <the complete new contents of that file>

No fences, no commentary. The file must be valid Python; if it is not, or if it
raises on import, your next evaluation scores zero.
"""


def propose_self_edit(agent_files: Dict[str, str], report: List[Tuple[str, bool, str]],
                      llm: Callable[[str], str]) -> Optional[str]:
    """One self-modification, as ``PATH: <p>\\n<body>``, or ``None``.

    The conditioning is the agent's **own evaluation log** -- which tasks it
    failed and what pytest said -- which is upstream's signal too
    (`DGM_outer.py` hands the parent its eval logs). What differs is that here
    the log is real output from real code.
    """
    sources = "\n\n".join(f"--- {p} ---\n{agent_files[p]}" for p in EDITABLE
                          if p in agent_files)
    log = "\n".join(f"[{'ok ' if ok else 'FAIL'}] {tid}: {detail}"
                    for tid, ok, detail in report) or "(no tasks run)"
    reply = llm(_PROPOSE_TMPL.format(
        sources=sources, n=len(report),
        resolved=sum(1 for _, ok, _ in report if ok),
        log=log, editable=list(EDITABLE)))
    if not reply.strip():
        # Say it rather than returning a quiet `None`. An empty completion and a
        # deliberate "no proposal" are the same value here and opposite
        # diagnoses: one is a starved token budget, the other is the search
        # having nothing to suggest. Measured, the first produced a run that
        # ended byte-identical to its seed and looked like the second.
        _EMPTY["n"] += 1
        if _EMPTY["n"] in (1, 5, 25):
            warnings.warn(
                f"the self-modification proposer returned an empty completion "
                f"({_EMPTY['n']} so far), so no child agent was produced. On a "
                f"reasoning model that means the token budget went to hidden "
                f"reasoning -- this module asks for a whole file and uses "
                f"max_tokens={MAX_TOKENS} for that reason.",
                RuntimeWarning, stacklevel=2)
        return None
    return reply.strip()


def parse_self_edit(proposal: str) -> Optional[Tuple[str, str]]:
    """``(path, body)`` from a proposal, or ``None`` if it is not usable.

    Rejects a path outside :data:`EDITABLE` rather than trusting it: this text
    becomes a file write. A malformed reply is *not* an error -- the model is
    allowed to fail, and the search simply gets no child that round.
    """
    text = proposal.strip()
    if not text.lower().startswith("path:"):
        return None
    head, _, body = text.partition("\n")
    path = head.split(":", 1)[1].strip()
    if path not in EDITABLE or not body.strip():
        return None
    return path, body


# ===========================================================================
# Driving DGM's own loop over the real objective
# ===========================================================================


def run_dgm_real(llm: Callable[[str], str], *, cases: Optional[List[TaskCase]] = None,
                 generations: int = 6, selfimprove_size: int = 2, seed: int = 0,
                 asynchronous: bool = False, async_ratio: int = 3,
                 max_seconds: float = 600.0, max_rollouts: Optional[int] = None,
                 staleness: str = "full", eval_concurrency: int = 4,
                 verbose: bool = False):
    """DGM over self-editing source, scored by pytest.

    Four things change against `run_dgm`'s surrogate mode -- the artifact (a file
    tree of source, not a capability set), `run`, `reward`, and `propose`. The
    parts that make it DGM do not: `DGMArchiveAggregator`'s keep-all archive, the
    `sigmoid(10*(s-0.5)) / (1+children)` parent draw, and the staged small->medium
    escalation are the same objects the surrogate mode uses.
    """
    import random as _random

    from agentdescent.evolution import Task, evolve
    from agentdescent.staleness import get_policy
    from agentdescent.treestrategy import FileTree

    from examples.dgm.dgm_self_improve import DGMContext

    cases = cases or load_tasks()
    if len(cases) < 4:
        raise ValueError("need at least four tasks for a train/held-out split")
    evaluate = make_real_evaluator(llm, concurrency=eval_concurrency)
    by_id = {c.id: c for c in cases}
    tasks = [Task(id=c.id, prompt=c.why, meta={"instance_id": c.id}) for c in cases]

    #: `rendered` is the agent's file tree; the aggregator scores whole agents,
    #: so the surrogate's `evaluate_fn(agent, instances)` signature is kept with
    #: `agent` now being `{path: source}`.
    ctx = DGMContext(rng=_random.Random(seed),
                     evaluate_fn=lambda agent, insts: evaluate(
                         agent, [by_id[i["instance_id"]] for i in insts]),
                     archive_mode="keep_all")

    # `editable` is the engine's own boundary on what a self-edit may touch, so
    # the agent's source is declared here rather than re-checked in the parser:
    # `to_diff` drops a path outside it, and a hostile one (`../..`) never
    # reaches `writable` because `safe_relpath` rejects it first.
    strategy = _FileTree(editable=list(EDITABLE))

    def run(rendered, task):
        """One rollout: this agent, this task, real pytest."""
        ok, detail = run_agent_on_task(parse_tree(rendered), by_id[task.meta["instance_id"]], llm)
        return f"{'resolved' if ok else 'unresolved'}: {detail}"

    def reward(task, output):
        return 1.0 if output.startswith("resolved") else 0.0

    def propose(rendered, task, output, score):
        """The self-modification. Conditioned on this agent's own run log."""
        files = parse_tree(rendered)
        report = [(task.meta["instance_id"], score >= 1.0, output)]
        proposal = propose_self_edit(files, report, llm)
        if not proposal:
            return None
        parsed = parse_self_edit(proposal)
        if parsed is None:
            return None
        path, body = parsed
        # `FileTree.to_diff` parses the shipped `<EDITS>` protocol -- a JSON
        # object of `{path: content}` -- not a rendered tree. Handing it a tree
        # returns `None` from `parse_edits`, silently, on every proposal: a
        # 13-rollout run then ends byte-identical to its seed with no warning
        # anywhere, which reads as "the search found nothing".
        return json.dumps({"edits": [{"path": path, "content": body}]})

    def factory(ledger, verifier, audit, config, policy):
        agg = SourceArchiveAggregator(ledger, verifier, ctx,
                                      artifact_id="coding_agent")
        agg.seed()          # measure the baseline before round 0, not on first card
        return agg

    result = evolve(
        tasks, reward, run=run, propose=propose, strategy=strategy,
        initial_state=dict(SEED_AGENT), blast_radius=0.6,
        artifact_id="coding_agent", rounds=generations,
        n_workers=selfimprove_size,
        max_concurrency=1 if asynchronous else selfimprove_size,
        asynchronous=asynchronous, async_ratio=async_ratio,
        max_seconds=max_seconds if asynchronous else None,
        held_out_frac=0.5, aggregator_factory=factory,
        staleness_policy=get_policy(staleness),
        eval_concurrency=eval_concurrency,
        verbose=verbose, max_rollouts=max_rollouts)
    # The `EvolutionResult` alone cannot answer "did the agent improve": the
    # seed's own score and the archive live on `ctx`, and returning only the
    # result leaves a run reporting a final number with nothing to compare it
    # against -- the same reporting gap ACE, EvoSkill and SkillOpt each had.
    result.dgm_seed_score = ctx.seed_score
    result.dgm_archive = ctx.archive
    return result


# ===========================================================================
# The archive, over source trees instead of capability sets
# ===========================================================================


@dataclass
class SourceAgent:
    """One DGM variant: the agent's own source, plus its lineage."""

    files: Dict[str, str]
    score: float = 0.0
    parent: Optional[int] = None
    children: int = 0
    generation: int = 0

    def key(self) -> str:
        import hashlib
        blob = "\0".join(f"{p}\0{self.files[p]}" for p in sorted(self.files))
        return hashlib.sha1(blob.encode()).hexdigest()[:16]


class SourceArchiveAggregator:
    """DGM's archive, reading a file tree where the shipped one reads a set.

    `DGMArchiveAggregator` builds `Agent(...)` straight from
    `state["capabilities"]` in three places, so it cannot be pointed at a
    different artifact -- the archive and the representation are coupled, which
    is only visible once you try. What is *not* coupled is everything that makes
    it DGM, and all three are reused here rather than reimplemented:

    * keep-all append (stepping stones stay, and under a real objective they can
      genuinely be worse -- which is the point);
    * `DGMParentSelection`, i.e. `sigmoid(10*(s-0.5)) / (1+children)`, at the
      standard selection seam;
    * `staged_evaluate`'s small -> medium ladder with `test_more_threshold=0.4`.
    """

    def __init__(self, ledger, verifier, ctx, artifact_id: str = "coding_agent",
                 selection=None) -> None:
        import threading

        from examples.dgm.dgm_self_improve import DGMParentSelection

        self.ledger = ledger
        self.verifier = verifier
        self.ctx = ctx
        self.aid = artifact_id
        self.selection = selection or DGMParentSelection(ctx.rng)
        self.cards: List = []
        self._lock = threading.Lock()
        self.head_index = 0
        self._seeded = False

    def seed(self) -> None:
        """Score the seed agent -- once, and *before* any card arrives.

        This lived inside `step()`, which only runs when a card reaches the
        merger. A run whose self-modifications all failed therefore never
        measured its own baseline, and `DGMContext.seed_score` defaults to 0.0 --
        so the report read `0.000 -> 0.844` for a seed that actually scored
        0.844 itself. The final number came from the engine and was right; only
        the thing it was compared against was invented. ADAS moved its seeding
        out of `step()` for the same reason, and EvoSkill and SkillOpt each
        shipped this exact bug.
        """
        if self._seeded:
            return
        from agentdescent.ledger import Ledger
        head = self.ledger.snapshot(Ledger.DEV).get(self.aid)
        seed = SourceAgent(dict(head.state), parent=None, generation=0)
        seed.score = self._staged(seed.files)
        self.ctx.archive.append(seed)
        self.ctx.seen.add(seed.key())
        self.ctx.seed_score = self.ctx.best_score = seed.score
        self.head_index, self._seeded = 0, True

    def _staged(self, files: Dict[str, str]) -> float:
        from examples.dgm.dgm_self_improve import (STAGE_MEDIUM, STAGE_SMALL,
                                                   staged_evaluate)
        insts = [{"instance_id": t.meta["instance_id"]}
                 for t in self.verifier.held_out]
        return staged_evaluate(files, self.ctx.evaluate_fn,
                               insts[:STAGE_SMALL], insts[:STAGE_MEDIUM], None)

    def ingest(self, card) -> None:
        with self._lock:
            self.cards.append(card)

    def step(self):
        from agentdescent.aggregator import MergeReport
        from agentdescent.evolvable import Diff
        from agentdescent.ledger import CASConflict, Ledger
        from agentdescent.selection import Candidate, SelectionContext

        snap = self.ledger.snapshot(Ledger.DEV)
        head = snap.get(self.aid)
        base_vv = {self.aid: snap.version.get(self.aid, 0)}
        self.seed()                                # no-op after construction

        parent_idx = self.head_index
        parent = self.ctx.archive[parent_idx]
        with self._lock:
            cards, self.cards = self.cards, []
        for card in cards:                        # each card is one self-modification
            child_files = dict(parent.files)
            child_files.update({p: c for p, c in card.diff.ops.items()
                                if c is not None})
            child = SourceAgent(child_files, parent=parent_idx,
                                generation=parent.generation + 1)
            if child.key() in self.ctx.seen:
                continue
            child.score = self._staged(child_files)
            self.ctx.archive.append(child)        # keep-all, worse ones included
            self.ctx.seen.add(child.key())
            self.ctx.best_score = max(self.ctx.best_score, child.score)
        parent.children += 1

        rows = [Candidate(artifact_id=self.aid, version=i, score=a.score,
                          selected=a.children)
                for i, a in enumerate(self.ctx.archive)]
        sel = SelectionContext(head=rows[self.head_index],
                               candidates=tuple(rows), n_workers=1)
        self.head_index = self.selection.select(sel, 1)[0].version
        target = self.ctx.archive[self.head_index].files
        committed = None
        if target != head.state:
            try:
                _, committed = self.ledger.commit(
                    head.apply(Diff(diff_id="dgm-select", target=self.aid,
                                    ops=dict(target), author="dgm")),
                    base_vv, branch=Ledger.DEV, message="dgm: select parent")
            except CASConflict:
                committed = None
        return [MergeReport(self.aid, None, False, len(cards), len(cards), 0, 0,
                            self.ctx.best_score, committed,
                            f"archive={len(self.ctx.archive)} "
                            f"best={self.ctx.best_score:.3f}")]
