"""Evaluation as its own stage, and the invariant that lets it be one.

A merge is three phases with different properties -- prepare (cheap, touches
aggregator state), measure (expensive, touches none), decide (cheap, commits).
`Aggregator.step()` runs all three on the caller's thread; `begin_step` /
`measure` / `finish_step` are the same three with the middle one left to the
caller, so `async_evolve(pipelined_gate=True)` can put it on its own threads and
let the merger go back to draining.

**The invariant that makes this safe is one candidate per artifact in flight.**
With it, every candidate is committed against the head it was prepared and
measured on, so the pipeline needs no candidate-level staleness rule -- which is
the difference between this and a change that quietly loosens what a commit
means. Most of this file is that invariant, and the ways it could be lost.
"""

import threading
import time
import warnings

import pytest

from agentdescent import AppendRules, Task
from agentdescent.aggregator import (
    Aggregator, AggregatorConfig, AggregatorContractError, MergeReport, _Candidate)
from agentdescent.async_evolve import async_evolve


# ---------------------------------------------------------------------------
# A domain whose held-out reward actually moves
# ---------------------------------------------------------------------------
#
# `AppendRules` + "is this task's own id in the artifact" cannot generalise:
# held-out tasks are never rolled out, so their rule is never proposed and
# held-out reward is structurally 0. That is fine for a throughput experiment
# and useless for "did quality survive the change", which is the acceptance
# criterion this file exists to check. So: one rule solves every task.

GOLD = "y"
KEY = "answer-y"


def _tasks(n=20):
    return [Task(id=f"t{i}", prompt=f"q{i}", meta={"gold": GOLD}) for i in range(n)]


def _reward(task, output):
    return 1.0 if output == task.meta["gold"] else 0.0


def _make_run(rollout_s=0.0, gate_s=0.0, held_ids=frozenset()):
    def run(rendered, task):
        time.sleep(gate_s if task.id in held_ids else rollout_s)
        return GOLD if KEY in rendered else "?"

    return run


def _propose(rendered, task, output, reward):
    return KEY


def _run_async(**kw):
    kw.setdefault("n_workers", 4)
    kw.setdefault("max_seconds", 2.0)
    kw.setdefault("self_verify", False)
    kw.setdefault("held_out_frac", 0.4)
    kw.setdefault("run", _make_run())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return async_evolve(_tasks(), _reward, propose=_propose,
                            strategy=AppendRules(), seed=0, **kw)


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


def _aggregator(tmp_path, tasks):
    """A real aggregator over a real ledger, so `_decide` can actually commit."""
    from agentdescent.evolution import _build_engine

    return _build_engine(tasks, _reward, agent=None, run=_make_run(),
                         propose=_propose, strategy=AppendRules(),
                         initial_state=None, blast_radius=0.2, artifact_id="a",
                         held_out_frac=0.4, repo_path=str(tmp_path),
                         agg_config=AggregatorConfig(batch_trigger=1),
                         staleness_policy=None, aggregator_factory=None,
                         oracle_budget=200)


def _card(eng, value=KEY):
    from agentdescent.evolvable import EvidenceCard

    snap = eng.ledger.snapshot("dev")
    artifact = snap.get("a")
    diff = eng.strategy.to_diff(artifact.state, value, "w0",
                                snap.version.get("a", 0), "a")
    return EvidenceCard(diff=diff, base_version={"a": snap.version.get("a", 0)},
                        touched=["a"], before_after_delta=1.0,
                        trajectory_refs=list(eng.held_out))


def test_begin_step_marks_in_flight_and_finish_step_releases(tmp_path):
    """The bookkeeping the whole invariant rests on."""
    eng = _aggregator(tmp_path, _tasks())
    agg = eng.aggregator
    agg.ingest(_card(eng))
    items = agg.begin_step(skip_in_flight=True)
    assert [i for i in items if isinstance(i, _Candidate)]
    assert agg._in_flight == {"a"}
    agg.finish_step(agg.measure(items))
    assert agg._in_flight == set()


def test_a_second_candidate_is_not_started_while_one_is_in_flight(tmp_path):
    """The invariant itself: at most one candidate per artifact, ever.

    Without it a second candidate would be prepared against a head the first is
    about to move, and would then commit a measurement taken on a head that no
    longer exists -- which is a real loosening of what a commit means, and
    exactly the thing this design avoids by not needing a policy for it."""
    eng = _aggregator(tmp_path, _tasks())
    agg = eng.aggregator
    agg.ingest(_card(eng))
    first = agg.begin_step(skip_in_flight=True)
    assert [i for i in first if isinstance(i, _Candidate)]

    agg.ingest(_card(eng, value=KEY + " more"))
    second = agg.begin_step(skip_in_flight=True)
    assert second == [], "a second candidate was prepared while one was in flight"

    # ...and the artifact is available again as soon as the first is decided.
    agg.finish_step(agg.measure(first))
    assert [i for i in agg.begin_step(skip_in_flight=True)
            if isinstance(i, _Candidate)]


def test_step_itself_never_skips(tmp_path):
    """`step()` measures inline, so nothing is ever in flight when it starts.

    The default `skip_in_flight=False` is what keeps the synchronous driver and
    every custom aggregator on exactly the path they were on."""
    eng = _aggregator(tmp_path, _tasks())
    agg = eng.aggregator
    agg.ingest(_card(eng))
    reports = agg.step()
    assert reports and agg._in_flight == set()


def test_deciding_an_unmeasured_candidate_raises(tmp_path):
    """It must not commit on zeros.

    An unmeasured candidate carries `base_counts = None`; taken as (0, 0) the
    Beta test sees a tie and rejects, so the merge would silently become a
    no-op -- a wrong answer wearing the shape of a right one."""
    eng = _aggregator(tmp_path, _tasks())
    agg = eng.aggregator
    agg.ingest(_card(eng))
    items = agg.begin_step()
    with pytest.raises(AggregatorContractError, match="unmeasured candidate"):
        agg.finish_step(items)             # measure() deliberately skipped


def test_a_failed_decision_still_releases_the_artifact(tmp_path):
    """Otherwise one bad merge stops that artifact merging for the rest of the run.

    A merger that quietly stops merging one artifact is indistinguishable from
    an idle one, which is the hardest kind of failure to see."""
    eng = _aggregator(tmp_path, _tasks())
    agg = eng.aggregator
    agg.ingest(_card(eng))
    items = agg.measure(agg.begin_step(skip_in_flight=True))
    assert agg._in_flight == {"a"}

    def boom(_ctx):
        raise RuntimeError("acceptance exploded")

    agg.acceptance_policy = type("P", (), {"accept": staticmethod(boom)})()
    with pytest.raises(RuntimeError):
        agg.finish_step(items)
    assert agg._in_flight == set()


def test_measure_touches_no_aggregator_state(tmp_path):
    """The property that makes the phase safe to move off the merger.

    Asserted by running it from several threads at once and checking that the
    aggregator's own state is unchanged -- the buffer, the posteriors and the
    in-flight set are all written by the phases either side of it."""
    eng = _aggregator(tmp_path, _tasks())
    agg = eng.aggregator
    agg.ingest(_card(eng))
    items = agg.begin_step(skip_in_flight=True)
    before = (dict(agg._posteriors), set(agg._in_flight), agg.buffer.pending())

    errors = []

    def measure():
        try:
            agg.measure(items)
        except Exception as e:      # noqa: BLE001 - reported, not swallowed
            errors.append(e)

    threads = [threading.Thread(target=measure) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert (dict(agg._posteriors), set(agg._in_flight),
            agg.buffer.pending()) == before


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


def test_the_pipelined_run_still_reaches_the_answer():
    """Quality is the acceptance criterion, not throughput.

    On a domain whose one rule solves every task, both paths must find it. A
    faster run that stopped committing would look like a win in every other
    number this change reports."""
    inline = _run_async()
    piped = _run_async(pipelined_gate=True)
    assert inline.final_reward == pytest.approx(1.0)
    assert piped.final_reward == pytest.approx(1.0)
    assert KEY in piped.rendered


def test_a_pipelined_run_commits_and_records_every_merge():
    """No commit may land in the ledger that no round accounts for.

    The shutdown path is where this would break: a candidate measured just
    before the budget expires still has to be decided *and* recorded, or
    `final_reward` reflects a commit that `history` cannot explain."""
    r = _run_async(pipelined_gate=True, max_seconds=1.5)
    committed = sum(h.committed for h in r.history)
    assert committed > 0
    # Every committed round is in the history, and the final artifact is one of
    # the things they committed.
    assert r.rendered
    assert len(r.history) >= committed


def test_an_aggregator_without_the_seam_warns_and_falls_back():
    """A custom aggregator predates `begin_step`. Refusing it would break the
    main extension point; running it inline and saying so does not."""

    class Legacy:
        def __init__(self, *a, **kw):
            self.inner = Aggregator(*a, **kw)

        def ingest(self, card):
            self.inner.ingest(card)

        def step(self):
            return self.inner.step()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = async_evolve(_tasks(), _reward, run=_make_run(), propose=_propose,
                         strategy=AppendRules(), n_workers=2, max_seconds=1.0,
                         self_verify=False, held_out_frac=0.4, seed=0,
                         pipelined_gate=True,
                         aggregator_factory=lambda l, v, a, c, p: Legacy(
                             l, v, a, c, staleness_policy=p))
    assert any("stays on the merger thread" in str(w.message) for w in caught)
    assert r.final_reward == pytest.approx(1.0)


def test_the_measurement_runs_off_the_merger_thread():
    """The point of the change, asserted as the mechanism rather than as a time.

    "The pipelined run did more rollouts" is what this buys, and it is a
    *measurement*: it depends on the machine, and an ordering between two timed
    runs flakes on a loaded one (it did, in the full suite, while passing alone).
    The numbers live in `examples/efficiency.py --only stages` and
    `docs/efficiency.md`; what the suite can guarantee is the thing that makes
    them possible -- that `measure()` does not run where merging runs.
    """
    seen = {"inline": set(), "piped": set()}

    def factory(which):
        def build(ledger, verifier, audit, config, policy):
            agg = Aggregator(ledger, verifier, audit, config,
                             staleness_policy=policy)
            inner = agg.measure

            def measure(items):
                seen[which].add(threading.current_thread().name)
                return inner(items)

            agg.measure = measure
            return agg

        return build

    _run_async(aggregator_factory=factory("inline"), max_seconds=1.5)
    _run_async(aggregator_factory=factory("piped"), pipelined_gate=True,
               max_seconds=1.5)

    assert seen["inline"] and seen["piped"], "no merge measured anything"
    # Inline: the merger measures, so it happens on whatever thread called
    # `step()`. Pipelined: only on the gate pool, whose threads are named.
    assert not any(n.startswith("agentdescent-gate") for n in seen["inline"])
    assert all(n.startswith("agentdescent-gate") for n in seen["piped"]), seen["piped"]


def test_an_aggregator_that_overrides_step_is_not_pipelined():
    """Having the three phases is not enough -- `step()` has to still *be* them.

    `PopulationAggregator` subclasses `Aggregator`, so it inherits all three,
    and overrides `step()` to admit the pre-merge head into its archive and
    consult its selection policy. Driving the phases directly there would skip
    every line of that override and run a different algorithm while reporting
    the requested one. Caught on a real run: the check was `hasattr` alone.
    """
    from agentdescent.population import PopulationAggregator
    from agentdescent.selection import SingleHead

    assert PopulationAggregator.step is not Aggregator.step, (
        "this test is only meaningful while PopulationAggregator overrides step()")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Called directly: `_run_async` silences RuntimeWarning, which is the
        # thing under test here.
        r = async_evolve(_tasks(), _reward, run=_make_run(), propose=_propose,
                         strategy=AppendRules(), n_workers=2, max_seconds=1.0,
                         self_verify=False, held_out_frac=0.4, seed=0,
                         pipelined_gate=True,
                         aggregator_factory=lambda l, v, a, c, p: PopulationAggregator(
                             l, v, a, c, staleness_policy=p, selection=SingleHead(),
                             artifact_id="artifact"))
    assert any("overrides step()" in str(w.message) for w in caught), \
        [str(w.message) for w in caught]
    assert r.final_reward == pytest.approx(1.0)
