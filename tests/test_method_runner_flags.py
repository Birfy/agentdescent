"""Flags this runner accepts, and whether it reads them.

`--reflective-merge` sat in the shared parser for all eleven MethodPolicy ports
and was never read: `run_port` used `policy.reflective`, a per-method constant.
A run could pass the flag, print `reflective_merge=True` because the *policy*
said so, and be byte-identical to a run that did not pass it.

That is not only a dead flag. A control experiment in this series varied exactly
that flag, found the two arms identical, and concluded the merge layer was not
the cause of a failure -- a refutation built on a no-op.

Three more were found the same way -- `--async-ratio`, `--eval-concurrency` and
`--eval-cache`, declared by `add_standard_args` and never passed to `run_port`
-- plus `--val-cap`, which these ports cannot honour at all because their splits
are frozen in `build()`. `--async-ratio` is the expensive one: every async row
in `bench/results/` records the value its command line asked for and ran at
`run_port`'s default of 1.

So this file no longer tests one flag. `test_every_declared_flag_is_honoured`
enumerates the parser and requires each flag to name where it is read, which
means the next flag added here cannot be decorative without a test failing.
"""
from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from examples import _method_runner as mr


def _args(**kw):
    base = dict(reflective_merge=False, no_reflective_merge=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _policy(reflective: bool):
    from examples.promptbreeder import promptbreeder_genetic_prompts as pb
    from dataclasses import replace
    return replace(pb.build(0), reflective=reflective)


def test_no_flag_keeps_the_methods_own_declaration():
    """`reflective` is a fidelity statement, not a knob: AFlow's contested
    workflow fields are model-merged because its optimizer rewrites a whole
    workflow, and Voyager's are not because its library overwrites a key."""
    assert mr._reflective_override(_args(), _policy(True)) is None
    assert mr._reflective_override(_args(), _policy(False)) is None


def test_the_flag_turns_it_on_and_the_negation_turns_it_off():
    assert mr._reflective_override(_args(reflective_merge=True), _policy(False)) is True
    assert mr._reflective_override(_args(no_reflective_merge=True), _policy(True)) is False


def test_both_flags_at_once_is_refused_rather_than_resolved():
    """A run that asked for the merge and against it has not said what it wants,
    and picking one is how a control arm ends up measuring the other."""
    with pytest.raises(SystemExit, match="contradictory"):
        mr._reflective_override(
            _args(reflective_merge=True, no_reflective_merge=True), _policy(True))


def test_run_port_reads_the_override_rather_than_the_policy_field():
    source = inspect.getsource(mr.run_port)
    assert "policy.reflective if reflective is None else" in source, (
        "the override is not threaded, so the flag is decorative again")
    assert "if policy.reflective:" not in source, (
        "the policy field is still deciding on its own")


def test_the_recorded_configuration_reports_what_actually_ran():
    """`reflective_merge` in the payload used to echo `policy.reflective`, so a
    run's own record could not distinguish an override from a declaration."""
    source = inspect.getsource(mr.run_port)
    assert '"reflective_merge": use_reflective' in source


def test_the_banner_says_when_it_was_overridden():
    source = inspect.getsource(mr.standard_main)
    assert "(overridden)" in source


# -- every flag the shared parser declares, and where it is read --------------


#: Declared flag -> the thing that honours it. A dest missing from this map is a
#: flag nobody has decided about; a dest here that no code reads is the defect
#: the whole file is about. Written from the parser rather than from prose,
#: because prose is what said `--gateless` worked while the parser rejected it.
HONOURED = {
    "provider": "completion_for",
    "model": "completion_for",
    "no_thinking": "completion_for",
    "seed": "build(args.seed) and run_port(seed=)",
    "asynchronous": "mode",
    "serial": "mode",
    "async_ratio": "run_port(async_ratio=)",
    "max_seconds": "run_port(max_seconds=)",
    "eval_concurrency": "run_port(eval_concurrency=)",
    "eval_cache": "run_port(eval_cache=)",
    "dry_run": "the early return",
    "yes": "confirm",
    "budget_rollouts": "mapped onto --candidates",
    "reflective_merge": "_reflective_override",
    "no_reflective_merge": "_reflective_override",
    "workers": "run_port(workers=)",
    "candidates": "run_port(candidate_budget=)",
    "staleness": "run_port(staleness=)",
    "temperature": "completion_for",
    "max_tokens": "completion_for",
    "timeout": "completion_for",
}


def _declared(extra_args=None):
    parser = mr.build_parser(extra_args)
    return {action.dest for action in parser._actions if action.dest != "help"}


def test_every_declared_flag_is_honoured():
    """The contract this file exists to keep: declaring is not honouring."""
    declared = _declared()
    assert declared - set(HONOURED) == set(), (
        "a flag reaches this runner with nowhere recorded that reads it")
    assert set(HONOURED) - declared == set(), (
        "this map names a flag the parser no longer declares")


def test_a_method_specific_flag_still_reaches_the_parser():
    """`extra_args` is Gödel Agent's `--gateless` seam, and it stays open."""
    from examples.godel_agent import godel_agent_self_modify as godel

    assert "gateless" in _declared(godel._gateless_flag)


def test_val_cap_is_refused_rather_than_accepted_and_ignored():
    """These ports freeze train/held-out/test in `build()`, before the parser is
    consulted, so there is no gate split left for a cap to trim. A flag that
    parses and moves nothing is worse than one that is not offered."""
    assert "val_cap" not in _declared()
    with pytest.raises(SystemExit):
        mr.build_parser().parse_args(["--val-cap", "8"])


def test_the_async_ratio_default_is_the_runners_own_rather_than_the_parsers():
    """1, not `add_standard_args`' 3, and the choice is the point.

    Until the flag was threaded, the parser's 3 reached nothing and every async
    run took `run_port`'s 1 -- including all fifteen rows in `bench/results/`
    whose config block records a number their command line only asked for.
    Adopting 3 here would have made fixing a dead flag silently change what
    `--async` does and make those rows irreproducible from the command line.
    """
    assert mr.DEFAULT_ASYNC_RATIO == 1
    assert mr.build_parser().parse_args([]).async_ratio == 1
    assert mr.build_parser().parse_args(["--async-ratio", "3"]).async_ratio == 3


def test_eval_concurrency_defaults_to_none_so_serial_keeps_its_control():
    """`--serial` is the control arm, and the published loop scores one task at
    a time. A plain default of 8 would make the control partly parallel, which
    is the confound `bench/matrix_run.py` documents for the other seven ports."""
    assert mr.build_parser().parse_args([]).eval_concurrency is None
    assert mr.build_parser().parse_args(["--eval-concurrency", "6"]).eval_concurrency == 6


# -- the flags reach the engine, not only the namespace ----------------------


class _Outcome(SimpleNamespace):
    """Just enough of a `MethodRunResult` for `standard_main`'s closing print."""

    def __init__(self):
        super().__init__(
            baseline_quality=0.0, final_quality=0.0,
            baseline_validation_quality=0.0, validation_quality=0.0,
            accepted=0, candidates=0, invalid_candidates=0,
            wall_seconds=0.0, engine_wall_seconds=0.0,
            model_usage={"calls": 0}, budget={"matched": True})


def _run_port_kwargs(monkeypatch, argv):
    """What `standard_main` would hand `run_port` for this command line."""
    from examples.promptbreeder import promptbreeder_genetic_prompts as pb

    seen = {}

    def fake_run_port(policy, recorder, **kwargs):
        seen.update(kwargs)
        return _Outcome()

    monkeypatch.setattr(mr, "run_port", fake_run_port)
    monkeypatch.setattr(mr, "completion_for",
                        lambda *a, **kw: (lambda prompt: ""))
    assert mr.standard_main(pb.build, ["--yes", *argv]) == 0
    return seen


def test_the_three_dropped_flags_reach_run_port(monkeypatch, tmp_path):
    """The regression itself: `standard_main` parsed all three and passed none,
    so a run that set every one of them was byte-identical to a run that set
    none."""
    seen = _run_port_kwargs(monkeypatch, [
        "--async", "--async-ratio", "4",
        "--eval-concurrency", "6",
        "--eval-cache", str(tmp_path)])
    assert seen["async_ratio"] == 4
    assert seen["eval_concurrency"] == 6
    assert seen["eval_cache"] == str(tmp_path)
    assert seen["mode"] == "async_pipeline"


def test_an_unset_flag_stays_unset_rather_than_becoming_a_default(monkeypatch):
    seen = _run_port_kwargs(monkeypatch, [])
    assert seen["async_ratio"] == 1
    assert seen["eval_concurrency"] is None
    assert seen["eval_cache"] == ""


def _spy_run(monkeypatch, *, mode, **kwargs):
    """One real offline `run_port`, with the engine entry point wiretapped."""
    from agentdescent.agents import Usage, metered
    from examples._measure import Recorder
    from examples.promptbreeder import promptbreeder_genetic_prompts as pb

    reply = json.dumps({"task_prompt": "State only the final number.",
                        "mutation_prompt": "Rewrite the instruction."})
    usage = Usage()
    recorder = Recorder(metered(lambda prompt: reply, usage), usage)

    name = "async_evolve" if mode == "async_pipeline" else "evolve"
    real = getattr(mr, name)
    seen = {}

    def spy(*args, **kw):
        seen.update(kw)
        return real(*args, **kw)

    monkeypatch.setattr(mr, name, spy)
    result = mr.run_port(pb.build(0), recorder, mode=mode, seed=0, workers=2,
                         candidate_budget=2, max_seconds=30.0,
                         shutdown_grace=5.0, **kwargs)
    return seen, result


def test_async_ratio_reaches_the_barrier_free_runtime(monkeypatch):
    """It is a real lag budget, not a label: it bounds both how far a worker's
    snapshot may drift and how many cards may sit un-merged ahead of it."""
    seen, result = _spy_run(monkeypatch, mode="async_pipeline", async_ratio=4)
    assert seen["async_ratio"] == 4
    assert result.framework["async_ratio"] == 4


def test_async_ratio_is_recorded_only_where_it_applies(monkeypatch):
    """A lag budget means nothing with a round barrier in the way, and a number
    printed anyway is one a reader would compare across arms."""
    _, result = _spy_run(monkeypatch, mode="sync_parallel", async_ratio=4)
    assert result.framework["async_ratio"] is None


def test_eval_concurrency_reaches_the_gate(monkeypatch):
    seen, result = _spy_run(monkeypatch, mode="sync_parallel",
                            eval_concurrency=5)
    assert seen["eval_concurrency"] == 5
    assert result.framework["eval_concurrency"] == 5


def test_without_the_flag_the_runners_own_rule_still_applies(monkeypatch):
    seen, _ = _spy_run(monkeypatch, mode="sync_parallel")
    assert seen["eval_concurrency"] == 2, "the worker count, as before"
    seen, _ = _spy_run(monkeypatch, mode="serial")
    assert seen["eval_concurrency"] == 1, "the control scores one at a time"


def test_an_explicit_eval_concurrency_overrides_serial_too(monkeypatch):
    """`bench/matrix_run.py` sets the control's gate concurrency deliberately
    (`--serial-eval-concurrency`), and that value has to arrive."""
    seen, _ = _spy_run(monkeypatch, mode="serial", eval_concurrency=4)
    assert seen["eval_concurrency"] == 4


def test_a_nonsense_gate_concurrency_is_refused():
    from examples.promptbreeder import promptbreeder_genetic_prompts as pb

    with pytest.raises(ValueError, match="eval_concurrency"):
        mr.run_port(pb.build(0), None, mode="serial", seed=0, workers=2,
                    candidate_budget=2, eval_concurrency=0)


def test_eval_cache_installs_a_file_cache_on_the_methods_own_bundle(
        monkeypatch, tmp_path):
    """Merged onto the bundle rather than built as a fresh `Policies`: the
    method's declared selection/acceptance fields travel in that same bundle,
    and `_common.eval_cache_kwargs` would have replaced them."""
    from agentdescent.evalcache import FileCache

    seen, result = _spy_run(monkeypatch, mode="sync_parallel",
                            eval_cache=str(tmp_path))
    bundle = seen["policies"]
    assert isinstance(bundle.eval_cache, FileCache)
    assert bundle.eval_cache.directory == str(tmp_path)
    assert result.framework["eval_cache"] is True
    assert list(tmp_path.glob("*.json")), "the gate was not memoised to disk"


def test_no_cache_directory_leaves_the_in_process_default(monkeypatch):
    """Off by default is not timidity: a persistent cache makes a rerun return
    the first run's numbers, which is wrong whenever the question is spread."""
    seen, result = _spy_run(monkeypatch, mode="sync_parallel")
    assert seen["policies"].eval_cache is None
    assert result.framework["eval_cache"] is False


# -- the source-level guards, for what a behaviour test cannot see ------------


def test_run_port_reads_the_flags_rather_than_recomputing_them():
    """Mirrors `test_run_port_reads_the_override_rather_than_the_policy_field`:
    a hardcoded expression back in place of a parameter is exactly how each of
    these started."""
    source = inspect.getsource(mr.run_port)
    assert '"eval_concurrency": gate_concurrency' in source, (
        "the gate concurrency is hardcoded again, so the flag is decorative")
    assert '"eval_concurrency": 1 if mode == "serial" else workers' not in source
    assert "async_ratio=async_ratio" in source, (
        "the lag budget is not threaded into async_evolve")


def test_standard_main_threads_each_flag_it_parses():
    source = inspect.getsource(mr.standard_main)
    for fragment in ("async_ratio=args.async_ratio",
                     "eval_concurrency=args.eval_concurrency",
                     "eval_cache=args.eval_cache"):
        assert fragment in source, f"{fragment} is not threaded into run_port"


def test_the_plan_states_what_the_live_run_would_do():
    """`--dry-run` is the only way to see a configuration without paying for it,
    and a flag absent from the plan is a flag nobody checks."""
    source = inspect.getsource(mr.standard_main)
    assert "eval_concurrency={gate_concurrency}" in source
    assert "async_ratio={args.async_ratio}" in source
    assert "eval_cache=on" in source


# -- clip_text: the shared truncation every port's artifact goes through ------


def test_clip_text_truncates_rather_than_discards():
    """It used to read `if not cleaned or len(cleaned) > max_len: return
    fallback` -- a function named `clip_text` that drops a 901-character answer
    and keeps a 900-character one.

    The cost lands twice: the proposal is lost, and it is counted as *invalid*,
    which reads in the metrics as the model producing junk. Measured on R-Zero,
    whose two update prompts ask for a policy statement and get one -- four of
    six replies ran 977-1632 characters, both fields came back empty, and the
    runs reported `invalid` of 41, 44 and 48 out of 80, against 2-11 for the
    ports whose prompts ask for a single sentence.
    """
    from examples._method_policy import clip_text

    long = "word " * 400
    out = clip_text(long)
    assert out, "a long but perfectly good answer was discarded"
    assert len(out) <= 900
    assert out.startswith("word word")

    # Truncation lands on a word boundary when one is near the limit.
    assert not out.endswith("wor")
    # ...and still returns something when there is no boundary to find.
    assert len(clip_text("x" * 2000)) == 900


def test_clip_text_still_rejects_what_is_actually_empty():
    from examples._method_policy import clip_text

    assert clip_text("") == ""
    assert clip_text("   \n  ") == ""
    assert clip_text(None) == ""
    assert clip_text(["not", "a", "string"]) == ""
    assert clip_text("", fallback="seed") == "seed"


def test_a_long_field_proposal_becomes_a_diff_rather_than_an_invalid_count():
    """The end-to-end consequence: `FieldSlots` counted a too-long value as an
    invalid proposal, so a run's `invalid` number measured verbosity."""
    import json

    from examples._measure import parse_json_object
    from examples._method_policy import FieldSlots

    slots = FieldSlots(fields={"memory": "seed"}, parse=parse_json_object)
    proposal = json.dumps({"memory": "sentence. " * 200})
    diff = slots.to_diff({"memory": "seed"}, proposal, "w", 1, "artifact")
    assert diff is not None, "a verbose but valid proposal was rejected"
    assert slots.invalid_proposals == 0
    assert len(diff.ops["memory"]) <= 900
