"""How a recorded row says it was produced, and why fifteen of them could not.

Every `bench/results/*.json` written from a MethodPolicy port has two halves: a
`cells` list of numbers, and a `config` block beside it. The numbers were
transcribed from the line `standard_main` prints. The config block was typed
from memory, because **the run printed nothing about its own setup** -- and
fifteen blocks then recorded `async_ratio: 2` for runs that took the runner's
default of 1, because `--async-ratio` was dropped before it reached the runtime
and no output contradicted the number the author had asked for.

The attribution is not a guess. Three things pin it:

* **Precision.** `standard_main` prints qualities at `.3f`, seconds at `.1f` and
  calls as an int. Across all 45 cells in those fifteen files, not one value
  carries more precision than that. `bench.candidate_methods` -- the only other
  caller of `run_port`, and the one that *does* pass `async_ratio` -- writes
  `MethodRunResult.compact()` straight to JSON: in the one file it produced,
  198 of 198 wall/engine values are full floats (9-14 decimals).
* **Schema.** `test`/`validation` pairs, `accepted`, `invalid`, `wall_s`,
  `engine_s`, `calls` are field-for-field that print line. `compact()` uses
  different names and carries `phase_summary`, `events` and `framework` too.
* **`staleness: full`.** `bench.candidate_methods` has no `--staleness` and
  never passes `staleness=` to `run_port`, so every run through it is `guarded`.
  Only the command line can produce `full`, and the commit messages introducing
  these rows describe them with exactly that flag.

So the fix is not a better guess: the run now states its own configuration, in
the key names those blocks use, on a line that can be copied instead of retyped.
"""
from __future__ import annotations

import glob
import json
import os
from decimal import Decimal
from types import SimpleNamespace

import pytest

from examples import _method_runner as mr


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _results():
    for path in sorted(glob.glob(os.path.join(ROOT, "bench", "results", "*.json"))):
        with open(path, encoding="utf-8") as handle:
            yield path, json.load(handle)


def _method_policy_rows():
    """The rows a MethodPolicy port produced: an arm and a rollout budget."""
    for path, data in _results():
        config = data.get("config") or {}
        if "arm" in config and "budget_rollouts" in config:
            yield path, data, config


def _places(value) -> int:
    exponent = Decimal(str(value)).as_tuple().exponent
    return -exponent if exponent < 0 else 0


def _args(**kw):
    base = dict(seed=0, candidates=80, workers=8, async_ratio=1,
                staleness="full", eval_concurrency=None, eval_cache="",
                temperature=0.7, no_thinking=True, provider="claude",
                model="deepseek-v4-flash")
    base.update(kw)
    return SimpleNamespace(**base)


def _policy():
    from examples.promptbreeder import promptbreeder_genetic_prompts as pb

    return pb.build(0)


# -- the run states its own configuration ------------------------------------


def test_run_config_states_every_field_a_results_block_records():
    config = mr.run_config(_args(), _policy(), "async_pipeline")
    missing = [f for f in mr.CONFIG_FIELDS if f not in config]
    assert not missing, f"a run cannot state {missing}, so a block must guess it"


def test_no_results_block_records_a_runner_field_the_run_cannot_state():
    """A block key that names a runner setting must be one `run_config` emits.

    Anything else is method-specific (`alpha`, `solver_samples`) or editorial
    (`domain`, `*_note`) and belongs to whoever wrote the file -- but a *runner*
    setting recorded here and unstateable there is the defect returning.
    """
    runner_owned = set(mr.CONFIG_FIELDS) | {
        # emitted under a different name, or by the engine-counter line
        "arm", "budget_rollouts",
    }
    editorial = {"domain", "selection", "memory", "proposal_calls_per_candidate"}
    for path, _data, config in _method_policy_rows():
        for key in config:
            if key.endswith("_note") or key in editorial:
                continue
            if key in runner_owned:
                continue
            # a method-specific knob: lower case, and not a runner concept
            assert key not in {"eval_concurrency", "eval_cache", "async_ratio",
                               "staleness", "reflective_merge", "self_verify"}, (
                f"{path}: {key} is a runner setting recorded outside CONFIG_FIELDS")


def test_the_live_run_prints_a_copyable_config_line(monkeypatch, capsys):
    """`--dry-run` printing the plan is not enough: the block sits beside numbers
    a *live* run produced, so the live run is what has to state it."""
    from examples.promptbreeder import promptbreeder_genetic_prompts as pb

    class _Outcome(SimpleNamespace):
        def __init__(self):
            super().__init__(
                baseline_quality=0.0, final_quality=0.0,
                baseline_validation_quality=0.0, validation_quality=0.0,
                accepted=0, candidates=0, invalid_candidates=0,
                wall_seconds=0.0, engine_wall_seconds=0.0,
                model_usage={"calls": 0}, budget={"matched": True})

    monkeypatch.setattr(mr, "run_port", lambda policy, recorder, **kw: _Outcome())
    monkeypatch.setattr(mr, "completion_for", lambda *a, **kw: (lambda p: ""))
    assert mr.standard_main(pb.build, [
        "--yes", "--async", "--async-ratio", "2", "--staleness", "full",
        "--workers", "8", "--budget-rollouts", "80", "--temperature", "0.7",
        "--no-thinking", "--provider", "claude", "--model", "deepseek-v4-flash",
    ]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if ln.startswith("config: ")]
    assert len(lines) == 1, "the live run printed no configuration"
    config = json.loads(lines[0][len("config: "):])
    assert config["async_ratio"] == 2
    assert config["staleness"] == "full"
    assert config["budget_rollouts"] == 80
    assert config["workers"] == 8
    assert config["arm"] == "async_pipeline"


def test_a_lag_budget_is_not_claimed_where_it_cannot_apply():
    assert mr.run_config(_args(), _policy(), "sync_parallel")["async_ratio"] is None
    assert mr.run_config(_args(), _policy(), "serial")["async_ratio"] is None


def test_thinking_is_not_claimed_disabled_on_the_path_that_drops_it():
    """`completion_for` silently ignores `--no-thinking` on the OpenAI-compatible
    path rather than guessing a per-vendor equivalent. A run that reported it
    disabled there would attribute the wall-clock to the wrong thing."""
    args = _args(provider="openai", no_thinking=True)
    assert mr.run_config(args, _policy(), "serial")["thinking"] == "default"
    assert mr.run_config(_args(no_thinking=False), _policy(),
                         "serial")["thinking"] == "default"


def test_the_run_records_which_staleness_policy_it_had(monkeypatch):
    """The one field that made those fifteen blocks unattributable: a block
    saying `full` cannot have come from `bench.candidate_methods`, and the run's
    own record used to be silent either way."""
    from agentdescent.agents import Usage, metered
    from examples._measure import Recorder

    reply = json.dumps({"task_prompt": "State only the final number.",
                        "mutation_prompt": "Rewrite the instruction."})
    usage = Usage()
    recorder = Recorder(metered(lambda prompt: reply, usage), usage)
    result = mr.run_port(_policy(), recorder, mode="sync_parallel", seed=0,
                         workers=2, candidate_budget=2, staleness="full")
    assert result.framework["staleness"] == "full"


# -- the forensic invariant, kept executable ---------------------------------


def test_every_methodpolicy_row_carries_the_precision_of_the_line_it_came_from():
    """These cells were transcribed from `standard_main`'s stdout, and that is
    checkable rather than assumed: `.3f` qualities, `.1f` seconds, integer calls.

    A future file whose cells carry more came from somewhere else, and its config
    block cannot be read as describing a command-line run -- which is exactly the
    inference that went wrong here.
    """
    rows = list(_method_policy_rows())
    assert rows, "no MethodPolicy results rows found"
    total = 0
    for path, data, _config in rows:
        for cell in data.get("cells", []):
            total += 1
            for key in ("test", "validation"):
                for value in cell.get(key, []):
                    assert _places(value) <= 3, f"{path}: {key}={value}"
            for key in ("wall_s", "engine_s"):
                if key in cell:
                    assert _places(cell[key]) <= 1, f"{path}: {key}={cell[key]}"
            if "calls" in cell:
                assert float(cell["calls"]).is_integer(), path
    assert total >= 45


def test_the_other_runner_writes_full_precision_which_is_the_contrast():
    """The control. `bench.candidate_methods` JSON-dumps `compact()` unrounded,
    so its file is distinguishable from a transcribed one at a glance -- which is
    what rules it out as the source of the fifteen."""
    path = os.path.join(ROOT, "bench", "results",
                        "candidate-methods-framework-final.json")
    if not os.path.exists(path):
        pytest.skip("the candidate_methods run is not in this checkout")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    observations = data.get("observations") or []
    assert observations, "the control file has no observations"
    unrounded = sum(1 for row in observations
                    for key in ("wall_seconds", "engine_wall_seconds")
                    if key in row and _places(row[key]) > 1)
    assert unrounded > len(observations), (
        "this file no longer looks like compact() output, so it is no longer "
        "the contrast the provenance argument rests on")
