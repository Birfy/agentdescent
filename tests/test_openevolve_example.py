"""Offline contract tests for the OpenEvolve AgentDescent port."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from bench import openevolve_agentdescent as bench
from examples.openevolve import _openevolve_support as evaluator
from examples.openevolve import openevolve_program_evolution as port


def test_source_gate_accepts_genome_and_rejects_unsafe_code():
    assert evaluator.validate_source(evaluator.INITIAL_PROGRAM)[0]
    assert not evaluator.validate_source(
        "import os\ndef search_algorithm(*args): return (0, 0)"
    )[0]
    assert not evaluator.validate_source(
        "def search_algorithm(*args):\n    return (-1.704, 0.678)\n"
    )[0]
    assert not evaluator.validate_source(
        "def search_algorithm(*args):\n    return open('/etc/passwd').read()\n"
    )[0]


def test_combined_metric_matches_pinned_upstream_fixture():
    trials = [
        {
            "success": True,
            "x": evaluator.GLOBAL_MIN_X,
            "y": evaluator.GLOBAL_MIN_Y,
            "seconds": 0.01,
            "objective_calls": 100,
        }
        for _ in range(10)
    ]
    metrics = evaluator.combined_metrics(trials)
    # OpenEvolve 411fb59 examples/function_minimization/evaluator.py:190-215.
    assert metrics["combined_score"] == pytest.approx(1.4997641483797084)
    assert metrics["distance_score"] == 1.0
    assert metrics["reliability_score"] == 1.0


def test_map_elites_islands_migrate_and_respect_candidate_cap():
    archive = port.OpenEvolveArchive(
        archive_size=8,
        num_islands=2,
        feature_bins=4,
        exploitation_ratio=1.0,
        migration_interval=2,
        candidate_limit=3,
    )

    def add(code, iteration, island, score, *, baseline=False):
        archive.add_program(
            evaluator.Program(
                evaluator.program_id(code),
                iteration,
                island,
                None,
                code,
                "fixture",
                {"combined_score": score},
                True,
            ),
            baseline=baseline,
        )

    add(evaluator.INITIAL_PROGRAM, 0, 0, 0.1, baseline=True)
    add(evaluator.INITIAL_PROGRAM + "\n# island one\n", 1, 1, 0.2)
    add(evaluator.INITIAL_PROGRAM + "\n# stronger island zero\n", 2, 0, 0.3)

    assert archive.migrations >= 1
    selections = [archive.select_parent() for _ in range(3)]
    assert [selection[1] for selection in selections] == [0, 1, 0]
    assert archive.select_parent() is None


@pytest.mark.skipif(evaluator.sandbox_backend() is None,
                    reason="no candidate isolation backend on this host")
def test_initial_program_runs_deterministically_in_sandbox():
    """Gated on `bwrap` until the Seatbelt backend existed, which meant the new
    backend had no test that actually executed anything through it -- the one
    test that runs a candidate end to end skipped on the only platform the new
    code path serves."""
    first = evaluator.evaluate_source(
        evaluator.INITIAL_PROGRAM, trials=3, budget=40, seed=7, timeout=5.0
    )
    second = evaluator.evaluate_source(
        evaluator.INITIAL_PROGRAM, trials=3, budget=40, seed=7, timeout=5.0
    )
    assert first[0] and second[0]
    assert first[1]["avg_value"] == second[1]["avg_value"]
    assert first[1]["avg_distance"] == second[1]["avg_distance"]
    assert first[1]["avg_objective_calls"] == 40


def test_dry_run_needs_no_key_network_or_sandbox(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(port, "evaluate_source", lambda *a, **k: pytest.fail("sandbox used"))
    assert port.main(["--dry-run", "--model", "glm-5.2"]) == 0
    output = capsys.readouterr().out
    assert "mode=sync" in output
    assert "no API, dataset, or sandbox" in output


def test_benchmark_dry_run_reports_reserved_calls_without_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert bench.main(["--dry-run", "--repeats", "2", "--iterations", "6"]) == 0
    assert "reserved model calls=36" in capsys.readouterr().out


def test_benchmark_speedups_pair_matching_repeats_and_exclude_missed_targets():
    rows = []
    for repeat, serial_wall, sync_wall, serial_ttq, sync_ttq in (
        (1, 12.0, 6.0, 8.0, 4.0),
        (2, 30.0, 10.0, 9.0, None),
    ):
        for mode, wall, ttq in (
            ("serial", serial_wall, serial_ttq),
            ("sync", sync_wall, sync_ttq),
        ):
            rows.append(
                {
                    "repeat": repeat,
                    "seed": repeat * 100,
                    "mode": mode,
                    "end_to_end_seconds": wall,
                    "time_to_quality_s": ttq,
                }
            )

    wall = bench._paired_speedup(
        rows,
        baseline_mode="serial",
        comparison_mode="sync",
        metric="end_to_end_seconds",
    )
    ttq = bench._paired_speedup(
        rows,
        baseline_mode="serial",
        comparison_mode="sync",
        metric="time_to_quality_s",
    )
    assert wall == {"paired_runs": 2, "speedup_min_median_max": [2.0, 2.5, 3.0]}
    assert ttq == {"paired_runs": 1, "speedup_min_median_max": [2.0, 2.0, 2.0]}


def test_agentdescent_serial_sync_and_async_run_real_port_offline():
    improved = (
        '"""improved"""\n'
        "def search_algorithm(objective, budget, rng, bounds):\n"
        "    return (-1.7, 0.68)\n"
    )

    calls = {"count": 0}

    def complete(prompt):
        calls["count"] += 1
        iteration = re.search(r"Iteration: (\d+)", prompt).group(1)
        return (
            f"<PROGRAM>\n{improved}</PROGRAM>\n"
            f"<CHANGE_SUMMARY>improved candidate {iteration}</CHANGE_SUMMARY>"
        )

    def fake_evaluate(source, *, trials, budget, seed, timeout, max_length):
        is_improved = '"""improved"""' in source
        rows = []
        for index in range(trials):
            x, y = ((-1.7, 0.68) if is_improved else (4.0, 4.0))
            rows.append(
                {
                    "seed": seed + index,
                    "success": True,
                    "x": x,
                    "y": y,
                    "seconds": 0.001,
                    "objective_calls": budget,
                }
            )
        return True, evaluator.combined_metrics(rows), "", rows

    for mode in ("serial", "sync", "async"):
        calls["count"] = 0
        run = port.run_agentdescent_openevolve(
            complete,
            mode=mode,
            iterations=4,
            workers=2,
            task_count=8,
            evaluator=fake_evaluate,
            max_seconds=5.0,
        )
        baseline = run.archive.baseline().metrics["framework_reward"]
        assert run.result.final_reward > baseline
        assert run.result.outcomes().get("committed") == 1
        assert run.result.stale_considered >= 4
        assert run.result.retired_workers == 0
        assert run.archive.summary()["programs_evaluated"] == 2
        assert calls["count"] == 4


def test_resource_limits_are_applied_in_runner_not_threaded_parent():
    parent_source = Path(evaluator.__file__).read_text(encoding="utf-8")
    runner_source = evaluator.RUNNER.read_text(encoding="utf-8")
    assert "preexec_fn" not in parent_source
    assert "start_new_session=True" in parent_source
    assert "resource.setrlimit" in runner_source


# -- candidate isolation: two backends, and no third "run it unconfined" -----


def test_the_backend_is_chosen_once_so_the_banner_cannot_claim_the_wrong_one():
    """`main` used to re-derive the choice with its own `shutil.which("bwrap")`.
    Two copies of one condition is how a banner ends up announcing Bubblewrap on
    a host that is actually running Seatbelt."""
    source = Path(port.__file__).read_text(encoding="utf-8")
    assert 'shutil.which("bwrap")' not in source, (
        "the CLI is deriving the backend itself instead of asking the dispatcher")
    assert evaluator.sandbox_backend() in (
        "Bubblewrap", "Seatbelt (sandbox-exec)", None)


def test_a_host_with_no_sandbox_refuses_rather_than_running_the_candidate(
    tmp_path, monkeypatch
):
    """The candidate is model-written Python that gets executed. Falling back to
    an unsandboxed subprocess to keep the example portable would be trading the
    only thing the sandbox is for."""
    monkeypatch.setattr(evaluator.shutil, "which", lambda _name: None)
    monkeypatch.setattr(evaluator.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="no candidate isolation"):
        evaluator.sandbox_command(tmp_path / "c.py", 1, 10, 0, 5.0, 512)


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
def test_bubblewrap_is_preferred_where_it_exists(tmp_path):
    command = evaluator.sandbox_command(tmp_path / "c.py", 1, 10, 0, 5.0, 512)
    assert command[0].endswith("bwrap")


def test_the_seatbelt_profile_denies_the_network_and_confines_writes(
    tmp_path, monkeypatch
):
    """Seatbelt is `(allow default)` plus denials, the inverse of bwrap's
    allowlist, so what it actually forbids is worth asserting rather than
    assuming from the fact that a profile file exists."""
    monkeypatch.setattr(evaluator.shutil, "which", lambda _name: None)
    monkeypatch.setattr(evaluator.sys, "platform", "darwin")
    monkeypatch.setattr(evaluator.Path, "exists", lambda _self: True)
    candidate = tmp_path / "c.py"
    command = evaluator.sandbox_command(candidate, 3, 200, 7, 15.0, 512)

    assert command[0] == "/usr/bin/sandbox-exec" and command[1] == "-f"
    profile = Path(command[2]).read_text(encoding="utf-8")
    assert "(deny network*)" in profile
    # Blanket deny first, scratch re-allowed after it -- Seatbelt takes the last
    # matching rule, so an `allow` above the `deny` would be silently overridden.
    assert profile.index("(deny file-write*)") < profile.index("(allow file-write*")
    assert f'(allow file-write* (subpath "{tmp_path.resolve()}"))' in profile
    # The same runner, with the same limits, on either backend.
    assert str(evaluator.RUNNER) in command
    assert "--cpu-seconds" in command and command[command.index("--cpu-seconds") + 1] == "15"


def test_a_platform_that_refuses_a_limit_says_so_instead_of_pretending(monkeypatch):
    """`set_resource_limits` used to call `setrlimit(RLIMIT_AS, ...)`
    unconditionally. Darwin has no usable `RLIMIT_AS` -- it raises whatever value
    you pass -- so every candidate died inside the runner's own
    `except BaseException` and was scored as a program that crashed: the sandbox
    failing, reported as the candidate failing.

    `setrlimit` is stubbed rather than called. Applying these to the pytest
    process is not a hypothetical hazard -- an earlier version of this test did,
    capping the whole run at 60 CPU seconds and 64 file descriptors, and the
    suite died of SIGXCPU two thirds of the way through.
    """
    import resource

    from examples.openevolve import _openevolve_runner as runner

    applied = {}

    def fake_setrlimit(which, value):
        if which == getattr(resource, "RLIMIT_AS", object()):
            raise ValueError("current limit exceeds maximum limit")
        applied[which] = value

    monkeypatch.setattr(resource, "setrlimit", fake_setrlimit)
    unavailable = runner.set_resource_limits(60, 512)

    assert unavailable == ["RLIMIT_AS"]
    assert applied[resource.RLIMIT_CPU] == (60, 61), (
        "the limit that actually stops a runaway candidate must still be applied")
    assert resource.RLIMIT_FSIZE in applied and resource.RLIMIT_NOFILE in applied


def test_the_runner_carries_the_refused_limits_out_of_the_sandbox():
    """A caller that cannot say which guarantees it got is a caller that will
    assume it got all of them."""
    payload = evaluator.RUNNER.read_text(encoding="utf-8")
    assert "limits_unavailable" in payload

def test_an_unparseable_mutation_warns_rather_than_scoring_as_no_improvement():
    """An empty completion and a mutation that genuinely failed to improve are
    the same `code=""` and opposite diagnoses: one is the token budget, the other
    is the search. Measured on deepseek-v4-flash, `--max-tokens 2048` gave 0 of 6
    parseable replies and the run reported `reward 0.7639 -> 0.7639`."""
    archive = port.OpenEvolveArchive(
        archive_size=8, num_islands=1, feature_bins=4,
        exploitation_ratio=1.0, migration_interval=99, candidate_limit=4)
    archive.add_program(
        evaluator.Program(
            evaluator.program_id(evaluator.INITIAL_PROGRAM),
            0, 0, None, evaluator.INITIAL_PROGRAM, "seed",
            {"combined_score": 1.0, "framework_reward": 0.5}, True, ""),
        baseline=True)

    port._EMPTY_PROPOSALS["n"] = 0
    propose = port.make_propose(
        archive, lambda _prompt: "", objective_budget=200, held_out_trials=2)
    with pytest.warns(RuntimeWarning, match="came back empty"):
        payload = propose("rendered", object(), "", 0.0)
    assert json.loads(payload)["code"] == ""


def test_thinking_disabled_reaches_an_anthropic_shaped_endpoint():
    """The option was gated on `is_openai_compatible(args) and model startswith
    glm`, so the `--thinking disabled` default silently did nothing everywhere
    else -- and on a reasoning model 96% of the output budget goes to thinking,
    which is what truncated every mutation above."""
    source = Path(port.__file__).read_text(encoding="utf-8")
    body = source.split("def _make_completion", 1)[1].split("\ndef ", 1)[0]
    assert "is_openai_compatible" not in body, (
        "the thinking option is still gated on the OpenAI-compatible path")
    assert 'args.thinking != "default"' in body


def test_reflective_merge_is_refused_rather_than_silently_ignored():
    """The shared parser offers `--reflective-merge` to every port. This one has
    no use for it -- fusing a round's candidates would collapse the MAP-Elites
    cells the archive exists to keep -- and a flag that is accepted and then
    does nothing is indistinguishable, from the command line, from one that
    worked."""
    with pytest.raises(SystemExit, match="not supported by the OpenEvolve port"):
        port.main(["--dry-run", "--reflective-merge"])


def _flat_evaluator(source, *, trials, budget, seed, timeout, max_length):
    """Enough shape for the archive to seed; the run never gets past `evolve`."""
    rows = [{"seed": seed + i, "success": True, "value": -1.0, "distance": 1.0,
             "x": 0.0, "y": 0.0, "runtime_seconds": 0.0, "objective_calls": budget}
            for i in range(trials)]
    return {"valid": True, "error": "",
            "metrics": {"combined_score": 0.5, "framework_reward": 0.25},
            "trials": rows}


def test_the_staleness_flag_reaches_the_engine(monkeypatch):
    """Five flags in this repository have been declared, documented, and never
    read. This one is checked at the seam it has to cross: the kwargs handed to
    `evolve`."""
    captured = {}

    def fake_evolve(*_args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop here -- the kwargs are the assertion")

    monkeypatch.setattr(port, "evolve", fake_evolve)
    with pytest.raises(RuntimeError, match="stop here"):
        port.run_agentdescent_openevolve(
            lambda _prompt: "", mode="serial", iterations=2, workers=1,
            task_count=8, test_trials=1, staleness="full",
            evaluator=_flat_evaluator)

    policy = captured.get("staleness_policy")
    assert policy is not None, "--staleness never reached the engine"
    assert type(policy) is type(port.get_policy("full"))
    assert type(policy) is not type(port.get_policy("guarded")), (
        "the default was handed through regardless of the flag")


@pytest.mark.skipif(evaluator.sandbox_backend() != "Seatbelt (sandbox-exec)",
                    reason="Seatbelt is not the backend on this host")
def test_seatbelt_actually_blocks_the_writes_the_profile_claims_to_block():
    """Asserting on the text of the profile only proves the string was written.
    The profile is `(allow default)` plus denials and rule order decides the
    outcome, so the two guarantees are checked by asking the kernel."""
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        candidate = Path(scratch) / "candidate.py"
        candidate.write_text("", encoding="utf-8")
        command = evaluator.sandbox_command(candidate, 1, 10, 0, 5.0, 512)
        profile = command[2]

        def under_sandbox(code):
            return subprocess.run(
                ["/usr/bin/sandbox-exec", "-f", profile, sys.executable, "-c", code],
                capture_output=True, text=True, timeout=60)

        outside = Path(tempfile.gettempdir()) / "agentdescent-sandbox-escape.txt"
        blocked = under_sandbox(f"open({str(outside)!r}, 'w').write('escaped')")
        assert blocked.returncode != 0, "a candidate could write outside its scratch dir"
        assert not outside.exists()

        allowed = under_sandbox(
            f"open({str(Path(scratch) / 'ok.txt')!r}, 'w').write('fine')")
        assert allowed.returncode == 0, (
            f"the runner cannot write its own scratch dir: {allowed.stderr[:400]}")

        # No network. `socket` is in the standard library, so this is the kernel
        # refusing the connection rather than the AST gate refusing the import.
        net = under_sandbox(
            "import socket; socket.create_connection(('1.1.1.1', 80), timeout=5)")
        assert net.returncode != 0, "a candidate could open a network connection"
