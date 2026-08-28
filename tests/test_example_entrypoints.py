"""The command-line contract shared by the eight faithful algorithm ports.

Eleven entry points, because ERA ships four tasks on one search: the Kaggle
regression upstream bundles, the paper's *numerical solution of integrals*,
double-precision evaluation of the Gauss hypergeometric function, equation
discovery on LLM-SRBench, and the AlgoTune speedup benchmark. The contract is
per entry point -- it is about the command line a user types -- so all five are
rows here, and none of them adds an algorithm to the fidelity table.

Ten of the eleven where numpy is absent: the LLM-SRBench task's scoring module
maps its grammar's function names onto numpy's, so its entry point cannot be
imported without it, and this repository treats numpy as optional.
"""

from __future__ import annotations

import argparse
import inspect
import pathlib
import socket
import urllib.request
from typing import Any, NamedTuple, Optional

import pytest

from examples.ace import ace_context_evolution as ace
from examples.adas import adas_meta_agent_search as adas
from examples.dgm import dgm_self_improve as dgm
from examples.era import era_algotune as era_algotune
from examples.era import era_empirical_software as era
from examples.era import era_hard_integrals as era_integrals
from examples.era import era_hypergeometric as era_hyp2f1

try:                                    # numpy is optional for this repo
    from examples.era import era_llm_srbench as era_srbench
except ImportError:                     # ...and the LLM-SRBench task needs it
    era_srbench = None
from examples.evoskill import evoskill_skill_discovery as evoskill
from examples.gepa import gepa_prompt_evolution as gepa
from examples.openevolve import openevolve_program_evolution as openevolve
from examples.skillopt import skillopt_skill_training as skillopt
from examples import _TEMPLATE as port_template
from examples import _common as common
from examples._common import add_standard_args


class Port(NamedTuple):
    """One port's entry in the contract.

    ``provider`` and ``async_ratio`` carry defaults because six of the eight
    ports take the shared ones; a port that names them is declaring a deliberate
    deviation, which is the only way one gets past this file.
    """

    module: Any
    iteration: str
    model: Optional[str]
    seconds: float
    loader: str
    provider: str = "claude"
    async_ratio: int = 3
    #: True when the port's own iteration flag already *is* the total rollout
    #: budget (`rounds = iterations // workers`), so `--budget-rollouts` maps
    #: onto it instead of adding a second budget beside it. Declared per row
    #: rather than tested by module identity: the property is what earns the
    #: exemption, and a second port with it should not have to be named in an
    #: `is` chain three hundred lines away.
    budget_is_iterations: bool = False


PORTS = (
    Port(ace, "rounds", "claude-haiku-4-5", 30.0, "load_dataset"),
    Port(gepa, "rounds", "claude-haiku-4-5", 45.0, "load_dataset"),
    Port(evoskill, "iterations", "claude-haiku-4-5", 40.0, "load_dataset"),
    Port(skillopt, "steps", "claude-haiku-4-5", 40.0, "load_dataset"),
    Port(adas, "generations", "claude-haiku-4-5", 60.0, "build_examples"),
    Port(dgm, "generations", None, 15.0, "load_dataset"),
    # OpenEvolve is measured against an OpenAI-compatible GLM endpoint, and its
    # sandboxed candidate evaluation makes a lag budget of 3 versions too loose.
    Port(openevolve, "iterations", "glm-5.2", 300.0, "build_tasks",
         provider="openai", async_ratio=1, budget_is_iterations=True),
    # ERA deviates for the same two reasons OpenEvolve does -- an
    # OpenAI-compatible endpoint, and sandboxed candidate execution that makes a
    # three-version lag budget too loose -- and for one of its own: a rollout
    # trains a model, so the async wall-clock budget is upstream's 60s timeout
    # times a tree's worth of expansions rather than a handful of API calls.
    Port(era, "iterations", "glm-5.2", 1800.0, "build_tasks",
         provider="openai", async_ratio=1, budget_is_iterations=True),
    # ERA's second task -- the paper's "numerical solution of integrals" -- on
    # the same search. Same deviations, and one loader of its own: the problems
    # are drawn rather than downloaded, so `prepare_suite` is the boundary a
    # dry-run must not cross.
    Port(era_integrals, "iterations", "glm-5.2", 1800.0, "prepare_suite",
         provider="openai", async_ratio=1, budget_is_iterations=True),
    # ERA's third task, 2F1 in double precision. Its loader reads a committed
    # file rather than drawing or downloading, so `load_suite` is the boundary
    # a dry-run must not cross.
    Port(era_hyp2f1, "iterations", "glm-5.2", 1800.0, "load_suite",
         provider="openai", async_ratio=1, budget_is_iterations=True),
    # ERA's AlgoTune task: one tree per benchmark task, scored in speedup over
    # the task's own reference implementation. Same deviations as its siblings;
    # `--iterations` is per task, because each task is its own search, and
    # `prepare_suite` is the boundary a dry-run must not cross.
    Port(era_algotune, "iterations", "glm-5.2", 1800.0, "prepare_suite",
         provider="openai", async_ratio=1, budget_is_iterations=True),
)

# ERA's fourth task, equation discovery on LLM-SRBench. Same deviations again,
# and a loader that downloads a benchmark, so `prepare_suite` is the boundary a
# dry-run must not cross. It is appended rather than declared inline because its
# scoring module maps this task's function names onto numpy's, so importing the
# entry point at all needs numpy -- which this repository treats as optional and
# a bare CI job does not install.
if era_srbench is not None:
    PORTS += (
        Port(era_srbench, "iterations", "glm-5.2", 1800.0, "prepare_suite",
             provider="openai", async_ratio=1, budget_is_iterations=True),
    )

PORT_IDS = tuple(port.module.__name__.rsplit(".", 1)[-1] for port in PORTS)


def test_standard_args_have_one_definition():
    parser = add_standard_args(argparse.ArgumentParser())
    args = parser.parse_args([
        "--provider", "openai",
        "--model", "test-model",
        "--seed", "7",
        "--async",
        "--async-ratio", "5",
        "--max-seconds", "9.5",
        "--dry-run",
        "--yes",
        "--serial",
        "--budget-rollouts", "64",
    ])
    assert args.provider == "openai"
    assert args.model == "test-model"
    assert args.seed == 7
    assert args.asynchronous is True
    assert args.async_ratio == 5
    assert args.max_seconds == 9.5
    assert args.dry_run is True
    assert args.yes is True
    assert args.serial is True
    assert args.budget_rollouts == 64


def test_a_budget_is_only_passed_when_one_was_asked_for():
    """`None` must not reach `evolve(max_rollouts=)` as a budget of nothing."""
    parser = add_standard_args(argparse.ArgumentParser())
    assert common.budget_kwargs(parser.parse_args([])) == {}
    assert common.budget_kwargs(parser.parse_args(["--budget-rollouts", "40"])) == {
        "max_rollouts": 40}


@pytest.mark.parametrize("port", PORTS, ids=PORT_IDS)
def test_every_port_can_hold_its_rollout_budget_fixed(port):
    """Without this, no speedup row in `docs/port-fidelity.md` means anything.

    Six of the eight ports pass a fixed `rounds` and let `n_workers` multiply it,
    so an `N=8` arm runs *eight times* the rollouts of the `--serial` arm.
    Comparing their wall-clocks reports eight times the model spend as parallel
    efficiency; comparing their final quality credits the extra spend to
    parallelism. `evolve(max_rollouts=)` has existed since the equal-budget work
    and **no port passed it** -- the same failure as `cheap_eval_tasks`, where the
    knob shipped and the default nobody sets stayed the only default there is.

    OpenEvolve and ERA are the exceptions: `rounds = iterations // workers`
    already fixes total work, so they map the shared flag onto `iterations`
    instead of adding a second budget beside it. That is `budget_is_iterations`
    on the row, not a module identity check here.
    """
    source = pathlib.Path(inspect.getfile(port.module)).read_text(encoding="utf-8")
    if port.budget_is_iterations:
        assert "args.iterations = args.budget_rollouts" in source, (
            f"{PORT_IDS[PORTS.index(port)]} declares its iteration budget is the "
            "rollout budget; map the shared flag onto it")
        return
    assert "budget_kwargs(args)" in source, (
        f"{PORT_IDS[PORTS.index(port)]} cannot hold its budget fixed, so its "
        "serial and parallel arms are not comparable")
    assert "max_rollouts" in source or "budget_kwargs(args))" in source


@pytest.mark.parametrize("port", PORTS, ids=PORT_IDS)
def test_every_port_uses_the_standard_contract(port):
    parser = port.module.build_parser()
    args = parser.parse_args([])
    assert args.provider == port.provider
    assert args.model == port.model
    assert args.seed == 0
    assert args.asynchronous is False
    assert args.async_ratio == port.async_ratio
    assert args.max_seconds == port.seconds
    assert args.dry_run is False
    assert args.yes is False
    assert args.serial is False
    assert hasattr(args, port.iteration)

    option_strings = [opt for action in parser._actions for opt in action.option_strings]
    for option in ("--provider", "--model", "--seed", "--async", "--async-ratio",
                   "--max-seconds", "--dry-run", "--yes", "--serial"):
        assert option_strings.count(option) == 1, f"{port.module.__name__}: {option}"
    iteration_options = {"--rounds", "--generations", "--iterations", "--steps"}
    assert iteration_options.intersection(option_strings) == {f"--{port.iteration}"}


@pytest.mark.parametrize("port", PORTS, ids=PORT_IDS)
def test_every_port_calls_the_shared_helper_once(port, monkeypatch):
    calls = []

    def recording_helper(parser, **kwargs):
        calls.append(kwargs)
        return common.add_standard_args(parser, **kwargs)

    monkeypatch.setattr(port.module, "add_standard_args", recording_helper)
    port.module.build_parser()
    assert len(calls) == 1


@pytest.mark.parametrize("port", PORTS, ids=PORT_IDS)
def test_dry_run_never_touches_data_network_or_models(port, monkeypatch, capsys):
    module = port.module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry-run crossed an external boundary")

    monkeypatch.setattr(module, port.loader, forbidden)
    monkeypatch.setattr(module, "completion_for", forbidden)
    monkeypatch.setattr(common, "claude", forbidden)
    monkeypatch.setattr(common, "openai_compatible", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    module.main(["--dry-run"])
    assert "dry-run" in capsys.readouterr().out.lower()


# ---------------------------------------------------------------------------
# The flags have to be *honoured*, not only declared. A port that grows a
# `--yes` it never reads is exactly the drift #74 was filed about, and a shape
# test over the parser cannot see it -- so these exercise the behaviour and
# then check no port reimplements it locally.
# ---------------------------------------------------------------------------


class _Stdin:
    def __init__(self, tty: bool):
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _never_asks(*_args, **_kwargs):
    raise AssertionError("confirm() prompted when it should not have")


@pytest.mark.parametrize("yes,tty", [(True, True), (True, False), (False, False)])
def test_confirm_does_not_prompt_when_yes_or_non_interactive(yes, tty, monkeypatch):
    monkeypatch.setattr(common.sys, "stdin", _Stdin(tty))
    monkeypatch.setattr("builtins.input", _never_asks)
    assert common.confirm(argparse.Namespace(yes=yes)) is True


@pytest.mark.parametrize("answer,expected", [
    ("y", True), ("YES", True), ("n", False), ("", False),
])
def test_confirm_honours_an_interactive_answer(answer, expected, monkeypatch, capsys):
    monkeypatch.setattr(common.sys, "stdin", _Stdin(True))
    monkeypatch.setattr("builtins.input", lambda *_args: answer)
    assert common.confirm(argparse.Namespace(yes=False)) is expected
    assert ("aborted." in capsys.readouterr().out) is not expected


@pytest.mark.parametrize("provider,factory", [
    ("claude", "claude"), ("openai", "openai_compatible"), ("glm", "openai_compatible"),
])
def test_completion_for_dispatches_on_provider(provider, factory, monkeypatch):
    seen = {}

    def record(name):
        def factory_stub(**kwargs):
            seen[name] = kwargs
            return name
        return factory_stub

    monkeypatch.setattr(common, "claude", record("claude"))
    monkeypatch.setattr(common, "openai_compatible", record("openai_compatible"))
    args = argparse.Namespace(provider=provider, model="test-model")

    assert common.completion_for(args, max_tokens=99) == factory
    assert seen == {factory: {"model": "test-model", "usage": None, "max_tokens": 99}}


@pytest.mark.parametrize(
    "module",
    [port.module for port in PORTS] + [port_template],
    ids=list(PORT_IDS) + ["_TEMPLATE"],
)
def test_no_port_reimplements_the_shared_behaviour(module):
    source = inspect.getsource(module)
    assert "Proceed with real API calls?" not in source, (
        f"{module.__name__}: use examples._common.confirm(args)")
    assert 'provider in ("openai", "glm")' not in source, (
        f"{module.__name__}: use examples._common.completion_for/is_openai_compatible")
    assert "confirm(args)" in source, f"{module.__name__}: --yes is declared but never read"
    assert "completion_for(" in source, f"{module.__name__}: --provider is never dispatched"
    assert "worker_count(" in source, (
        f"{module.__name__}: --serial is declared but never read, so the port has "
        "no serial baseline and any speedup it reports has nothing to be a "
        "speedup over")


# -- the serial control -----------------------------------------------------
#
# #73 needs one row per port comparing the upstream serial algorithm against the
# parallelised one. Until this flag existed, no port could run the first of
# those, so every parallelisation claim in the repository was one-armed.


def test_serial_collapses_to_one_worker():
    assert common.worker_count(argparse.Namespace(serial=True, asynchronous=False),
                               8) == 1


def test_without_serial_the_requested_count_is_untouched():
    assert common.worker_count(argparse.Namespace(serial=False, asynchronous=False),
                               8) == 8


def test_serial_and_async_are_refused_rather_than_silently_reconciled():
    """A one-worker *asynchronous* run is not the upstream serial algorithm: its
    diffs can still go stale against a head the merger moved. Putting staleness
    into the control arm is the one thing the control must not have."""
    with pytest.raises(SystemExit) as excinfo:
        common.worker_count(argparse.Namespace(serial=True, asynchronous=True), 8)
    assert "contradictory" in str(excinfo.value)


@pytest.mark.parametrize("port", PORTS, ids=PORT_IDS)
def test_serial_reaches_the_plan_every_port_prints(port, monkeypatch, capsys):
    """Honoured, not merely accepted -- and visible in the run's own output.

    A port that collapsed its worker count inside `evolve()` while still printing
    the parallel plan would be reporting one run and performing another.
    """
    module = port.module
    monkeypatch.setattr(module, port.loader,
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("dry-run crossed a boundary")))
    module.main(["--dry-run", "--serial"])
    plan = capsys.readouterr().out
    module.main(["--dry-run"])
    parallel = capsys.readouterr().out
    assert plan != parallel, (
        f"{module.__name__}: --serial changed nothing the run reports")


def test_ports_table_covers_every_standardised_entrypoint():
    """An eighth port must join PORTS, or it escapes the whole contract.

    Each port owns a directory now, so this walks the tree instead of the top
    level. A glob that stopped at ``examples/*.py`` would have gone quietly
    empty the moment the ports moved down a level -- passing by finding nothing.
    """
    examples_dir = pathlib.Path(common.__file__).resolve().parent
    # Underscore-prefixed modules are shared infrastructure, not ports: _common
    # defines the flags, _TEMPLATE demonstrates them, and _method_runner is the
    # single entry point behind the candidate-method ports -- whose own
    # contract (MethodPolicy, budgets, modes) lives in
    # tests/test_candidate_methods.py rather than in PORTS.
    on_disk = {
        path.stem for path in examples_dir.rglob("*.py")
        if not path.stem.startswith("_")
        and "add_standard_args" in path.read_text()
    }
    listed = {port.module.__name__.rsplit(".", 1)[-1] for port in PORTS}
    if era_srbench is None:
        # numpy is optional in this repository and the LLM-SRBench entry point
        # cannot be imported without it, so its row is absent -- for that reason
        # and no other. Named here rather than filtered by a rule, so a port that
        # goes missing for any *different* reason still fails this assertion.
        on_disk.discard("era_llm_srbench")
    assert on_disk == listed
    assert len(on_disk) > 1, "the walk found nothing to check"


@pytest.mark.parametrize("port", PORTS, ids=PORT_IDS)
def test_every_port_owns_a_directory_with_a_readme(port):
    """The layout is the contract: one algorithm, one folder, one entry point.

    A port dropped back at the top of ``examples/`` still imports and still
    runs, so nothing else in the suite would notice it had skipped the layout.
    """
    module_path = pathlib.Path(inspect.getfile(port.module)).resolve()
    folder = module_path.parent
    assert folder.name != "examples", \
        f"{port.module.__name__} sits at the top of examples/, not in its own folder"
    assert folder.parent.name == "examples", \
        f"{port.module.__name__} is nested deeper than examples/<algorithm>/"
    assert (folder / "README.md").exists(), f"{folder.name}/ has no README.md"
    assert (folder / "__init__.py").exists(), f"{folder.name}/ is not a package"


def test_porting_checklist_stays_short():
    root = pathlib.Path(__file__).resolve().parent.parent
    lines = (root / "docs" / "porting-checklist.md").read_text().splitlines()
    assert len(lines) <= 20


def test_port_template_is_importable_and_offline(capsys):
    args = port_template.build_parser().parse_args([])
    assert args.iterations == 6
    assert args.provider == "claude"
    port_template.main(["--dry-run"])
    assert "no dataset or model api was accessed" in capsys.readouterr().out.lower()


# -- the reported metric was the silent half of a run ------------------------


def test_score_tasks_matches_the_sequential_loop_it_replaces():
    """Concurrency here must change wall-clock and nothing else."""
    tasks = [type("T", (), {"id": str(i), "gold": i % 3})() for i in range(12)]

    def solve(artifact, task):
        return f"{artifact}-{task.gold}"

    def reward(task, output):
        return 1.0 if output.endswith(f"-{task.gold}") else 0.0

    serial_score = sum(reward(t, solve("x", t)) for t in tasks) / len(tasks)
    assert common.score_tasks(solve, "x", tasks, reward, concurrency=8) == serial_score
    assert common.score_tasks(solve, "x", tasks, reward, concurrency=1) == serial_score


def test_score_tasks_runs_them_concurrently():
    """The point of it. A sequential loop over a 20-task split at ~38s a call is
    thirteen minutes of silence, paid twice per arm."""
    import threading
    import time

    peak = [0]
    live = [0]
    lock = threading.Lock()

    def solve(artifact, task):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.02)
        with lock:
            live[0] -= 1
        return "ok"

    tasks = [type("T", (), {"id": str(i)})() for i in range(8)]
    common.score_tasks(solve, "x", tasks, lambda t, o: 1.0, concurrency=8)
    assert peak[0] > 1, "the split was scored one task at a time"


def test_an_empty_split_scores_zero_rather_than_dividing_by_it():
    assert common.score_tasks(lambda a, t: "x", "a", [], lambda t, o: 1.0) == 0.0


@pytest.mark.parametrize("port", PORTS, ids=PORT_IDS)
def test_every_name_main_uses_is_actually_imported(port):
    """`--dry-run` returns before `main()` reaches its real body, so the only
    thing exercising the rest of it is a paid run.

    A helper added to a port's `main` without its import passes every test here
    and then raises `NameError` twenty minutes into a sweep, after the first cell
    has been paid for. That is exactly how `capped_val` shipped. Resolving the
    names statically costs nothing and catches it before the API key is read.

    Deliberately conservative: it collects every binding anywhere in the module
    rather than tracking scopes, so it cannot flag a real name and will miss a
    name bound in the wrong scope. Catching the import that is simply absent is
    the whole job.
    """
    import ast
    import builtins

    source = pathlib.Path(inspect.getfile(port.module)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    bound = set(dir(builtins)) | {"__name__", "__file__", "__doc__"}

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Global):
            bound |= set(node.names)

    used = {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    unresolved = sorted(used - bound)
    assert not unresolved, (
        f"{PORT_IDS[PORTS.index(port)]} reads names it never binds: {unresolved}")
