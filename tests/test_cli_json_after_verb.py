"""``--json`` works after the verb as well as before it (#513)."""

import json

import pytest

from agentdescent import cli

from tests.test_runstore_cli import _spec_file  # noqa: F401
from tests.test_evolvespec import _WORDS  # noqa: F401


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTDESCENT_HOME", str(tmp_path / "home"))
    return tmp_path


def _run(capsys, argv):
    try:
        code = cli.main(argv)
    except SystemExit as e:  # argparse errors
        code = e.code
    out, err = capsys.readouterr()
    return code, out, err


def _verbs(tmp_path):
    spec, _ = _spec_file(tmp_path)
    store = str(tmp_path / "runs")
    return [
        ["doctor"],
        ["status"],
        ["show", "no-such-run"],
        ["plan", spec],
        ["portfolio"],
        ["audit", "no-such-run"],
        ["safety"],
        ["algorithms", "list"],
    ], store


def test_json_before_and_after_the_verb_agree(home, capsys):
    verbs, store = _verbs(home)
    for verb in verbs:
        before = _run(capsys, ["--store", store, "--json", *verb])
        after = _run(capsys, ["--store", store, *verb, "--json"])
        assert after[0] != 2 or "unrecognized arguments" not in after[2], verb
        assert before == after, verb


def test_json_after_the_verb_is_really_json(home, capsys):
    code, out, _ = _run(capsys, ["--store", str(home / "runs"), "status", "--json"])
    assert code == 0
    json.loads(out)


def test_without_json_the_flag_stays_off():
    a = cli.build_parser().parse_args(["status"])
    assert a.json is False
    a = cli.build_parser().parse_args(["--json", "status"])
    assert a.json is True
    a = cli.build_parser().parse_args(["status", "--json"])
    assert a.json is True


def test_help_does_not_leak_suppress(capsys):
    with pytest.raises(SystemExit):
        cli.main(["status", "--help"])
    assert "==SUPPRESS==" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "==SUPPRESS==" not in capsys.readouterr().out
