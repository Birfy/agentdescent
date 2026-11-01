"""argparse usage errors honour ``--json`` (#664)."""

import json

import pytest

from agentdescent import cli


def _run(capsys, argv):
    with pytest.raises(SystemExit) as ei:
        cli.main(argv)
    out, err = capsys.readouterr()
    return ei.value.code, out, err


@pytest.mark.parametrize("argv", [
    ["--json", "init"],                          # missing required path
    ["init", "--json"],                          # --json after the verb
    ["init", "x", "--kind", "nope", "--json"],   # invalid choice
    ["--json", "status", "--no-such-flag"],      # unknown option
    ["--json", "no-such-verb"],                  # invalid sub-command
    ["--json"],                                  # missing sub-command
])
def test_usage_error_is_one_json_object(capsys, argv):
    code, out, err = _run(capsys, argv)
    assert code == 2
    payload = json.loads(out)
    assert payload["ok"] is False
    assert payload["error"]
    assert payload["usage"].startswith("usage: agentdescent")
    assert err == ""


def test_plain_usage_error_unchanged(capsys):
    code, out, err = _run(capsys, ["init"])
    assert code == 2
    assert out == ""
    assert err.startswith("usage: agentdescent")
    assert "error:" in err


def test_json_after_double_dash_is_not_an_option(capsys):
    code, out, err = _run(capsys, ["init", "--", "--json", "extra"])
    assert code == 2
    assert out == ""
    assert "error:" in err


def test_flag_reset_after_parse(capsys):
    _run(capsys, ["--json", "init"])
    assert cli._CliParser.json_errors is False
    code, out, err = _run(capsys, ["init"])
    assert out == "" and err.startswith("usage:")


@pytest.mark.parametrize("flag", ["--j", "--js", "--jso"])
def test_abbreviated_json_flag_gets_json_usage_error(capsys, flag):
    # argparse resolves the prefix to --json, so the error must match (#666).
    code, out, err = _run(capsys, [flag, "status", "--no-such-flag"])
    assert code == 2
    payload = json.loads(out)
    assert payload["ok"] is False
    assert "--no-such-flag" in payload["error"]
    assert err == ""


@pytest.mark.parametrize("argv, expected", [
    (["--js", "status"], True),
    (["status", "--jso"], True),
    (["--", "--js"], False),
    (["--", "--json"], False),
    (["-", "status"], False),
    (["--jsonx", "status"], False),
    (["--store", "x"], False),
])
def test_wants_json_prefixes(argv, expected):
    assert cli._wants_json(argv) is expected
