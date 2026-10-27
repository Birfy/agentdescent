"""A bare ``agentdescent algorithms`` lists the algorithms instead of erroring."""
import json

from agentdescent import cli


def test_bare_algorithms_lists_like_list(capsys):
    assert cli.main(["algorithms"]) == 0
    bare = capsys.readouterr().out
    assert cli.main(["algorithms", "list"]) == 0
    assert bare == capsys.readouterr().out
    assert "gepa" in bare


def test_bare_algorithms_json_either_side(capsys):
    assert cli.main(["algorithms", "--json"]) == 0
    after = json.loads(capsys.readouterr().out)
    assert cli.main(["--json", "algorithms"]) == 0
    before = json.loads(capsys.readouterr().out)
    assert after == before
    assert any(card["name"] == "gepa" for card in after)


def test_show_still_needs_a_name(capsys):
    assert cli.main(["algorithms", "show", "gepa"]) == 0
    assert "gepa" in capsys.readouterr().out
