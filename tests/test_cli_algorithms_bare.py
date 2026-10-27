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


def test_bare_algorithms_takes_kind_like_list(capsys):
    # #549: `algorithms --kind text` used to exit 2 with "invalid choice: 'text'".
    assert cli.main(["algorithms", "--kind", "agent_dir"]) == 0
    bare = capsys.readouterr().out
    assert cli.main(["algorithms", "list", "--kind", "agent_dir"]) == 0
    assert bare == capsys.readouterr().out
    assert "evoskill" in bare and "gepa" not in bare
    # Given before the action, the group's --kind must survive into `list`.
    assert cli.main(["algorithms", "--kind", "agent_dir", "list"]) == 0
    assert capsys.readouterr().out == bare
    assert cli.main(["algorithms", "--kind", "agent_dir", "--json"]) == 0
    names = [card["name"] for card in json.loads(capsys.readouterr().out)]
    assert "evoskill" in names and "gepa" not in names


def test_bare_algorithms_bad_kind_names_the_flag(capsys):
    assert cli.main(["algorithms", "--kind", "nope"]) == 2
    assert "--kind must be one of" in capsys.readouterr().err


def test_show_refuses_kind(capsys):
    assert cli.main(["algorithms", "--kind", "text", "show", "gepa"]) == 2
    assert "--kind" in capsys.readouterr().err
