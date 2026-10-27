"""`apply --from` installs an on-disk artifact, e.g. a campaign's result (#533)."""

from __future__ import annotations

import json
import os
import shlex

from agentdescent import cli
from tests.test_evolvespec import _dir_spec


def _next_commands(out: str):
    lines = out.splitlines()
    i = lines.index("next:")
    return [ln.strip() for ln in lines[i + 1:] if ln.strip()]


def test_campaign_prints_an_apply_path_that_works(tmp_path, capsys):
    spec = _dir_spec(str(tmp_path))
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec.to_dict()))
    assert cli.main(["campaign", str(path), "--sessions", "1", "--rounds", "1",
                     "--workdir", str(tmp_path / "c")]) == 0
    nxt = _next_commands(capsys.readouterr().out)
    dry = next(c for c in nxt if "--dry-run" in c)
    argv = shlex.split(dry.split("   ")[0])
    assert argv[:2] == ["agentdescent", "apply"]
    i = argv.index("--from")
    final = argv[i + 1]
    assert argv[argv.index("--to") + 1] == os.path.abspath(spec.target)
    # make the artifact differ so apply has something to write
    with open(os.path.join(final, "rules.md"), "a", encoding="utf-8") as fh:
        fh.write("\n- evolved line\n")
    original = open(os.path.join(spec.target, "rules.md")).read()

    assert cli.main(argv[1:]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["written"] == ["rules.md"] and plan["noop"] is False
    assert open(os.path.join(spec.target, "rules.md")).read() == original

    assert cli.main([a for a in argv[1:] if a != "--dry-run"]) == 0
    done = json.loads(capsys.readouterr().out)
    assert done["backup"] == [os.path.abspath(spec.target) + ".bak-0"]
    assert open(os.path.join(done["backup"][0], "rules.md")).read() == original
    assert "evolved line" in open(os.path.join(spec.target, "rules.md")).read()

    # applying again is a no-op: nothing written, no second backup
    assert cli.main([a for a in argv[1:] if a != "--dry-run"]) == 0
    again = json.loads(capsys.readouterr().out)
    assert again["noop"] is True
    assert not os.path.exists(os.path.abspath(spec.target) + ".bak-1")


def test_campaign_json_carries_the_next_commands(tmp_path, capsys):
    spec = _dir_spec(str(tmp_path))
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec.to_dict()))
    assert cli.main(["--json", "campaign", str(path), "--sessions", "1",
                     "--rounds", "1", "--workdir", str(tmp_path / "c")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["target"] == os.path.abspath(spec.target)
    assert out["next"][0].endswith("--dry-run")
    assert "--from" in out["next"][1] and "--to" in out["next"][1]


def test_apply_from_a_text_file_backs_up_and_extra_files_are_left(tmp_path, capsys):
    src = tmp_path / "evolved.md"
    src.write_text("new prompt\n")
    dest = tmp_path / "prompt.md"
    dest.write_text("old prompt\n")
    assert cli.main(["apply", "--from", str(src), "--to", str(dest)]) == 0
    res = json.loads(capsys.readouterr().out)
    assert dest.read_text() == "new prompt\n"
    assert res["backup"] == [str(dest) + ".bak-0"]
    assert (tmp_path / "prompt.md.bak-0").read_text() == "old prompt\n"

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    (a / "SKILL.md").write_text("x")
    (b / "SKILL.md").write_text("y")
    (b / "keep.txt").write_text("mine")
    assert cli.main(["apply", "--from", str(a), "--to", str(b)]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["extra"] == ["keep.txt"] and (b / "keep.txt").read_text() == "mine"
    assert (b / "SKILL.md").read_text() == "x"


def test_apply_from_argument_errors(tmp_path, capsys):
    src = tmp_path / "x.md"
    src.write_text("x")
    assert cli.main(["apply", "--from", str(src)]) == 2
    assert "--to" in capsys.readouterr().err
    assert cli.main(["apply", "someid", "--from", str(src), "--to",
                     str(tmp_path / "y.md")]) == 2
    assert cli.main(["apply"]) == 2
    assert cli.main(["apply", "--from", str(tmp_path / "missing"), "--to",
                     str(tmp_path / "y.md")]) != 0
    assert "does not exist" in capsys.readouterr().err

def test_apply_from_refuses_symlinks_in_directory_artifact(tmp_path, capsys):
    """Directory artifacts must not dereference links into --to (#538 review).

    A campaign workspace is written by CLI workers; a planted ``abs ->
    /etc/hostname`` (or ``~/.ssh/id_ed25519``) used to become a regular file
    under the user's skill/plugin dir via ``shutil.copy2``. ``apply <run_id>``
    never materialises symlinks (FileTree skips them at load), so ``--from``
    refuses them instead of following.
    """
    src, dest = tmp_path / "final", tmp_path / "skill"
    src.mkdir(); dest.mkdir()
    (src / "SKILL.md").write_text("ok\n")
    (dest / "SKILL.md").write_text("old\n")
    link = src / "abs"
    link.symlink_to("/etc/hostname")
    assert link.is_symlink()

    assert cli.main(["apply", "--from", str(src), "--to", str(dest)]) != 0
    err = capsys.readouterr().err
    assert "symlink" in err and "abs" in err
    # Must not have materialised /etc/hostname into the destination.
    assert not (dest / "abs").exists()
    assert (dest / "SKILL.md").read_text() == "old\n"
    assert not list(tmp_path.glob("skill.bak-*"))

    # --from itself as a symlink is also refused (isfile would follow it).
    file_src = tmp_path / "evolved.md"
    file_src.write_text("new\n")
    alias = tmp_path / "alias.md"
    alias.symlink_to(file_src)
    target = tmp_path / "prompt.md"
    target.write_text("old\n")
    assert cli.main(["apply", "--from", str(alias), "--to", str(target)]) != 0
    assert "symlink" in capsys.readouterr().err
    assert target.read_text() == "old\n"

