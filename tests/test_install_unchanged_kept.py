"""Re-running install must not claim \"updated\" for identical files (#555)."""
import os
import time

from agentdescent.integrations import _Writer, install


def test_writer_keeps_identical_bytes_without_rewrite(tmp_path):
    path = tmp_path / "skill.md"
    path.write_text("same\n", encoding="utf-8")
    mtime_before = path.stat().st_mtime_ns
    time.sleep(0.02)
    w = _Writer(dry_run=False)
    w.write(str(path), "same\n")
    assert w.lines == [f"kept {path} (unchanged)"]
    assert path.read_text(encoding="utf-8") == "same\n"
    assert path.stat().st_mtime_ns == mtime_before


def test_writer_updates_when_content_differs(tmp_path):
    path = tmp_path / "skill.md"
    path.write_text("old\n", encoding="utf-8")
    w = _Writer(dry_run=False)
    w.write(str(path), "new\n")
    assert w.lines == [f"updated {path}"]
    assert path.read_text(encoding="utf-8") == "new\n"


def test_writer_dry_run_still_reports_would_write_for_a_change(tmp_path):
    path = tmp_path / "skill.md"
    path.write_text("old\n", encoding="utf-8")
    w = _Writer(dry_run=True)
    w.write(str(path), "new\n")
    assert w.lines == [f"would write {path}"]
    assert path.read_text(encoding="utf-8") == "old\n"


def test_writer_dry_run_keeps_identical_bytes(tmp_path):
    path = tmp_path / "skill.md"
    path.write_text("same\n", encoding="utf-8")
    w = _Writer(dry_run=True)
    w.write(str(path), "same\n")
    assert w.lines == [f"kept {path} (unchanged)"]


def test_install_rerun_keeps_unchanged_scaffold_files(tmp_path):
    home = str(tmp_path)
    first = install("grok-bot", home=home)
    assert any(line.startswith("wrote ") for line in first)
    second = install("grok-bot", home=home)
    kept = [line for line in second if line.startswith("kept ") and "(unchanged)" in line]
    updated = [line for line in second if line.startswith("updated ")]
    assert len(kept) >= 3, second
    assert updated == [], second
