"""Changelog fragments: one file per PR, folded into CHANGELOG.md at release.

Every PR used to add a bullet at the top of ``[Unreleased]``, so each merge made
all the other open PRs conflict on ``CHANGELOG.md``. A PR now adds
``changelog.d/<number>-<slug>.md`` instead: the first line is the category, the
rest is the entry text. See ``changelog.d/README.md``.

    python -m tools.changelog_fragments check
    python -m tools.changelog_fragments release
    python -m tools.changelog_fragments release --version X.Y.Z --date YYYY-MM-DD

Standard library only. ``--root`` points both commands at another checkout.
"""
from __future__ import annotations

import argparse
import datetime
import re
import sys
import textwrap
from pathlib import Path

# Keep a Changelog order; also the order new headings are created in.
CATEGORIES = ("Added", "Changed", "Deprecated", "Removed", "Fixed", "Security")
FRAGMENT_DIR = "changelog.d"
README_NAME = "README.md"
CHANGELOG_NAME = "CHANGELOG.md"

_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
_UNRELEASED_RE = re.compile(r"^## \[Unreleased\]\s*$", re.IGNORECASE)
_LINK_RE = re.compile(r"^\[([^\]]+)\]:\s*(\S+)\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


class Fragment:
    """One validated ``changelog.d`` file."""

    def __init__(self, name, category, text):
        self.name = name
        self.category = category
        self.text = text

    def bullet(self):
        """The entry as Markdown bullet lines (``- first`` / ``  continued``)."""
        lines = self.text.split("\n")
        first = lines[0]
        rest = textwrap.dedent("\n".join(lines[1:])).split("\n") if len(lines) > 1 else []
        out = ["- " + first]
        out.extend(("  " + line) if line.strip() else "" for line in rest)
        return out


def collect(root):
    """Return ``(fragments, errors)`` for ``<root>/changelog.d``.

    Fragments are sorted by file name, which makes folding deterministic. The
    README and dotfiles are ignored; any other non-``.md`` file is an error so a
    misnamed fragment is never silently dropped.
    """
    directory = Path(root) / FRAGMENT_DIR
    fragments, errors = [], []
    if not directory.is_dir():
        return fragments, errors
    for path in sorted(directory.iterdir(), key=lambda p: p.name):
        name = path.name
        if name == README_NAME or name.startswith(".") or path.is_dir():
            continue
        if path.suffix != ".md":
            errors.append(f"{FRAGMENT_DIR}/{name}: fragments must be .md files")
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"{FRAGMENT_DIR}/{name}: cannot read ({exc})")
            continue
        fragment, problem = _parse(name, raw)
        if problem:
            errors.append(f"{FRAGMENT_DIR}/{name}: {problem}")
        else:
            fragments.append(fragment)
    return fragments, errors


def _parse(name, raw):
    lines = raw.lstrip("\ufeff").replace("\r\n", "\n").split("\n")
    category = lines[0].strip()
    if category not in CATEGORIES:
        shown = category if category else "<empty>"
        return None, (
            f"first line must be one of {', '.join(CATEGORIES)}; got {shown!r}"
        )
    body = "\n".join(lines[1:]).strip("\n").rstrip()
    if body.lstrip() == "-" or body.lstrip().startswith("- "):
        body = body.lstrip()[1:].lstrip(" ")  # tolerate an author-written bullet marker
    if not body.strip():
        return None, "entry text is empty"
    return Fragment(name, category, body), None


def _scan(lines, start, end):
    """Indices of ``### `` headings in ``lines[start:end]``, outside code fences."""
    found, fenced = [], False
    for i in range(start, end):
        if _FENCE_RE.match(lines[i]):
            fenced = not fenced
        elif not fenced and lines[i].startswith("### "):
            found.append(i)
    return found


def _unreleased_bounds(lines):
    """``(heading_index, end_index)`` of the [Unreleased] section, or None."""
    fenced, start = False, None
    for i, line in enumerate(lines):
        if _FENCE_RE.match(line):
            fenced = not fenced
            continue
        if fenced:
            continue
        if start is None:
            if _UNRELEASED_RE.match(line):
                start = i
        elif line.startswith("## ") or _LINK_RE.match(line):
            return start, i
    return (start, len(lines)) if start is not None else None


def fold(text, fragments):
    """Return CHANGELOG ``text`` with ``fragments`` appended under [Unreleased]."""
    if not fragments:
        return text
    fragments = sorted(fragments, key=lambda f: f.name)
    lines = text.split("\n")
    if _unreleased_bounds(lines) is None:
        raise ValueError("CHANGELOG.md has no '## [Unreleased]' section")
    for category in CATEGORIES:
        entries = [f for f in fragments if f.category == category]
        if not entries:
            continue
        bullets = []
        for fragment in entries:
            if bullets:
                bullets.append("")
            bullets.extend(fragment.bullet())
        start, end = _unreleased_bounds(lines)
        headings = _scan(lines, start + 1, end)
        titles = {i: lines[i][4:].strip() for i in headings}
        existing = [i for i in headings if titles[i] == category]
        if existing:
            head = existing[0]
            later = [i for i in headings if i > head]
            block_end = later[0] if later else end
            pos = _last_content(lines, head, block_end)
            lines[pos:pos] = [""] + bullets
            continue
        block = ["### " + category, ""] + bullets
        rank = CATEGORIES.index(category)
        later = [
            i for i in headings
            if titles[i] in CATEGORIES and CATEGORIES.index(titles[i]) > rank
        ]
        if later:
            lines[later[0] : later[0]] = block + [""]
            continue
        pos = _last_content(lines, start, end)
        insert = ([""] if lines[pos - 1].strip() else []) + block
        if pos < len(lines) and lines[pos].strip():
            insert.append("")
        lines[pos:pos] = insert
    return "\n".join(lines)


def _last_content(lines, head, end):
    """Index just past the last non-blank line in ``lines[head:end]``."""
    pos = head + 1
    for i in range(head + 1, end):
        if lines[i].strip():
            pos = i + 1
    return pos


def cut_release(text, version, date):
    """Rename [Unreleased] to a dated release and add a fresh empty one.

    Matches how 0.5.1 was cut: ``## [X.Y.Z] — YYYY-MM-DD`` under an empty
    ``## [Unreleased]``; the compare links at the bottom move along with it.
    An empty [Unreleased] is left alone (nothing to release).
    """
    lines = text.split("\n")
    bounds = _unreleased_bounds(lines)
    if bounds is None:
        raise ValueError("CHANGELOG.md has no '## [Unreleased]' section")
    start, end = bounds
    if not any(line.strip() for line in lines[start + 1 : end]):
        return text, False
    if any(re.match(rf"^## \[{re.escape(version)}\]", line) for line in lines):
        raise ValueError(f"CHANGELOG.md already has a section for {version}")
    lines[start : start + 1] = ["## [Unreleased]", "", f"## [{version}] \u2014 {date}"]
    for i, line in enumerate(lines):
        m = _LINK_RE.match(line)
        if m and m.group(1).lower() == "unreleased":
            prev = re.search(r"/compare/(v[^./]+(?:\.[^./]+)*)\.\.\.HEAD$", m.group(2))
            if prev:
                base = m.group(2)[: m.group(2).index("/compare/")]
                lines[i : i + 1] = [
                    f"[Unreleased]: {base}/compare/v{version}...HEAD",
                    f"[{version}]: {base}/compare/{prev.group(1)}...v{version}",
                ]
            break
    return "\n".join(lines), True


def run_check(root, out=None, err=None):
    out, err = out or sys.stdout, err or sys.stderr
    fragments, errors = collect(root)
    for message in errors:
        print(message, file=err)
    if errors:
        return 1
    print(f"{len(fragments)} changelog fragment(s) OK", file=out)
    return 0


def run_release(root, version=None, date=None, out=None, err=None):
    out, err = out or sys.stdout, err or sys.stderr
    fragments, errors = collect(root)
    for message in errors:
        print(message, file=err)
    if errors:
        print("no changes made", file=err)
        return 1
    path = Path(root) / CHANGELOG_NAME
    text = path.read_text(encoding="utf-8")
    try:
        new = fold(text, fragments)
        if version:
            new, cut = cut_release(new, version, date)
        else:
            cut = False
    except ValueError as exc:
        print(f"error: {exc}; no changes made", file=err)
        return 1
    if new != text:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(new)
    # Delete only after CHANGELOG.md is written: a crash leaves duplicates to
    # remove, never lost entries.
    for fragment in fragments:
        (Path(root) / FRAGMENT_DIR / fragment.name).unlink()
    if fragments:
        print(f"folded {len(fragments)} fragment(s) into {CHANGELOG_NAME}", file=out)
    else:
        print("no fragments to fold", file=out)
    if version:
        if cut:
            print(f"cut [{version}] \u2014 {date}", file=out)
        else:
            print(f"[Unreleased] is empty; no [{version}] section created", file=out)
    return 0


def _build_parser():
    parser = argparse.ArgumentParser(prog="python -m tools.changelog_fragments")
    parser.add_argument(
        "--root", default=str(Path(__file__).resolve().parent.parent),
        help="repository root (default: this checkout)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="validate every fragment in changelog.d/")
    release = sub.add_parser(
        "release", help="fold fragments into CHANGELOG.md [Unreleased] and delete them"
    )
    release.add_argument("--version", help="also cut a release section, X.Y.Z")
    release.add_argument("--date", help="release date, YYYY-MM-DD (with --version)")
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "check":
        return run_check(args.root)
    if bool(args.version) != bool(args.date):
        parser.error("--version and --date must be given together")
    if args.version:
        if not _VERSION_RE.match(args.version):
            parser.error(f"--version must look like X.Y.Z, got {args.version!r}")
        try:
            if not re.match(r"^\d{4}-\d{2}-\d{2}$", args.date):
                raise ValueError(args.date)
            datetime.date.fromisoformat(args.date)
        except ValueError:
            parser.error(f"--date must be YYYY-MM-DD, got {args.date!r}")
    return run_release(args.root, args.version, args.date)


if __name__ == "__main__":
    sys.exit(main())
