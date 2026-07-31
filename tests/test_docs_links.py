"""Internal documentation links must resolve.

The docs are restructured periodically -- pages split, merge and move -- and a
broken cross-reference is invisible until a reader hits it. mkdocs --strict does
not validate heading anchors at all, so both halves are checked here.
"""
import pathlib
import re

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"

# [text](target) where target is not http(s), a mailto, or a bare anchor
LINK = re.compile(r"\[[^\]]*\]\((?!https?://|mailto:)([^)\s]+)\)")


def _anchor(heading: str) -> str:
    """Python-Markdown's `toc` slugify, exactly.

    Worth copying rather than approximating: it drops every non-word character
    (so backticks, parentheses and em dashes vanish) but **keeps underscores**,
    because `_` is a word character. Collapsing them to dashes reports every
    `async_evolve()` heading as a broken anchor.
    """
    text = re.sub(r"[^\w\s-]", "", heading.strip()).strip().lower()
    return re.sub(r"[-\s]+", "-", text)


def _anchors_of(path: pathlib.Path) -> set:
    out = set()
    for line in path.read_text().splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            out.add(_anchor(m.group(2)))
    return out


def _links():
    for page in sorted(DOCS.glob("*.md")):
        for target in LINK.findall(page.read_text()):
            yield page, target


def test_every_internal_link_points_at_a_real_page():
    missing = []
    for page, target in _links():
        file_part = target.split("#", 1)[0]
        if not file_part:
            continue                       # same-page anchor, checked below
        if not (DOCS / file_part).exists():
            missing.append(f"{page.name} -> {target}")
    assert not missing, "broken page links:\n" + "\n".join(missing)


def test_every_internal_anchor_exists():
    """mkdocs --strict does not check these, so a renamed heading breaks silently."""
    missing = []
    for page, target in _links():
        file_part, _, anchor = target.partition("#")
        if not anchor:
            continue
        target_page = (DOCS / file_part) if file_part else page
        if not target_page.exists():
            continue                       # reported by the test above
        if anchor not in _anchors_of(target_page):
            missing.append(f"{page.name} -> {target}")
    assert not missing, "broken anchors:\n" + "\n".join(missing)


def test_every_page_is_in_the_nav():
    """An orphaned page is invisible on the site even though it renders."""
    nav = (DOCS.parent / "mkdocs.yml").read_text()
    orphans = [p.name for p in sorted(DOCS.glob("*.md")) if p.name not in nav]
    assert not orphans, f"pages not reachable from the nav: {orphans}"
