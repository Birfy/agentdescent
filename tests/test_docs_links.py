"""Internal documentation links must resolve.

The docs are restructured periodically -- pages split, merge and move -- and a
broken cross-reference is invisible until a reader hits it. mkdocs --strict does
not validate heading anchors at all, so both halves are checked here.
"""
from html.parser import HTMLParser
import pathlib
import re

import markdown
import pytest
import yaml

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"

# [text](target) where target is not http(s), a mailto, or a bare anchor
LINK = re.compile(r"\[[^\]]*\]\((?!https?://|mailto:)([^)\s]+)\)")


class _HTMLAnchors(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        ids = [value for name, value in attrs if name == "id"]
        assert len(ids) <= 1, "an element must not repeat its id attribute"
        for value in ids:
            assert value and not any(c.isspace() for c in value), "invalid HTML id"
            assert value not in self.ids, f"duplicate HTML id: {value}"
            self.ids.add(value)


def _anchors_of(path: pathlib.Path) -> set:
    # Use MkDocs' underlying renderer, not regexes over Markdown source. Code
    # examples must remain escaped, while raw HTML IDs and generated heading
    # IDs are both actual browser targets. These are the core extensions used
    # for the same constructs in mkdocs.yml.
    source = path.read_text()
    # MkDocs consumes a leading YAML mapping before passing text to Markdown.
    # Otherwise its closing delimiter can turn metadata into a setext heading
    # and invent a link target that is absent from the rendered site.
    frontmatter = re.match(
        r"\A---[ \t]*\n(.*?)\n(?:---|\.\.\.)[ \t]*\n", source, re.DOTALL)
    if frontmatter:
        try:
            metadata = yaml.safe_load(frontmatter.group(1))
        except yaml.YAMLError:
            metadata = None
        if isinstance(metadata, dict):
            source = source[frontmatter.end():].lstrip("\n")
    html = markdown.markdown(source, extensions=[
        "meta", "toc", "fenced_code", "attr_list", "md_in_html",
    ])
    parser = _HTMLAnchors()
    parser.feed(html)
    return parser.ids


@pytest.mark.parametrize("end", ["---", "..."])
def test_anchor_guard_ignores_yaml_metadata(tmp_path, end):
    page = tmp_path / "page.md"
    page.write_text(
        '---\ndescription: <div id="fake">metadata</div>\n' + end + '\n## Real\n')
    assert _anchors_of(page) == {"real"}


@pytest.mark.parametrize("metadata", [
    '---\ndescription: Example: <div id="fake">metadata</div>\n---\n',
    'description: <div id="fake">metadata</div>\n\n',
])
def test_anchor_guard_matches_site_meta_extension(tmp_path, metadata):
    page = tmp_path / "page.md"
    page.write_text(metadata + '## Real\n')
    assert _anchors_of(page) == {"real"}


def test_anchor_guard_recognizes_markdown_and_real_html_ids(tmp_path):
    page = tmp_path / "page.md"
    page.write_text('## Source examples\n<details id="building-the-docs">Text</details>')
    assert _anchors_of(page) == {"source-examples", "building-the-docs"}


@pytest.mark.parametrize("example", [
    '<!-- <details id="fake">example</details> -->',
    '```html\n<details id="fake">example</details>\n```',
    '~~~html\n<details id="fake">example</details>\n~~~',
    '`<details id="fake">example</details>`',
    '&lt;details id="fake"&gt;example&lt;/details&gt;',
    '    <div id="fake">indented example</div>',
    '```html\n``` trailing text\n<div id="fake">still fenced</div>\n```',
])
def test_anchor_guard_rejects_ids_that_are_only_examples(tmp_path, example):
    page = tmp_path / "page.md"
    page.write_text(example)
    assert "fake" not in _anchors_of(page)


def test_anchor_guard_matches_renderer_for_a_backslash_before_raw_html(tmp_path):
    # Python-Markdown does not backslash-escape '<': the emitted raw HTML
    # remains a browser anchor. Entity-escaped HTML is rejected above.
    page = tmp_path / "page.md"
    page.write_text('\\<div id="real">raw HTML</div>')
    assert _anchors_of(page) == {"real"}


def test_anchor_guard_preserves_backticks_in_real_html_ids(tmp_path):
    page = tmp_path / "page.md"
    page.write_text('<div id="first`id"></div><div id="second`id"></div>')
    assert _anchors_of(page) == {"first`id", "second`id"}


@pytest.mark.parametrize("html", [
    '<div id="same"></div><details id="same"></details>',
    '<div id="first" id="second"></div>',
    '<div id=""></div>',
    '<div id="has spaces"></div>',
    '## Same\n<div id="same"></div>',
])
def test_anchor_guard_rejects_duplicate_or_malformed_ids(tmp_path, html):
    page = tmp_path / "page.md"
    page.write_text(html)
    with pytest.raises(AssertionError):
        _anchors_of(page)


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
