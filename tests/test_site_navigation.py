"""Keep the demo in the Material drawer when the desktop header disappears.

Source checks run in pytest; ``python tests/test_site_navigation.py site`` also
checks every rendered documentation page after the strict docs build.
"""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def config():
    # BaseLoader understands configuration structure without importing MkDocs's
    # tagged Markdown extension callbacks into the ordinary offline test suite.
    return yaml.load((ROOT / "mkdocs.yml").read_text(), Loader=yaml.BaseLoader)


class Navigation(HTMLParser):
    def __init__(self):
        super().__init__()
        self.primary_depth = 0
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "nav":
            if self.primary_depth or "md-nav--primary" in attrs.get("class", "").split():
                self.primary_depth += 1
        if tag == "a" and self.primary_depth:
            self.links.append(attrs)

    def handle_endtag(self, tag):
        if tag == "nav" and self.primary_depth:
            self.primary_depth -= 1


def test_demo_is_a_top_level_drawer_destination():
    cfg = config()
    assert {"Interactive demo": cfg["site_url"] + "#demo"} in cfg["nav"]
    assert 'id="demo"' in (ROOT / "docs/index.md").read_text()


def test_header_breakpoint_keeps_the_drawer_as_the_mobile_route():
    css = (ROOT / "docs/assets/docs-theme.css").read_text()
    assert "@media(max-width:900px)" in css
    assert ".ad-header-links{display:none}" in css
    header = (ROOT / "site_overrides/partials/header.html").read_text()
    assert 'for="__drawer"' in header
    assert '{{ nav.homepage.url | url }}#demo' in header


def verify_rendered_site(site):
    cfg = config()
    destination = cfg["site_url"] + "#demo"
    homepage = Navigation()
    homepage.feed((site / "index.html").read_text())
    assert "demo" in homepage.ids
    checked = 0
    for path in site.rglob("*.html"):
        html = path.read_text()
        if 'class="ad-header-links"' not in html:
            continue  # Homepage uses its own responsive navigation.
        parsed = Navigation()
        parsed.feed(html)
        page_url = urljoin(cfg["site_url"], path.relative_to(site).as_posix())
        links = [a for a in parsed.links if urljoin(page_url, a.get("href", "")) == destination]
        assert len(links) == 1, f"{path}: expected one primary drawer demo link"
        assert links[0].get("tabindex", "0") == "0", f"{path}: demo is not keyboard focusable"
        assert "hidden" not in links[0], f"{path}: demo link is hidden"
        checked += 1
    assert checked > 0, "No documentation pages were checked"
    return checked


if __name__ == "__main__":
    count = verify_rendered_site(Path(sys.argv[1]))
    print(f"Demo navigation verified on {count} rendered documentation pages")
