"""Rendered CSS regression check, run against the built site in docs CI.

Install playwright and its Chromium browser, then run:
    python tools/check_site_layout.py site [screenshot-directory]
This is deliberately separate from offline core pytest: it needs a real browser.
"""
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
import sys


@contextmanager
def serve(site):
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(site)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def check(site, screenshots):
    from playwright.sync_api import sync_playwright

    screenshots.mkdir(parents=True, exist_ok=True)
    count = 0
    with serve(site) as base, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            for width in (1440, 1024, 901, 900, 768, 390, 320):
                for theme in ("light", "dark"):
                    for route, name in (("/", "home"), ("/install/", "quickstart")):
                        context = browser.new_context(viewport={"width": width, "height": 900}, color_scheme=theme)
                        page = context.new_page()
                        label = f"{name}-{width}-{theme}"
                        try:
                            page.goto(base + route, wait_until="networkidle")
                            # Use the page's real theme control so CSS and JS agree.
                            if page.locator(".experience").get_attribute("data-theme") != theme:
                                page.locator(".theme-toggle").click()
                            assert page.locator(".experience").get_attribute("data-theme") == theme, label
                            assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth"), label
                            assert page.locator(".experience").evaluate("e => {const r=e.getBoundingClientRect(); return r.left === 0 && r.right === document.documentElement.clientWidth}"), f"{label}: unwanted outer gutters"
                            menu = page.locator(".exp-mobile-menu")
                            summary = menu.locator("summary")
                            if width > 900:
                                assert not menu.is_visible(), f"{label}: mobile menu leaked onto desktop"
                                assert page.locator(".exp-nav-links").is_visible(), label
                            else:
                                assert menu.is_visible(), f"{label}: mobile menu missing"
                                assert not page.locator(".exp-nav-links").is_visible(), label
                                summary.focus()
                                summary.press("Enter")
                                assert menu.get_attribute("open") is not None, label
                                assert menu.get_by_role("link", name="Interactive demo").is_visible(), label
                                style = summary.evaluate("e => ({bg:getComputedStyle(e).backgroundColor,before:getComputedStyle(e,'::before').display,after:getComputedStyle(e,'::after').display})")
                                assert style == {"bg": "rgba(0, 0, 0, 0)", "before": "none", "after": "none"}, f"{label}: Material decoration leaked: {style}"
                                assert menu.locator(":scope > div").evaluate("e => {const r=e.getBoundingClientRect(); return r.left >= 0 && r.right <= document.documentElement.clientWidth}"), f"{label}: dropdown clipped"
                                summary.press("Enter")
                                assert menu.get_attribute("open") is None, label
                                summary.press("Space")
                                assert menu.get_attribute("open") is not None, label
                                summary.press("Space")
                                assert menu.get_attribute("open") is None, label
                            if name == "home":
                                code = page.locator(".hero-command code")
                                assert code.evaluate("e => getComputedStyle(e).whiteSpace") == "nowrap", label
                                assert code.evaluate("e => getComputedStyle(e).overflowX") == "auto", label
                                assert code.get_attribute("tabindex") == "0", label
                                assert page.locator(".hero-command button").evaluate("e=>e.getBoundingClientRect().right <= document.documentElement.clientWidth"), label
                                code.focus()
                                assert code.evaluate("e => document.activeElement === e"), label
                            else:
                                faq = page.locator(".qs3-more details").first
                                faq_summary = faq.locator("summary")
                                faq_summary.click()
                                assert faq.get_attribute("open") is not None, label
                                assert faq.locator("p").is_visible(), label
                                assert faq_summary.evaluate("e => getComputedStyle(e).backgroundColor") == "rgba(0, 0, 0, 0)", label
                                assert faq_summary.evaluate("e => getComputedStyle(e,'::before').display") == "none", label
                                faq_summary.click()
                                assert faq.get_attribute("open") is None, label
                            page.evaluate("window.scrollTo(0,0)")
                            if width in (1440, 390):
                                page.screenshot(path=str(screenshots / f"{label}.png"), full_page=True)
                            count += 1
                        except Exception:
                            page.screenshot(path=str(screenshots / f"FAILED-{label}.png"), full_page=True)
                            raise
                        finally:
                            context.close()
        finally:
            browser.close()
    print(f"Rendered layout passed: {count} page/viewport/theme cases")


if __name__ == "__main__":
    check(Path(sys.argv[1]).resolve(), Path(sys.argv[2] if len(sys.argv) > 2 else "layout-screenshots"))
