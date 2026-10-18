"""What a `pip install agentdescent` actually gives you.

Checked by installing the built wheel into a fresh venv outside the repo: the
library imports and works, the dataloader caches under ~/.cache (not the repo),
but `examples/` is deliberately NOT shipped -- while README and docs contain ~30
`python -m examples.…` commands. That mismatch is a first-hour failure for anyone
who follows the install instructions, so the docs must say a checkout is needed.
"""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_packaging_only_ships_the_library():
    """Guard the decision: a top-level `examples` package would squat the name."""
    cfg = (ROOT / "pyproject.toml").read_text()
    m = re.search(r"include\s*=\s*\[([^\]]*)\]", cfg)
    assert m, "expected a packages.find include list"
    includes = m.group(1)
    assert "agentdescent" in includes
    assert "examples" not in includes, \
        "shipping a top-level `examples` package would collide with other projects"


def _assert_example_checkout_caveat(readme):
    """Validate the first shell install command and its source-example caveat."""
    install_at = None
    # Inspect shell blocks, not a quoted command in explanatory prose. The first
    # pip install is onboarding: a later correct command cannot excuse a typo.
    for block in re.finditer(r"```(?:bash|sh)\n(.*?)```", readme, re.DOTALL):
        install = re.search(r"(?m)^pip install[^\n]*$", block.group(1))
        if install:
            package = (r"agentdescent(?:\[[A-Za-z0-9_,.-]+\])?"
                       r"(?:(?:===|==|!=|~=|>=|<=|>|<)[A-Za-z0-9.*+_-]+"
                       r"(?:,(?:===|==|!=|~=|>=|<=|>|<)[A-Za-z0-9.*+_-]+)*)?")
            bare = r"agentdescent(?:\[[A-Za-z0-9_,.-]+\])?(?:==[A-Za-z0-9.*+_-]+)?"
            command = rf"pip install (?:\"{package}\"|'{package}'|{bare})[ \t]*"
            assert re.fullmatch(command, install.group()), \
                "onboarding must install agentdescent; quote shell version constraints"
            install_at = block.start(1) + install.start()
            break
    assert install_at is not None, "the README must show a shell pip install command"
    assert "python -m examples" in readme, "premise: the README advertises examples"
    nearby = readme[install_at:install_at + 1200].lower()
    caveat_at = nearby.find("examples are outside the wheel")
    assert caveat_at >= 0, "explain beside the install that examples are source-only"
    assert "clone the repo" in nearby[caveat_at:], \
        "the source-example checkout instruction belongs beside the install"


def test_readme_tells_pip_users_the_examples_need_a_checkout():
    _assert_example_checkout_caveat((ROOT / "README.md").read_text())


@pytest.mark.parametrize("requirement", [
    "agentdescent", "agentdescent==0.5.1", '\"agentdescent>=0.5.1\"',
    "'agentdescent[mcp]>=0.5.1,<0.6'",
])
def test_checkout_guard_accepts_supported_install_syntax(requirement):
    _assert_example_checkout_caveat(
        f"```bash\npip install {requirement}\n```\n"
        "Research examples are outside the wheel. To run python -m examples.run_demo, "
        "clone the repo.")


@pytest.mark.parametrize("readme", [
    # No caveat; a clone elsewhere does not explain the source-only examples.
    "```bash\npip install agentdescent\n```\nclone the repo; python -m examples.run_demo",
    # A valid-looking command for a different package cannot satisfy onboarding.
    "```bash\npip install agentdescent-other\n```\n"
    "Examples are outside the wheel; clone the repo; python -m examples.run_demo",
    # Shell redirection is not a version constraint unless quoted.
    "```bash\npip install agentdescent>=0.5.1\n```\n"
    "Examples are outside the wheel; clone the repo; python -m examples.run_demo",
    # A caveat too far from the install still fails the positioning invariant.
    "```bash\npip install agentdescent\n```\n" + "x" * 1200 +
    "Examples are outside the wheel; clone the repo; python -m examples.run_demo",
])
def test_checkout_guard_rejects_missing_or_misleading_onboarding(readme):
    with pytest.raises(AssertionError):
        _assert_example_checkout_caveat(readme)


def test_usage_guide_repeats_the_caveat_before_the_demos():
    usage = (ROOT / "docs" / "usage.md").read_text()
    # Match the heading, not its number: the page is renumbered whenever a
    # section is added, and this test is about the caveat's *position*.
    demos_at = usage.index("Run the demos")
    caveat_at = usage.index("checkout", demos_at)
    assert caveat_at - demos_at < 400, "warn before listing commands that need a clone"


def test_dataloader_caches_outside_the_repo():
    """A cache under the repo would break for an installed (read-only) package."""
    from agentdescent import dataloader

    root = pathlib.Path(dataloader.CACHE_ROOT).expanduser()
    assert not str(root).startswith(str(ROOT)), \
        f"cache must not live inside the package tree: {root}"
    assert ".cache" in str(root)


def test_apache_code_in_the_package_is_declared_licensed_and_shipped():
    """``agentdescent/algorithms/rrsi`` carries files adapted from google-research/rrsi
    (Apache-2.0), so the wheel must say so and ship the text (#386 review)."""
    cfg = (ROOT / "pyproject.toml").read_text()
    assert re.search(r'(?m)^license\s*=\s*"MIT AND Apache-2\.0"', cfg)
    m = re.search(r"license-files\s*=\s*\[([^\]]*)\]", cfg)
    assert m, "license-files must list the license texts and NOTICE"
    for entry in ("LICENSE", "NOTICE", "LICENSES/*"):
        assert f'"{entry}"' in m.group(1)
    assert "License :: OSI Approved :: MIT License" not in cfg    # clashes with an SPDX expression
    text = (ROOT / "LICENSES" / "Apache-2.0.txt").read_text()
    assert "Apache License" in text and "Version 2.0, January 2004" in text
    notice = (ROOT / "NOTICE").read_text()
    # every packaged file that keeps an Apache header is named in NOTICE
    apache = [p for p in (ROOT / "agentdescent").rglob("*.py")
              if "Licensed under the Apache License" in p.read_text()]
    assert apache, "expected the adapted RRSI files"
    for path in apache:
        rel = path.relative_to(ROOT / "agentdescent" / "algorithms" / "rrsi")
        assert str(rel) in notice and "google-research/rrsi" in notice, path
