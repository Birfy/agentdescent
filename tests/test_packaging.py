"""What a `pip install agentdescent` actually gives you.

Checked by installing the built wheel into a fresh venv outside the repo: the
library imports and works, the dataloader caches under ~/.cache (not the repo),
but `examples/` is deliberately NOT shipped -- while README and docs contain ~30
`python -m examples.…` commands. That mismatch is a first-hour failure for anyone
who follows the install instructions, so the docs must say a checkout is needed.
"""

import pathlib
import re

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


def test_readme_tells_pip_users_the_examples_need_a_checkout():
    readme = (ROOT / "README.md").read_text()
    assert "python -m examples" in readme, "premise: the README does advertise them"
    lowered = readme.lower()
    assert "clone" in lowered, "the README must say a checkout is required"
    # the explanation must sit near the install instructions, not buried at the end
    install_at = readme.index("pip install agentdescent")
    clone_at = lowered.index("clone the repo")
    assert 0 < clone_at - install_at < 1200, \
        "the checkout caveat belongs beside the install instructions"


def test_usage_guide_repeats_the_caveat_before_the_demos():
    usage = (ROOT / "docs" / "usage.md").read_text()
    demos_at = usage.index("## 2. Run the demos")
    caveat_at = usage.index("checkout", demos_at)
    assert caveat_at - demos_at < 400, "warn before listing commands that need a clone"


def test_dataloader_caches_outside_the_repo():
    """A cache under the repo would break for an installed (read-only) package."""
    from agentdescent import dataloader

    root = pathlib.Path(dataloader.CACHE_ROOT).expanduser()
    assert not str(root).startswith(str(ROOT)), \
        f"cache must not live inside the package tree: {root}"
    assert ".cache" in str(root)
