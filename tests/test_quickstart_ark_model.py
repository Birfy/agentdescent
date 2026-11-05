"""The quickstart's Ark Coding example names the model the Coding plan serves (#754).

`docs/plugin-quickstart.md` is the entry page a new user copies from. Its
only-`ARK_API_KEY` example used `doubao-seed-code`, which the Coding endpoint
(`/api/coding/v3`) refuses with 404 UnsupportedModel, while the OpenCode Coding
recipe in `docs/plugins.md` already pins `ark/deepseek-v4-flash`. Pin the two
pages together so the copy-paste path cannot drift back to a dead model name.
"""

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
CODING = "https://ark.cn-beijing.volces.com/api/coding/v3"


def _recipe_model() -> str:
    text = (DOCS / "plugins.md").read_text(encoding="utf-8")
    found = set(re.findall(r'"-m", "ark/([^"]+)"', text))
    assert len(found) == 1, f"plugins.md OpenCode Coding recipe models: {found}"
    return found.pop()


def test_quickstart_ark_coding_example_uses_the_recipe_model():
    model = _recipe_model()
    text = (DOCS / "plugin-quickstart.md").read_text(encoding="utf-8")
    cli = re.findall(r"--model (\S+) --base-url " + re.escape(CODING), text)
    spec = re.findall(r'"model": "([^"]+)",\s*"api_key_env": "ARK_API_KEY",\s*'
                      r'"default_base_url": "' + re.escape(CODING) + '"', text)
    assert cli and spec, "the quickstart lost its only-ARK_API_KEY example"
    assert set(cli) == {model}, cli
    assert set(spec) == {model}, spec


def test_quickstart_never_names_a_model_the_coding_plan_refuses():
    text = (DOCS / "plugin-quickstart.md").read_text(encoding="utf-8")
    assert "doubao-seed-code" not in text
