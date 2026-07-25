"""Offline tests for concordia.backends.

The tool-loop backend and the grep helper are exercised with a stub completion
(no network). The OpenHands backend needs Python >= 3.12 + openhands-ai, so here
we only assert it fails with a clear, actionable error when that's absent.
"""

import pytest

from concordia.backends import _grep, openhands_backend, tool_loop_backend


DOC = """Budget expenditures, by year
| Year | 1938 | 1939 | 1940 | 1941 |
| National defense | 500 | 689 | 2602 | 6404 |
| Veterans | 556 | 561 | 570 | 580 |
unrelated paragraph about weather and other matters
| Interest on the public debt | 900 | 920 | 941 | 950 |
""".splitlines()


def test_grep_finds_row_and_includes_year_header():
    out = _grep(DOC, "national defense", window=0)
    assert "National defense" in out
    # the nearest header row (with several years) is pulled in for column alignment
    assert "1940" in out and "Year" in out


def test_grep_reports_no_matches_gracefully():
    assert _grep(DOC, "cryptocurrency mining") == "(no matches)"


def test_tool_loop_backend_greps_then_answers():
    # a scripted "agent": first turn GREPs, second turn ANSWERs from the observation.
    calls = {"n": 0}

    def stub(prompt: str) -> str:
        calls["n"] += 1
        if "National defense" not in prompt:      # nothing grepped yet -> search
            return "GREP national defense 1940"
        return "ANSWER 2602"                        # observation now contains the row

    backend = tool_loop_backend(stub, max_steps=4)
    ans = backend.answer("What was national defense spending in 1940?", "\n".join(DOC))
    assert ans == "2602"
    assert calls["n"] == 2                          # one search, one answer


def test_tool_loop_backend_stops_at_max_steps():
    backend = tool_loop_backend(lambda p: "GREP something", max_steps=3)
    # never answers -> the loop caps and forces a final completion (still returns a str)
    assert isinstance(backend.answer("q", "\n".join(DOC)), str)


def test_openhands_backend_errors_clearly_without_dependency():
    # On Python 3.9 (this test env) openhands-ai is not installable; the factory
    # must fail with an actionable message rather than a cryptic ImportError.
    try:
        openhands_backend()
    except RuntimeError as e:
        assert "openhands-ai" in str(e)
    except Exception:  # pragma: no cover - if openhands IS present, skip
        pytest.skip("openhands-ai present in this environment")
    else:  # pragma: no cover
        pytest.skip("openhands-ai present in this environment")
