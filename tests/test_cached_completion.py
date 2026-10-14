"""meta.cached_completion: the cache is best effort and never costs a paid call.

Regressions for #269. A cache-write failure must return the response the
caller already paid for, leave no ``*.partial`` sibling behind, and a corrupt
or non-str entry is a miss -- counted once, as a miss.
"""

import glob
import json
import os

import pytest

from agentdescent import meta
from agentdescent.meta import cached_completion


def _partials(d):
    return glob.glob(os.path.join(str(d), "*.partial"))


def _entries(d):
    return glob.glob(os.path.join(str(d), "*.json"))


def test_write_failure_returns_the_paid_response_and_leaves_no_partial(tmp_path, monkeypatch):
    d = tmp_path / "c"
    calls = {"n": 0}

    def complete(prompt):
        calls["n"] += 1
        return f"answer-{calls['n']}"

    cached = cached_completion(complete, str(d))

    real_fdopen = os.fdopen

    class _Full:
        def __init__(self, fh):
            self._fh = fh

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._fh.close()
            return False

        def write(self, _):
            raise OSError(28, "No space left on device")

    monkeypatch.setattr(meta.os, "fdopen", lambda fd, *a, **k: _Full(real_fdopen(fd, *a, **k)))
    assert cached("p") == "answer-1"
    assert _partials(d) == []
    assert _entries(d) == []
    assert cached.stats == {"hits": 0, "misses": 1}


def test_replace_failure_leaves_no_partial(tmp_path, monkeypatch):
    d = tmp_path / "c"
    cached = cached_completion(lambda p: "paid", str(d))

    def boom(src, dst):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(meta.os, "replace", boom)
    assert cached("p") == "paid"
    assert _partials(d) == []
    assert _entries(d) == []


def test_missing_cache_dir_does_not_raise(tmp_path):
    d = tmp_path / "c"
    cached = cached_completion(lambda p: "paid", str(d))
    os.rmdir(d)                              # removed mid-run
    assert cached("p") == "paid"
    assert not d.exists()


@pytest.mark.parametrize("response", [None, 3, {"text": "x"}, object()])
def test_a_non_str_response_is_returned_uncached(tmp_path, response):
    d = tmp_path / "c"
    calls = {"n": 0}

    def complete(prompt):
        calls["n"] += 1
        return response

    cached = cached_completion(complete, str(d))
    assert cached("p") is response
    assert cached("p") is response
    assert calls["n"] == 2, "a non-str response must not be served from the cache"
    assert _partials(d) == [] and _entries(d) == []


@pytest.mark.parametrize("content", [
    "{ truncated",
    json.dumps({"response": None}),
    json.dumps({"response": 7}),
    json.dumps(["response", "x"]),
    json.dumps({"other": "x"}),
])
def test_a_corrupt_entry_is_a_miss_only(tmp_path, content):
    d = tmp_path / "c"
    seed = cached_completion(lambda p: "first", str(d))
    assert seed("p") == "first"
    (path,) = _entries(d)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)

    cached = cached_completion(lambda p: "recomputed", str(d))
    assert cached("p") == "recomputed"
    assert cached.stats == {"hits": 0, "misses": 1}
    # The recomputed response replaced the bad entry and is now a real hit.
    assert cached("p") == "recomputed"
    assert cached.stats == {"hits": 1, "misses": 1}
    assert _partials(d) == []


def test_happy_path_still_round_trips(tmp_path):
    d = tmp_path / "c"
    calls = {"n": 0}

    def complete(prompt):
        calls["n"] += 1
        return "ünïcode ✓"

    cached = cached_completion(complete, str(d), key_extra="m|0")
    assert [cached("q") for _ in range(3)] == ["ünïcode ✓"] * 3
    assert calls["n"] == 1
    assert cached.stats == {"hits": 2, "misses": 1}
    assert _partials(d) == []
