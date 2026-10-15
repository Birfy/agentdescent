"""Offline, real-cache regressions for datasets-server page filenames (#261)."""

import json
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from agentdescent import dataloader as dl


@pytest.fixture
def rows_cache(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setattr(dl, "CACHE_ROOT", str(root))
    cache_path = dl.cache_path
    paths, requests = [], []

    def checked_path(subdir, filename):
        path = cache_path(subdir, filename)
        # Check before the real reader/writer sees the path. Even the buggy
        # implementation cannot read or write outside this disposable fixture.
        assert os.path.commonpath((os.path.normpath(path), str(root))) == str(root)
        assert Path(path).parent == root / "hf" / "example_ds"
        assert len(filename.encode("ascii")) <= 200
        assert re.fullmatch(r"[A-Za-z0-9._~-]+", filename)
        paths.append(Path(path))
        return path

    def read(url, timeout):
        query = parse_qs(urlsplit(url).query, keep_blank_values=True)
        requests.append((url, timeout, query))
        return json.dumps({"rows": [{"row": {"split": query["split"][0]}}]}).encode()

    monkeypatch.setattr(dl, "cache_path", checked_path)
    monkeypatch.setattr(dl, "_read_url", read)
    return root, paths, requests


@pytest.mark.parametrize("split", [
    "../../../../escape", "a/b/c", r"a\b\c", 'a:b?c*"<>|\0',
    "训练/测试", "s" * 1000,
], ids=["traversal", "slash", "backslash", "nonportable", "unicode", "long"])
def test_hf_rows_cache_filename_is_flat_bounded_and_reusable(rows_cache, split):
    root, paths, requests = rows_cache
    expected = [{"split": split}]
    assert dl.hf_rows("example/ds", split, limit=1, timeout=7.5) == expected
    assert dl.hf_rows("example/ds", split, limit=1, timeout=7.5) == expected
    assert len(requests) == 1
    url, timeout, query = requests[0]
    assert url == dl.rows_url("example/ds", "default", split, 0, 1)
    assert timeout == 7.5
    assert query == {"dataset": ["example/ds"], "config": ["default"],
                     "split": [split], "offset": ["0"], "length": ["1"]}
    assert paths[0] == paths[1]
    assert list(root.rglob("*.json")) == [paths[0]]
    assert json.loads(paths[0].read_bytes()) == {"rows": [{"row": expected[0]}]}


@pytest.mark.parametrize("config,split,offset,length", [
    ("c" * 1000, "train", 0, 1),
    ("c" * 1000, "s" * 1000, 0, 1),
    ("c" * 1000, "s" * 1000, 10 ** 100, 10 ** 100),
], ids=["long-config", "both-long", "all-components-long"])
def test_rows_cache_bounds_the_whole_basename(rows_cache, config, split, offset, length):
    _, paths, requests = rows_cache
    for _ in range(2):
        assert dl._rows_payload("example/ds", config, split, offset, length,
                                cache=True, timeout=9) == {
            "rows": [{"row": {"split": split}}]}
    assert len(requests) == 1
    assert requests[0][0] == dl.rows_url("example/ds", config, split, offset, length)
    assert paths[0] == paths[1]


@pytest.mark.parametrize("config,split", [
    ("default", "train"), ("en-US", "validation.small"),
    ("cfg/v2", "test_v1"), ("c" * 48, "s" * 48),
    ("default", "_train"), ("default", "train_"), ("default", "__"),
])
def test_rows_cache_preserves_existing_normal_filenames(rows_cache, config, split):
    root, paths, requests = rows_cache
    path = root / "hf" / "example_ds" / f"{dl._slug(config)}_{split}_100_10.json"
    path.parent.mkdir(parents=True)
    payload = {"rows": [{"row": {"existing": "cached page"}}]}
    path.write_bytes(json.dumps(payload).encode())
    assert dl._rows_payload("example/ds", config, split, 100, 10,
                            cache=True, timeout=9) == payload
    assert paths == [path]
    assert requests == []


@pytest.mark.parametrize("splits", [
    ["a/b", "a:b", "a_b", r"a\b"],
    ["", "/", ":", "☃"],
    ["s" * 1000 + "a", "s" * 1000 + "b", "s" * 48],
    ["_train", "train_", "train"],
], ids=["same-slug", "empty-slugs", "same-prefix", "stripped-underscores"])
def test_distinct_split_inputs_never_share_a_cache_entry(rows_cache, splits):
    _, paths, requests = rows_cache
    for split in splits + list(reversed(splits)):
        assert dl.hf_rows("example/ds", split, limit=1) == [{"split": split}]
    assert len(requests) == len(splits)
    assert len(set(paths)) == len(splits)


def test_truncated_config_names_keep_distinct_entries(rows_cache):
    _, paths, requests = rows_cache
    configs = ["c" * 1000 + "a", "c" * 1000 + "b"]
    for config in configs + list(reversed(configs)):
        assert dl.hf_rows("example/ds", "train", config=config, limit=1) == [
            {"split": "train"}]
    assert len(requests) == 2
    assert len(set(paths)) == 2
    assert [query["config"][0] for _, _, query in requests] == configs


def test_long_configs_that_slug_identically_keep_distinct_entries(rows_cache):
    """A bounded config is digested from the raw config, not its lossy slug:
    ``a/x...`` and ``a:x...`` both slug to ``a_x...`` (review on #262)."""
    _, paths, requests = rows_cache
    configs = ["a/" + "x" * 1000, "a:" + "x" * 1000]
    for config in configs + list(reversed(configs)):
        assert dl.hf_rows("example/ds", "train", config=config, limit=1) == [
            {"split": "train"}]
    assert len(requests) == 2
    assert len(set(paths)) == 2
    assert [query["config"][0] for _, _, query in requests] == configs


def test_transformed_split_cannot_alias_its_own_cache_component(rows_cache):
    _, paths, requests = rows_cache
    split = "a/b"
    assert dl.hf_rows("example/ds", split, limit=1) == [{"split": split}]
    # A suffix that is itself a valid unmodified slug would let this second
    # request read the first request's page. Reserve a separate namespace.
    component = paths[0].name[len("default_"):-len("_0_1.v2.json")]
    assert component != split
    assert dl.hf_rows("example/ds", component, limit=1) == [{"split": component}]
    assert len(requests) == 2
    assert paths[0] != paths[1]


def test_transformed_split_does_not_read_an_alias_from_a_legacy_cache(rows_cache):
    root, paths, requests = rows_cache
    split = "a/b"
    component = dl._rows_cache_component(split)
    # The old writer accepted this literal split and used the unversioned
    # name below. Its data must not be returned for the distinct split `a/b`.
    legacy = root / "hf" / "example_ds" / f"default_{component}_0_1.json"
    legacy.parent.mkdir(parents=True)
    payload = {"rows": [{"row": {"split": component}}]}
    old_bytes = json.dumps(payload).encode()
    legacy.write_bytes(old_bytes)

    for _ in range(2):
        assert dl.hf_rows("example/ds", split, limit=1) == [{"split": split}]
    assert len(requests) == 1
    assert paths[0] == paths[1]
    assert paths[0] != legacy
    assert paths[0].name.endswith("_0_1.v2.json")
    assert legacy.read_bytes() == old_bytes
