"""Offline tests for the concordia.dataloader data layer.

The URL/paging helpers are pure; the HTTP fetch is monkeypatched so hf_rows /
fetch_text are exercised without a network.
"""

import json
import os

from concordia import dataloader as dl


def test_page_offsets_splits_into_capped_pages():
    assert dl.page_offsets(250) == [(0, 100), (100, 100), (200, 50)]
    assert dl.page_offsets(80) == [(0, 80)]
    assert dl.page_offsets(0) == []
    assert dl.page_offsets(100) == [(0, 100)]


def test_rows_url_is_well_formed():
    url = dl.rows_url("a/b", "cfg", "train", 0, 10)
    assert url.startswith(dl.ROWS_URL + "?")
    assert "dataset=a%2Fb" in url and "config=cfg" in url
    assert "offset=0" in url and "length=10" in url


def test_slug_is_filesystem_safe():
    assert dl._slug("princeton-nlp/SWE-bench_Verified") == "princeton-nlp_SWE-bench_Verified"


def test_cache_path_creates_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CACHE_ROOT", str(tmp_path / "c"))
    p = dl.cache_path("sub", "f.json")
    assert os.path.isdir(os.path.dirname(p))


def _fake_rows_payload(n):
    return {"rows": [{"row": {"i": i}} for i in range(n)],
            "features": [{"name": "lbl", "type": {"names": ["a", "b"]}}]}


def test_hf_rows_pages_and_stops_when_exhausted(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CACHE_ROOT", str(tmp_path))
    calls = []

    def fake_read(url, timeout):
        # parse the requested length out of the url; return that many rows,
        # except make the 3rd page short to signal exhaustion.
        length = int(url.split("length=")[1].split("&")[0])
        calls.append(length)
        n = length if len(calls) < 3 else 20
        return json.dumps(_fake_rows_payload(n)).encode()

    monkeypatch.setattr(dl, "_read_url", fake_read)
    rows = dl.hf_rows("ds", "train", limit=250)
    # pages: 100, 100, then a short page (20 < 50) -> stop.
    assert len(rows) == 100 + 100 + 20
    assert calls == [100, 100, 50]


def test_hf_rows_uses_cache_on_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CACHE_ROOT", str(tmp_path))
    n_fetches = {"n": 0}

    def fake_read(url, timeout):
        n_fetches["n"] += 1
        return json.dumps(_fake_rows_payload(10)).encode()

    monkeypatch.setattr(dl, "_read_url", fake_read)
    dl.hf_rows("ds", "train", limit=10)
    dl.hf_rows("ds", "train", limit=10)      # served from cache
    assert n_fetches["n"] == 1


def test_hf_feature_names_extracts_nested_class_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CACHE_ROOT", str(tmp_path))
    payload = {"rows": [{"row": {}}],
               "features": [{"name": "ner_tags",
                             "type": {"_type": "Sequence",
                                      "feature": {"names": ["O", "B-X"]}}}]}
    monkeypatch.setattr(dl, "_read_url", lambda url, timeout: json.dumps(payload).encode())
    assert dl.hf_feature_names("ds", "train", "ner_tags") == ["O", "B-X"]


def test_fetch_text_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CACHE_ROOT", str(tmp_path))
    n = {"n": 0}

    def fake_read(url, timeout):
        n["n"] += 1
        return b"hello\nworld"

    monkeypatch.setattr(dl, "_read_url", fake_read)
    assert dl.fetch_text("http://x/y", filename="y.txt").splitlines() == ["hello", "world"]
    dl.fetch_text("http://x/y", filename="y.txt")
    assert n["n"] == 1


def test_load_gated_hf_returns_none_without_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert dl.load_gated_hf("some/gated", "test") is None


# -- Dataset + split_dataset -------------------------------------------------


def test_split_dataset_ratios_and_determinism():
    ds = dl.split_dataset(range(100), ratios=(0.6, 0.2, 0.2), seed=1)
    assert ds.sizes() == (60, 20, 20)
    assert len(ds) == 100
    # splits are disjoint and cover everything
    assert set(ds.train) | set(ds.val) | set(ds.test) == set(range(100))
    assert not (set(ds.train) & set(ds.val)) and not (set(ds.val) & set(ds.test))
    # deterministic given the seed
    again = dl.split_dataset(range(100), ratios=(0.6, 0.2, 0.2), seed=1)
    assert (ds.train, ds.val, ds.test) == (again.train, again.val, again.test)


def test_dataset_trainval_and_val_frac():
    ds = dl.Dataset(train=[1, 2, 3, 4, 5, 6], val=[7, 8], test=[9, 10])
    assert ds.trainval == [1, 2, 3, 4, 5, 6, 7, 8]
    assert abs(ds.val_frac - 2 / 8) < 1e-9      # |val| / |train+val|


def test_dataset_map_shapes_every_split():
    ds = dl.Dataset(train=[1], val=[2], test=[3]).map(lambda x: x * 10)
    assert (ds.train, ds.val, ds.test) == ([10], [20], [30])


def test_split_dataset_stratified_keeps_classes_in_each_split():
    items = [{"c": "a"}] * 10 + [{"c": "b"}] * 10
    ds = dl.split_dataset(items, ratios=(0.6, 0.2, 0.2), seed=0,
                          stratify_key=lambda x: x["c"])
    for split in (ds.train, ds.val, ds.test):
        classes = {x["c"] for x in split}
        assert classes == {"a", "b"}           # both classes present in each split


def test_dataset_from_splits():
    ds = dl.dataset_from_splits([1, 2], [3], [4, 5], name="x")
    assert ds.sizes() == (2, 1, 2) and ds.name == "x"
