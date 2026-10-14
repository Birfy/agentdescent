"""Offline tests for the agentdescent.dataloader data layer.

The URL/paging helpers are pure; the HTTP fetch is monkeypatched so hf_rows /
fetch_text are exercised without a network.
"""

import json
import os

from agentdescent import dataloader as dl


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


def test_a_stratified_split_never_puts_an_item_in_both_val_and_test():
    """`test` is documented as never seen by the optimizer; `val` is what it gates on.

    A class too small for the ratios used to have its item *copied* into val
    while test kept it too -- and a rare class is exactly what stratifying is
    for (a thin FiNER tag, a low-resource MGSM language).
    """
    for rare_n in (1, 2, 3):
        items = ([{"c": "rare", "i": i} for i in range(rare_n)]
                 + [{"c": "common", "i": i} for i in range(10)])
        ds = dl.split_dataset(items, ratios=(0.6, 0.2, 0.2), seed=0,
                              stratify_key=lambda x: x["c"])
        ids = [{(x["c"], x["i"]) for x in s} for s in (ds.train, ds.val, ds.test)]
        train, val, test = ids
        assert not (val & test), f"rare_n={rare_n}: {sorted(val & test)} is in val AND test"
        assert not (train & val) and not (train & test)
        assert len(train | val | test) == len(items), "an item was dropped"
        assert "rare" in {c for c, _ in val}, "stratification must still reach val"


def test_dataset_from_splits():
    ds = dl.dataset_from_splits([1, 2], [3], [4, 5], name="x")
    assert ds.sizes() == (2, 1, 2) and ds.name == "x"


def test_val_frac_round_trips_through_evolve():
    """Dataset.val_frac promises the engine's held-out split IS the Dataset's val.

    Float truncation (13.9999 -> 13) silently moved one train item into held-out
    for many dataset sizes, so the two splits disagreed.
    """
    from agentdescent.dataloader import split_dataset
    from agentdescent.evolution import Task, _build_engine

    for n in (29, 87, 100, 149, 200):
        rows = [{"i": i} for i in range(n)]
        ds = split_dataset(rows, ratios=(0.5, 0.25, 0.25), seed=0)
        tasks = [Task(id=f"t{i}", prompt="q") for i in range(len(ds.trainval))]

        eng = _build_engine(
            tasks, lambda t, o: 1.0, agent=None,
            run=lambda r, t: "x", propose=lambda r, t, o, s: None,
            strategy=None, initial_state=None, blast_radius=0.2,
            artifact_id="a", held_out_frac=ds.val_frac, repo_path=None,
            agg_config=None, staleness_policy=None, aggregator_factory=None,
            oracle_budget=10)
        assert len(eng.train) == len(ds.train), (
            f"n={n}: engine train={len(eng.train)} but Dataset train={len(ds.train)}")
        assert len(eng.held_out) == len(ds.val)


def test_modelscope_context_pairs_are_normalised_to_dict():
    """The ModelScope HotpotQA mirror stores ``context`` as the original
    ``[[title, [sentences]], ...]`` pairs; examples read it as the dict shape
    the HuggingFace datasets-server yields. The loader must convert, once, so
    GEPA's `_render_context` works unchanged on either source."""
    pairs = [["Ed Wood (film)", ["sentence one.", " sentence two."]],
             ["Scott Derrickson", ["Scott is an American director."]]]
    out = dl._hotpotqa_context_to_dict(pairs)
    assert out == {"title": ["Ed Wood (film)", "Scott Derrickson"],
                   "sentences": [["sentence one.", " sentence two."],
                                 ["Scott is an American director."]]}
    # already-a-dict passes through untouched (the HF datasets-server shape)
    already = {"title": ["x"], "sentences": [["y"]]}
    assert dl._hotpotqa_context_to_dict(already) is already


def test_modelscope_rows_are_cached(tmp_path, monkeypatch):
    """`hf_rows_modelscope` fetches the ~46 MB raw file once and serves repeat
    calls from cache; the HTTP round-trip is exercised without a network."""
    monkeypatch.setattr(dl, "CACHE_ROOT", str(tmp_path))
    n_fetch = {"n": 0}

    def fake_read(url, timeout):
        n_fetch["n"] += 1
        return json.dumps([
            {"question": "Q1", "answer": "A1",
             "context": [["t1", ["s1"]]]},
        ]).encode()

    monkeypatch.setattr(dl, "_read_url", fake_read)
    rows1 = dl.hf_rows_modelscope(limit=10)
    rows2 = dl.hf_rows_modelscope(limit=10)      # served from cache
    assert n_fetch["n"] == 1
    assert len(rows1) == 1 and len(rows2) == 1
    assert rows1[0]["context"] == {"title": ["t1"], "sentences": [["s1"]]}

def test_modelscope_cache_false_bypasses_stale_cache(tmp_path, monkeypatch):
    """``cache=False`` must not reuse a populated ModelScope cache entry.

    A caller asking for fresh rows after a cached load previously still got the
    old bytes because ``hf_rows_modelscope`` ignored its ``cache`` flag when
    calling ``fetch_bytes`` (#229).
    """
    monkeypatch.setattr(dl, "CACHE_ROOT", str(tmp_path))
    state = {"answer": "old"}
    calls = []

    def fake_read(url, timeout):
        calls.append(url)
        return json.dumps([
            {"question": "Q", "answer": state["answer"],
             "context": [["T", ["S"]]]},
        ]).encode()

    monkeypatch.setattr(dl, "_read_url", fake_read)
    url = "https://example.invalid/hotpotqa.json"
    first = dl.hf_rows_modelscope(url=url, limit=1, cache=True)
    state["answer"] = "fresh"
    bypass = dl.hf_rows_modelscope(url=url, limit=1, cache=False)
    assert first[0]["answer"] == "old"
    assert bypass[0]["answer"] == "fresh"
    assert len(calls) == 2
    # Bypass must not overwrite the on-disk entry (same contract as hf_rows).
    cached_again = dl.hf_rows_modelscope(url=url, limit=1, cache=True)
    assert cached_again[0]["answer"] == "old"
    assert len(calls) == 2


def test_modelscope_repeated_cache_false_sees_changing_responses(tmp_path, monkeypatch):
    """Repeated uncached ModelScope calls observe each mocked response."""
    monkeypatch.setattr(dl, "CACHE_ROOT", str(tmp_path))
    answers = iter(["a", "b", "c"])
    calls = []

    def fake_read(url, timeout):
        calls.append(url)
        return json.dumps([
            {"question": "Q", "answer": next(answers),
             "context": [["T", ["S"]]]},
        ]).encode()

    monkeypatch.setattr(dl, "_read_url", fake_read)
    url = "https://example.invalid/hotpotqa-repeat.json"
    seen = [dl.hf_rows_modelscope(url=url, limit=1, cache=False)[0]["answer"]
            for _ in range(3)]
    assert seen == ["a", "b", "c"]
    assert len(calls) == 3


def test_fetch_bytes_cache_false_skips_read_and_write(tmp_path, monkeypatch):
    """``fetch_bytes(..., cache=False)`` ignores and does not refresh the file."""
    monkeypatch.setattr(dl, "CACHE_ROOT", str(tmp_path))
    payloads = iter([b"first", b"second"])
    n = {"n": 0}

    def fake_read(url, timeout):
        n["n"] += 1
        return next(payloads)

    monkeypatch.setattr(dl, "_read_url", fake_read)
    assert dl.fetch_bytes("http://x/y", filename="y.bin") == b"first"
    assert dl.fetch_bytes("http://x/y", filename="y.bin", cache=False) == b"second"
    assert n["n"] == 2
    assert dl.fetch_bytes("http://x/y", filename="y.bin") == b"first"
    assert n["n"] == 2



# ---------------------------------------------------------------------------
# Cache robustness (#255): torn entries, non-JSON bodies, locale-free reads
# ---------------------------------------------------------------------------


def _cache_files(root):
    return sorted(os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs)


def _counting_reader(body):
    calls = {"n": 0}

    def fake_read(url, timeout):
        calls["n"] += 1
        return body() if callable(body) else body

    return fake_read, calls


def test_a_torn_hf_rows_page_is_refetched_and_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CACHE_ROOT", str(tmp_path))
    good = json.dumps(_fake_rows_payload(3)).encode()
    fake, calls = _counting_reader(good)
    monkeypatch.setattr(dl, "_read_url", fake)
    dl.hf_rows("ds", "train", limit=3)
    (path,) = _cache_files(tmp_path)
    with open(path, "wb") as fh:
        fh.write(good[:20])                       # interrupted write, old version
    assert len(dl.hf_rows("ds", "train", limit=3)) == 3
    assert calls["n"] == 2
    with open(path, "rb") as fh:
        assert fh.read() == good                  # entry repaired
    assert len(dl.hf_rows("ds", "train", limit=3)) == 3
    assert calls["n"] == 2                        # and served from cache again


def test_a_torn_modelscope_entry_is_refetched_and_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "CACHE_ROOT", str(tmp_path))
    good = json.dumps([{"question": "Q", "answer": "A", "context": [["t", ["s"]]]}]).encode()
    fake, calls = _counting_reader(good)
    monkeypatch.setattr(dl, "_read_url", fake)
    url = "https://example.invalid/hotpotqa.json"
    dl.hf_rows_modelscope(url=url, limit=1)
    (path,) = _cache_files(tmp_path)
    with open(path, "wb") as fh:
        fh.write(good[:10])
    assert dl.hf_rows_modelscope(url=url, limit=1)[0]["answer"] == "A"
    assert dl.hf_rows_modelscope(url=url, limit=1)[0]["answer"] == "A"
    assert calls["n"] == 2


def test_a_non_json_response_raises_and_is_never_cached(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(dl, "CACHE_ROOT", str(tmp_path))
    fake, calls = _counting_reader(b"<html>502 Bad Gateway</html>")
    monkeypatch.setattr(dl, "_read_url", fake)
    with pytest.raises(ValueError):
        dl.hf_rows("ds", "train", limit=3)
    with pytest.raises(ValueError):
        dl.hf_rows_modelscope(url="https://example.invalid/x.json", limit=1)
    assert _cache_files(tmp_path) == []           # nothing poisoned, no temp left


def test_a_failed_cache_write_leaves_no_temp_file(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(dl, "CACHE_ROOT", str(tmp_path))
    monkeypatch.setattr(dl, "_read_url", lambda url, timeout: b"payload")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(dl.os, "replace", boom)
    with pytest.raises(OSError):
        dl.fetch_bytes("http://x/y", filename="y.bin")
    assert _cache_files(tmp_path) == []


def test_a_cache_hit_does_not_depend_on_the_locale_encoding(tmp_path):
    """The first call parses bytes; the cached call used text-mode ``open()``,
    which decodes with the locale codec (cp1252 on Windows) and failed on
    non-ASCII rows. Simulated with an ASCII locale and UTF-8 mode off."""
    import subprocess
    import sys
    script = (
        "import json, sys\n"
        "from agentdescent import dataloader as dl\n"
        "dl.CACHE_ROOT = sys.argv[1]\n"
        "body = json.dumps({'rows': [{'row': {'q': '\\u65e5\\u672c\\u8a9e'}}]},"
        " ensure_ascii=False).encode('utf-8')\n"
        "dl._read_url = lambda url, timeout: body\n"
        "a = dl.hf_rows('ds', 'train', limit=1)\n"
        "b = dl.hf_rows('ds', 'train', limit=1)\n"
        "assert a == b, (ascii(a), ascii(b))\n"
        "print(ascii(b[0]['q']))\n"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONCOERCECLOCALE="0", LC_ALL="C", LANG="C",
               PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.pop("PYTHONUTF8", None)
    proc = subprocess.run([sys.executable, "-X", "utf8=0", "-c", script, str(tmp_path)],
                          env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "'\\u65e5\\u672c\\u8a9e'"
