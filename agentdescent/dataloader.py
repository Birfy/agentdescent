"""Dependency-free dataset loading -- the *data layer* for examples/experiments.

Just as :mod:`agentdescent.agents` is the provider-agnostic "talk to a model" layer,
this is the "load a dataset" layer. It is deliberately kept **out of** the
evolution engine: which benchmark you evolve against has nothing to do with the
framework. The examples used to each re-implement the same HuggingFace
datasets-server paging + on-disk caching; this module centralises it.

The whole surface is small:

* :func:`hf_rows` -- pull rows of any public dataset from the HuggingFace
  **datasets-server** JSON API (``/rows``), paged and cached, using only
  ``urllib`` (no ``datasets`` dependency);
* :func:`hf_feature_names` -- the label vocabulary of a ClassLabel feature
  (e.g. FiNER's XBRL tag names);
* :func:`fetch_text` / :func:`fetch_bytes` -- a cached raw-URL fetch (for data
  hosted as plain files on GitHub, etc.);
* :func:`load_gated_hf` -- a best-effort loader for **gated** datasets via a lazy
  ``datasets`` import + an ``HF_TOKEN``; returns ``None`` if unavailable.

Everything caches under ``~/.cache/agentdescent/<subdir>``. The URL/paging helpers
are pure so they can be unit-tested without a network.
"""

from __future__ import annotations

import json
import os
import random
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

ROWS_URL = "https://datasets-server.huggingface.co/rows"
CACHE_ROOT = os.path.expanduser("~/.cache/agentdescent")
# The datasets-server caps a single /rows request at 100 rows.
PAGE_MAX = 100


# ---------------------------------------------------------------------------
# Dataset: a train / val / test partition
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    """A dataset partitioned into **train / val / test** splits.

    The examples all share the same discipline: fit on ``train``, select/gate on
    ``val`` (the held-out set `evolve()` optimises against), and report a final
    number on ``test`` (fully held out, never seen by the optimizer). ``map``
    shapes every split at once (e.g. raw rows -> ``Task`` objects)."""

    train: List[Any] = field(default_factory=list)
    val: List[Any] = field(default_factory=list)
    test: List[Any] = field(default_factory=list)
    name: str = ""

    def sizes(self) -> Tuple[int, int, int]:
        return (len(self.train), len(self.val), len(self.test))

    def __len__(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)

    def map(self, fn: Callable[[Any], Any]) -> "Dataset":
        """Apply ``fn`` to every item in every split, returning a new Dataset."""
        return Dataset([fn(x) for x in self.train], [fn(x) for x in self.val],
                       [fn(x) for x in self.test], self.name)

    @property
    def trainval(self) -> List[Any]:
        """``train + val`` -- what `evolve()` consumes as its ``tasks``."""
        return list(self.train) + list(self.val)

    @property
    def val_frac(self) -> float:
        """``|val| / |train+val|`` -- pass as `evolve(held_out_frac=...)` so the
        engine's held-out split is exactly this Dataset's ``val``."""
        n = len(self.train) + len(self.val)
        return len(self.val) / n if n else 0.0


def _cut_points(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int]:
    tr, va, _ = ratios
    a = min(n, max(0, int(round(n * tr))))
    b = min(n, max(a, a + int(round(n * va))))
    return a, b


def split_dataset(items: Sequence[Any], *, ratios: Tuple[float, float, float] = (0.6, 0.2, 0.2),
                  seed: int = 0, shuffle: bool = True,
                  stratify_key: Optional[Callable[[Any], Any]] = None,
                  name: str = "") -> Dataset:
    """Partition ``items`` into a :class:`Dataset` by ``ratios`` (train, val, test).

    Deterministic given ``seed``. With ``stratify_key`` each group (e.g. a task
    category / difficulty) is split by the same ratios so every split stays
    class-balanced."""
    items = list(items)
    rng = random.Random(seed)
    if stratify_key is not None:
        groups: dict = {}
        for it in items:
            groups.setdefault(stratify_key(it), []).append(it)
        train: List[Any] = []
        val: List[Any] = []
        test: List[Any] = []
        for key in sorted(groups, key=lambda k: str(k)):
            g = list(groups[key])
            if shuffle:
                rng.shuffle(g)
            a, b = _cut_points(len(g), ratios)
            train += g[:a]
            val += g[a:b] or g[a:a + 1]        # guarantee a non-empty val per class
            test += g[b:]
        for split in (train, val, test):
            if shuffle:
                rng.shuffle(split)
        return Dataset(train, val, test, name)
    if shuffle:
        rng.shuffle(items)
    a, b = _cut_points(len(items), ratios)
    return Dataset(items[:a], items[a:b], items[b:], name)


def dataset_from_splits(train: Sequence[Any], val: Sequence[Any],
                        test: Sequence[Any] = (), name: str = "") -> Dataset:
    """Build a :class:`Dataset` from splits a source already provides (e.g. HF
    native ``train`` / ``validation`` splits)."""
    return Dataset(list(train), list(val), list(test), name)


# ---------------------------------------------------------------------------
# Pure helpers (no network -- unit-testable)
# ---------------------------------------------------------------------------


def _slug(name: str) -> str:
    """A filesystem-safe cache subdir derived from a dataset name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def rows_url(dataset: str, config: str, split: str, offset: int, length: int) -> str:
    """Build the datasets-server ``/rows`` URL (pure)."""
    q = urllib.parse.urlencode({"dataset": dataset, "config": config,
                                "split": split, "offset": offset, "length": length})
    return f"{ROWS_URL}?{q}"


def page_offsets(limit: int, page: int = PAGE_MAX) -> List[Tuple[int, int]]:
    """Split ``limit`` rows into ``(offset, length)`` pages of at most ``page`` (pure)."""
    return [(off, min(page, limit - off)) for off in range(0, max(0, limit), page)]


# ---------------------------------------------------------------------------
# Cache + fetch
# ---------------------------------------------------------------------------


def cache_path(subdir: str, filename: str) -> str:
    d = os.path.join(CACHE_ROOT, subdir)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)


def _read_url(url: str, timeout: float) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def fetch_bytes(url: str, *, cache_subdir: str = "raw", filename: Optional[str] = None,
                timeout: float = 60.0) -> bytes:
    """Fetch a raw URL, caching the bytes on disk (keyed by ``filename``)."""
    filename = filename or _slug(url)[-180:]
    path = cache_path(cache_subdir, filename)
    if not os.path.exists(path):
        data = _read_url(url, timeout)
        with open(path, "wb") as f:
            f.write(data)
    with open(path, "rb") as f:
        return f.read()


def fetch_text(url: str, *, cache_subdir: str = "raw", filename: Optional[str] = None,
               timeout: float = 60.0, encoding: str = "utf-8") -> str:
    """Fetch a raw URL as decoded text, cached on disk."""
    return fetch_bytes(url, cache_subdir=cache_subdir, filename=filename,
                       timeout=timeout).decode(encoding, errors="ignore")


# ---------------------------------------------------------------------------
# HuggingFace datasets-server (dependency-free)
# ---------------------------------------------------------------------------


def _rows_payload(dataset: str, config: str, split: str, offset: int, length: int,
                  *, cache: bool, timeout: float) -> dict:
    subdir = os.path.join("hf", _slug(dataset))
    fname = f"{_slug(config)}_{split}_{offset}_{length}.json"
    path = cache_path(subdir, fname)
    if cache and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    data = _read_url(rows_url(dataset, config, split, offset, length), timeout)
    if cache:
        with open(path, "wb") as f:
            f.write(data)
    return json.loads(data)


def hf_rows(dataset: str, split: str, *, config: str = "default", limit: int = 100,
            cache: bool = True, timeout: float = 60.0) -> List[dict]:
    """Return up to ``limit`` rows of a public dataset via the datasets-server.

    Pages the ``/rows`` endpoint (≤100 rows each) and caches each page. Each
    returned element is the dataset row dict (the API's ``rows[i]["row"]``)."""
    out: List[dict] = []
    for offset, length in page_offsets(limit, PAGE_MAX):
        payload = _rows_payload(dataset, config, split, offset, length,
                                cache=cache, timeout=timeout)
        batch = [r["row"] for r in payload.get("rows", [])]
        out.extend(batch)
        if len(batch) < length:                       # dataset exhausted
            break
    return out


def hf_feature_names(dataset: str, split: str, feature: str, *,
                     config: str = "default", timeout: float = 60.0) -> List[str]:
    """The class-label names of a (possibly nested Sequence) ClassLabel feature.

    E.g. ``hf_feature_names("nlpaueb/finer-139", "validation", "ner_tags",
    config="finer-139")`` -> the 279 BIO XBRL tag names."""
    payload = _rows_payload(dataset, config, split, 0, 1, cache=True, timeout=timeout)
    for feat in payload["features"]:
        if feat["name"] == feature:
            t = feat["type"]
            if t.get("_type") == "Sequence" or "feature" in t:   # token-level labels
                return t["feature"]["names"]
            return t["names"]
    raise KeyError(f"feature {feature!r} not found in {dataset}")


# ---------------------------------------------------------------------------
# Gated datasets (optional, lazy `datasets` import)
# ---------------------------------------------------------------------------


def load_gated_hf(dataset: str, split: str, *,
                  token_envs: Tuple[str, ...] = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
                  ) -> Optional[List[dict]]:
    """Best-effort load of a **gated** dataset via the ``datasets`` library.

    Returns a list of row dicts, or ``None`` if no token is set, the library is
    missing, or access is denied -- so callers can fall back gracefully."""
    token = next((os.environ[e] for e in token_envs if os.environ.get(e)), None)
    if not token:
        return None
    try:
        from datasets import load_dataset  # lazy, optional dependency
        ds = load_dataset(dataset, split=split, token=token)
        return [dict(r) for r in ds]
    except Exception:  # noqa: BLE001 - any auth/library failure -> fall back
        return None
