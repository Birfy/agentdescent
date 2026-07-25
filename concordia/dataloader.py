"""Dependency-free dataset loading -- the *data layer* for examples/experiments.

Just as :mod:`concordia.agents` is the provider-agnostic "talk to a model" layer,
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

Everything caches under ``~/.cache/concordia/<subdir>``. The URL/paging helpers
are pure so they can be unit-tested without a network.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from typing import List, Optional, Tuple

ROWS_URL = "https://datasets-server.huggingface.co/rows"
CACHE_ROOT = os.path.expanduser("~/.cache/concordia")
# The datasets-server caps a single /rows request at 100 rows.
PAGE_MAX = 100


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
