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
* :func:`fetch_text` / :func:`fetch_bytes` -- a raw-URL fetch with optional
  on-disk cache (for data hosted as plain files on GitHub, etc.);
* :func:`load_gated_hf` -- a best-effort loader for **gated** datasets via a lazy
  ``datasets`` import + an ``HF_TOKEN``; returns ``None`` if unavailable.

Everything caches under ``~/.cache/agentdescent/<subdir>``. The URL/paging helpers
are pure so they can be unit-tested without a network.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
import random
import re
import tempfile
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
            # A group too small for the ratios has to give val an item from
            # somewhere -- but by *moving* one, not copying it. `g[a:b] or
            # g[a:a+1]` did the latter: for a two-item class the cut points
            # collapse to a == b == 1, so val borrowed g[1] while test still took
            # g[b:] == [g[1]]. The same item then sat in the split the optimizer
            # gates on *and* in the one this class documents as "fully held out,
            # never seen by the optimizer", which is the one promise a test split
            # makes. Rare classes are exactly where this bites, and stratifying is
            # what you do when you have them (FiNER tags, MGSM languages).
            if a == b:
                if b < len(g):
                    b += 1              # take val's item from test
                elif a > 0:
                    a -= 1              # nothing left there: take it from train
            train += g[:a]
            val += g[a:b]
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


def _rows_cache_component(value: str, identity: Optional[str] = None) -> str:
    """Bound a page-filename component without aliasing lossy conversions.

    ``identity`` is the raw string the digest must distinguish when ``value``
    is itself already a lossy rendering (the config slug): two configs that
    slug identically must still hash apart once the name has to be bounded.
    """
    slug = _slug(value)
    if re.fullmatch(r"[A-Za-z0-9._-]*", value) and len(value) <= 48:
        return value
    # Retain the original identity when sanitizing or truncating. `~` is
    # portable but absent from _slug's alphabet, so an ordinary split cannot
    # impersonate a transformed component by spelling out its digest suffix.
    raw = value if identity is None else identity
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"{slug[:12]}~{digest}"


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
                timeout: float = 60.0, cache: bool = True) -> bytes:
    """Fetch a raw URL, optionally caching the bytes on disk (keyed by ``filename``).

    With ``cache=True`` (default), an existing cache entry is returned and a miss
    is written after the download. With ``cache=False``, an existing cache entry
    is ignored and the response is not written — matching :func:`hf_rows` /
    ``_rows_payload`` bypass semantics.
    """
    path = _raw_cache_path(url, cache_subdir, filename)
    if cache and os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    data = _read_url(url, timeout)
    if cache:
        _write_cache(path, data)
    return data


def _raw_cache_path(url: str, cache_subdir: str, filename: Optional[str]) -> str:
    return cache_path(cache_subdir, filename or _slug(url)[-180:])


def _write_cache(path: str, data: bytes) -> None:
    """Write a cache entry atomically: readers see the whole response or none.

    Writing straight into ``path`` left a truncated file after Ctrl-C, OOM or a
    full disk -- and a concurrent reader could see a half-written one -- and
    since a cache hit is just ``os.path.exists``, that file was served forever.
    """
    # A short unique sibling, not `path + suffix`: an entry name near NAME_MAX
    # (255 bytes on most filesystems) must not fail only because of the temp.
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".ad-",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def fetch_text(url: str, *, cache_subdir: str = "raw", filename: Optional[str] = None,
               timeout: float = 60.0, encoding: str = "utf-8",
               cache: bool = True) -> str:
    """Fetch a raw URL as decoded text, optionally cached on disk.

    ``cache`` has the same meaning as in :func:`fetch_bytes`: ``False`` bypasses
    and does not refresh the on-disk entry.
    """
    return fetch_bytes(url, cache_subdir=cache_subdir, filename=filename,
                       timeout=timeout, cache=cache).decode(encoding, errors="ignore")


# ---------------------------------------------------------------------------
# HuggingFace datasets-server (dependency-free)
# ---------------------------------------------------------------------------


def _rows_payload(dataset: str, config: str, split: str, offset: int, length: int,
                  *, cache: bool, timeout: float) -> dict:
    subdir = os.path.join("hf", _slug(dataset))
    # Preserve existing short config slugs; only the filename is transformed,
    # never the request. Ordinary filenames retain their existing cache hits.
    parts = (_slug(config), split, str(offset), str(length))
    # The config slug is already lossy ("a/x" and "a:x" both become "a_x"), so
    # when it must be bounded the digest is taken over the raw config, not the
    # slug -- otherwise two long configs share one cache page (review on #262).
    components = (_rows_cache_component(parts[0], identity=config),) + tuple(
        _rows_cache_component(part) for part in parts[1:])
    # Legacy names always end in an integer length followed by `.json`.
    # A suffix after that field prevents an old literal split from aliasing a
    # transformed component. At least one component is <=45 bytes in this
    # case, so the longer suffix still keeps the full basename <=200 bytes.
    suffix = ".json" if components == parts else ".v2.json"
    fname = "_".join(components) + suffix
    return _cached_json(cache_path(subdir, fname),
                        rows_url(dataset, config, split, offset, length),
                        cache=cache, timeout=timeout)


def _cached_json(path: str, url: str, *, cache: bool, timeout: float) -> Any:
    """``url``'s JSON body, served from / stored at the cache entry ``path``.

    * The entry is read as bytes: it holds the server's raw UTF-8, and a
      text-mode ``open()`` decodes with the locale's codec (cp1252 / cp936 on
      Windows), so a cache hit used to fail on text the first call returned.
    * An entry that does not parse (torn by an interrupted write, or a non-JSON
      body cached by an older version) is refetched and replaced, not raised on
      forever.
    * A fresh response is parsed *before* it is cached, so a non-JSON body
      raises and never reaches the cache.
    * ``cache=False`` neither reads nor writes the entry.
    """
    if cache and os.path.exists(path):
        with open(path, "rb") as f:
            cached = f.read()
        try:
            return json.loads(cached)
        except ValueError:
            pass
    data = _read_url(url, timeout)
    payload = json.loads(data)
    if cache:
        _write_cache(path, data)
    return payload


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
# ModelScope mirror (for datasets whose HuggingFace endpoint is unreachable)
# ---------------------------------------------------------------------------

#: A ModelScope-hosted copy of HotpotQA, fetched as a plain JSON file. The
#: HuggingFace datasets-server can be unreachable from some networks while
#: ModelScope stays up, so :func:`hf_rows_modelscope` is the fallback that
#: loads the *same* rows from a different mirror.
MODELSCOPE_HOTPOTQA_URL = (
    "https://www.modelscope.cn/api/v1/datasets/OpenDataLab/HotpotQA/repo"
    "?Revision=master&FilePath=raw%2Fhotpot_dev_distractor_v1.json"
)


def _hotpotqa_context_to_dict(context) -> dict:
    """Normalise a HotpotQA ``context`` to the dict shape the engine expects.

    The HuggingFace datasets-server already yields ``{"title": [...],
    "sentences": [[...], ...]}``; the raw JSON mirror yields the original
    ``[[title, [sentence, ...]], ...]`` pairs. Callers (GEPA's
    ``_render_context``) read ``context["title"]`` and ``context["sentences"]``,
    so the pairs are converted here, once, at load time.
    """
    if isinstance(context, dict):
        return context
    titles, sents = [], []
    for title, sentences in context:
        titles.append(title)
        sents.append(list(sentences))
    return {"title": titles, "sentences": sents}


def hf_rows_modelscope(url: str = MODELSCOPE_HOTPOTQA_URL, *,
                       limit: int = 100, cache: bool = True,
                       timeout: float = 60.0) -> List[dict]:
    """Load HotpotQA rows from a ModelScope mirror (plain JSON file).

    Equivalent to :func:`hf_rows` for the ``hotpotqa/hotpot_qa`` validation
    split, but via a ModelScope-hosted raw file instead of the HuggingFace
    datasets-server -- for networks where one reachable mirror and not the
    other. Rows are the same fields (``context`` / ``question`` / ``answer``),
    with ``context`` normalised to the ``{"title": ..., "sentences": ...}``
    dict the examples read.

    With ``cache=True`` (default), responses live under
    ``~/.cache/agentdescent/modelscope/`` keyed by the URL so a repeat call
    does not re-download the ~46 MB file. With ``cache=False``, any existing
    cache entry is ignored and the fresh response is not written back — the
    same bypass contract as :func:`hf_rows`.
    """
    rows = _cached_json(_raw_cache_path(url, "modelscope", None), url,
                        cache=cache, timeout=timeout)
    out: List[dict] = []
    for r in rows[:limit]:
        out.append({**r, "context": _hotpotqa_context_to_dict(r.get("context", []))})
    return out


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

def select_hard(items: Sequence[Any], score: Callable[[Any], float],
                keep: Optional[int] = None, threshold: float = 0.999,
                concurrency: int = 8, min_items: int = 12,
                passes: int = 1) -> List[Any]:
    """Keep the items a baseline gets **wrong** -- headroom, from a saturated set.

    A benchmark a model already solves cannot demonstrate a skill: there is
    nothing to add, and a correct implementation commits nothing. Measured with
    ``deepseek-v4-flash``, three of the shipped ports sit at 0.9-1.0 out of the
    box (FiNER-139 at the default concept count, SearchQA, MGSM).

    Swapping in another dataset breaks fidelity to the paper being ported, so the
    other lever is to keep the dataset and drop the items that carry no signal.
    One baseline pass, scored concurrently, keeps whatever falls below
    ``threshold``:

        items = select_hard(items, lambda it: score(solve(it), it["answer"]))

    This makes the *benchmark* harder, so numbers from it are not comparable with
    numbers from the full set -- say which you used. ``keep`` caps the result
    (the hardest are kept in their original order); if nothing fails, everything
    is returned rather than an empty set.

    ``passes`` is how many independent baseline measurements an item has to fail
    before it counts as hard, and 1 is not enough for a non-deterministic model.
    Selecting on a single noisy measurement selects the *unlucky* answers, and
    re-measuring them regresses to the mean -- measured on SkillOpt/SearchQA with
    ``deepseek-v4-flash``: the filter kept only items the seed skill got wrong,
    and the gate then scored that same split at **1.000**. Every one of them was
    answered correctly the second time. A run set up that way cannot show
    anything: the strict gate needs `candidate > current` and there is nothing
    above 1.000. Raise this to 2 or 3 where the actor is a sampled model; the
    filter costs `passes` times as much and stops selecting for luck.
    """
    from concurrent.futures import ThreadPoolExecutor

    items = list(items)
    if not items:
        return items
    with ThreadPoolExecutor(max(1, min(concurrency, len(items)))) as pool:
        # An item is hard only if it fails *every* pass; `scores` keeps the best
        # one it ever achieved, so the top-up below still orders by difficulty.
        scores = list(pool.map(score, items))
        for _ in range(max(0, passes - 1)):
            scores = [max(prev, new) for prev, new
                      in zip(scores, pool.map(score, items))]
    hard = [it for it, sc in zip(items, scores) if sc < threshold]
    if not hard:
        return items                      # nothing failed: caller keeps the full set
    if len(hard) < min_items:
        # The whole point is a benchmark that is nearly saturated, so the survivors
        # can be a handful -- and a handful splits into a 2-item validation set that
        # measures nothing, or crashes the engine outright. Say so, and top up with
        # the easiest items rather than returning something unusable.
        warnings.warn(
            f"select_hard kept only {len(hard)} of {len(items)} items "
            f"({1 - len(hard)/len(items):.0%} of the pool is already solved). "
            f"Topping up to {min_items}; pass a larger pool for a subset that is "
            f"entirely hard.", RuntimeWarning, stacklevel=2)
        rest = [it for it, sc in zip(items, scores) if sc >= threshold]
        hard = hard + rest[:min_items - len(hard)]
    return hard[:keep] if keep else hard
