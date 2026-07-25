# Loading datasets

> **The data layer.** Just as [`concordia.agents`](agents.md) is the "talk to a
> model" layer, `concordia.dataloader` is the "load a dataset" layer. It is
> deliberately separate from the evolution engine — *which* benchmark you evolve
> against has nothing to do with the framework — and every
> [self-evolution example](self-evolution-examples.md) loads its data through it.

The examples each need a public benchmark (FiNER, HotpotQA, SearchQA, MGSM,
SWE-bench Verified, OfficeQA). Rather than re-implement HuggingFace paging and
on-disk caching in every file, that boilerplate lives here — dependency-free
(`urllib` only), cached under `~/.cache/concordia/`.

## The surface

```python
from concordia.dataloader import (
    hf_rows, hf_feature_names, fetch_text, fetch_bytes, load_gated_hf)
```

| Function | What it does |
|---|---|
| `hf_rows(dataset, split, *, config, limit)` | rows of any **public** dataset via the HF **datasets-server** `/rows` API — paged (≤100/req) and cached |
| `hf_feature_names(dataset, split, feature, *, config)` | the label vocabulary of a `ClassLabel` (or nested `Sequence[ClassLabel]`) feature |
| `fetch_text(url, *, cache_subdir, filename)` / `fetch_bytes(...)` | a cached raw-URL fetch (data hosted as plain files, e.g. on GitHub) |
| `load_gated_hf(dataset, split)` | best-effort load of a **gated** dataset via a lazy `datasets` import + `HF_TOKEN`; returns `None` if unavailable |

`rows_url(...)` and `page_offsets(...)` are pure helpers (no network), unit-tested
in `tests/test_dataloader.py`.

## Examples

```python
from concordia.dataloader import hf_rows, hf_feature_names, fetch_text, load_gated_hf

# Public dataset via the datasets-server (paged + cached), any split/config:
rows = hf_rows("hotpotqa/hotpot_qa", "validation", config="distractor", limit=200)

# A token-classification label vocabulary (FiNER's 279 BIO XBRL tags):
names = hf_feature_names("nlpaueb/finer-139", "validation", "ner_tags", config="finer-139")

# Data hosted as a raw file (ADAS ships MGSM as TSVs on GitHub):
tsv = fetch_text("https://raw.githubusercontent.com/ShengranHu/ADAS/main/"
                 "dataset/mgsm/mgsm_en.tsv", cache_subdir="mgsm", filename="mgsm_en.tsv")

# A gated dataset (falls back to None so callers can degrade gracefully):
rows = load_gated_hf("databricks/officeqa", "test")   # needs HF_TOKEN, else None
```

## How the examples use it

Each example keeps only its **dataset-specific shaping** (turning rows into
`Task`s, building the reward) and delegates the fetch/cache to the data layer:

```python
# examples/gepa_prompt_evolution.py
from concordia.dataloader import hf_rows

HOTPOTQA = ("hotpotqa/hotpot_qa", "validation", "distractor")

def download_hotpotqa(limit):
    dataset, split, config = HOTPOTQA
    return hf_rows(dataset, split, config=config, limit=limit)
```

| Example | Data layer call |
|---|---|
| ACE (FiNER-139) | `hf_rows(..., config="finer-139")` + `hf_feature_names(...)` |
| GEPA (HotpotQA) | `hf_rows(..., config="distractor")` |
| SkillOpt (SearchQA) | `hf_rows("lucadiliello/searchqa", ...)` |
| DGM (SWE-bench Verified) | `hf_rows("princeton-nlp/SWE-bench_Verified", "test")` |
| ADAS (MGSM) | `fetch_text(<raw TSV url>)` |
| EvoSkill (OfficeQA) | `load_gated_hf(...)` → falls back to `fetch_text(<bundled CSV>)` |

## Design notes

* **Dependency-free public path.** `hf_rows` / `fetch_text` use only `urllib`, so
  the examples install nothing extra. The `datasets` library is imported lazily,
  and *only* inside `load_gated_hf`, for gated datasets that need auth.
* **Cache-first.** Every page and file is cached under `~/.cache/concordia/`;
  re-runs and `--dry-run` are offline after the first fetch.
* **Not in the engine.** Nothing in `concordia.evolution` / `concordia.aggregator`
  imports this — it is a convenience for examples and experiments, exactly like
  `concordia.agents`.
