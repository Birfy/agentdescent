# Loading datasets

> **The data layer.** Just as [`agentdescent.agents`](agents.md) is the "talk to a
> model" layer, `agentdescent.dataloader` is the "load a dataset" layer. It is
> deliberately separate from the evolution engine — *which* benchmark you evolve
> against has nothing to do with the framework — and every
> dataset-backed [self-evolution example](self-evolution-examples.md) loads its data
> through it (the seven benchmark-faithful ports; the eleven MethodPolicy ports
> run bundled deterministic domains instead).

The examples each need a public benchmark (FiNER, HotpotQA, SearchQA, MGSM,
SWE-bench Verified, OfficeQA). Rather than re-implement HuggingFace paging and
on-disk caching in every file, that boilerplate lives here — dependency-free
(`urllib` only), cached under `~/.cache/agentdescent/`.

## The surface

```python
from agentdescent.dataloader import (
    Dataset, split_dataset, dataset_from_splits,          # the train/val/test layer
    hf_rows, hf_feature_names, fetch_text, fetch_bytes, load_gated_hf)   # loaders
```

| Function | What it does |
|---|---|
| `Dataset` | a **train / val / test** partition, with `.trainval`, `.val_frac`, `.map`, `.sizes()` |
| `split_dataset(items, *, ratios, seed, stratify_key)` | partition items into a `Dataset` (optionally class-stratified) |
| `dataset_from_splits(train, val, test)` | build a `Dataset` from splits a source already provides |
| `hf_rows(dataset, split, *, config, limit)` | rows of any **public** dataset via the HF **datasets-server** `/rows` API — paged (≤100/req) and cached |
| `hf_feature_names(dataset, split, feature, *, config)` | the label vocabulary of a `ClassLabel` (or nested `Sequence[ClassLabel]`) feature |
| `fetch_text(url, *, cache_subdir, filename)` / `fetch_bytes(...)` | a cached raw-URL fetch (data hosted as plain files, e.g. on GitHub) |
| `load_gated_hf(dataset, split)` | best-effort load of a **gated** dataset via a lazy `datasets` import + `HF_TOKEN`; returns `None` if unavailable |

`rows_url(...)`, `page_offsets(...)`, `split_dataset(...)` are pure (no network),
unit-tested in `tests/test_dataloader.py`.

## `Dataset` — a train / val / test partition

Every self-evolution example follows the same discipline: **fit on `train`, gate
/ select on `val`** (the held-out set `evolve()` optimises against), and **report
a final number on `test`** (fully held out, never seen by the optimizer).

```python
from agentdescent.dataloader import split_dataset

ds = split_dataset(tasks, ratios=(0.5, 0.25, 0.25), seed=0,
                   stratify_key=lambda t: t.meta["target"])   # optional class balance
ds.sizes()          # (n_train, n_val, n_test)

# run the optimizer on train+val so evolve()'s held-out split IS ds.val:
result = evolve(ds.trainval, reward, agent=agent, held_out_frac=ds.val_frac, ...)

# then score the evolved artifact on the untouched test split:
test_metric = evaluate(agent, result.rendered, ds.test, reward)
```

`ds.val_frac` is `|val| / |train+val|`, so passing it as `held_out_frac` makes the
engine's internal held-out split exactly `ds.val`. When a source ships native
splits (e.g. SearchQA's `train` / `validation`), build the `Dataset` with
`dataset_from_splits(...)` instead of re-splitting.

## Examples

```python
from agentdescent.dataloader import hf_rows, hf_feature_names, fetch_text, load_gated_hf

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

## How the dataset-backed ports use it

Each example keeps only its **dataset-specific shaping** (turning rows into
`Task`s, building the reward) and delegates the fetch/cache to the data layer:

```python
# examples/gepa/gepa_prompt_evolution.py
from agentdescent.dataloader import hf_rows

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
* **Cache-first.** Every page and file is cached under `~/.cache/agentdescent/`;
  real re-runs are offline after the first fetch. Faithful-port `--dry-run`
  returns before the loader and is offline even with an empty cache.
* **Not in the engine.** Nothing in `agentdescent.evolution` / `agentdescent.aggregator`
  imports this — it is a convenience for examples and experiments, exactly like
  `agentdescent.agents`.

## Turning a saturated benchmark into one with headroom — `select_hard`

A benchmark your model already solves cannot demonstrate a skill: there is nothing
to add, and a correct implementation commits nothing. Measured with
`deepseek-v4-flash`, three of the shipped ports sit at 0.9–1.0 out of the box
(FiNER-139 at the default concept count, SearchQA, MGSM).

Swapping datasets breaks fidelity to the paper being ported, so the other lever is
to keep the dataset and drop the items that carry no signal:

```python
from agentdescent.dataloader import select_hard

items = select_hard(items, lambda it: score(solve(it), it["answer"]))
```

One baseline pass, scored concurrently, keeping whatever falls below `threshold`
(default: anything not fully correct). `keep=` caps the result; if nothing fails
it returns the full set rather than nothing.

!!! warning "This changes the benchmark, not just the sample"
    Numbers from a hard subset are not comparable with numbers from the full set.
    Report which one you used — [Measured results](results.md) does.
