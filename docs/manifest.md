# Run manifest — what made this run reproducible

A run is reproducible only if the spec, data, model configuration and code that
produced it are identifiable. `RunManifest` hashes those inputs into one stable
id plus a field-by-field record; `verify_manifest` says which input changed.

```python
from agentdescent import build_manifest, verify_manifest

manifest = build_manifest(
    spec,
    data_paths=[spec.data["path"]],
    model_config={"model": "m", "api_key": "sk-..."},
    code_version="0.8.0")
manifest.manifest_id
changed = verify_manifest(spec, manifest, data_paths=[...], code_version="0.8.0")
```

- **Spec** is the canonical `spec.to_dict()` hash.
- **Data** is a sorted set of content hashes (a directory is a sorted relative
  path/content tree), so identical bytes in a different directory share the
  hash; absolute paths are kept only as informational metadata, and a missing
  path is recorded explicitly.
- **Model config** is canonical JSON with secret-named keys (`api_key`,
  `access_token`, `token`, `password`, ...) and obvious secret value patterns
  (`Bearer ...`, `sk-...`, `api_key=...`) redacted before hashing; the raw
  secret is never stored. A normal key like `max_tokens` is **not** redacted.
- **Seeds/algorithm/code version** are explicit fields; `extra` is redacted and
  hashed.
- `manifest_id` is the hash of those fields, so two runs with identical inputs
  share it. `diff_manifest` / `verify_manifest` return the changed field names,
  which is what a replay/audit report needs.

A manifest is metadata only: it contains no artifact and no secret. Store it
beside a decision record or in a run directory.
