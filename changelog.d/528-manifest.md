Added
`RunManifest` / `build_manifest` / `verify_manifest` / `diff_manifest`: a
content-addressed manifest of a run's spec, data files or directory tree, model
configuration (secret keys redacted), seeds, algorithm and code version. The
manifest id is stable for identical inputs, and verification names the fields
that changed.
