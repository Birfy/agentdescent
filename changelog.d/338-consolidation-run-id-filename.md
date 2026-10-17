Fixed
`ConsolidationStore` filenames hash the complete `run_id`, so distinct ids
such as `a/b` and `ab` no longer share `runs/ab.json` and overwrite each
other's archived products; `find` still accepts a pre-#338 sanitized file
only when the stored identity matches (#338).
