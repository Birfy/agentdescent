Fixed
`OpsFabric.record` now closes the `ops.record` span as `error` for a
failed run and `cancelled` for a cancelled one, instead of `ok` for every
terminal run, so telemetry error counts match the console (#630).
