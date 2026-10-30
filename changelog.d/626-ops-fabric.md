Added
`OpsFabric`: one finished `LoopReport` is written into the operations console,
the lineage graph, the telemetry trace and an optional QD snapshot. Generations
become lineage nodes (chained from the last *accepted* artifact) and telemetry
events. Re-recording the same `(run_id, generation count, status)` is a no-op;
a later terminal report updates the run and appends new generations. Unknown
statuses are refused, and a caller-supplied empty `LineageGraph` is preserved.
