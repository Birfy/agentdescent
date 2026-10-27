Added
`run_guarded`, one entry point for a session that cannot skip the kill switch or
the decision record. The injected acceptance is chained **before** the spec's
own acceptance policy or the shipped default statistical gate, so guards can
only tighten adoption; stateful checkpoint/restore/screen-reject hooks are
forwarded. The decision record says whether a candidate actually committed and
carries the error separately, a failing log does not hide the result, and an
`aggregator_factory` algorithm is called out in `notes`. The kill switch is an
explicit pre-flight check.
