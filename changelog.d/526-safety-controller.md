Added
`KillSwitch`, `canary` and `rollback_plan`: a file-based global stop, a paired
exact sign test that recommends promote/hold/rollback from incumbent vs candidate
per-task scores, and a pure state diff for rollback. Missing candidate tasks
block promotion and are reported, significant losses use a separate
`max_regression` bar, a delete-only rollback is not a no-op, and the switch
treats a directory at its path as engaged and writes atomically. These are
deployment operations separate from the acceptance gate, so a committed change
can still be canaried and reverted deliberately.
