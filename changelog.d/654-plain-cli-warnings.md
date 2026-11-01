Fixed
Foreground `campaign` and `tournament` print AgentDescent warnings as one
`warning: …` line (no `file:line` or source), matching `evolve`. The early-stop
warning is no longer echoed on the terminal when the status/summary line already
shows the error; library `evolve()` callers keep standard `warnings` behaviour
(#654).
