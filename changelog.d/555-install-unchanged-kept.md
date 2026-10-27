Fixed
`agentdescent install` no longer reports `updated` for files whose contents are
already identical; it prints `kept <path> (unchanged)` and leaves the file
untouched (#555).
