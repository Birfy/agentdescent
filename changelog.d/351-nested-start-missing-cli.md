Fixed
MCP `start` in nested mode (`AGENTDESCENT_NESTED=1`) still returns the
documented refused stub when the proposed `agent` / `reflect` CLI is absent
from `PATH`; the missing-CLI fail-closed gate runs only for non-nested starts
(#351).
