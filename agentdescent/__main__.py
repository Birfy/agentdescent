"""``python -m agentdescent``: the same entry point as the ``agentdescent`` script (#510)."""
import sys

from .cli import main

if __name__ == "__main__":  # pragma: no cover - exercised in a subprocess
    sys.exit(main())
