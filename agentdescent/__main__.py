"""``python -m agentdescent``: the same entry point as the ``agentdescent`` script (#510)."""
import sys

from .cli import main

if __name__ == "__main__":  # pragma: no cover - covered by tests/test_python_m_main.py
    sys.exit(main())
