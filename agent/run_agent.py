"""Human-friendly command-line entry point for the Sari agent."""

import sys
from pathlib import Path

_AGENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _AGENT_DIR.parent
for _path in (str(_REPO_ROOT), str(_AGENT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from orchestrator.cli import main


if __name__ == "__main__":
    main()
