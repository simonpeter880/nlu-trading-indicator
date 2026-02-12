"""Compatibility shim for indicator.integrations.rsi_timing_integration."""

import runpy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from indicator.integrations.rsi_timing_integration import *  # noqa: F401,F403

if __name__ == "__main__":
    runpy.run_module("indicator.integrations.rsi_timing_integration", run_name="__main__")
