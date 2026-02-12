"""Compatibility shim for indicator.apps.runner."""

import runpy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from indicator.apps.runner import *  # noqa: F401,F403

if __name__ == "__main__":
    runpy.run_module("indicator.apps.runner", run_name="__main__")
