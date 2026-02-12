"""Compatibility shim for `nlu_analyzer.tools.roc_momentum_compact_print`."""

import runpy

from nlu_analyzer.tools.roc_momentum_compact_print import *  # noqa: F401,F403

if __name__ == "__main__":
    runpy.run_module("nlu_analyzer.tools.roc_momentum_compact_print", run_name="__main__")
