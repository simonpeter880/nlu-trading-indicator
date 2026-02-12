"""Compatibility shim for `nlu_analyzer.integrations.trend_strength_integration`."""

import runpy

from nlu_analyzer.integrations.trend_strength_integration import *  # noqa: F401,F403


if __name__ == "__main__":
    runpy.run_module("nlu_analyzer.integrations.trend_strength_integration", run_name="__main__")
