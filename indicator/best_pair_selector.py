"""Compatibility shim for indicator.engines.best_pair_selector."""

try:
    from engines.best_pair_selector import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.best_pair_selector import *  # noqa: F401,F403
