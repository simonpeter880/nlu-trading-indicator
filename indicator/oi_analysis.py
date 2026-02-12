"""Compatibility shim for indicator.engines.oi_analysis."""

try:
    from engines.oi_analysis import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.oi_analysis import *  # noqa: F401,F403
