"""Compatibility shim for indicator.engines.volume_analysis."""

try:
    from engines.volume_analysis import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.volume_analysis import *  # noqa: F401,F403
