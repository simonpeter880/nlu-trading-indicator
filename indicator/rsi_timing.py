"""Compatibility shim for indicator.engines.rsi_timing."""

try:
    from engines.rsi_timing import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.rsi_timing import *  # noqa: F401,F403
