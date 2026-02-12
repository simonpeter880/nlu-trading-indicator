"""Compatibility shim for indicator.engines.stoch_rsi_timing."""

try:
    from engines.stoch_rsi_timing import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.stoch_rsi_timing import *  # noqa: F401,F403
