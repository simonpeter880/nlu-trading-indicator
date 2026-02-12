"""Compatibility shim for indicator.engines.macd_histogram."""

try:
    from engines.macd_histogram import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.macd_histogram import *  # noqa: F401,F403
