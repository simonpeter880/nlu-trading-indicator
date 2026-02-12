"""Compatibility shim for indicator.engines.orderbook_analysis."""

try:
    from engines.orderbook_analysis import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.orderbook_analysis import *  # noqa: F401,F403
