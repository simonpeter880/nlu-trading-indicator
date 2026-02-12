"""Compatibility shim for indicator.engines.data_fetcher."""

try:
    from engines.data_fetcher import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.data_fetcher import *  # noqa: F401,F403
