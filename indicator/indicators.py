"""Compatibility shim for indicator.engines.indicators."""

try:
    from engines.indicators import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.indicators import *  # noqa: F401,F403
