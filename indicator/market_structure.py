"""Compatibility shim for indicator.engines.market_structure."""

try:
    from engines.market_structure import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.market_structure import *  # noqa: F401,F403
