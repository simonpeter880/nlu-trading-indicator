"""Compatibility shim for indicator.engines.atr_expansion."""

try:
    from engines.atr_expansion import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.atr_expansion import *  # noqa: F401,F403
