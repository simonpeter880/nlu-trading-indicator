"""Compatibility shim for indicator.engines.volume_engine."""

try:
    from engines.volume_engine import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.volume_engine import *  # noqa: F401,F403
