"""Compatibility shim for indicator.engines.ema_filter."""

try:
    from engines.ema_filter import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.ema_filter import *  # noqa: F401,F403
