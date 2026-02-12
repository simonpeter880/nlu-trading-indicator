"""Compatibility shim for indicator.engines.bollinger_bandwidth."""

try:
    from engines.bollinger_bandwidth import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.bollinger_bandwidth import *  # noqa: F401,F403
