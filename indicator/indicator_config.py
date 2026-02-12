"""Compatibility shim for indicator.engines.indicator_config."""

try:
    from engines.indicator_config import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.indicator_config import *  # noqa: F401,F403
