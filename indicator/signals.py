"""Compatibility shim for indicator.engines.signals."""

try:
    from engines.signals import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.signals import *  # noqa: F401,F403
