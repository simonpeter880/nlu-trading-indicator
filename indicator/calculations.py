"""Compatibility shim for indicator.engines.calculations."""

try:
    from engines.calculations import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.calculations import *  # noqa: F401,F403
