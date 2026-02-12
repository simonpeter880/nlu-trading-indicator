"""Compatibility shim for indicator.engines.choppiness."""

try:
    from engines.choppiness import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.choppiness import *  # noqa: F401,F403
