"""Compatibility shim for indicator.engines.unified_score."""

try:
    from engines.unified_score import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.unified_score import *  # noqa: F401,F403
