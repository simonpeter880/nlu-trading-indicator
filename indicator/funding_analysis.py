"""Compatibility shim for indicator.engines.funding_analysis."""

try:
    from engines.funding_analysis import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.funding_analysis import *  # noqa: F401,F403
