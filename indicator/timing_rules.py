"""Compatibility shim for indicator.engines.timing_rules."""

try:
    from engines.timing_rules import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.timing_rules import *  # noqa: F401,F403
