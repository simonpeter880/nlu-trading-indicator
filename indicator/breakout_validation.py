"""Compatibility shim for indicator.engines.breakout_validation."""

try:
    from engines.breakout_validation import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.breakout_validation import *  # noqa: F401,F403
