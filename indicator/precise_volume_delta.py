"""Compatibility shim for indicator.engines.precise_volume_delta."""

try:
    from engines.precise_volume_delta import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.precise_volume_delta import *  # noqa: F401,F403
