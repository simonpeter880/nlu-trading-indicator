"""Compatibility shim for indicator.engines.institutional_structure."""

try:
    from engines.institutional_structure import *  # noqa: F401,F403
except ImportError:
    from indicator.engines.institutional_structure import *  # noqa: F401,F403
