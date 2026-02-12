"""Shared signal enums and helpers to avoid stringly-typed signals."""

from enum import Enum
from typing import Union


class Signal(Enum):
    """Allowed signal values across indicator modules."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    CAUTION = "caution"
    WARNING = "warning"
    TRAP = "trap"
    SUSPICIOUS = "suspicious"

    def __str__(self) -> str:
        return self.value


SignalLike = Union[Signal, str]


def signal_value(signal: SignalLike) -> str:
    """Normalize signal-like values to their string representation."""
    return signal.value if isinstance(signal, Signal) else str(signal)


def coerce_signal(signal: SignalLike, default: Signal = Signal.NEUTRAL) -> Signal:
    """Convert a string to Signal, falling back to default for unknown values."""
    if isinstance(signal, Signal):
        return signal
    try:
        return Signal(str(signal))
    except ValueError:
        return default
