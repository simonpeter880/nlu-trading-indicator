"""Formatting utilities for indicator analysis display."""

from typing import Union

from .colors import Colors
from signals import Signal, signal_value


def signal_color(signal: Union[Signal, str]) -> str:
    """Get color for signal type."""
    value = signal_value(signal)
    if value == Signal.BULLISH.value:
        return Colors.GREEN
    elif value == Signal.BEARISH.value:
        return Colors.RED
    elif value == Signal.SUSPICIOUS.value:
        return Colors.MAGENTA
    else:
        return Colors.YELLOW


def strength_bar(strength: float, width: int = 20) -> str:
    """Create a visual strength bar.

    Args:
        strength: Value from 0-100
        width: Bar width in characters

    Returns:
        Colored bar string
    """
    filled = int(strength / 100 * width)
    empty = width - filled

    if strength >= 70:
        color = Colors.GREEN
    elif strength >= 50:
        color = Colors.YELLOW
    else:
        color = Colors.RED

    return f"{color}{'█' * filled}{Colors.DIM}{'░' * empty}{Colors.RESET}"


def score_bar(score: float, width: int = 20) -> str:
    """Create a visual bar for score in [-1, +1] range.

    Args:
        score: Value from -1.0 to +1.0
        width: Bar width in characters

    Returns:
        Colored bar string with center marker
    """
    # Map score from [-1, +1] to [0, width]
    center = width // 2
    filled = int((score + 1.0) / 2.0 * width)

    # Build bar
    bar = ""
    for i in range(width):
        if i == center:
            # Center marker
            bar += Colors.DIM + "│" + Colors.RESET
        elif i < center:
            # Left side (bearish)
            if i < filled:
                bar += Colors.RED + "█" + Colors.RESET
            else:
                bar += Colors.DIM + "░" + Colors.RESET
        else:
            # Right side (bullish)
            if i < filled:
                bar += Colors.GREEN + "█" + Colors.RESET
            else:
                bar += Colors.DIM + "░" + Colors.RESET

    return bar
