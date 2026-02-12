"""
Shared math utilities for indicator calculations.
"""

import math
from typing import Iterable, List, Tuple


def simple_average(values: Iterable[float], default: float = 0.0) -> float:
    """Return the arithmetic mean of values or a default if empty."""
    values_list = list(values)
    if not values_list:
        return default
    return sum(values_list) / len(values_list)


def average_last(values: List[float], window: int, default: float = 0.0) -> float:
    """Return the average of the last window values (or all values if shorter)."""
    if not values:
        return default
    if window <= 0:
        return default
    slice_vals = values[-window:]
    return sum(slice_vals) / len(slice_vals)


def calculate_sma(prices: List[float], period: int) -> List[float]:
    """Calculate Simple Moving Average."""
    if len(prices) < period or period <= 0:
        return []
    window_sum = sum(prices[:period])
    sma = [window_sum / period]
    for i in range(period, len(prices)):
        window_sum += prices[i] - prices[i - period]
        sma.append(window_sum / period)
    return sma


def calculate_ema(prices: List[float], period: int) -> List[float]:
    """Calculate Exponential Moving Average."""
    if len(prices) < period or period <= 0:
        return []

    multiplier = 2 / (period + 1)
    ema = [sum(prices[:period]) / period]  # Start with SMA

    for price in prices[period:]:
        ema.append((price - ema[-1]) * multiplier + ema[-1])

    return ema


def calculate_rolling_mean_std(prices: List[float], period: int) -> Tuple[List[float], List[float]]:
    """
    Calculate rolling mean and standard deviation.

    Uses rolling sums for O(n) performance. Standard deviation uses
    population variance (divide by period), matching existing usage.
    """
    if len(prices) < period or period <= 0:
        return [], []

    window_sum = sum(prices[:period])
    window_sumsq = sum(p * p for p in prices[:period])

    means: List[float] = []
    stds: List[float] = []

    for i in range(period, len(prices) + 1):
        mean = window_sum / period
        variance = (window_sumsq / period) - (mean * mean)
        if variance < 0:
            variance = 0.0
        means.append(mean)
        stds.append(math.sqrt(variance))

        if i < len(prices):
            outgoing = prices[i - period]
            incoming = prices[i]
            window_sum += incoming - outgoing
            window_sumsq += incoming * incoming - outgoing * outgoing

    return means, stds
