"""
Choppiness Index (CHOP) Module - Regime Filter

Detects whether the market is in a trending or range-bound (choppy) state.
Uses O(1) incremental updates with monotonic deques for efficiency.

CHOP ranges from 0-100:
- High CHOP (>61.8) => Range-bound, choppy, use mean reversion
- Low CHOP (<38.2) => Trending, use trend continuation
- Mid CHOP (38.2-61.8) => Transition, mixed regime

Formula (exact TradingView implementation):
    CHOP = 100 * log10(SUM(TR, n) / (MaxHigh(n) - MinLow(n) + eps)) / log10(n)

Where:
- TR = True Range
- SUM(TR, n) = Sum of TR over last n bars
- MaxHigh(n) = Highest high over last n bars
- MinLow(n) = Lowest low over last n bars
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Candle:
    """OHLCV candle data."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class ChoppinessConfig:
    """Configuration for Choppiness Index calculation."""

    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "1h"])
    period_by_tf: Dict[str, int] = field(default_factory=dict)  # Override per TF
    default_period: int = 14
    chop_high: float = 61.8  # Above => CHOP (range-bound)
    chop_low: float = 38.2  # Below => TREND
    slope_smooth: int = 1  # EMA smoothing for slope (1=no smoothing)
    window_crossings: int = 20  # Optional crossings window
    eps: float = 1e-12

    def get_period(self, timeframe: str) -> int:
        """Get period for a timeframe."""
        return self.period_by_tf.get(timeframe, self.default_period)


@dataclass
class ChopState:
    """Choppiness Index state for a timeframe."""

    tr: float
    sum_tr: Optional[float]
    hh: Optional[float]  # Highest high over n
    ll: Optional[float]  # Lowest low over n
    range: Optional[float]  # hh - ll
    chop: Optional[float]  # Choppiness value (0-100)
    chop_prev: Optional[float]  # Previous chop value
    chop_slope: Optional[float]  # chop - chop_prev
    chop_state: str  # WARMUP / CHOP / TRANSITION / TREND
    chop_score_0_100: Optional[float]  # Same as chop
    debug: Dict


class _TimeframeState:
    """Internal state for a single timeframe."""

    def __init__(self, period: int, config: ChoppinessConfig):
        self.period = period
        self.config = config

        # Previous close for TR calculation
        self.prev_close: Optional[float] = None

        # Bar index for monotonic deques
        self.bar_index = 0

        # Rolling TR sum
        self.tr_deque: deque = deque(maxlen=period)
        self.tr_sum = 0.0

        # Monotonic deques for HH/LL (amortized O(1))
        # Elements: (bar_index, value)
        self.max_high_deque: deque = deque()  # Decreasing highs
        self.min_low_deque: deque = deque()  # Increasing lows

        # CHOP history
        self.chop_prev: Optional[float] = None
        self.chop_smooth: Optional[float] = None  # For smoothed slope


class ChoppinessEngine:
    """
    Choppiness Index Engine with O(1) incremental updates.

    Uses monotonic deques for efficient rolling max/min operations.
    """

    def __init__(self, config: Optional[ChoppinessConfig] = None):
        """Initialize engine with config."""
        self.config = config or ChoppinessConfig()
        self._states: Dict[str, _TimeframeState] = {}

        # Initialize states for configured timeframes
        for tf in self.config.timeframes:
            period = self.config.get_period(tf)
            self._states[tf] = _TimeframeState(period, self.config)

    def on_candle_close(self, timeframe: str, candle: Candle) -> ChopState:
        """
        Process a candle close for a timeframe.

        Args:
            timeframe: Timeframe identifier
            candle: OHLCV candle

        Returns:
            ChopState with current choppiness values
        """
        # Ensure state exists
        if timeframe not in self._states:
            period = self.config.get_period(timeframe)
            self._states[timeframe] = _TimeframeState(period, self.config)

        state = self._states[timeframe]

        # Calculate TR
        tr = self._true_range(candle.high, candle.low, state.prev_close)
        state.prev_close = candle.close

        # Update TR rolling sum
        if len(state.tr_deque) == state.period:
            # Deque is full, subtract oldest
            oldest = state.tr_deque[0]
            state.tr_sum -= oldest

        state.tr_deque.append(tr)
        state.tr_sum += tr

        # Update HH monotonic deque
        state.bar_index += 1
        self._update_max_high(state, candle.high)
        self._update_min_low(state, candle.low)

        # Compute CHOP if we have enough data
        chop_state = self._compute_chop(state, timeframe)

        return chop_state

    def warmup(self, candles_by_tf: Dict[str, List[Candle]]) -> Dict[str, ChopState]:
        """
        Warmup engine with historical candles.

        Args:
            candles_by_tf: Dict mapping timeframe -> list of candles

        Returns:
            Dict mapping timeframe -> ChopState
        """
        states = {}

        for tf, candles in candles_by_tf.items():
            for candle in candles:
                state = self.on_candle_close(tf, candle)
            states[tf] = state

        return states

    def update(self, candles_by_tf: Dict[str, List[Candle]]) -> Dict[str, ChopState]:
        """Alias for warmup (for consistency with other engines)."""
        return self.warmup(candles_by_tf)

    def get_state(self, timeframe: str) -> Optional[ChopState]:
        """Get current state for a timeframe."""
        if timeframe not in self._states:
            return None

        state = self._states[timeframe]
        return self._compute_chop(state, timeframe)

    def reset(self, timeframe: str) -> None:
        """Reset state for a timeframe."""
        if timeframe in self._states:
            period = self.config.get_period(timeframe)
            self._states[timeframe] = _TimeframeState(period, self.config)

    def _true_range(self, high: float, low: float, prev_close: Optional[float]) -> float:
        """Calculate True Range."""
        if prev_close is None:
            return high - low

        hl = high - low
        hc = abs(high - prev_close)
        lc = abs(low - prev_close)

        return max(hl, hc, lc)

    def _update_max_high(self, state: _TimeframeState, high: float) -> None:
        """Update monotonic deque for rolling max (HH)."""
        # Remove elements from back that are <= current high (decreasing order)
        while state.max_high_deque and state.max_high_deque[-1][1] <= high:
            state.max_high_deque.pop()

        # Append current
        state.max_high_deque.append((state.bar_index, high))

        # Remove elements from front that are outside window
        while state.max_high_deque and state.max_high_deque[0][0] <= state.bar_index - state.period:
            state.max_high_deque.popleft()

    def _update_min_low(self, state: _TimeframeState, low: float) -> None:
        """Update monotonic deque for rolling min (LL)."""
        # Remove elements from back that are >= current low (increasing order)
        while state.min_low_deque and state.min_low_deque[-1][1] >= low:
            state.min_low_deque.pop()

        # Append current
        state.min_low_deque.append((state.bar_index, low))

        # Remove elements from front that are outside window
        while state.min_low_deque and state.min_low_deque[0][0] <= state.bar_index - state.period:
            state.min_low_deque.popleft()

    def _compute_chop(self, state: _TimeframeState, timeframe: str) -> ChopState:
        """Compute Choppiness Index from current state."""
        tr = state.tr_deque[-1] if state.tr_deque else 0.0

        # Check if we have enough data
        if len(state.tr_deque) < state.period:
            return ChopState(
                tr=tr,
                sum_tr=None,
                hh=None,
                ll=None,
                range=None,
                chop=None,
                chop_prev=None,
                chop_slope=None,
                chop_state="WARMUP",
                chop_score_0_100=None,
                debug={
                    "bars_collected": len(state.tr_deque),
                    "bars_needed": state.period,
                    "timeframe": timeframe,
                },
            )

        # Get HH and LL
        hh = state.max_high_deque[0][1] if state.max_high_deque else None
        ll = state.min_low_deque[0][1] if state.min_low_deque else None

        if hh is None or ll is None:
            return ChopState(
                tr=tr,
                sum_tr=state.tr_sum,
                hh=hh,
                ll=ll,
                range=None,
                chop=None,
                chop_prev=None,
                chop_slope=None,
                chop_state="WARMUP",
                chop_score_0_100=None,
                debug={
                    "bars_collected": len(state.tr_deque),
                    "hh": hh,
                    "ll": ll,
                },
            )

        # Calculate CHOP
        range_n = hh - ll
        ratio = state.tr_sum / (range_n + self.config.eps)

        # CHOP formula: 100 * log10(ratio) / log10(n)
        try:
            chop = 100.0 * math.log10(ratio) / math.log10(state.period)
        except (ValueError, ZeroDivisionError):
            chop = 50.0  # Fallback to neutral

        # Clamp to [0, 100]
        chop = max(0.0, min(100.0, chop))

        # Calculate slope
        chop_slope = None
        if state.chop_prev is not None:
            # Optional: smooth chop before computing slope
            if self.config.slope_smooth > 1:
                alpha = 2.0 / (self.config.slope_smooth + 1.0)
                if state.chop_smooth is None:
                    state.chop_smooth = chop
                else:
                    state.chop_smooth = alpha * chop + (1 - alpha) * state.chop_smooth
                chop_slope = state.chop_smooth - state.chop_prev
            else:
                chop_slope = chop - state.chop_prev

        # Determine state
        if chop >= self.config.chop_high:
            chop_state_label = "CHOP"
        elif chop <= self.config.chop_low:
            chop_state_label = "TREND"
        else:
            chop_state_label = "TRANSITION"

        # Update history
        state.chop_prev = chop

        return ChopState(
            tr=tr,
            sum_tr=state.tr_sum,
            hh=hh,
            ll=ll,
            range=range_n,
            chop=chop,
            chop_prev=state.chop_prev if state.chop_prev != chop else None,
            chop_slope=chop_slope,
            chop_state=chop_state_label,
            chop_score_0_100=chop,
            debug={
                "timeframe": timeframe,
                "period": state.period,
                "chop_high_thr": self.config.chop_high,
                "chop_low_thr": self.config.chop_low,
                "trendiness_score": 100.0 - chop,
                "ratio": ratio,
                "eps": self.config.eps,
            },
        )


def print_choppiness(states: Dict[str, ChopState]) -> None:
    """
    Print choppiness states in compact format.

    Args:
        states: Dict mapping timeframe -> ChopState
    """
    print("CHOPPINESS")
    for tf, state in sorted(states.items()):
        if state.chop is None:
            print(f"{tf}: state=WARMUP")
            continue

        parts = [f"{tf}:"]
        parts.append(f"chop={state.chop:.1f}")
        parts.append(f"state={state.chop_state}")

        if state.chop_slope is not None:
            sign = "+" if state.chop_slope >= 0 else ""
            parts.append(f"slope={sign}{state.chop_slope:.1f}")

        if state.sum_tr is not None:
            parts.append(f"sumTR={state.sum_tr:.1f}")

        if state.range is not None:
            parts.append(f"range={state.range:.1f}")

        print(" ".join(parts))


def format_chop_state(timeframe: str, state: ChopState) -> str:
    """
    Format a single chop state for display.

    Args:
        timeframe: Timeframe identifier
        state: ChopState to format

    Returns:
        Formatted string
    """
    if state.chop is None:
        return f"{timeframe}: state=WARMUP"

    parts = [f"{timeframe}:"]
    parts.append(f"chop={state.chop:.1f}")
    parts.append(f"state={state.chop_state}")

    if state.chop_slope is not None:
        sign = "+" if state.chop_slope >= 0 else ""
        parts.append(f"slope={sign}{state.chop_slope:.1f}")

    if state.sum_tr is not None:
        parts.append(f"sumTR={state.sum_tr:.1f}")

    if state.range is not None:
        parts.append(f"range={state.range:.1f}")

    return " ".join(parts)


def interpret_chop(state: ChopState) -> str:
    """
    Interpret chop state and provide actionable insight.

    Args:
        state: ChopState to interpret

    Returns:
        Interpretation string
    """
    if state.chop is None:
        return "Insufficient data for choppiness analysis"

    if state.chop_state == "CHOP":
        return "Range-bound market - prefer mean reversion, VWAP fades, avoid breakouts"
    elif state.chop_state == "TREND":
        return "Trending market - allow trend continuation, breakout attempts"
    else:  # TRANSITION
        return "Mixed regime - use cautious position sizing, wait for clarity"
