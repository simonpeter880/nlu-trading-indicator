"""
Bollinger Bandwidth Engine - Volatility Compression/Expansion Detector

Computes standard Bollinger Bands and Bandwidth with O(1) incremental updates
using rolling sums for mean/variance calculations.

CRITICAL:
- O(1) per candle close (no window scans)
- Stable ratio-based state classification
- Bandwidth normalized by mid (works across price scales)
- Detects compression (squeeze setup) and expansion (breakout)

States:
- COMPRESSED: bw_ratio < 0.80 (squeeze building)
- NORMAL: 0.80 ≤ bw_ratio < 1.20 (typical volatility)
- EXPANDING: 1.20 ≤ bw_ratio < 1.60 (breakout in progress)
- EXTREME: bw_ratio ≥ 1.60 (exhaustion risk)
- FADE_RISK: EXPANDING/EXTREME but slope negative (losing steam)
- WARMUP: Not enough data yet

Usage:
    config = BollingerBandwidthConfig(timeframes=["1m", "5m"])
    engine = BollingerBandwidthEngine(config)

    state = engine.on_candle_close("1m", candle)
    print(f"BW State: {state.bw_state}, Ratio: {state.bw_ratio:.2f}")

    if state.bw_state == "COMPRESSED":
        # Squeeze setup - arm for breakout
    elif state.bw_state == "EXPANDING":
        # Breakout window - volatility expanding
    elif state.bw_state == "EXTREME" or state.bw_state == "FADE_RISK":
        # Exhaustion risk - be cautious
"""

from dataclasses import dataclass, field
from collections import deque
from typing import Optional
import math


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Candle:
    """OHLCV candle."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class BollingerBandwidthConfig:
    """Configuration for Bollinger Bandwidth engine."""

    # Timeframes
    timeframes: list[str] = field(default_factory=lambda: ["1m", "5m", "1h"])

    # Bollinger Bands parameters
    bb_period: int = 20
    bb_k: float = 2.0

    # Bandwidth smoothing and lookback
    bw_sma_period: int = 50               # SMA for bw_ratio
    smooth_bw_ema_period: int = 3         # EMA smoothing for bandwidth (1 disables)
    eps: float = 1e-12

    # State thresholds (ratio-based)
    ratio_compress: float = 0.80
    ratio_expand: float = 1.20
    ratio_extreme: float = 1.60
    fade_slope_thr: float = -0.02         # negative slope threshold for FADE_RISK


@dataclass
class BollingerBandwidthState:
    """Bollinger Bandwidth state output."""

    mid: Optional[float]                  # SMA(close, n)
    std: Optional[float]                  # stdev(close, n)
    upper: Optional[float]                # mid + k*std
    lower: Optional[float]                # mid - k*std
    bandwidth: Optional[float]            # (upper - lower) / mid
    bandwidth_smooth: Optional[float]     # EMA-smoothed bandwidth
    bandwidth_slope: Optional[float]      # bandwidth_smooth - prev_bandwidth_smooth
    bw_sma: Optional[float]               # SMA(bandwidth_smooth, m)
    bw_ratio: Optional[float]             # bandwidth_smooth / bw_sma
    bw_state: str                         # WARMUP/COMPRESSED/NORMAL/EXPANDING/EXTREME/FADE_RISK
    bw_score_0_100: Optional[float]       # 0-100 score
    debug: dict


# ============================================================================
# INTERNAL STATE
# ============================================================================

@dataclass
class _TimeframeState:
    """Internal state per timeframe (O(1) operations)."""

    # Rolling stats for close window (bb_period)
    close_deque: deque = field(default_factory=deque)
    sum_x: float = 0.0                    # sum of close
    sum_x2: float = 0.0                   # sum of close^2

    # Bollinger Bands
    mid: Optional[float] = None
    std: Optional[float] = None
    upper: Optional[float] = None
    lower: Optional[float] = None
    bandwidth: Optional[float] = None

    # Bandwidth smoothing
    bw_ema: Optional[float] = None
    bandwidth_smooth: Optional[float] = None
    prev_bandwidth_smooth: Optional[float] = None
    bandwidth_slope: Optional[float] = None

    # Rolling stats for bandwidth SMA (bw_sma_period)
    bw_deque: deque = field(default_factory=deque)
    bw_sum: float = 0.0
    bw_sma: Optional[float] = None
    bw_ratio: Optional[float] = None

    # State classification
    bw_state: str = "WARMUP"
    bw_score_0_100: Optional[float] = None

    # Bar counter
    bar_index: int = 0


# ============================================================================
# ENGINE
# ============================================================================

class BollingerBandwidthEngine:
    """
    Bollinger Bandwidth Engine - Volatility compression/expansion detector.

    O(1) per candle close with rolling sums for mean/variance calculations.
    """

    def __init__(self, config: BollingerBandwidthConfig):
        self.config = config
        self.states: dict[str, _TimeframeState] = {}

        for tf in config.timeframes:
            self.states[tf] = _TimeframeState(
                close_deque=deque(maxlen=config.bb_period),
                bw_deque=deque(maxlen=config.bw_sma_period)
            )

    def reset(self, timeframe: str) -> None:
        """Reset state for a timeframe."""
        if timeframe in self.states:
            self.states[timeframe] = _TimeframeState(
                close_deque=deque(maxlen=self.config.bb_period),
                bw_deque=deque(maxlen=self.config.bw_sma_period)
            )

    def warmup(self, candles_by_tf: dict[str, list[Candle]]) -> dict[str, BollingerBandwidthState]:
        """
        Warmup engine with historical candles.

        Args:
            candles_by_tf: Dict mapping timeframe to list of candles

        Returns:
            Dict mapping timeframe to final BollingerBandwidthState
        """
        results = {}

        for tf, candles in candles_by_tf.items():
            if tf not in self.states:
                self.states[tf] = _TimeframeState(
                    close_deque=deque(maxlen=self.config.bb_period),
                    bw_deque=deque(maxlen=self.config.bw_sma_period)
                )

            state = None
            for candle in candles:
                state = self.on_candle_close(tf, candle)

            if state is not None:
                results[tf] = state

        return results

    def on_candle_close(self, timeframe: str, candle: Candle) -> BollingerBandwidthState:
        """
        Process candle close and update Bollinger Bandwidth.

        Args:
            timeframe: Timeframe identifier
            candle: OHLCV candle

        Returns:
            BollingerBandwidthState with bands, bandwidth, ratio, and state
        """
        if timeframe not in self.states:
            self.states[timeframe] = _TimeframeState(
                close_deque=deque(maxlen=self.config.bb_period),
                bw_deque=deque(maxlen=self.config.bw_sma_period)
            )

        state = self.states[timeframe]
        state.bar_index += 1

        close = candle.close

        # Update rolling close window
        self._update_close_window(state, close)

        # Compute Bollinger Bands
        self._compute_bollinger_bands(state)

        # Compute bandwidth
        self._compute_bandwidth(state)

        # Smooth bandwidth (optional EMA)
        self._smooth_bandwidth(state)

        # Compute bandwidth slope
        self._compute_bandwidth_slope(state)

        # Update bandwidth SMA
        self._update_bandwidth_sma(state)

        # Compute bandwidth ratio
        self._compute_bandwidth_ratio(state)

        # Classify state
        self._classify_state(state)

        # Compute score
        self._compute_score(state)

        # Build output
        return self._build_state(state, timeframe)

    def get_state(self, timeframe: str) -> Optional[BollingerBandwidthState]:
        """Get current state for timeframe (or None if never updated)."""
        if timeframe not in self.states:
            return None

        state = self.states[timeframe]
        if state.bar_index == 0:
            return None

        return self._build_state(state, timeframe)

    # ========================================================================
    # ROLLING CLOSE WINDOW
    # ========================================================================

    def _update_close_window(self, state: _TimeframeState, close: float) -> None:
        """Update rolling close window with O(1) sum updates."""
        # Remove oldest value if at capacity
        if len(state.close_deque) == self.config.bb_period:
            old_close = state.close_deque[0]
            state.sum_x -= old_close
            state.sum_x2 -= old_close * old_close

        # Add new value
        state.close_deque.append(close)
        state.sum_x += close
        state.sum_x2 += close * close

    # ========================================================================
    # BOLLINGER BANDS
    # ========================================================================

    def _compute_bollinger_bands(self, state: _TimeframeState) -> None:
        """Compute Bollinger Bands (mid, std, upper, lower)."""
        n = len(state.close_deque)

        if n < self.config.bb_period:
            state.mid = None
            state.std = None
            state.upper = None
            state.lower = None
            return

        # Mean
        mean = state.sum_x / n

        # Variance (using sum of squares)
        var = (state.sum_x2 / n) - (mean * mean)
        var = max(var, 0.0)  # Handle numerical errors

        # Standard deviation
        std = math.sqrt(var)

        # Bollinger Bands
        upper = mean + self.config.bb_k * std
        lower = mean - self.config.bb_k * std

        state.mid = mean
        state.std = std
        state.upper = upper
        state.lower = lower

    # ========================================================================
    # BANDWIDTH
    # ========================================================================

    def _compute_bandwidth(self, state: _TimeframeState) -> None:
        """Compute bandwidth = (upper - lower) / mid."""
        if state.upper is None or state.lower is None or state.mid is None:
            state.bandwidth = None
            return

        # Normalized bandwidth
        bandwidth = (state.upper - state.lower) / (state.mid + self.config.eps)
        state.bandwidth = bandwidth

    # ========================================================================
    # BANDWIDTH SMOOTHING
    # ========================================================================

    def _smooth_bandwidth(self, state: _TimeframeState) -> None:
        """Apply optional EMA smoothing to bandwidth."""
        if state.bandwidth is None:
            state.bandwidth_smooth = None
            return

        if self.config.smooth_bw_ema_period <= 1:
            # No smoothing
            state.bandwidth_smooth = state.bandwidth
        else:
            # EMA smoothing
            alpha = 2.0 / (self.config.smooth_bw_ema_period + 1.0)

            if state.bw_ema is None:
                # Seed with first bandwidth
                state.bw_ema = state.bandwidth
            else:
                # Update EMA
                state.bw_ema = alpha * state.bandwidth + (1.0 - alpha) * state.bw_ema

            state.bandwidth_smooth = state.bw_ema

    # ========================================================================
    # BANDWIDTH SLOPE
    # ========================================================================

    def _compute_bandwidth_slope(self, state: _TimeframeState) -> None:
        """Compute bandwidth slope (bandwidth_smooth - prev_bandwidth_smooth)."""
        if state.bandwidth_smooth is None:
            state.bandwidth_slope = None
            return

        if state.prev_bandwidth_smooth is not None:
            state.bandwidth_slope = state.bandwidth_smooth - state.prev_bandwidth_smooth
        else:
            state.bandwidth_slope = None

        # Update prev
        state.prev_bandwidth_smooth = state.bandwidth_smooth

    # ========================================================================
    # BANDWIDTH SMA
    # ========================================================================

    def _update_bandwidth_sma(self, state: _TimeframeState) -> None:
        """Update rolling SMA of bandwidth_smooth."""
        if state.bandwidth_smooth is None:
            return

        # Remove oldest if at capacity
        if len(state.bw_deque) == self.config.bw_sma_period:
            old_bw = state.bw_deque[0]
            state.bw_sum -= old_bw

        # Add new value
        state.bw_deque.append(state.bandwidth_smooth)
        state.bw_sum += state.bandwidth_smooth

        # Compute SMA
        if len(state.bw_deque) == self.config.bw_sma_period:
            state.bw_sma = state.bw_sum / self.config.bw_sma_period
        else:
            state.bw_sma = None

    # ========================================================================
    # BANDWIDTH RATIO
    # ========================================================================

    def _compute_bandwidth_ratio(self, state: _TimeframeState) -> None:
        """Compute bandwidth ratio = bandwidth_smooth / bw_sma."""
        if state.bandwidth_smooth is None or state.bw_sma is None:
            state.bw_ratio = None
            return

        state.bw_ratio = state.bandwidth_smooth / (state.bw_sma + self.config.eps)

    # ========================================================================
    # STATE CLASSIFIER
    # ========================================================================

    def _classify_state(self, state: _TimeframeState) -> None:
        """Classify bandwidth state."""
        if state.bandwidth is None:
            state.bw_state = "WARMUP"
            return

        if state.bw_sma is None:
            state.bw_state = "WARMUP"
            return

        ratio = state.bw_ratio

        # Base classification
        if ratio < self.config.ratio_compress:
            state.bw_state = "COMPRESSED"
        elif ratio >= self.config.ratio_extreme:
            state.bw_state = "EXTREME"
        elif ratio >= self.config.ratio_expand:
            state.bw_state = "EXPANDING"
        else:
            state.bw_state = "NORMAL"

        # FADE_RISK override
        if ratio >= self.config.ratio_expand:
            if state.bandwidth_slope is not None and state.bandwidth_slope <= self.config.fade_slope_thr:
                state.bw_state = "FADE_RISK"

    # ========================================================================
    # SCORE
    # ========================================================================

    def _compute_score(self, state: _TimeframeState) -> None:
        """Compute bandwidth score 0-100."""
        if state.bw_ratio is None:
            state.bw_score_0_100 = None
            return

        # Base score (0-1 range)
        numerator = state.bw_ratio - self.config.ratio_compress
        denominator = self.config.ratio_extreme - self.config.ratio_compress + self.config.eps
        base = max(0.0, min(1.0, numerator / denominator))

        # Scale to 0-100
        score = 100.0 * base

        # Cap by state
        if state.bw_state == "COMPRESSED":
            score = min(score, 30.0)
        elif state.bw_state == "NORMAL":
            score = min(score, 60.0)
        elif state.bw_state == "EXPANDING":
            score = min(score, 85.0)
        # EXTREME can go to 100

        # Penalty for FADE_RISK
        if state.bw_state == "FADE_RISK":
            score = max(0.0, score - 10.0)

        state.bw_score_0_100 = score

    # ========================================================================
    # OUTPUT
    # ========================================================================

    def _build_state(self, state: _TimeframeState, timeframe: str) -> BollingerBandwidthState:
        """Build output state."""
        debug = {
            "timeframe": timeframe,
            "bar_index": state.bar_index,
            "bb_period": self.config.bb_period,
            "bw_sma_period": self.config.bw_sma_period,
            "close_window_len": len(state.close_deque),
            "bw_window_len": len(state.bw_deque),
        }

        return BollingerBandwidthState(
            mid=state.mid,
            std=state.std,
            upper=state.upper,
            lower=state.lower,
            bandwidth=state.bandwidth,
            bandwidth_smooth=state.bandwidth_smooth,
            bandwidth_slope=state.bandwidth_slope,
            bw_sma=state.bw_sma,
            bw_ratio=state.bw_ratio,
            bw_state=state.bw_state,
            bw_score_0_100=state.bw_score_0_100,
            debug=debug
        )


# ============================================================================
# PRINT HELPERS
# ============================================================================

def print_bollinger_bandwidth(states: dict[str, BollingerBandwidthState]) -> None:
    """
    Print Bollinger Bandwidth states in compact format.

    Example output:
    BOLLINGER BW
    1m: bw=0.0123 ratio=0.74 state=COMPRESSED slope=-0.0012 score=18
    5m: bw=0.0198 ratio=1.28 state=EXPANDING  slope=+0.0021 score=73
    1h: bw=0.0410 ratio=1.67 state=EXTREME    slope=-0.0040 score=90
    """
    print("BOLLINGER BW")
    for tf, state in states.items():
        print(format_bandwidth_state(tf, state))


def format_bandwidth_state(timeframe: str, state: BollingerBandwidthState) -> str:
    """Format single bandwidth state."""
    if state.bandwidth_smooth is None:
        return f"{timeframe}: state={state.bw_state}"

    bw_str = f"{state.bandwidth_smooth:.4f}"
    ratio_str = f"{state.bw_ratio:.2f}" if state.bw_ratio is not None else "--"
    slope_str = f"{state.bandwidth_slope:+.4f}" if state.bandwidth_slope is not None else "    --"
    score_str = f"{state.bw_score_0_100:>2.0f}" if state.bw_score_0_100 is not None else "--"

    return (
        f"{timeframe}: bw={bw_str} ratio={ratio_str:>4s} state={state.bw_state:<12s} "
        f"slope={slope_str} score={score_str}"
    )


def interpret_bandwidth(state: BollingerBandwidthState) -> str:
    """Interpret bandwidth state in human-readable format."""
    bw_state = state.bw_state

    if bw_state == "WARMUP":
        return "Warming up - not enough data yet"
    elif bw_state == "COMPRESSED":
        return f"Compressed (ratio={state.bw_ratio:.2f}) - squeeze building, potential breakout setup"
    elif bw_state == "NORMAL":
        return f"Normal (ratio={state.bw_ratio:.2f}) - typical volatility"
    elif bw_state == "EXPANDING":
        return f"Expanding (ratio={state.bw_ratio:.2f}) - breakout in progress, volatility increasing"
    elif bw_state == "EXTREME":
        return f"Extreme (ratio={state.bw_ratio:.2f}) - high volatility, exhaustion risk"
    elif bw_state == "FADE_RISK":
        return f"Fade risk (ratio={state.bw_ratio:.2f}, slope={state.bandwidth_slope:.4f}) - expansion losing steam"
    else:
        return "Unknown state"


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "BOLLINGER BANDWIDTH ENGINE" + " " * 22 + "║")
    print("╚" + "=" * 68 + "╝\n")

    config = BollingerBandwidthConfig(
        timeframes=["1m", "5m"],
        bb_period=20,
        bb_k=2.0,
        bw_sma_period=50,
        smooth_bw_ema_period=3,
    )
    engine = BollingerBandwidthEngine(config)

    print("Example 1: Compression (Squeeze Building)")
    print("-" * 70)

    # Simulate tight range (compression)
    price = 100.0
    for i in range(80):
        # Tight oscillation
        price += 0.1 if i % 2 == 0 else -0.1
        candle = Candle(i * 60000, price, price + 0.2, price - 0.2, price, 1000)
        state = engine.on_candle_close("1m", candle)

    print(format_bandwidth_state("1m", state))
    print(f"→ {interpret_bandwidth(state)}\n")

    print("Example 2: Expansion (Breakout)")
    print("-" * 70)

    # Simulate breakout (expansion)
    for i in range(80, 100):
        price += 2.0  # Strong move
        candle = Candle(i * 60000, price, price + 3, price - 1, price, 2000)
        state = engine.on_candle_close("1m", candle)

    print(format_bandwidth_state("1m", state))
    print(f"→ {interpret_bandwidth(state)}\n")

    print("Example 3: Multiple Timeframes")
    print("-" * 70)

    # Simulate different volatility on 5m
    engine.reset("5m")
    price = 100.0
    for i in range(100):
        if i < 70:
            # Normal volatility
            price += 0.5 if i % 3 == 0 else -0.3
        else:
            # Extreme volatility
            price += 4.0 if i % 2 == 0 else -3.5

        candle = Candle(i * 300000, price, price + 2, price - 2, price, 1500)
        state_5m = engine.on_candle_close("5m", candle)

    states = {
        "1m": engine.get_state("1m"),
        "5m": engine.get_state("5m"),
    }

    print_bollinger_bandwidth(states)
    print()

    for tf, st in states.items():
        if st is not None:
            print(f"{tf}: {interpret_bandwidth(st)}")

    print("\n" + "=" * 70)
    print("✅ Bollinger Bandwidth Engine Ready!")
    print("=" * 70)
