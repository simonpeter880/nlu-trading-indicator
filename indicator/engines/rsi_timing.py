"""
RSI Timing Module - Divergence Detection & Regime Labeling

Implements Wilder RSI with:
- Regular divergence (potential reversal)
- Hidden divergence (trend continuation)
- Regime bands (55/45) for timing confirmation
- Failure swings (exhaustion warnings)
- O(1) incremental updates

IMPORTANT: This is a TIMING tool, NOT an entry signal.
Use divergence for:
- Exhaustion warnings (regular divergence)
- Continuation timing (hidden divergence)
- Structure confirmation only

Always combine with price action, volume, and structure.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
class SwingPoint:
    """Swing high/low point for divergence detection."""

    ts: int  # Timestamp
    price: float  # Swing price
    rsi: float  # RSI at swing
    index: int  # Bar index


@dataclass
class RSITimingConfig:
    """Configuration for RSI Timing module."""

    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "1h"])

    # RSI calculation
    rsi_period: int = 14

    # Regime bands (timing confirmation, NOT entry)
    regime_high: float = 55.0  # Above => bullish regime
    regime_low: float = 45.0  # Below => bearish regime
    extreme_high: float = 75.0  # Warning only
    extreme_low: float = 25.0  # Warning only

    # Divergence settings
    min_bars_between_swings_by_tf: Dict[str, int] = field(default_factory=dict)
    min_bars_fallback: int = 3
    min_price_diff_atrp_factor: float = 0.15  # % of ATR
    atr_period: int = 14
    use_external_atr_percent: bool = True

    # Failure swing (optional exhaustion detector)
    enable_failure_swing: bool = True
    fs_high: float = 60.0
    fs_mid_low: float = 45.0
    fs_low_break_margin: float = 1.0

    # Internal pivot detector (optional, default OFF)
    enable_internal_pivots: bool = False
    pivot_left: int = 2
    pivot_right: int = 2

    def get_min_bars(self, timeframe: str) -> int:
        """Get minimum bars between swings for a timeframe."""
        return self.min_bars_between_swings_by_tf.get(timeframe, self.min_bars_fallback)


@dataclass
class RSITimingState:
    """RSI timing state for a timeframe."""

    rsi: Optional[float]
    rsi_regime: str  # WARMUP / BULLISH / BEARISH / RANGE
    divergence: str  # NONE / REG_BULL / REG_BEAR / HID_BULL / HID_BEAR
    div_strength_0_100: Optional[float]
    failure_swing: str  # NONE / BULL / BEAR
    debug: Dict


class _FailureSwingDetector:
    """Detects RSI failure swings (exhaustion patterns)."""

    def __init__(self, config: RSITimingConfig):
        self.config = config
        self.reset()

    def reset(self):
        """Reset detector state."""
        # Bear failure swing tracking
        self.bear_stage = 0  # 0=none, 1=above_high, 2=pullback, 3=lower_high
        self.bear_peak = None
        self.bear_pullback_low = None

        # Bull failure swing tracking
        self.bull_stage = 0
        self.bull_low = None
        self.bull_pullback_high = None

    def update(self, rsi: float) -> str:
        """
        Update failure swing detection.

        Returns:
            "BEAR" / "BULL" / "NONE"
        """
        if not self.config.enable_failure_swing:
            return "NONE"

        result = "NONE"

        # Bear failure swing detection
        if self.bear_stage == 0:
            if rsi >= self.config.fs_high:
                self.bear_stage = 1
                self.bear_peak = rsi
        elif self.bear_stage == 1:
            if rsi > self.bear_peak:
                self.bear_peak = rsi
            elif rsi < self.config.fs_mid_low:
                # Pulled back below mid-low, start tracking
                self.bear_stage = 2
                self.bear_pullback_low = rsi
        elif self.bear_stage == 2:
            if rsi < self.bear_pullback_low:
                self.bear_pullback_low = rsi
            elif rsi > self.config.fs_mid_low:
                # Bouncing back up
                if rsi < self.bear_peak:
                    # Lower high formed
                    self.bear_stage = 3
                else:
                    # Not a lower high, reset
                    self.bear_stage = 0
        elif self.bear_stage == 3:
            # Waiting for break below pullback low
            if rsi < (self.bear_pullback_low - self.config.fs_low_break_margin):
                result = "BEAR"
                self.bear_stage = 0  # Reset after trigger
            elif rsi > self.config.fs_high:
                # Reset if going back above high
                self.bear_stage = 0

        # Bull failure swing detection
        if self.bull_stage == 0:
            if rsi <= (100 - self.config.fs_high):  # Mirror: low threshold
                self.bull_stage = 1
                self.bull_low = rsi
        elif self.bull_stage == 1:
            if rsi < self.bull_low:
                self.bull_low = rsi
            elif rsi > (100 - self.config.fs_mid_low):  # Mirror: high threshold
                self.bull_stage = 2
                self.bull_pullback_high = rsi
        elif self.bull_stage == 2:
            if rsi > self.bull_pullback_high:
                self.bull_pullback_high = rsi
            elif rsi < (100 - self.config.fs_mid_low):
                # Pulling back down
                if rsi > self.bull_low:
                    # Higher low formed
                    self.bull_stage = 3
                else:
                    self.bull_stage = 0
        elif self.bull_stage == 3:
            # Waiting for break above pullback high
            if rsi > (self.bull_pullback_high + self.config.fs_low_break_margin):
                result = "BULL"
                self.bull_stage = 0
            elif rsi < (100 - self.config.fs_high):
                self.bull_stage = 0

        return result


class _TimeframeState:
    """Internal state for a single timeframe."""

    def __init__(self, config: RSITimingConfig, timeframe: str):
        self.config = config
        self.timeframe = timeframe

        # RSI calculation state
        self.prev_close: Optional[float] = None
        self.avg_gain: Optional[float] = None
        self.avg_loss: Optional[float] = None
        self.seed_gains: deque = deque(maxlen=config.rsi_period)
        self.seed_losses: deque = deque(maxlen=config.rsi_period)
        self.rsi: Optional[float] = None

        # ATR% calculation (if needed)
        self.prev_close_for_atr: Optional[float] = None
        self.atr: Optional[float] = None
        self.tr_seed: deque = deque(maxlen=config.atr_period)
        self.atr_percent: Optional[float] = None

        # Swing tracking
        self.last_swing_high: Optional[SwingPoint] = None
        self.prev_swing_high: Optional[SwingPoint] = None
        self.last_swing_low: Optional[SwingPoint] = None
        self.prev_swing_low: Optional[SwingPoint] = None
        self.bar_index = 0

        # Divergence state
        self.current_divergence = "NONE"
        self.div_strength: Optional[float] = None

        # Failure swing detector
        self.fs_detector = _FailureSwingDetector(config)


class RSITimingEngine:
    """
    RSI Timing Engine with divergence detection and regime labeling.

    Features:
    - Wilder RSI calculation (O(1) incremental)
    - Regular divergence (reversal warnings)
    - Hidden divergence (continuation signals)
    - Regime bands (55/45 for timing confirmation)
    - Failure swings (exhaustion patterns)
    """

    def __init__(self, config: Optional[RSITimingConfig] = None):
        """Initialize engine with config."""
        self.config = config or RSITimingConfig()
        self._states: Dict[str, _TimeframeState] = {}

        # Initialize states for configured timeframes
        for tf in self.config.timeframes:
            self._states[tf] = _TimeframeState(self.config, tf)

    def on_candle_close(
        self, timeframe: str, candle: Candle, atr_percent: Optional[float] = None
    ) -> RSITimingState:
        """
        Process candle close and update RSI state.

        Args:
            timeframe: Timeframe identifier
            candle: OHLCV candle
            atr_percent: Optional external ATR% (preferred)

        Returns:
            RSITimingState with current values
        """
        # Ensure state exists
        if timeframe not in self._states:
            self._states[timeframe] = _TimeframeState(self.config, timeframe)

        state = self._states[timeframe]
        state.bar_index += 1

        # Update RSI
        self._update_rsi(state, candle.close)

        # Update ATR% if needed and not provided
        if atr_percent is not None:
            state.atr_percent = atr_percent
        elif not self.config.use_external_atr_percent or state.atr_percent is None:
            self._update_atr_percent(state, candle.high, candle.low, candle.close)

        # Update failure swing detector
        failure_swing = "NONE"
        if state.rsi is not None:
            failure_swing = state.fs_detector.update(state.rsi)

        return self._build_state(state)

    def on_swing_high(self, timeframe: str, ts: int, price_high: float) -> None:
        """
        Register a swing high from external market structure.

        Args:
            timeframe: Timeframe identifier
            ts: Timestamp of swing
            price_high: Price at swing high
        """
        if timeframe not in self._states:
            return

        state = self._states[timeframe]

        if state.rsi is None:
            return  # Skip if RSI not ready

        # Create swing point
        swing = SwingPoint(ts=ts, price=price_high, rsi=state.rsi, index=state.bar_index)

        # Shift swings
        state.prev_swing_high = state.last_swing_high
        state.last_swing_high = swing

        # Check for divergence
        self._check_divergence_high(state)

    def on_swing_low(self, timeframe: str, ts: int, price_low: float) -> None:
        """
        Register a swing low from external market structure.

        Args:
            timeframe: Timeframe identifier
            ts: Timestamp of swing
            price_low: Price at swing low
        """
        if timeframe not in self._states:
            return

        state = self._states[timeframe]

        if state.rsi is None:
            return

        swing = SwingPoint(ts=ts, price=price_low, rsi=state.rsi, index=state.bar_index)

        state.prev_swing_low = state.last_swing_low
        state.last_swing_low = swing

        self._check_divergence_low(state)

    def warmup(
        self,
        candles_by_tf: Dict[str, List[Candle]],
        atr_percent_by_tf: Optional[Dict[str, float]] = None,
    ) -> Dict[str, RSITimingState]:
        """
        Warmup engine with historical candles.

        Args:
            candles_by_tf: Dict mapping timeframe -> list of candles
            atr_percent_by_tf: Optional dict of external ATR% values

        Returns:
            Dict mapping timeframe -> RSITimingState
        """
        states = {}

        for tf, candles in candles_by_tf.items():
            atr_pct = atr_percent_by_tf.get(tf) if atr_percent_by_tf else None

            for candle in candles:
                state = self.on_candle_close(tf, candle, atr_percent=atr_pct)

            states[tf] = state

        return states

    def get_state(self, timeframe: str) -> Optional[RSITimingState]:
        """Get current state for a timeframe."""
        if timeframe not in self._states:
            return None

        return self._build_state(self._states[timeframe])

    def reset(self, timeframe: str) -> None:
        """Reset state for a timeframe."""
        if timeframe in self._states:
            self._states[timeframe] = _TimeframeState(self.config, timeframe)

    def _update_rsi(self, state: _TimeframeState, close: float) -> None:
        """Update RSI calculation (Wilder method)."""
        if state.prev_close is None:
            state.prev_close = close
            return

        # Calculate change
        change = close - state.prev_close
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        # Seed period
        if len(state.seed_gains) < self.config.rsi_period:
            state.seed_gains.append(gain)
            state.seed_losses.append(loss)

            if len(state.seed_gains) == self.config.rsi_period:
                # Seed complete
                state.avg_gain = sum(state.seed_gains) / self.config.rsi_period
                state.avg_loss = sum(state.seed_losses) / self.config.rsi_period
        else:
            # Wilder smoothing
            n = self.config.rsi_period
            state.avg_gain = (state.avg_gain * (n - 1) + gain) / n
            state.avg_loss = (state.avg_loss * (n - 1) + loss) / n

        # Calculate RSI
        if state.avg_gain is not None and state.avg_loss is not None:
            # Handle edge case: both avg_gain and avg_loss are 0 (no movement)
            if state.avg_gain == 0.0 and state.avg_loss == 0.0:
                state.rsi = 50.0  # Neutral RSI when no movement
            else:
                rs = state.avg_gain / (state.avg_loss + 1e-12)
                state.rsi = 100.0 - (100.0 / (1.0 + rs))

        state.prev_close = close

    def _update_atr_percent(
        self, state: _TimeframeState, high: float, low: float, close: float
    ) -> None:
        """Update ATR% calculation (only if needed for divergence threshold)."""
        # Calculate TR
        if state.prev_close_for_atr is None:
            tr = high - low
        else:
            hl = high - low
            hc = abs(high - state.prev_close_for_atr)
            lc = abs(low - state.prev_close_for_atr)
            tr = max(hl, hc, lc)

        state.prev_close_for_atr = close

        # Seed period
        if len(state.tr_seed) < self.config.atr_period:
            state.tr_seed.append(tr)

            if len(state.tr_seed) == self.config.atr_period:
                # Seed ATR
                state.atr = sum(state.tr_seed) / self.config.atr_period
        else:
            # Wilder smoothing
            n = self.config.atr_period
            state.atr = (state.atr * (n - 1) + tr) / n

        # Calculate ATR%
        if state.atr is not None:
            state.atr_percent = state.atr / (close + 1e-12)

    def _check_divergence_high(self, state: _TimeframeState) -> None:
        """Check for divergence at swing high."""
        if state.prev_swing_high is None:
            state.current_divergence = "NONE"
            state.div_strength = None
            return

        new_swing = state.last_swing_high
        prev_swing = state.prev_swing_high

        # Check separation
        min_bars = self.config.get_min_bars(state.timeframe)
        if (new_swing.index - prev_swing.index) < min_bars:
            return

        # Check price difference threshold
        price_delta = abs(new_swing.price - prev_swing.price)
        min_threshold = self._get_min_price_threshold(state, new_swing.price)

        if price_delta < min_threshold:
            return

        # Calculate deltas
        price_higher = new_swing.price > prev_swing.price
        rsi_higher = new_swing.rsi > prev_swing.rsi

        # Detect divergence
        divergence = "NONE"

        if price_higher and not rsi_higher:
            # Regular bearish divergence
            divergence = "REG_BEAR"
        elif not price_higher and rsi_higher:
            # Hidden bearish divergence
            divergence = "HID_BEAR"

        # Avoid noise: require clear difference
        if divergence != "NONE":
            # If price_delta is marginal, skip
            if price_delta < (2.0 * min_threshold):
                rsi_delta = abs(new_swing.rsi - prev_swing.rsi)
                if rsi_delta < 3.0:  # Less than 3 RSI points
                    divergence = "NONE"

        if divergence != "NONE":
            state.current_divergence = divergence
            state.div_strength = self._calculate_div_strength(
                price_delta, min_threshold, abs(new_swing.rsi - prev_swing.rsi)
            )
        else:
            state.current_divergence = "NONE"
            state.div_strength = None

    def _check_divergence_low(self, state: _TimeframeState) -> None:
        """Check for divergence at swing low."""
        if state.prev_swing_low is None:
            state.current_divergence = "NONE"
            state.div_strength = None
            return

        new_swing = state.last_swing_low
        prev_swing = state.prev_swing_low

        # Check separation
        min_bars = self.config.get_min_bars(state.timeframe)
        if (new_swing.index - prev_swing.index) < min_bars:
            return

        # Check price difference
        price_delta = abs(new_swing.price - prev_swing.price)
        min_threshold = self._get_min_price_threshold(state, new_swing.price)

        if price_delta < min_threshold:
            return

        # Calculate deltas
        price_lower = new_swing.price < prev_swing.price
        rsi_lower = new_swing.rsi < prev_swing.rsi

        # Detect divergence
        divergence = "NONE"

        if price_lower and not rsi_lower:
            # Regular bullish divergence
            divergence = "REG_BULL"
        elif not price_lower and rsi_lower:
            # Hidden bullish divergence
            divergence = "HID_BULL"

        # Avoid noise
        if divergence != "NONE":
            if price_delta < (2.0 * min_threshold):
                rsi_delta = abs(new_swing.rsi - prev_swing.rsi)
                if rsi_delta < 3.0:
                    divergence = "NONE"

        if divergence != "NONE":
            state.current_divergence = divergence
            state.div_strength = self._calculate_div_strength(
                price_delta, min_threshold, abs(new_swing.rsi - prev_swing.rsi)
            )
        else:
            state.current_divergence = "NONE"
            state.div_strength = None

    def _get_min_price_threshold(self, state: _TimeframeState, price: float) -> float:
        """Get minimum price difference threshold for divergence."""
        if state.atr_percent is not None:
            return self.config.min_price_diff_atrp_factor * state.atr_percent * price
        else:
            # Fallback: 0.1% of price
            return price * 0.001

    def _calculate_div_strength(
        self, price_delta: float, min_threshold: float, rsi_delta: float
    ) -> float:
        """Calculate divergence strength (0-100)."""
        # Normalize price delta (0-3 range, clip at 3x threshold)
        price_norm = min(price_delta / (min_threshold + 1e-12), 3.0) / 3.0

        # Normalize RSI delta (10 points = strong)
        rsi_norm = min(rsi_delta / 10.0, 3.0) / 3.0

        # Combine: 50% price, 50% RSI
        strength = 100.0 * (0.5 * price_norm + 0.5 * rsi_norm)

        return max(0.0, min(100.0, strength))

    def _build_state(self, state: _TimeframeState) -> RSITimingState:
        """Build RSITimingState from internal state."""
        # Determine regime
        if state.rsi is None:
            regime = "WARMUP"
        elif state.rsi >= self.config.regime_high:
            regime = "BULLISH"
        elif state.rsi <= self.config.regime_low:
            regime = "BEARISH"
        else:
            regime = "RANGE"

        # Get failure swing
        failure_swing = "NONE"
        if state.rsi is not None:
            failure_swing = state.fs_detector.update(state.rsi)

        return RSITimingState(
            rsi=state.rsi,
            rsi_regime=regime,
            divergence=state.current_divergence,
            div_strength_0_100=state.div_strength,
            failure_swing=failure_swing,
            debug={
                "timeframe": state.timeframe,
                "avg_gain": state.avg_gain,
                "avg_loss": state.avg_loss,
                "atr_percent": state.atr_percent,
                "regime_high": self.config.regime_high,
                "regime_low": self.config.regime_low,
                "last_swing_high": (
                    {"price": state.last_swing_high.price, "rsi": state.last_swing_high.rsi}
                    if state.last_swing_high
                    else None
                ),
                "last_swing_low": (
                    {"price": state.last_swing_low.price, "rsi": state.last_swing_low.rsi}
                    if state.last_swing_low
                    else None
                ),
            },
        )


def print_rsi_timing(states: Dict[str, RSITimingState]) -> None:
    """
    Print RSI timing states in compact format.

    Args:
        states: Dict mapping timeframe -> RSITimingState
    """
    print("RSI TIMING")
    for tf, state in sorted(states.items()):
        if state.rsi is None:
            print(f"{tf}: regime=WARMUP")
            continue

        parts = [f"{tf}:"]
        parts.append(f"rsi={state.rsi:.1f}")
        parts.append(f"regime={state.rsi_regime}")
        parts.append(f"div={state.divergence}")

        if state.div_strength_0_100 is not None:
            parts.append(f"strength={state.div_strength_0_100:.0f}")
        else:
            parts.append("strength=--")

        parts.append(f"fs={state.failure_swing}")

        print(" ".join(parts))


def format_rsi_state(timeframe: str, state: RSITimingState) -> str:
    """Format a single RSI state for display."""
    if state.rsi is None:
        return f"{timeframe}: regime=WARMUP"

    parts = [f"{timeframe}:"]
    parts.append(f"rsi={state.rsi:.1f}")
    parts.append(f"regime={state.rsi_regime}")
    parts.append(f"div={state.divergence}")

    if state.div_strength_0_100 is not None:
        parts.append(f"strength={state.div_strength_0_100:.0f}")
    else:
        parts.append("strength=--")

    parts.append(f"fs={state.failure_swing}")

    return " ".join(parts)


def interpret_rsi(state: RSITimingState) -> str:
    """Provide interpretation of RSI state."""
    if state.rsi is None:
        return "Insufficient data for RSI analysis"

    interpretations = []

    # Regime
    if state.rsi_regime == "BULLISH":
        interpretations.append("Bullish regime (RSI > 55) - trend favors longs")
    elif state.rsi_regime == "BEARISH":
        interpretations.append("Bearish regime (RSI < 45) - trend favors shorts")
    else:
        interpretations.append("Range regime (45-55) - no clear trend bias")

    # Divergence
    if state.divergence == "REG_BULL":
        interpretations.append("⚠️ Regular bullish divergence - potential reversal UP")
    elif state.divergence == "REG_BEAR":
        interpretations.append("⚠️ Regular bearish divergence - potential reversal DOWN")
    elif state.divergence == "HID_BULL":
        interpretations.append("✅ Hidden bullish divergence - continuation signal UP")
    elif state.divergence == "HID_BEAR":
        interpretations.append("✅ Hidden bearish divergence - continuation signal DOWN")

    # Failure swing
    if state.failure_swing == "BULL":
        interpretations.append("🔥 Bullish failure swing - strong reversal UP")
    elif state.failure_swing == "BEAR":
        interpretations.append("🔥 Bearish failure swing - strong reversal DOWN")

    return " | ".join(interpretations) if interpretations else "No significant signals"
