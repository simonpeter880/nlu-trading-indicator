"""
MACD Histogram Module - Momentum Shift Detector

Implements MACD histogram for detecting momentum phase shifts and
strength changes. This is a MOMENTUM SHIFT tool, NOT an entry signal.

CRITICAL: MACD histogram alone is NOT an entry - use it for:
- Phase identification (bullish vs bearish momentum)
- Shift detection (momentum changing direction)
- Weakening warnings (momentum losing strength)

Always combine with trend structure, volume, and other confirmations.

Formula:
- MACD Line = EMA(fast) - EMA(slow)
- Signal Line = EMA(signal_period, MACD Line)
- Histogram = MACD Line - Signal Line
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from collections import deque


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
class MACDHistConfig:
    """Configuration for MACD Histogram."""
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "1h"])

    # MACD periods (standard: 12, 26, 9)
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9

    # Confirmation (anti-noise)
    confirm_bars_shift: int = 2     # Bars to confirm shift
    confirm_bars_weaken: int = 2    # Bars to confirm weakening

    # Normalization
    normalize: str = "atr"  # "atr" / "price" / "none"
    eps: float = 1e-12
    clip_norm: float = 3.0  # Clip normalized values

    # Gating (optional)
    suppress_when_chop: bool = True
    suppress_when_squeeze: bool = False

    # Thresholds (for normalized or raw hist/slope)
    slope_thr_norm: float = 0.05   # Minimum slope magnitude
    hist_thr_norm: float = 0.02    # Near-zero threshold


@dataclass
class MACDHistState:
    """MACD Histogram state for a timeframe."""
    ema_fast: Optional[float]
    ema_slow: Optional[float]
    macd: Optional[float]
    signal: Optional[float]
    hist: Optional[float]
    hist_slope: Optional[float]
    hist_accel: Optional[float]  # Second derivative (optional)
    hist_norm: Optional[float]   # Normalized histogram
    phase: str                   # BULL / BEAR / WARMUP
    event: str                   # BULL_SHIFT / BEAR_SHIFT / BULL_WEAKEN / BEAR_WEAKEN / NONE
    event_confidence_0_100: Optional[float]
    debug: Dict


class _EMAState:
    """State for incremental EMA calculation."""

    def __init__(self, period: int):
        self.period = period
        self.alpha = 2.0 / (period + 1.0)
        self.value: Optional[float] = None
        self.seed_buffer: deque = deque(maxlen=period)

    def update(self, x: float) -> Optional[float]:
        """Update EMA with new value."""
        if self.value is None:
            # Seeding phase
            self.seed_buffer.append(x)

            if len(self.seed_buffer) == self.period:
                # Seed with SMA
                self.value = sum(self.seed_buffer) / self.period

            return self.value
        else:
            # Incremental update
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
            return self.value


class _TimeframeState:
    """Internal state for a single timeframe."""

    def __init__(self, config: MACDHistConfig, timeframe: str):
        self.config = config
        self.timeframe = timeframe

        # EMA states
        self.ema_fast = _EMAState(config.fast_period)
        self.ema_slow = _EMAState(config.slow_period)
        self.ema_signal = _EMAState(config.signal_period)

        # Current values
        self.macd: Optional[float] = None
        self.signal: Optional[float] = None
        self.hist: Optional[float] = None
        self.hist_prev: Optional[float] = None
        self.hist_slope: Optional[float] = None
        self.hist_slope_prev: Optional[float] = None

        # Normalized values
        self.hist_norm: Optional[float] = None
        self.hist_norm_prev: Optional[float] = None

        # Event tracking
        self.shift_candidate = "NONE"
        self.shift_count = 0
        self.weaken_candidate = "NONE"
        self.weaken_count = 0

        self.bar_index = 0


class MACDHistogramEngine:
    """
    MACD Histogram Engine for momentum shift detection.

    Features:
    - O(1) incremental EMA updates
    - Phase identification (bull/bear momentum)
    - Shift detection (momentum reversals)
    - Weakening detection (momentum deceleration)
    - Optional normalization by ATR or price
    - Anti-noise confirmation periods
    """

    def __init__(self, config: Optional[MACDHistConfig] = None):
        """Initialize engine with config."""
        self.config = config or MACDHistConfig()
        self._states: Dict[str, _TimeframeState] = {}

        # Initialize states for configured timeframes
        for tf in self.config.timeframes:
            self._states[tf] = _TimeframeState(self.config, tf)

    def on_candle_close(
        self,
        timeframe: str,
        candle: Candle,
        atr: Optional[float] = None,
        atr_percent: Optional[float] = None,
        chop_state: Optional[str] = None,
        atr_exp_state: Optional[str] = None
    ) -> MACDHistState:
        """
        Process candle close and update MACD histogram state.

        Args:
            timeframe: Timeframe identifier
            candle: OHLCV candle
            atr: ATR value (for normalization)
            atr_percent: ATR% value (fallback for normalization)
            chop_state: Choppiness state ("CHOP"/"TREND"/etc.)
            atr_exp_state: ATR expansion state

        Returns:
            MACDHistState with current values
        """
        # Ensure state exists
        if timeframe not in self._states:
            self._states[timeframe] = _TimeframeState(self.config, timeframe)

        state = self._states[timeframe]
        state.bar_index += 1

        # Update EMAs
        close = candle.close
        ema_fast_val = state.ema_fast.update(close)
        ema_slow_val = state.ema_slow.update(close)

        # Calculate MACD line
        if ema_fast_val is not None and ema_slow_val is not None:
            state.macd = ema_fast_val - ema_slow_val

            # Update signal line EMA
            signal_val = state.ema_signal.update(state.macd)

            if signal_val is not None:
                state.signal = signal_val

                # Calculate histogram
                state.hist_prev = state.hist
                state.hist = state.macd - state.signal

                # Calculate slope
                if state.hist_prev is not None:
                    state.hist_slope_prev = state.hist_slope
                    state.hist_slope = state.hist - state.hist_prev

                    # Calculate acceleration (optional)
                    if state.hist_slope_prev is not None:
                        hist_accel = state.hist_slope - state.hist_slope_prev
                    else:
                        hist_accel = None
                else:
                    hist_accel = None

                # Normalize histogram
                self._normalize_hist(state, candle.close, atr, atr_percent)

                # Detect events
                self._detect_events(
                    state,
                    chop_state=chop_state,
                    atr_exp_state=atr_exp_state
                )

        return self._build_state(state)

    def warmup(
        self,
        candles_by_tf: Dict[str, List[Candle]],
        atr_by_tf: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, MACDHistState]:
        """
        Warmup engine with historical candles.

        Args:
            candles_by_tf: Dict mapping timeframe -> list of candles
            atr_by_tf: Optional dict mapping timeframe -> list of ATR values

        Returns:
            Dict mapping timeframe -> MACDHistState
        """
        states = {}

        for tf, candles in candles_by_tf.items():
            atr_values = atr_by_tf.get(tf) if atr_by_tf else None

            for i, candle in enumerate(candles):
                atr_val = atr_values[i] if atr_values and i < len(atr_values) else None
                state = self.on_candle_close(tf, candle, atr=atr_val)

            states[tf] = state

        return states

    def get_state(self, timeframe: str) -> Optional[MACDHistState]:
        """Get current state for a timeframe."""
        if timeframe not in self._states:
            return None

        return self._build_state(self._states[timeframe])

    def reset(self, timeframe: str) -> None:
        """Reset state for a timeframe."""
        if timeframe in self._states:
            self._states[timeframe] = _TimeframeState(self.config, timeframe)

    def _normalize_hist(
        self,
        state: _TimeframeState,
        close: float,
        atr: Optional[float],
        atr_percent: Optional[float]
    ) -> None:
        """Normalize histogram by ATR or price."""
        if state.hist is None:
            return

        if self.config.normalize == "price":
            # Normalize by price
            denom = close + self.config.eps
            state.hist_norm_prev = state.hist_norm
            state.hist_norm = state.hist / denom
            # Clip
            state.hist_norm = max(-self.config.clip_norm,
                                 min(self.config.clip_norm, state.hist_norm))

        elif self.config.normalize == "atr":
            # Normalize by ATR
            if atr is not None:
                denom = atr + self.config.eps
            elif atr_percent is not None:
                denom = (atr_percent * close) + self.config.eps
            else:
                # No ATR available, skip normalization
                state.hist_norm_prev = state.hist_norm
                state.hist_norm = None
                return

            state.hist_norm_prev = state.hist_norm
            state.hist_norm = state.hist / denom
            # Clip
            state.hist_norm = max(-self.config.clip_norm,
                                 min(self.config.clip_norm, state.hist_norm))

        else:  # "none"
            state.hist_norm_prev = state.hist_norm
            state.hist_norm = None

    def _detect_events(
        self,
        state: _TimeframeState,
        chop_state: Optional[str],
        atr_exp_state: Optional[str]
    ) -> None:
        """Detect momentum shift and weakening events."""
        if state.hist is None or state.hist_slope is None:
            state.shift_candidate = "NONE"
            state.shift_count = 0
            state.weaken_candidate = "NONE"
            state.weaken_count = 0
            return

        # Use normalized values if available, else raw
        if state.hist_norm is not None:
            h = state.hist_norm
            h_prev = state.hist_norm_prev if state.hist_norm_prev is not None else 0.0
            s = h - h_prev if state.hist_norm_prev is not None else 0.0
        else:
            h = state.hist
            h_prev = state.hist_prev if state.hist_prev is not None else 0.0
            s = state.hist_slope

        # Detect shift candidates
        shift_candidate = "NONE"

        # BULL_SHIFT: crosses above 0 with positive slope
        if h_prev <= 0 and h > 0 and s > self.config.slope_thr_norm:
            shift_candidate = "BULL_SHIFT"

        # BEAR_SHIFT: crosses below 0 with negative slope
        elif h_prev >= 0 and h < 0 and s < -self.config.slope_thr_norm:
            shift_candidate = "BEAR_SHIFT"

        # Detect weakening candidates
        weaken_candidate = "NONE"

        # BULL_WEAKEN: hist > 0 but slope negative
        if h > 0 and s < -self.config.slope_thr_norm:
            weaken_candidate = "BULL_WEAKEN"

        # BEAR_WEAKEN: hist < 0 but slope positive
        elif h < 0 and s > self.config.slope_thr_norm:
            weaken_candidate = "BEAR_WEAKEN"

        # Update shift confirmation
        if shift_candidate != "NONE":
            if shift_candidate == state.shift_candidate:
                state.shift_count += 1
            else:
                state.shift_candidate = shift_candidate
                state.shift_count = 1
        else:
            state.shift_candidate = "NONE"
            state.shift_count = 0

        # Update weaken confirmation
        if weaken_candidate != "NONE":
            if weaken_candidate == state.weaken_candidate:
                state.weaken_count += 1
            else:
                state.weaken_candidate = weaken_candidate
                state.weaken_count = 1
        else:
            state.weaken_candidate = "NONE"
            state.weaken_count = 0

    def _calculate_event_confidence(
        self,
        state: _TimeframeState,
        event: str,
        h: float,
        s: float
    ) -> Optional[float]:
        """Calculate event confidence (0-100)."""
        if event == "NONE":
            return None

        # Magnitude component
        mag = abs(h) / (2.0 * self.config.hist_thr_norm + self.config.eps)
        mag = min(mag, 2.0) / 2.0  # Normalize to [0, 1]

        # Slope component
        slo = abs(s) / (2.0 * self.config.slope_thr_norm + self.config.eps)
        slo = min(slo, 2.0) / 2.0  # Normalize to [0, 1]

        # Combined confidence
        conf = 100.0 * (0.6 * mag + 0.4 * slo)

        # Cap at 50 if near zero
        if abs(h) < self.config.hist_thr_norm:
            conf = min(conf, 50.0)

        return conf

    def _build_state(self, state: _TimeframeState) -> MACDHistState:
        """Build MACDHistState from internal state."""
        # Determine phase
        if state.hist is None:
            phase = "WARMUP"
        elif state.hist > 0:
            phase = "BULL"
        else:
            phase = "BEAR"

        # Determine event (with confirmation and gating)
        event = "NONE"

        # Check shift first (higher priority)
        if state.shift_count >= self.config.confirm_bars_shift:
            event = state.shift_candidate

        # Then check weakening
        elif state.weaken_count >= self.config.confirm_bars_weaken:
            event = state.weaken_candidate

        # Apply gates (suppress events but keep values)
        gates_suppress = False

        if (self.config.suppress_when_chop and
            hasattr(state, '_chop_state') and
            getattr(state, '_chop_state') == "CHOP"):
            gates_suppress = True

        if (self.config.suppress_when_squeeze and
            hasattr(state, '_atr_exp_state') and
            getattr(state, '_atr_exp_state') == "SQUEEZE"):
            gates_suppress = True

        if gates_suppress:
            event = "NONE"

        # Calculate confidence
        if state.hist is not None and state.hist_slope is not None:
            h = state.hist_norm if state.hist_norm is not None else state.hist
            s = (state.hist_norm - state.hist_norm_prev
                 if state.hist_norm is not None and state.hist_norm_prev is not None
                 else state.hist_slope)
            event_conf = self._calculate_event_confidence(state, event, h, s)
        else:
            event_conf = None

        # Calculate acceleration
        hist_accel = None
        if state.hist_slope is not None and state.hist_slope_prev is not None:
            hist_accel = state.hist_slope - state.hist_slope_prev

        return MACDHistState(
            ema_fast=state.ema_fast.value,
            ema_slow=state.ema_slow.value,
            macd=state.macd,
            signal=state.signal,
            hist=state.hist,
            hist_slope=state.hist_slope,
            hist_accel=hist_accel,
            hist_norm=state.hist_norm,
            phase=phase,
            event=event,
            event_confidence_0_100=event_conf,
            debug={
                "timeframe": state.timeframe,
                "fast_period": self.config.fast_period,
                "slow_period": self.config.slow_period,
                "signal_period": self.config.signal_period,
                "normalize": self.config.normalize,
                "shift_candidate": state.shift_candidate,
                "shift_count": state.shift_count,
                "weaken_candidate": state.weaken_candidate,
                "weaken_count": state.weaken_count,
                "confirm_bars_shift": self.config.confirm_bars_shift,
                "confirm_bars_weaken": self.config.confirm_bars_weaken,
            }
        )


def print_macd_histogram(states: Dict[str, MACDHistState]) -> None:
    """
    Print MACD histogram states in compact format.

    Args:
        states: Dict mapping timeframe -> MACDHistState
    """
    print("MACD HIST")
    for tf, state in sorted(states.items()):
        if state.hist is None:
            print(f"{tf}: phase=WARMUP")
            continue

        parts = [f"{tf}:"]
        parts.append(f"phase={state.phase}")

        # Histogram value
        sign = "+" if state.hist >= 0 else ""
        parts.append(f"hist={sign}{state.hist:.2f}")

        # Slope
        if state.hist_slope is not None:
            sign = "+" if state.hist_slope >= 0 else ""
            parts.append(f"slope={sign}{state.hist_slope:.2f}")

        # Normalized
        if state.hist_norm is not None:
            sign = "+" if state.hist_norm >= 0 else ""
            parts.append(f"norm={sign}{state.hist_norm:.2f}")

        # Event
        parts.append(f"event={state.event}")

        # Confidence
        if state.event_confidence_0_100 is not None:
            parts.append(f"conf={state.event_confidence_0_100:.0f}")
        else:
            parts.append("conf=--")

        print(" ".join(parts))


def format_macd_state(timeframe: str, state: MACDHistState) -> str:
    """Format a single MACD state for display."""
    if state.hist is None:
        return f"{timeframe}: phase=WARMUP"

    parts = [f"{timeframe}:"]
    parts.append(f"phase={state.phase}")

    sign = "+" if state.hist >= 0 else ""
    parts.append(f"hist={sign}{state.hist:.2f}")

    if state.hist_slope is not None:
        sign = "+" if state.hist_slope >= 0 else ""
        parts.append(f"slope={sign}{state.hist_slope:.2f}")

    if state.hist_norm is not None:
        sign = "+" if state.hist_norm >= 0 else ""
        parts.append(f"norm={sign}{state.hist_norm:.2f}")

    parts.append(f"event={state.event}")

    if state.event_confidence_0_100 is not None:
        parts.append(f"conf={state.event_confidence_0_100:.0f}")
    else:
        parts.append("conf=--")

    return " ".join(parts)


def interpret_macd(state: MACDHistState) -> str:
    """Provide interpretation of MACD histogram state."""
    if state.hist is None:
        return "Insufficient data for MACD analysis"

    interpretations = []

    # Phase
    if state.phase == "BULL":
        interpretations.append("Bullish momentum (histogram > 0)")
    elif state.phase == "BEAR":
        interpretations.append("Bearish momentum (histogram < 0)")

    # Events
    if state.event == "BULL_SHIFT":
        interpretations.append("✅ BULL SHIFT - Momentum turning bullish")
    elif state.event == "BEAR_SHIFT":
        interpretations.append("✅ BEAR SHIFT - Momentum turning bearish")
    elif state.event == "BULL_WEAKEN":
        interpretations.append("⚠️ BULL WEAKENING - Bullish momentum losing strength")
    elif state.event == "BEAR_WEAKEN":
        interpretations.append("⚠️ BEAR WEAKENING - Bearish momentum losing strength")

    return " | ".join(interpretations) if interpretations else "Neutral momentum"
