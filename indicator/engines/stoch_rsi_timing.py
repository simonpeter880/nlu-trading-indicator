"""
Stochastic RSI Timing Module - Micro Timing of Pullback Completion

Implements StochRSI with O(1) incremental updates for detecting pullback completion
in trending markets. This is a MICRO TIMING tool, NOT a standalone entry signal.

CRITICAL: Only use StochRSI signals when:
- Trend permission granted (EMA/ribbon/supertrend confirms trend)
- Directional bias established (+1 bull / -1 bear)
- Optionally: price near VWAP/EMA support/resistance
- Optionally: not in choppy regime (CHOP filter)

StochRSI alone is NOT an entry - it's the final timing confirmation
after all other conditions are met.
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
class StochRSIConfig:
    """Configuration for Stochastic RSI timing."""

    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "1h"])

    # RSI settings (if computing internally)
    rsi_period: int = 14
    use_external_rsi: bool = True  # Prefer external RSI from rsi_timing.py

    # StochRSI settings
    stoch_period: int = 14  # Lookback for RSI min/max
    k_smooth: int = 3  # Smoothing for %K
    d_smooth: int = 3  # Smoothing for %D

    # Zones
    oversold: float = 0.20
    overbought: float = 0.80

    # Timing confirmation (anti-noise)
    confirm_bars: int = 2  # Require signal to persist N bars

    # Noise guards
    disable_when_chop: bool = True
    disable_when_atr_squeeze: bool = False
    eps: float = 1e-12


@dataclass
class StochRSIState:
    """Stochastic RSI state for a timeframe."""

    rsi: Optional[float]
    stochrsi: Optional[float]  # 0-1
    k: Optional[float]  # Smoothed %K (0-1)
    d: Optional[float]  # Smoothed %D (0-1)
    zone: str  # OVERSOLD / NEUTRAL / OVERBOUGHT
    micro_timing: str  # PULLBACK_DONE_UP / PULLBACK_DONE_DOWN / NONE
    timing_conf_0_100: Optional[float]
    debug: Dict


class _TimeframeState:
    """Internal state for a single timeframe."""

    def __init__(self, config: StochRSIConfig, timeframe: str):
        self.config = config
        self.timeframe = timeframe

        # RSI state (if computing internally)
        self.prev_close: Optional[float] = None
        self.avg_gain: Optional[float] = None
        self.avg_loss: Optional[float] = None
        self.seed_gains: deque = deque(maxlen=config.rsi_period)
        self.seed_losses: deque = deque(maxlen=config.rsi_period)
        self.rsi: Optional[float] = None

        # Monotonic deques for RSI min/max (O(1) amortized)
        self.rsi_min_deque: deque = deque()  # (index, rsi) increasing
        self.rsi_max_deque: deque = deque()  # (index, rsi) decreasing
        self.bar_index = 0

        # StochRSI smoothing (rolling SMA)
        self.stochrsi_deque: deque = deque(maxlen=config.k_smooth)
        self.stochrsi_sum = 0.0

        self.k_deque: deque = deque(maxlen=config.d_smooth)
        self.k_sum = 0.0

        # Current values
        self.stochrsi: Optional[float] = None
        self.k: Optional[float] = None
        self.d: Optional[float] = None
        self.k_prev: Optional[float] = None
        self.d_prev: Optional[float] = None

        # Timing confirmation
        self.timing_candidate = "NONE"
        self.confirm_count = 0


class StochRSITimingEngine:
    """
    Stochastic RSI Timing Engine for micro timing of pullback completion.

    Features:
    - O(1) incremental StochRSI calculation
    - Monotonic deques for rolling RSI min/max
    - Pullback completion detection (oversold -> turn up, etc.)
    - Multi-gate confirmation (trend, bias, VWAP, CHOP, ATR)
    - Anti-noise confirmation period
    """

    def __init__(self, config: Optional[StochRSIConfig] = None):
        """Initialize engine with config."""
        self.config = config or StochRSIConfig()
        self._states: Dict[str, _TimeframeState] = {}

        # Initialize states for configured timeframes
        for tf in self.config.timeframes:
            self._states[tf] = _TimeframeState(self.config, tf)

    def on_candle_close(
        self,
        timeframe: str,
        candle: Candle,
        rsi_value: Optional[float] = None,
        trend_permission: Optional[bool] = None,
        bias: Optional[int] = None,
        bias_str: Optional[str] = None,
        vwap_near: Optional[bool] = None,
        chop_state: Optional[str] = None,
        atr_exp_state: Optional[str] = None,
    ) -> StochRSIState:
        """
        Process candle close and update StochRSI state.

        Args:
            timeframe: Timeframe identifier
            candle: OHLCV candle
            rsi_value: External RSI value (preferred if available)
            trend_permission: Trend gate (from EMA/ribbon/etc.)
            bias: Directional bias (+1 bull, -1 bear, 0 neutral)
            bias_str: Alternative bias as string ("BULL"/"BEAR"/"NEUTRAL")
            vwap_near: Price near VWAP/support/resistance
            chop_state: Choppiness state ("CHOP"/"TREND"/etc.)
            atr_exp_state: ATR expansion state ("SQUEEZE"/"EXPANSION"/etc.)

        Returns:
            StochRSIState with current values
        """
        # Ensure state exists
        if timeframe not in self._states:
            self._states[timeframe] = _TimeframeState(self.config, timeframe)

        state = self._states[timeframe]
        state.bar_index += 1

        # Parse bias
        if bias_str is not None:
            if bias_str == "BULL":
                bias = 1
            elif bias_str == "BEAR":
                bias = -1
            else:
                bias = 0

        # Update RSI (use external if provided, else compute)
        if rsi_value is not None and self.config.use_external_rsi:
            state.rsi = rsi_value
        else:
            self._update_rsi(state, candle.close)

        # Update StochRSI
        if state.rsi is not None:
            self._update_stochrsi(state)

        # Check micro timing
        self._check_micro_timing(
            state,
            trend_permission=trend_permission,
            bias=bias,
            vwap_near=vwap_near,
            chop_state=chop_state,
            atr_exp_state=atr_exp_state,
        )

        return self._build_state(state)

    def warmup(
        self,
        candles_by_tf: Dict[str, List[Candle]],
        rsi_by_tf: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, StochRSIState]:
        """
        Warmup engine with historical candles.

        Args:
            candles_by_tf: Dict mapping timeframe -> list of candles
            rsi_by_tf: Optional dict mapping timeframe -> list of RSI values

        Returns:
            Dict mapping timeframe -> StochRSIState
        """
        states = {}

        for tf, candles in candles_by_tf.items():
            rsi_values = rsi_by_tf.get(tf) if rsi_by_tf else None

            for i, candle in enumerate(candles):
                rsi_val = rsi_values[i] if rsi_values and i < len(rsi_values) else None
                state = self.on_candle_close(tf, candle, rsi_value=rsi_val)

            states[tf] = state

        return states

    def get_state(self, timeframe: str) -> Optional[StochRSIState]:
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
            if state.avg_gain == 0.0 and state.avg_loss == 0.0:
                state.rsi = 50.0  # Neutral when no movement
            else:
                rs = state.avg_gain / (state.avg_loss + self.config.eps)
                state.rsi = 100.0 - (100.0 / (1.0 + rs))

        state.prev_close = close

    def _update_rsi_min_max(self, state: _TimeframeState, rsi: float) -> None:
        """Update monotonic deques for rolling RSI min/max."""
        # Update min deque (increasing order)
        while state.rsi_min_deque and state.rsi_min_deque[-1][1] >= rsi:
            state.rsi_min_deque.pop()
        state.rsi_min_deque.append((state.bar_index, rsi))

        # Remove old elements from min deque
        while (
            state.rsi_min_deque
            and state.rsi_min_deque[0][0] <= state.bar_index - self.config.stoch_period
        ):
            state.rsi_min_deque.popleft()

        # Update max deque (decreasing order)
        while state.rsi_max_deque and state.rsi_max_deque[-1][1] <= rsi:
            state.rsi_max_deque.pop()
        state.rsi_max_deque.append((state.bar_index, rsi))

        # Remove old elements from max deque
        while (
            state.rsi_max_deque
            and state.rsi_max_deque[0][0] <= state.bar_index - self.config.stoch_period
        ):
            state.rsi_max_deque.popleft()

    def _update_stochrsi(self, state: _TimeframeState) -> None:
        """Update StochRSI calculation."""
        rsi = state.rsi

        # Update rolling min/max
        self._update_rsi_min_max(state, rsi)

        # Get min/max from deques
        if not state.rsi_min_deque or not state.rsi_max_deque:
            return

        rsi_min = state.rsi_min_deque[0][1]
        rsi_max = state.rsi_max_deque[0][1]

        # Calculate StochRSI
        rsi_range = rsi_max - rsi_min
        if rsi_range < self.config.eps:
            stochrsi = 0.5  # Neutral when no range
        else:
            stochrsi = (rsi - rsi_min) / (rsi_range + self.config.eps)

        # Clamp to [0, 1]
        stochrsi = max(0.0, min(1.0, stochrsi))
        state.stochrsi = stochrsi

        # Update %K (smoothed StochRSI)
        if len(state.stochrsi_deque) == self.config.k_smooth:
            # Remove oldest from sum
            state.stochrsi_sum -= state.stochrsi_deque[0]

        state.stochrsi_deque.append(stochrsi)
        state.stochrsi_sum += stochrsi

        if len(state.stochrsi_deque) == self.config.k_smooth:
            state.k_prev = state.k
            state.k = state.stochrsi_sum / self.config.k_smooth

            # Update %D (smoothed %K)
            if len(state.k_deque) == self.config.d_smooth:
                # Remove oldest from sum
                state.k_sum -= state.k_deque[0]

            state.k_deque.append(state.k)
            state.k_sum += state.k

            if len(state.k_deque) == self.config.d_smooth:
                state.d_prev = state.d
                state.d = state.k_sum / self.config.d_smooth

    def _check_micro_timing(
        self,
        state: _TimeframeState,
        trend_permission: Optional[bool],
        bias: Optional[int],
        vwap_near: Optional[bool],
        chop_state: Optional[str],
        atr_exp_state: Optional[str],
    ) -> None:
        """Check for pullback completion micro timing."""
        # Reset timing if prerequisites not met
        if state.k is None or state.d is None:
            state.timing_candidate = "NONE"
            state.confirm_count = 0
            return

        # Apply gates
        gates_ok = True

        # Trend permission gate
        if trend_permission is not None and not trend_permission:
            gates_ok = False

        # Bias gate
        if bias is not None and bias == 0:
            gates_ok = False

        # VWAP proximity gate (optional)
        if vwap_near is not None and not vwap_near:
            gates_ok = False

        # CHOP noise guard
        if self.config.disable_when_chop and chop_state is not None and chop_state == "CHOP":
            gates_ok = False

        # ATR squeeze guard (optional)
        if (
            self.config.disable_when_atr_squeeze
            and atr_exp_state is not None
            and atr_exp_state == "SQUEEZE"
        ):
            gates_ok = False

        if not gates_ok:
            state.timing_candidate = "NONE"
            state.confirm_count = 0
            return

        # Detect pullback completion
        candidate = "NONE"

        # Bull pullback completion (bias = +1)
        if bias is not None and bias == 1:
            # Turn up from oversold
            if (
                state.k_prev is not None
                and state.k_prev < self.config.oversold
                and state.k > self.config.oversold
                and state.k > state.k_prev
            ):
                candidate = "PULLBACK_DONE_UP"

            # K crosses above D in oversold zone
            elif (
                state.k_prev is not None
                and state.d_prev is not None
                and state.k <= 0.30
                and state.d <= 0.30
                and state.k_prev <= state.d_prev
                and state.k > state.d
            ):
                candidate = "PULLBACK_DONE_UP"

        # Bear pullback completion (bias = -1)
        elif bias is not None and bias == -1:
            # Turn down from overbought
            if (
                state.k_prev is not None
                and state.k_prev > self.config.overbought
                and state.k < self.config.overbought
                and state.k < state.k_prev
            ):
                candidate = "PULLBACK_DONE_DOWN"

            # K crosses below D in overbought zone
            elif (
                state.k_prev is not None
                and state.d_prev is not None
                and state.k >= 0.70
                and state.d >= 0.70
                and state.k_prev >= state.d_prev
                and state.k < state.d
            ):
                candidate = "PULLBACK_DONE_DOWN"

        # Confirmation logic (anti-noise)
        if candidate != "NONE":
            if candidate == state.timing_candidate:
                # Same candidate, increment count
                state.confirm_count += 1
            else:
                # New candidate, reset
                state.timing_candidate = candidate
                state.confirm_count = 1
        else:
            # No candidate, reset
            state.timing_candidate = "NONE"
            state.confirm_count = 0

    def _calculate_timing_conf(self, state: _TimeframeState, micro_timing: str) -> Optional[float]:
        """Calculate timing confidence (0-100)."""
        if micro_timing == "NONE" or state.k is None:
            return None

        if micro_timing == "PULLBACK_DONE_UP":
            # Base confidence: how far from oversold to neutral
            base = (state.k - self.config.oversold) / (
                0.50 - self.config.oversold + self.config.eps
            )
            base = max(0.0, min(1.0, base))

            # Cross bonus if K crossed D
            cross_bonus = 0.0
            if state.d is not None and state.k > state.d:
                cross_bonus = 0.2

            conf = 100.0 * min(base + cross_bonus, 1.0)
            return conf

        elif micro_timing == "PULLBACK_DONE_DOWN":
            # Base confidence: how far from overbought to neutral
            base = (self.config.overbought - state.k) / (
                self.config.overbought - 0.50 + self.config.eps
            )
            base = max(0.0, min(1.0, base))

            # Cross bonus if K crossed below D
            cross_bonus = 0.0
            if state.d is not None and state.k < state.d:
                cross_bonus = 0.2

            conf = 100.0 * min(base + cross_bonus, 1.0)
            return conf

        return None

    def _build_state(self, state: _TimeframeState) -> StochRSIState:
        """Build StochRSIState from internal state."""
        # Determine zone
        if state.k is None:
            zone = "WARMUP"
        elif state.k <= self.config.oversold:
            zone = "OVERSOLD"
        elif state.k >= self.config.overbought:
            zone = "OVERBOUGHT"
        else:
            zone = "NEUTRAL"

        # Determine micro timing (require confirmation)
        if state.confirm_count >= self.config.confirm_bars:
            micro_timing = state.timing_candidate
        else:
            micro_timing = "NONE"

        # Calculate confidence
        timing_conf = self._calculate_timing_conf(state, micro_timing)

        return StochRSIState(
            rsi=state.rsi,
            stochrsi=state.stochrsi,
            k=state.k,
            d=state.d,
            zone=zone,
            micro_timing=micro_timing,
            timing_conf_0_100=timing_conf,
            debug={
                "timeframe": state.timeframe,
                "rsi_min": state.rsi_min_deque[0][1] if state.rsi_min_deque else None,
                "rsi_max": state.rsi_max_deque[0][1] if state.rsi_max_deque else None,
                "oversold_thr": self.config.oversold,
                "overbought_thr": self.config.overbought,
                "timing_candidate": state.timing_candidate,
                "confirm_count": state.confirm_count,
                "confirm_bars": self.config.confirm_bars,
            },
        )


def print_stoch_rsi_timing(states: Dict[str, StochRSIState]) -> None:
    """
    Print Stochastic RSI timing states in compact format.

    Args:
        states: Dict mapping timeframe -> StochRSIState
    """
    print("STOCH RSI TIMING")
    for tf, state in sorted(states.items()):
        if state.k is None:
            print(f"{tf}: zone=WARMUP")
            continue

        parts = [f"{tf}:"]
        parts.append(f"K={state.k:.2f}")
        parts.append(f"D={state.d:.2f}" if state.d is not None else "D=--")
        parts.append(f"zone={state.zone}")
        parts.append(f"micro={state.micro_timing}")

        if state.timing_conf_0_100 is not None:
            parts.append(f"conf={state.timing_conf_0_100:.0f}")
        else:
            parts.append("conf=--")

        print(" ".join(parts))


def format_stoch_rsi_state(timeframe: str, state: StochRSIState) -> str:
    """Format a single StochRSI state for display."""
    if state.k is None:
        return f"{timeframe}: zone=WARMUP"

    parts = [f"{timeframe}:"]
    parts.append(f"K={state.k:.2f}")
    parts.append(f"D={state.d:.2f}" if state.d is not None else "D=--")
    parts.append(f"zone={state.zone}")
    parts.append(f"micro={state.micro_timing}")

    if state.timing_conf_0_100 is not None:
        parts.append(f"conf={state.timing_conf_0_100:.0f}")
    else:
        parts.append("conf=--")

    return " ".join(parts)


def interpret_stoch_rsi(state: StochRSIState) -> str:
    """Provide interpretation of StochRSI state."""
    if state.k is None:
        return "Insufficient data for StochRSI analysis"

    interpretations = []

    # Zone
    if state.zone == "OVERSOLD":
        interpretations.append("Oversold zone - potential bounce IF trend supports")
    elif state.zone == "OVERBOUGHT":
        interpretations.append("Overbought zone - potential pullback IF trend supports")
    else:
        interpretations.append("Neutral zone - no extreme")

    # Micro timing
    if state.micro_timing == "PULLBACK_DONE_UP":
        interpretations.append("✅ Pullback completion UP - micro timing for long entry")
    elif state.micro_timing == "PULLBACK_DONE_DOWN":
        interpretations.append("✅ Pullback completion DOWN - micro timing for short entry")

    return " | ".join(interpretations) if interpretations else "No significant timing signal"
