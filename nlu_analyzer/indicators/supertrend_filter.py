"""
Supertrend Filter Module - Regime Labeling and Directional Bias

Real-time Supertrend Engine used ONLY as:
- Regime label (TREND vs CHOP)
- Directional bias filter (UP vs DOWN)

NOT used as entry trigger. Provides context for other trading systems.

Features:
- Incremental O(1) updates per candle
- Wilder ATR smoothing
- Band locking (non-repainting)
- Flip-rate based regime detection
- Hysteresis for stability.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional


class Direction(Enum):
    """Supertrend direction."""

    UP = "UP"
    DOWN = "DOWN"


class Regime(Enum):
    """Market regime classification."""

    TREND = "TREND"
    CHOP = "CHOP"
    TRANSITION = "TRANSITION"


@dataclass
class Candle:
    """OHLCV candle structure."""

    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class SupertrendConfig:
    """Configuration for Supertrend Engine."""

    # Core Supertrend parameters
    atr_period: int = 10
    multiplier: float = 3.0
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "1h"])

    # Regime detection parameters
    flip_window: int = 20
    flip_rate_chop: float = 0.10  # If flip_rate > this => CHOP
    flip_rate_trend: float = 0.05  # If flip_rate < this => TREND-friendly
    min_hold_bars: int = 3  # Require N bars same direction for TREND
    st_distance_factor: float = 0.15  # Avg distance >= factor*ATR% for clean trend
    atrp_min_chop: float = 0.0010  # ATR% below this => more likely CHOP

    # Strength component weights
    strength_weight_hold: float = 0.35
    strength_weight_flip: float = 0.35
    strength_weight_dist: float = 0.30

    # Per-timeframe overrides
    per_tf_overrides: Dict[str, Dict] = field(default_factory=dict)

    def get_atr_period(self, tf: str) -> int:
        """Get ATR period for timeframe (with override support)"""
        if tf in self.per_tf_overrides:
            return self.per_tf_overrides[tf].get("atr_period", self.atr_period)
        return self.atr_period

    def get_multiplier(self, tf: str) -> float:
        """Get multiplier for timeframe (with override support)"""
        if tf in self.per_tf_overrides:
            return self.per_tf_overrides[tf].get("multiplier", self.multiplier)
        return self.multiplier


@dataclass
class SupertrendState:
    """State output for Supertrend per timeframe."""

    # Core Supertrend values
    st_direction: Direction
    st_line: float
    final_upper: float
    final_lower: float
    basic_upper: float
    basic_lower: float

    # ATR values
    atr: float
    atr_percent: float

    # Flip tracking
    flip_event: bool
    flips_last_n: int
    flip_rate: float

    # Regime classification
    regime: Regime
    regime_strength_0_100: float

    # Additional metrics
    direction_hold_count: int
    distance_avg: float

    # Debug information
    debug: Dict = field(default_factory=dict)


class _TimeframeState:
    """Internal state for a single timeframe."""

    def __init__(self, config: SupertrendConfig, tf: str):
        self.config = config
        self.tf = tf

        # ATR tracking
        self.atr: Optional[float] = None
        self.prev_close: Optional[float] = None
        self.tr_values: List[float] = []

        # Supertrend bands
        self.basic_upper: float = 0.0
        self.basic_lower: float = 0.0
        self.final_upper: float = 0.0
        self.final_lower: float = 0.0

        # Direction tracking
        self.direction: Direction = Direction.UP
        self.prev_direction: Direction = Direction.UP
        self.direction_hold_count: int = 0

        # Flip tracking
        self.flip_deque: Deque[bool] = deque(maxlen=config.flip_window)
        self.distance_deque: Deque[float] = deque(maxlen=config.flip_window)

        # State
        self.candle_count: int = 0
        self.is_ready: bool = False

    def compute_true_range(self, candle: Candle) -> float:
        """
        Compute True Range.

        TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
        """
        if self.prev_close is None:
            # First candle, use high-low
            return candle.high - candle.low

        tr1 = candle.high - candle.low
        tr2 = abs(candle.high - self.prev_close)
        tr3 = abs(candle.low - self.prev_close)

        return max(tr1, tr2, tr3)

    def update_atr(self, tr: float) -> None:
        """
        Update ATR using Wilder smoothing.

        First: ATR = SMA(TR, atr_period)
        Then: ATR = (ATR_prev * (n-1) + TR) / n.
        """
        atr_period = self.config.get_atr_period(self.tf)

        if self.atr is None:
            # Accumulate for seed
            self.tr_values.append(tr)

            if len(self.tr_values) >= atr_period:
                # Seed with SMA
                self.atr = sum(self.tr_values[-atr_period:]) / atr_period
                self.is_ready = True
        else:
            # Wilder smoothing
            self.atr = (self.atr * (atr_period - 1) + tr) / atr_period


class SupertrendEngine:
    """
    Real-time Supertrend Engine for regime labeling and directional bias.

    Provides TREND/CHOP regime classification and UP/DOWN directional filter.
    NOT used as entry trigger.
    """

    def __init__(self, config: Optional[SupertrendConfig] = None):
        """
        Initialize Supertrend Engine.

        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or SupertrendConfig()
        self._states: Dict[str, _TimeframeState] = {}

    def _get_or_create_state(self, tf: str) -> _TimeframeState:
        """Get or create state for a timeframe."""
        if tf not in self._states:
            self._states[tf] = _TimeframeState(self.config, tf)
        return self._states[tf]

    def _compute_bands(self, state: _TimeframeState, candle: Candle) -> None:
        """
        Compute basic and final Supertrend bands.

        Basic bands:
            basic_upper = HL2 + multiplier * ATR
            basic_lower = HL2 - multiplier * ATR

        Final bands (with locking):
            See formulas in docstring.
        """
        if state.atr is None or not state.is_ready:
            return

        multiplier = self.config.get_multiplier(state.tf)
        hl2 = (candle.high + candle.low) / 2.0

        # Basic bands
        basic_upper = hl2 + multiplier * state.atr
        basic_lower = hl2 - multiplier * state.atr

        state.basic_upper = basic_upper
        state.basic_lower = basic_lower

        # Final bands (with locking)
        if state.candle_count <= 1:
            # First time, just use basic
            state.final_upper = basic_upper
            state.final_lower = basic_lower
        else:
            # Upper band locking
            if basic_upper < state.final_upper or state.prev_close > state.final_upper:
                state.final_upper = basic_upper
            # else: keep prev_final_upper

            # Lower band locking
            if basic_lower > state.final_lower or state.prev_close < state.final_lower:
                state.final_lower = basic_lower
            # else: keep prev_final_lower

    def _update_direction(self, state: _TimeframeState, candle: Candle) -> bool:
        """
        Update direction and detect flips.

        Returns:
            True if direction flipped.
        """
        if not state.is_ready:
            return False

        state.prev_direction = state.direction

        # Determine new direction
        if candle.close > state.final_upper:
            state.direction = Direction.UP
        elif candle.close < state.final_lower:
            state.direction = Direction.DOWN
        # else: persist previous direction

        # Check for flip
        flip_event = state.direction != state.prev_direction

        # Update hold count
        if flip_event:
            state.direction_hold_count = 1
        else:
            state.direction_hold_count += 1

        # Track flips
        state.flip_deque.append(flip_event)

        return flip_event

    def _compute_regime(self, state: _TimeframeState, candle: Candle) -> tuple[Regime, float]:
        """
        Compute regime classification and strength.

        Returns:
            (regime, strength_0_100)
        """
        if not state.is_ready or len(state.flip_deque) < self.config.flip_window:
            return Regime.CHOP, 0.0

        eps = 1e-10

        # Compute metrics
        flips_count = sum(state.flip_deque)
        flip_rate = flips_count / self.config.flip_window

        atr_percent = state.atr / (candle.close + eps)

        # Current distance to st_line
        st_line = state.final_lower if state.direction == Direction.UP else state.final_upper
        distance = abs(candle.close - st_line) / (candle.close + eps)
        state.distance_deque.append(distance)

        # Average distance
        distance_avg = (
            sum(state.distance_deque) / len(state.distance_deque) if state.distance_deque else 0.0
        )

        # Classify regime
        is_chop = False

        # High flip rate => CHOP
        if flip_rate > self.config.flip_rate_chop:
            is_chop = True

        # Low ATR + moderate flips => CHOP
        if atr_percent < self.config.atrp_min_chop and flip_rate > 0.05:
            is_chop = True

        # Hugging line => CHOP
        if distance_avg < self.config.st_distance_factor * atr_percent:
            is_chop = True

        # Check for TREND criteria
        is_trend = (
            flip_rate < self.config.flip_rate_trend
            and state.direction_hold_count >= self.config.min_hold_bars
            and distance_avg >= self.config.st_distance_factor * atr_percent
        )

        if is_chop:
            regime = Regime.CHOP
        elif is_trend:
            regime = Regime.TREND
        else:
            regime = Regime.CHOP  # Conservative default

        # Compute strength
        hold_comp = min(1.0, state.direction_hold_count / (self.config.min_hold_bars * 2))
        flip_comp = 1.0 - min(1.0, flip_rate / self.config.flip_rate_chop)
        dist_comp = min(
            1.0, distance_avg / (2 * self.config.st_distance_factor * atr_percent + eps)
        )

        strength = 100.0 * (
            self.config.strength_weight_hold * hold_comp
            + self.config.strength_weight_flip * flip_comp
            + self.config.strength_weight_dist * dist_comp
        )

        # Cap strength for CHOP
        if regime == Regime.CHOP:
            strength = min(strength, 40.0)

        return regime, strength

    def warmup(self, candles_by_tf: Dict[str, List[Candle]]) -> Dict[str, SupertrendState]:
        """
        Warm up the engine with historical candles.

        Args:
            candles_by_tf: Dictionary of candle lists by timeframe

        Returns:
            Dictionary of SupertrendState by timeframe.
        """
        results = {}

        for tf, candles in candles_by_tf.items():
            state = self._get_or_create_state(tf)

            for candle in candles:
                # This will build up state incrementally
                result = self.on_candle_close(tf, candle)

            if state.is_ready:
                results[tf] = result

        return results

    def on_candle_close(self, tf: str, candle: Candle) -> SupertrendState:
        """
        Process a candle close and return Supertrend state.

        Args:
            tf: Timeframe string
            candle: Closed candle

        Returns:
            SupertrendState with all metrics.
        """
        state = self._get_or_create_state(tf)

        # Compute True Range
        tr = state.compute_true_range(candle)

        # Update ATR
        state.update_atr(tr)

        # Compute bands
        self._compute_bands(state, candle)

        # Update direction
        flip_event = self._update_direction(state, candle)

        # Compute regime
        regime, strength = self._compute_regime(state, candle)

        # Prepare output
        eps = 1e-10
        atr_percent = state.atr / (candle.close + eps) if state.atr else 0.0

        st_line = state.final_lower if state.direction == Direction.UP else state.final_upper

        flips_count = sum(state.flip_deque) if state.flip_deque else 0
        flip_rate = flips_count / self.config.flip_window if state.flip_deque else 0.0

        distance_avg = (
            sum(state.distance_deque) / len(state.distance_deque) if state.distance_deque else 0.0
        )

        # Build debug info
        debug_info = {
            "is_ready": state.is_ready,
            "candle_count": state.candle_count,
            "tr": tr,
            "flip_window": self.config.flip_window,
            "flip_rate_chop_thr": self.config.flip_rate_chop,
            "flip_rate_trend_thr": self.config.flip_rate_trend,
            "min_hold_bars": self.config.min_hold_bars,
            "st_distance_factor": self.config.st_distance_factor,
            "atrp_min_chop": self.config.atrp_min_chop,
        }

        result = SupertrendState(
            st_direction=state.direction,
            st_line=st_line,
            final_upper=state.final_upper,
            final_lower=state.final_lower,
            basic_upper=state.basic_upper,
            basic_lower=state.basic_lower,
            atr=state.atr if state.atr else 0.0,
            atr_percent=atr_percent,
            flip_event=flip_event,
            flips_last_n=flips_count,
            flip_rate=flip_rate,
            regime=regime,
            regime_strength_0_100=strength,
            direction_hold_count=state.direction_hold_count,
            distance_avg=distance_avg,
            debug=debug_info,
        )

        # Update state for next iteration
        state.prev_close = candle.close
        state.candle_count += 1

        return result

    def update(self, candles_by_tf: Dict[str, List[Candle]]) -> Dict[str, SupertrendState]:
        """
        Convenience method to update all timeframes.

        Args:
            candles_by_tf: Dictionary of candle lists by timeframe (latest candle last)

        Returns:
            Dictionary of SupertrendState by timeframe.
        """
        results = {}

        for tf, candles in candles_by_tf.items():
            if not candles:
                continue

            latest_candle = candles[-1]
            results[tf] = self.on_candle_close(tf, latest_candle)

        return results

    def get_state(self, tf: str) -> Optional[SupertrendState]:
        """
        Get current state for a timeframe.

        Args:
            tf: Timeframe string

        Returns:
            SupertrendState if available, None otherwise.
        """
        if tf not in self._states:
            return None

        state = self._states[tf]
        if not state.is_ready:
            return None

        # Return last computed state (requires calling on_candle_close first)
        # This is a simplified getter; in practice you'd cache the last result
        return None


def format_supertrend_output(states: Dict[str, SupertrendState], compact: bool = True) -> str:
    """
    Format Supertrend states for display.

    Args:
        states: Dictionary of SupertrendState by timeframe
        compact: If True, use compact single-line format per TF

    Returns:
        Formatted string.
    """
    lines = ["SUPERTREND"]

    for tf in sorted(states.keys()):
        state = states[tf]

        if compact:
            # Check if hugging line
            hugging = "YES" if state.distance_avg < 0.001 else "NO"

            line = (
                f"{tf}: dir={state.st_direction.value} "
                f"regime={state.regime.value:5s} "
                f"strength={state.regime_strength_0_100:.0f} "
                f"line={state.st_line:.2f} "
                f"atr%={state.atr_percent*100:.2f}% "
                f"flips{state.debug['flip_window']}={state.flips_last_n} "
                f"({state.flip_rate:.2f}) "
                f"hold={state.direction_hold_count} "
                f"dist_avg={state.distance_avg*100:.2f}%"
            )

            if hugging == "YES":
                line += " hugging_line=YES"

            lines.append(line)
        else:
            # Verbose format
            lines.append(f"\n{tf}:")
            lines.append(f"  Direction: {state.st_direction.value}")
            lines.append(f"  Regime: {state.regime.value}")
            lines.append(f"  Strength: {state.regime_strength_0_100:.1f}")
            lines.append(f"  ST Line: {state.st_line:.2f}")
            lines.append(f"  ATR: {state.atr:.2f} ({state.atr_percent*100:.2f}%)")
            lines.append(
                f"  Flips (last {state.debug['flip_window']}): {state.flips_last_n} ({state.flip_rate:.2%})"
            )
            lines.append(f"  Hold Count: {state.direction_hold_count}")
            lines.append(f"  Distance Avg: {state.distance_avg*100:.2f}%")

    return "\n".join(lines)
