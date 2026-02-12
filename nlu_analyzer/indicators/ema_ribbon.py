"""
EMA Ribbon Module - Trend Health Analysis

Real-time EMA Ribbon Engine that provides trend health metrics without entry signals.
Designed to integrate with existing EMA System outputs.

Features:
- Incremental EMA updates (O(1) per period)
- ATR-adaptive thresholds with static fallbacks
- Multi-timeframe support
- Trend health classification (HEALTHY/WEAKENING/EXHAUSTING/CHOP)
- Stack scoring, ribbon width analysis, center slope tracking.
"""

import statistics
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RibbonDirection(Enum):
    """Ribbon direction based on EMA stack alignment."""

    BULL = "BULL"
    BEAR = "BEAR"
    NEUTRAL = "NEUTRAL"


class RibbonState(Enum):
    """Ribbon health state classification."""

    HEALTHY = "HEALTHY"
    WEAKENING = "WEAKENING"
    EXHAUSTING = "EXHAUSTING"
    CHOP = "CHOP"


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
class EMARibbonConfig:
    """Configuration for EMA Ribbon Engine."""

    # Ribbon periods (must be sorted ascending, unique)
    ribbon_periods: List[int] = field(
        default_factory=lambda: [9, 12, 15, 18, 21, 25, 30, 35, 40, 50]
    )

    # ATR configuration
    atr_period: int = 14

    # Slope lookback by timeframe (bars to look back for center slope)
    slope_lookback_by_tf: Dict[str, int] = field(
        default_factory=lambda: {"1m": 15, "5m": 10, "1h": 4}
    )
    slope_lookback_default: int = 10

    # Width smoothing period
    width_smooth_period: int = 5

    # ATR%-adaptive threshold factors
    width_thr_factor: float = 0.10  # width_thr = 0.10 * ATR%
    slope_thr_factor: float = 0.15  # slope_thr = 0.15 * ATR%
    pullback_band_factor: float = 0.30  # pullback band = 0.30 * ATR%
    compress_rate_thr: float = -0.15  # width_rate threshold for compression
    expand_rate_thr: float = 0.15  # width_rate threshold for expansion

    # Static fallbacks if ATR% not available
    width_thr_static_by_tf: Dict[str, float] = field(
        default_factory=lambda: {"1m": 0.0008, "5m": 0.0012, "1h": 0.0020}
    )
    slope_thr_static_by_tf: Dict[str, float] = field(
        default_factory=lambda: {"1m": 0.00025, "5m": 0.00035, "1h": 0.0010}
    )
    pullback_band_static_by_tf: Dict[str, float] = field(
        default_factory=lambda: {"1m": 0.0010, "5m": 0.0020, "1h": 0.0060}
    )

    # Stack score thresholds
    stack_strong_thr: float = 0.80
    stack_ok_thr: float = 0.70
    stack_direction_thr: float = 0.60

    # Strength component weights
    strength_weight_stack: float = 0.30
    strength_weight_width: float = 0.25
    strength_weight_slope: float = 0.25
    strength_weight_expansion: float = 0.20

    def __post_init__(self):
        """Validate configuration."""
        # Validate ribbon periods
        if not self.ribbon_periods:
            raise ValueError("ribbon_periods cannot be empty")

        sorted_periods = sorted(set(self.ribbon_periods))
        if sorted_periods != list(self.ribbon_periods):
            raise ValueError("ribbon_periods must be sorted ascending and unique")

        if any(p <= 0 for p in self.ribbon_periods):
            raise ValueError("All ribbon_periods must be positive")

        # Validate weights sum to 1.0
        weight_sum = (
            self.strength_weight_stack
            + self.strength_weight_width
            + self.strength_weight_slope
            + self.strength_weight_expansion
        )
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"Strength weights must sum to 1.0, got {weight_sum}")


@dataclass
class RibbonStateOutput:
    """Output structure for ribbon state per timeframe."""

    # Core outputs
    ribbon_periods_used: List[int]
    emas: Dict[int, float]
    ribbon_direction: RibbonDirection
    stack_score: float
    ribbon_width: float
    ribbon_width_smooth: float
    width_rate: float
    ribbon_center: float
    ribbon_center_slope: float
    ribbon_state: RibbonState
    ribbon_strength_0_100: float
    pullback_into_ribbon: bool

    # Debug information
    debug: Dict[str, Any] = field(default_factory=dict)


class _TimeframeState:
    """Internal state for a single timeframe."""

    def __init__(self, config: EMARibbonConfig, tf: str):
        self.config = config
        self.tf = tf

        # EMA values by period
        self.ema_values: Dict[int, float] = {}

        # Width smoothing
        self.width_smooth: Optional[float] = None
        self.width_smooth_alpha: float = 2.0 / (config.width_smooth_period + 1)

        # History for slope and rate calculations
        slope_lookback = config.slope_lookback_by_tf.get(tf, config.slope_lookback_default)
        self.center_history: deque = deque(maxlen=slope_lookback + 1)
        self.width_smooth_history: deque = deque(maxlen=2)  # Only need previous value

        # Previous slope for exhaustion detection
        self.prev_center_slope: Optional[float] = None

        # Candle count for warmup
        self.candle_count: int = 0
        self.max_period: int = max(config.ribbon_periods)
        self.is_ready: bool = False

    def update_emas(self, close: float) -> None:
        """Incrementally update all EMA values."""
        for period in self.config.ribbon_periods:
            alpha = 2.0 / (period + 1)

            if period not in self.ema_values:
                # Seed with first close
                self.ema_values[period] = close
            else:
                # Incremental update
                prev_ema = self.ema_values[period]
                self.ema_values[period] = alpha * close + (1 - alpha) * prev_ema

        self.candle_count += 1
        if self.candle_count >= self.max_period:
            self.is_ready = True


class EMARibbonEngine:
    """
    Real-time EMA Ribbon Engine for trend health analysis.

    Provides incremental updates with O(1) complexity per period.
    Supports ATR-adaptive thresholds and multi-timeframe analysis.
    """

    def __init__(self, config: Optional[EMARibbonConfig] = None):
        """
        Initialize EMA Ribbon Engine.

        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or EMARibbonConfig()
        self._states: Dict[str, _TimeframeState] = {}

    def _get_or_create_state(self, tf: str) -> _TimeframeState:
        """Get or create state for a timeframe."""
        if tf not in self._states:
            self._states[tf] = _TimeframeState(self.config, tf)
        return self._states[tf]

    def _get_thresholds(self, tf: str, atr_percent: Optional[float]) -> Dict[str, float]:
        """
        Calculate adaptive thresholds based on ATR% or use static fallbacks.

        Args:
            tf: Timeframe string
            atr_percent: ATR as percentage of close (decimal, e.g., 0.01 = 1%)

        Returns:
            Dictionary with width_thr, slope_thr, pullback_band.
        """
        if atr_percent is not None and atr_percent > 0:
            return {
                "width_thr": self.config.width_thr_factor * atr_percent,
                "slope_thr": self.config.slope_thr_factor * atr_percent,
                "pullback_band": self.config.pullback_band_factor * atr_percent,
            }
        else:
            return {
                "width_thr": self.config.width_thr_static_by_tf.get(tf, 0.0010),
                "slope_thr": self.config.slope_thr_static_by_tf.get(tf, 0.0005),
                "pullback_band": self.config.pullback_band_static_by_tf.get(tf, 0.0020),
            }

    def _compute_stack_score(self, emas: Dict[int, float]) -> tuple[float, RibbonDirection]:
        """
        Compute stack score and direction.

        Args:
            emas: Dictionary of EMA values by period

        Returns:
            (stack_score, direction) where stack_score is 0..1.
        """
        periods = sorted(self.config.ribbon_periods)

        if len(periods) < 2:
            return 0.0, RibbonDirection.NEUTRAL

        # Count correct adjacencies for bull and bear
        correct_bull = 0
        correct_bear = 0
        total_pairs = len(periods) - 1

        for i in range(total_pairs):
            p_fast = periods[i]
            p_slow = periods[i + 1]

            ema_fast = emas.get(p_fast, 0)
            ema_slow = emas.get(p_slow, 0)

            if ema_fast > ema_slow:
                correct_bull += 1
            elif ema_fast < ema_slow:
                correct_bear += 1

        stack_score_bull = correct_bull / total_pairs if total_pairs > 0 else 0
        stack_score_bear = correct_bear / total_pairs if total_pairs > 0 else 0

        # Determine direction
        if (
            stack_score_bull >= self.config.stack_direction_thr
            and stack_score_bull > stack_score_bear
        ):
            direction = RibbonDirection.BULL
            stack_score = stack_score_bull
        elif (
            stack_score_bear >= self.config.stack_direction_thr
            and stack_score_bear > stack_score_bull
        ):
            direction = RibbonDirection.BEAR
            stack_score = stack_score_bear
        else:
            direction = RibbonDirection.NEUTRAL
            stack_score = max(stack_score_bull, stack_score_bear)

        return stack_score, direction

    def _compute_ribbon_width(
        self, emas: Dict[int, float], close: float, state: _TimeframeState
    ) -> tuple[float, float, float]:
        """
        Compute ribbon width, smoothed width, and width rate.

        Args:
            emas: Dictionary of EMA values
            close: Current close price
            state: Timeframe state

        Returns:
            (width, width_smooth, width_rate)
        """
        eps = 1e-10
        ema_values = list(emas.values())

        if not ema_values:
            return 0.0, 0.0, 0.0

        # Raw width
        width = (max(ema_values) - min(ema_values)) / (close + eps)

        # Smooth width with EMA
        if state.width_smooth is None:
            width_smooth = width
        else:
            width_smooth = (
                state.width_smooth_alpha * width
                + (1 - state.width_smooth_alpha) * state.width_smooth
            )

        # Compute width rate
        if len(state.width_smooth_history) > 0:
            prev_width_smooth = state.width_smooth_history[-1]
            width_rate = (width_smooth - prev_width_smooth) / (abs(prev_width_smooth) + eps)
        else:
            width_rate = 0.0

        # Update state
        state.width_smooth_history.append(width_smooth)
        state.width_smooth = width_smooth

        return width, width_smooth, width_rate

    def _compute_ribbon_center(
        self, emas: Dict[int, float], state: _TimeframeState
    ) -> tuple[float, float]:
        """
        Compute ribbon center and center slope.

        Args:
            emas: Dictionary of EMA values
            state: Timeframe state

        Returns:
            (center, center_slope)
        """
        eps = 1e-10
        ema_values = list(emas.values())

        if not ema_values:
            return 0.0, 0.0

        # Use median as center
        center = statistics.median(ema_values)

        # Compute slope
        if len(state.center_history) >= state.center_history.maxlen:
            # We have enough history
            center_old = state.center_history[0]
            center_slope = (center - center_old) / (abs(center_old) + eps)
        else:
            center_slope = 0.0

        # Update history
        state.center_history.append(center)

        # Store for exhaustion detection
        if state.prev_center_slope is not None:
            # Could use this for detecting slope decay
            pass
        state.prev_center_slope = center_slope

        return center, center_slope

    def _check_pullback(self, close: float, center: float, pullback_band: float) -> bool:
        """
        Check if price is pulling back into ribbon.

        Args:
            close: Current close price
            center: Ribbon center
            pullback_band: Pullback threshold band (as fraction of close price)

        Returns:
            True if pullback detected.
        """
        # pullback_band is a percentage (e.g., 0.003 = 0.3%)
        # Convert to absolute distance: band * close
        band_absolute = pullback_band * close
        return abs(close - center) <= band_absolute

    def _classify_state(
        self,
        direction: RibbonDirection,
        stack_score: float,
        width_smooth: float,
        center_slope: float,
        width_rate: float,
        thresholds: Dict[str, float],
        state: _TimeframeState,
    ) -> RibbonState:
        """
        Classify ribbon state based on metrics.

        Args:
            direction: Ribbon direction
            stack_score: Stack alignment score
            width_smooth: Smoothed ribbon width
            center_slope: Center slope
            width_rate: Width rate of change
            thresholds: Threshold dictionary
            state: Timeframe state

        Returns:
            RibbonState classification.
        """
        width_thr = thresholds["width_thr"]
        slope_thr = thresholds["slope_thr"]

        # Check for CHOP
        if stack_score < self.config.stack_direction_thr or (
            width_smooth < 0.7 * width_thr and abs(center_slope) < 0.5 * slope_thr
        ):
            return RibbonState.CHOP

        # Remaining states require BULL or BEAR direction
        if direction == RibbonDirection.NEUTRAL:
            return RibbonState.CHOP

        # Check quality indicators
        stack_good = stack_score >= self.config.stack_strong_thr
        stack_ok = stack_score >= self.config.stack_ok_thr
        width_good = width_smooth >= width_thr

        # Slope must align with direction
        if direction == RibbonDirection.BULL:
            slope_good = center_slope >= slope_thr
        else:  # BEAR
            slope_good = center_slope <= -slope_thr

        # HEALTHY: Strong trend with good metrics
        if stack_good and width_good and slope_good and width_rate >= self.config.compress_rate_thr:
            return RibbonState.HEALTHY

        # EXHAUSTING: Compression + slope decay
        exhaustion_compress = self.config.compress_rate_thr - 0.05
        if stack_ok and width_rate <= exhaustion_compress:
            # Check for slope decay if we have history
            slope_decaying = False
            if state.prev_center_slope is not None:
                if direction == RibbonDirection.BULL:
                    slope_decaying = center_slope < state.prev_center_slope
                else:
                    slope_decaying = center_slope > state.prev_center_slope

            if slope_decaying or abs(center_slope) < slope_thr:
                return RibbonState.EXHAUSTING

        # WEAKENING: Trend present but deteriorating
        if stack_ok and (
            width_rate < self.config.compress_rate_thr or abs(center_slope) < slope_thr
        ):
            return RibbonState.WEAKENING

        # Default to WEAKENING if we're trending but not healthy
        return RibbonState.WEAKENING

    def _compute_strength(
        self,
        stack_score: float,
        width_smooth: float,
        center_slope: float,
        width_rate: float,
        thresholds: Dict[str, float],
        ribbon_state: RibbonState,
    ) -> float:
        """
        Compute ribbon strength score (0-100).

        Args:
            stack_score: Stack alignment score
            width_smooth: Smoothed ribbon width
            center_slope: Center slope
            width_rate: Width rate of change
            thresholds: Threshold dictionary
            ribbon_state: Classified ribbon state

        Returns:
            Strength score 0-100.
        """
        eps = 1e-10

        # Component calculations (clip to 0..1)
        stack_component = max(0.0, min(1.0, (stack_score - 0.5) / 0.5))

        width_component = max(0.0, min(1.0, width_smooth / (2 * thresholds["width_thr"] + eps)))

        slope_component = max(
            0.0, min(1.0, abs(center_slope) / (2 * thresholds["slope_thr"] + eps))
        )

        expansion_range = self.config.expand_rate_thr - self.config.compress_rate_thr
        expansion_component = max(
            0.0, min(1.0, (width_rate - self.config.compress_rate_thr) / (expansion_range + eps))
        )

        # Weighted strength
        strength = 100 * (
            self.config.strength_weight_stack * stack_component
            + self.config.strength_weight_width * width_component
            + self.config.strength_weight_slope * slope_component
            + self.config.strength_weight_expansion * expansion_component
        )

        # Apply state caps
        if ribbon_state == RibbonState.CHOP:
            strength = min(strength, 35.0)
        elif ribbon_state == RibbonState.EXHAUSTING:
            strength = min(strength, 60.0)

        return strength

    def warmup(
        self,
        candles_by_tf: Dict[str, List[Candle]],
        atr_percent_by_tf: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Warm up the engine with historical candles.

        Args:
            candles_by_tf: Dictionary of candle lists by timeframe
            atr_percent_by_tf: Optional ATR% values by timeframe.
        """
        for tf, candles in candles_by_tf.items():
            state = self._get_or_create_state(tf)

            for candle in candles:
                state.update_emas(candle.close)

    def on_candle_close(
        self,
        tf: str,
        candle: Candle,
        atr_percent: Optional[float] = None,
        ema_system_state: Optional[Dict[str, Any]] = None,
    ) -> RibbonStateOutput:
        """
        Process a candle close and return ribbon state.

        Args:
            tf: Timeframe string
            candle: Closed candle
            atr_percent: Optional ATR as percentage of close
            ema_system_state: Optional state from EMA system (unused currently)

        Returns:
            RibbonStateOutput with all metrics.
        """
        state = self._get_or_create_state(tf)

        # Update EMAs
        state.update_emas(candle.close)

        if not state.is_ready:
            # Return default output during warmup
            return RibbonStateOutput(
                ribbon_periods_used=self.config.ribbon_periods,
                emas=state.ema_values.copy(),
                ribbon_direction=RibbonDirection.NEUTRAL,
                stack_score=0.0,
                ribbon_width=0.0,
                ribbon_width_smooth=0.0,
                width_rate=0.0,
                ribbon_center=0.0,
                ribbon_center_slope=0.0,
                ribbon_state=RibbonState.CHOP,
                ribbon_strength_0_100=0.0,
                pullback_into_ribbon=False,
                debug={"warmup": True, "candles": state.candle_count},
            )

        # Get thresholds
        thresholds = self._get_thresholds(tf, atr_percent)

        # Compute metrics
        stack_score, direction = self._compute_stack_score(state.ema_values)
        width, width_smooth, width_rate = self._compute_ribbon_width(
            state.ema_values, candle.close, state
        )
        center, center_slope = self._compute_ribbon_center(state.ema_values, state)
        pullback = self._check_pullback(candle.close, center, thresholds["pullback_band"])

        # Classify state
        ribbon_state = self._classify_state(
            direction, stack_score, width_smooth, center_slope, width_rate, thresholds, state
        )

        # Compute strength
        strength = self._compute_strength(
            stack_score, width_smooth, center_slope, width_rate, thresholds, ribbon_state
        )

        # Build debug info
        debug_info = {
            "thresholds": thresholds,
            "atr_percent": atr_percent,
            "slope_lookback": state.center_history.maxlen - 1,
            "candles_processed": state.candle_count,
        }

        return RibbonStateOutput(
            ribbon_periods_used=self.config.ribbon_periods,
            emas=state.ema_values.copy(),
            ribbon_direction=direction,
            stack_score=stack_score,
            ribbon_width=width,
            ribbon_width_smooth=width_smooth,
            width_rate=width_rate,
            ribbon_center=center,
            ribbon_center_slope=center_slope,
            ribbon_state=ribbon_state,
            ribbon_strength_0_100=strength,
            pullback_into_ribbon=pullback,
            debug=debug_info,
        )

    def update(
        self,
        candles_by_tf: Dict[str, List[Candle]],
        atr_percent_by_tf: Optional[Dict[str, float]] = None,
        ema_system_state_by_tf: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, RibbonStateOutput]:
        """
        Convenience method to update all timeframes.

        Args:
            candles_by_tf: Dictionary of candle lists by timeframe (latest candle last)
            atr_percent_by_tf: Optional ATR% values by timeframe
            ema_system_state_by_tf: Optional EMA system states by timeframe

        Returns:
            Dictionary of RibbonStateOutput by timeframe.
        """
        results = {}

        for tf, candles in candles_by_tf.items():
            if not candles:
                continue

            latest_candle = candles[-1]
            atr_percent = atr_percent_by_tf.get(tf) if atr_percent_by_tf else None
            ema_state = ema_system_state_by_tf.get(tf) if ema_system_state_by_tf else None

            results[tf] = self.on_candle_close(tf, latest_candle, atr_percent, ema_state)

        return results


def format_ribbon_output(ribbon_states: Dict[str, RibbonStateOutput], compact: bool = True) -> str:
    """
    Format ribbon states for display.

    Args:
        ribbon_states: Dictionary of RibbonStateOutput by timeframe
        compact: If True, use compact single-line format per TF

    Returns:
        Formatted string.
    """
    lines = ["EMA RIBBON"]

    for tf, state in sorted(ribbon_states.items()):
        if compact:
            # Compact format
            pullback_str = "YES" if state.pullback_into_ribbon else "NO"
            width_thr = state.debug.get("thresholds", {}).get("width_thr", 0)

            line = (
                f"{tf}: dir={state.ribbon_direction.value} "
                f"state={state.ribbon_state.value} "
                f"strength={state.ribbon_strength_0_100:.0f} "
                f"stack={state.stack_score:.2f} "
                f"width={state.ribbon_width_smooth:.4f} (thr={width_thr:.4f}) "
                f"rate={state.width_rate:+.2f} "
                f"slope={state.ribbon_center_slope:+.5f} "
                f"pullback={pullback_str}"
            )
            lines.append(line)
        else:
            # Verbose format
            lines.append(f"\n{tf}:")
            lines.append(f"  Direction: {state.ribbon_direction.value}")
            lines.append(f"  State: {state.ribbon_state.value}")
            lines.append(f"  Strength: {state.ribbon_strength_0_100:.1f}")
            lines.append(f"  Stack Score: {state.stack_score:.2f}")
            lines.append(
                f"  Width: {state.ribbon_width_smooth:.4f} (rate: {state.width_rate:+.2%})"
            )
            lines.append(f"  Center Slope: {state.ribbon_center_slope:+.4%}")
            lines.append(f"  Pullback: {state.pullback_into_ribbon}")

    # Add notes if any timeframe shows concerning signals
    notes = []
    for tf, state in ribbon_states.items():
        if state.ribbon_state == RibbonState.EXHAUSTING:
            notes.append(f"{tf} exhausting -> high risk")
        elif state.ribbon_state == RibbonState.WEAKENING and state.width_rate < -0.10:
            notes.append(f"{tf} width compressing -> exhaustion risk")

    if notes:
        lines.append("Notes: " + ", ".join(notes))

    return "\n".join(lines)
