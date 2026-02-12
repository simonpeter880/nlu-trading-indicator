"""
Trend Strength Composite Module - Lag-Free Trend Measurement

Real-time trend strength calculation combining:
1. EMA slope magnitude (normalized by ATR%)
2. EMA ribbon expansion/width momentum
3. Relative Volume (RV)
4. Open Interest (OI) expansion rate

Features:
- Incremental O(1) updates per candle
- ATR% normalization for cross-instrument comparability
- Component weighting with missing-value handling
- Light EMA smoothing to reduce noise without lag
- Bucketing: WEAK / EMERGING / STRONG.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional


class Bucket(Enum):
    """Trend strength bucket classification."""

    WEAK = "WEAK"
    EMERGING = "EMERGING"
    STRONG = "STRONG"


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
class TrendStrengthConfig:
    """Configuration for Trend Strength Engine."""

    # Timeframes
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "1h"])

    # Smoothing and lookback
    smooth_period: int = 5
    atr_period: int = 14
    rv_period: int = 20

    # EMA slope normalization
    ema_slope_k_by_tf: Dict[str, int] = field(default_factory=lambda: {"1m": 15, "5m": 10, "1h": 4})
    ema_slope_k_default: int = 10
    ema_slope_strong_factor: float = 0.20  # Slope strong if ~0.20 × ATR% over k bars

    # Ribbon width rate normalization
    ribbon_wr_low: float = -0.10
    ribbon_wr_high: float = 0.20

    # Relative volume normalization
    rv_low: float = 0.8
    rv_high: float = 2.0

    # OI expansion normalization
    oi_ref_by_tf: Dict[str, float] = field(
        default_factory=lambda: {"1m": 0.003, "5m": 0.006, "1h": 0.020}
    )
    oi_ref_default: float = 0.006

    # Component weights
    w_ema_slope: float = 0.35
    w_ribbon: float = 0.25
    w_rv: float = 0.20
    w_oi: float = 0.20

    # Safety caps (optional hooks)
    cap_when_structure_range: int = 50
    cap_when_supertrend_chop: int = 50
    cap_when_rv_dead: int = 25

    # Bucket thresholds
    bucket_weak_max: float = 30.0
    bucket_emerging_max: float = 60.0

    def __post_init__(self):
        """Validate weights."""
        total_weight = self.w_ema_slope + self.w_ribbon + self.w_rv + self.w_oi
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")


@dataclass
class TrendStrengthState:
    """State output for Trend Strength per timeframe."""

    # Strength scores
    strength_raw: float
    strength_smooth: float
    strength_signed: float  # -100..+100 (direction_bias * strength_smooth)
    direction_bias: int  # -1 (bear), 0 (neutral), +1 (bull)
    bucket: Bucket

    # Normalized components (0..1)
    components_norm: Dict[str, float]

    # Raw components
    components_raw: Dict[str, float]

    # Weights used
    weights: Dict[str, float]

    # Debug information
    debug: Dict = field(default_factory=dict)


def parse_bias(bias_str: Optional[str]) -> int:
    """
    Parse bias string to integer direction.

    Args:
        bias_str: "BULL", "BEAR", or other

    Returns:
        +1 for BULL, -1 for BEAR, 0 otherwise.
    """
    if bias_str is None:
        return 0
    bias_upper = str(bias_str).upper()
    if bias_upper == "BULL":
        return 1
    elif bias_upper == "BEAR":
        return -1
    else:
        return 0


class _TimeframeState:
    """Internal state for a single timeframe."""

    def __init__(self, config: TrendStrengthConfig, tf: str):
        self.config = config
        self.tf = tf

        # ATR tracking (Wilder smoothing)
        self.atr: Optional[float] = None
        self.prev_close: Optional[float] = None
        self.tr_values: List[float] = []

        # RV tracking (volume SMA)
        self.vol_deque: Deque[float] = deque(maxlen=config.rv_period)
        self.vol_sum: float = 0.0

        # Strength smoothing
        self.strength_smooth: Optional[float] = None
        self.smooth_alpha: float = 2.0 / (config.smooth_period + 1)

        # State
        self.candle_count: int = 0
        self.is_ready: bool = False

    def compute_true_range(self, candle: Candle) -> float:
        """Compute True Range."""
        if self.prev_close is None:
            return candle.high - candle.low

        tr1 = candle.high - candle.low
        tr2 = abs(candle.high - self.prev_close)
        tr3 = abs(candle.low - self.prev_close)

        return max(tr1, tr2, tr3)

    def update_atr(self, tr: float) -> None:
        """Update ATR using Wilder smoothing."""
        atr_period = self.config.atr_period

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

    def update_rv_tracking(self, volume: float) -> None:
        """Update volume tracking for RV calculation."""
        # Remove old value from sum if deque is full
        if len(self.vol_deque) == self.config.rv_period:
            self.vol_sum -= self.vol_deque[0]

        # Add new value
        self.vol_deque.append(volume)
        self.vol_sum += volume


class TrendStrengthEngine:
    """
    Real-time Trend Strength Engine combining multiple momentum indicators.

    Provides lag-free trend strength measurement (0-100) by normalizing
    and weighting EMA slope, ribbon expansion, relative volume, and OI growth.
    """

    def __init__(self, config: Optional[TrendStrengthConfig] = None):
        """
        Initialize Trend Strength Engine.

        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config or TrendStrengthConfig()
        self._states: Dict[str, _TimeframeState] = {}
        self._eps = 1e-12

    def _get_or_create_state(self, tf: str) -> _TimeframeState:
        """Get or create state for a timeframe."""
        if tf not in self._states:
            self._states[tf] = _TimeframeState(self.config, tf)
        return self._states[tf]

    def _clip(self, value: float, lo: float, hi: float) -> float:
        """Clip value to range [lo, hi]."""
        return max(lo, min(hi, value))

    def _compute_atr_percent(
        self,
        state: _TimeframeState,
        close: float,
        atr_provided: Optional[float],
        atr_percent_provided: Optional[float],
    ) -> float:
        """
        Get ATR% either from provided values or internal computation.

        Returns:
            ATR as percentage of close (decimal)
        """
        if atr_percent_provided is not None:
            return atr_percent_provided

        if atr_provided is not None:
            return atr_provided / (close + self._eps)

        if state.atr is not None:
            return state.atr / (close + self._eps)

        return 0.0

    def _normalize_ema_slope(
        self,
        slope: Optional[float],
        ema50_now: Optional[float],
        ema50_k: Optional[float],
        atr_percent: float,
        tf: str,
    ) -> Optional[float]:
        """
        Normalize EMA slope magnitude.

        Args:
            slope: Direct slope if provided
            ema50_now: Current EMA50 value
            ema50_k: EMA50 value k bars ago
            atr_percent: ATR as percentage
            tf: Timeframe

        Returns:
            Normalized slope (0..1) or None if unavailable.
        """
        # Use provided slope if available
        if slope is not None:
            slope_abs = abs(slope)
        elif ema50_now is not None and ema50_k is not None:
            # Compute slope from EMA values
            slope_abs = abs((ema50_now - ema50_k) / (ema50_k + self._eps))
        else:
            return None

        # Normalize by ATR%
        denom = self.config.ema_slope_strong_factor * atr_percent + self._eps
        slope_norm = self._clip(slope_abs / denom, 0.0, 1.0)

        return slope_norm

    def _normalize_ribbon(self, ribbon_width_rate: Optional[float]) -> Optional[float]:
        """
        Normalize ribbon width rate.

        Args:
            ribbon_width_rate: Width rate from ribbon engine

        Returns:
            Normalized ribbon (0..1) or None.
        """
        if ribbon_width_rate is None:
            return None

        wr_range = self.config.ribbon_wr_high - self.config.ribbon_wr_low + self._eps
        ribbon_norm = self._clip(
            (ribbon_width_rate - self.config.ribbon_wr_low) / wr_range, 0.0, 1.0
        )

        return ribbon_norm

    def _compute_rv(self, state: _TimeframeState, volume: float) -> Optional[float]:
        """
        Compute relative volume.

        Args:
            state: Timeframe state
            volume: Current volume

        Returns:
            RV value or None if not enough history.
        """
        if len(state.vol_deque) < self.config.rv_period:
            return None

        vol_avg = state.vol_sum / len(state.vol_deque)
        rv = volume / (vol_avg + self._eps)

        return rv

    def _normalize_rv(
        self, rv_provided: Optional[float], state: _TimeframeState, volume: float
    ) -> Optional[float]:
        """
        Normalize relative volume.

        Args:
            rv_provided: RV if provided externally
            state: Timeframe state
            volume: Current volume

        Returns:
            Normalized RV (0..1) or None.
        """
        # Use provided RV or compute
        if rv_provided is not None:
            rv = rv_provided
        else:
            rv = self._compute_rv(state, volume)

        if rv is None:
            return None

        # Normalize with saturation
        rv_range = self.config.rv_high - self.config.rv_low + self._eps
        rv_norm = self._clip((rv - self.config.rv_low) / rv_range, 0.0, 1.0)

        return rv_norm

    def _normalize_oi(
        self, oi_now: Optional[float], oi_prev: Optional[float], tf: str
    ) -> Optional[float]:
        """
        Normalize OI expansion rate.

        Args:
            oi_now: Current OI
            oi_prev: Previous OI
            tf: Timeframe

        Returns:
            Normalized OI (0..1) or None.
        """
        if oi_now is None or oi_prev is None:
            return None

        # Compute absolute % change
        d_oi = abs((oi_now - oi_prev) / (oi_prev + self._eps))

        # Get reference
        oi_ref = self.config.oi_ref_by_tf.get(tf, self.config.oi_ref_default)

        # Normalize
        oi_norm = self._clip(d_oi / (oi_ref + self._eps), 0.0, 1.0)

        return oi_norm

    def _compute_weighted_strength(
        self, components_norm: Dict[str, Optional[float]]
    ) -> tuple[float, Dict[str, float]]:
        """
        Compute weighted strength from normalized components.

        Handles missing components by renormalizing weights.

        Returns:
            (strength_raw, weights_used)
        """
        # Collect available components and their weights
        available = {}
        weights_map = {
            "ema_slope": self.config.w_ema_slope,
            "ribbon": self.config.w_ribbon,
            "rv": self.config.w_rv,
            "oi": self.config.w_oi,
        }

        for key, value in components_norm.items():
            if value is not None:
                available[key] = value

        if not available:
            return 0.0, {}

        # Renormalize weights
        total_weight = sum(weights_map[k] for k in available.keys())
        weights_used = {k: weights_map[k] / total_weight for k in available.keys()}

        # Compute weighted sum
        strength_raw = 100.0 * sum(weights_used[k] * available[k] for k in available.keys())

        return strength_raw, weights_used

    def _smooth_strength(self, state: _TimeframeState, strength_raw: float) -> float:
        """
        Apply EMA smoothing to strength.

        Args:
            state: Timeframe state
            strength_raw: Raw strength value

        Returns:
            Smoothed strength.
        """
        if state.strength_smooth is None:
            state.strength_smooth = strength_raw
        else:
            state.strength_smooth = (
                state.smooth_alpha * strength_raw + (1 - state.smooth_alpha) * state.strength_smooth
            )

        return state.strength_smooth

    def _apply_safety_caps(
        self, strength: float, flags: Optional[Dict[str, bool]], rv: Optional[float]
    ) -> float:
        """
        Apply optional safety caps based on external conditions.

        Args:
            strength: Current strength value
            flags: Optional dict with condition flags
            rv: Current RV value

        Returns:
            Capped strength.
        """
        if flags is None:
            flags = {}

        # Structure range cap
        if flags.get("structure_is_range", False):
            strength = min(strength, self.config.cap_when_structure_range)

        # Supertrend chop cap
        if flags.get("supertrend_is_chop", False):
            strength = min(strength, self.config.cap_when_supertrend_chop)

        # RV dead cap
        if rv is not None and rv < 0.3:
            strength = min(strength, self.config.cap_when_rv_dead)

        return strength

    def _classify_bucket(self, strength: float) -> Bucket:
        """
        Classify strength into bucket.

        Args:
            strength: Strength value (0-100)

        Returns:
            Bucket classification.
        """
        if strength <= self.config.bucket_weak_max:
            return Bucket.WEAK
        elif strength <= self.config.bucket_emerging_max:
            return Bucket.EMERGING
        else:
            return Bucket.STRONG

    def warmup(
        self,
        candles_by_tf: Dict[str, List[Candle]],
        ema_states_by_tf: Optional[Dict] = None,
        ribbon_states_by_tf: Optional[Dict] = None,
        oi_series_by_tf: Optional[Dict] = None,
    ) -> Dict[str, TrendStrengthState]:
        """
        Warm up the engine with historical candles.

        Args:
            candles_by_tf: Dictionary of candle lists by timeframe
            ema_states_by_tf: Optional EMA states (not used in warmup)
            ribbon_states_by_tf: Optional ribbon states (not used in warmup)
            oi_series_by_tf: Optional OI series (not used in warmup)

        Returns:
            Dictionary of TrendStrengthState by timeframe.
        """
        results = {}

        for tf, candles in candles_by_tf.items():
            state = self._get_or_create_state(tf)

            for candle in candles:
                # Build up internal state (ATR, RV tracking)
                tr = state.compute_true_range(candle)
                state.update_atr(tr)
                state.update_rv_tracking(candle.volume)
                state.prev_close = candle.close
                state.candle_count += 1

        return results

    def on_candle_close(
        self,
        tf: str,
        candle: Candle,
        ema50_now: Optional[float] = None,
        ema50_k: Optional[float] = None,
        slope_50: Optional[float] = None,
        ribbon_width_rate: Optional[float] = None,
        rv: Optional[float] = None,
        oi_now: Optional[float] = None,
        oi_prev: Optional[float] = None,
        atr: Optional[float] = None,
        atr_percent: Optional[float] = None,
        bias: Optional[str] = None,
        direction_bias: Optional[int] = None,
        flags: Optional[Dict[str, bool]] = None,
    ) -> TrendStrengthState:
        """
        Process a candle close and return trend strength state.

        Args:
            tf: Timeframe string
            candle: Closed candle
            ema50_now: Current EMA50 value
            ema50_k: EMA50 value k bars ago
            slope_50: Direct slope if available
            ribbon_width_rate: Ribbon width rate
            rv: Relative volume if pre-computed
            oi_now: Current open interest
            oi_prev: Previous open interest
            atr: ATR value if available
            atr_percent: ATR as percentage if available
            bias: Optional bias string ("BULL", "BEAR", "NEUTRAL")
            direction_bias: Optional direction as int (-1, 0, +1). Takes precedence over bias.
            flags: Optional safety flags

        Returns:
            TrendStrengthState with all metrics.
        """
        state = self._get_or_create_state(tf)

        # Update internal state
        tr = state.compute_true_range(candle)
        state.update_atr(tr)
        state.update_rv_tracking(candle.volume)

        # Compute ATR%
        atr_pct = self._compute_atr_percent(state, candle.close, atr, atr_percent)

        # Normalize components
        ema_slope_norm = self._normalize_ema_slope(slope_50, ema50_now, ema50_k, atr_pct, tf)
        ribbon_norm = self._normalize_ribbon(ribbon_width_rate)
        rv_norm = self._normalize_rv(rv, state, candle.volume)
        oi_norm = self._normalize_oi(oi_now, oi_prev, tf)

        components_norm = {
            "ema_slope": ema_slope_norm,
            "ribbon": ribbon_norm,
            "rv": rv_norm,
            "oi": oi_norm,
        }

        # Compute raw strength
        strength_raw, weights_used = self._compute_weighted_strength(components_norm)

        # Smooth strength
        strength_smooth = self._smooth_strength(state, strength_raw)

        # Apply safety caps
        rv_actual = rv if rv is not None else self._compute_rv(state, candle.volume)
        strength_smooth = self._apply_safety_caps(strength_smooth, flags, rv_actual)

        # Determine direction bias
        if direction_bias is not None:
            # direction_bias parameter takes precedence
            dir_bias = int(direction_bias)
            if dir_bias not in [-1, 0, 1]:
                dir_bias = 0  # Default to neutral if invalid
        else:
            # Parse bias string
            dir_bias = parse_bias(bias)

        # Calculate signed strength
        strength_signed = dir_bias * strength_smooth

        # Classify bucket
        bucket = self._classify_bucket(strength_smooth)

        # Build raw components dict
        components_raw = {}
        if slope_50 is not None:
            components_raw["slope"] = slope_50
        elif ema50_now is not None and ema50_k is not None:
            components_raw["slope"] = abs((ema50_now - ema50_k) / (ema50_k + self._eps))

        if ribbon_width_rate is not None:
            components_raw["wr"] = ribbon_width_rate

        if rv_actual is not None:
            components_raw["RV"] = rv_actual

        if oi_now is not None and oi_prev is not None:
            components_raw["dOI"] = abs((oi_now - oi_prev) / (oi_prev + self._eps))

        # Build debug info
        debug_info = {
            "atr_percent": atr_pct,
            "is_ready": state.is_ready,
            "candle_count": state.candle_count,
            "missing_components": [k for k, v in components_norm.items() if v is None],
            "ema_slope_strong_factor": self.config.ema_slope_strong_factor,
            "ribbon_wr_range": (self.config.ribbon_wr_low, self.config.ribbon_wr_high),
            "rv_range": (self.config.rv_low, self.config.rv_high),
            "oi_ref": self.config.oi_ref_by_tf.get(tf, self.config.oi_ref_default),
        }

        result = TrendStrengthState(
            strength_raw=strength_raw,
            strength_smooth=strength_smooth,
            strength_signed=strength_signed,
            direction_bias=dir_bias,
            bucket=bucket,
            components_norm={k: v for k, v in components_norm.items() if v is not None},
            components_raw=components_raw,
            weights=weights_used,
            debug=debug_info,
        )

        # Update state for next iteration
        state.prev_close = candle.close
        state.candle_count += 1

        return result

    def get_state(self, tf: str) -> Optional[TrendStrengthState]:
        """
        Get current state for a timeframe.

        Args:
            tf: Timeframe string

        Returns:
            TrendStrengthState if available, None otherwise.
        """
        if tf not in self._states:
            return None

        state = self._states[tf]
        if not state.is_ready:
            return None

        # Note: This returns None; actual state comes from on_candle_close
        return None


def format_trend_strength_output(
    states: Dict[str, TrendStrengthState], compact: bool = True
) -> str:
    """
    Format trend strength states for display.

    Args:
        states: Dictionary of TrendStrengthState by timeframe
        compact: If True, use compact format

    Returns:
        Formatted string.
    """
    lines = ["TREND STRENGTH"]

    for tf in sorted(states.keys()):
        state = states[tf]

        if compact:
            # Compact format with directional signing
            sign_str = "+" if state.direction_bias >= 0 else ""
            line = (
                f"{tf}: {sign_str}{state.strength_signed:.0f} ({state.bucket.value}) "
                f"raw={state.strength_raw:.0f} smooth={state.strength_smooth:.0f} "
                f"bias={state.direction_bias:+d}"
            )
            lines.append(line)

            # Components normalized
            comp_norm_strs = []
            for key in ["ema_slope", "ribbon", "rv", "oi"]:
                if key in state.components_norm:
                    val = state.components_norm[key]
                    label = key if key != "ema_slope" else "slope"
                    comp_norm_strs.append(f"{label}={val:.2f}")

            if comp_norm_strs:
                lines.append(f"  comps_norm: {' '.join(comp_norm_strs)}")

            # Components raw
            comp_raw_strs = []
            for key in ["slope", "wr", "RV", "dOI"]:
                if key in state.components_raw:
                    val = state.components_raw[key]
                    if key in ["slope", "dOI"]:
                        comp_raw_strs.append(f"{key}={val:.5f}")
                    elif key == "wr":
                        comp_raw_strs.append(f"{key}={val:+.2f}")
                    else:
                        comp_raw_strs.append(f"{key}={val:.2f}")

            if comp_raw_strs:
                lines.append(f"  comps_raw : {' '.join(comp_raw_strs)}")

        else:
            # Verbose format
            lines.append(f"\n{tf}:")
            lines.append(f"  Strength: {state.strength_smooth:.1f} ({state.bucket.value})")
            lines.append(f"  Raw: {state.strength_raw:.1f}")
            lines.append("  Components (normalized):")
            for key, val in state.components_norm.items():
                lines.append(f"    {key}: {val:.2f}")

    return "\n".join(lines)
