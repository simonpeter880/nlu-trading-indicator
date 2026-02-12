"""
EMA Trend Filter - Real-Time Dynamic Trend Detection

Provides per-timeframe EMA analysis with incremental updates:
- EMA bias (BULL/BEAR/NEUTRAL)
- EMA regime (TREND/RANGE)
- EMA alignment (STACKED_UP/STACKED_DOWN/MIXED)
- Slope analysis (slope_21, slope_50)
- Ribbon width (normalized)
- Extension from EMA21
- Pullback zone detection
- Trend strength score (0-100)

ARCHITECTURE:
- Supports both batch warmup and O(1) incremental updates
- Uses ringbuffers for EMA history (slope calculation)
- Adaptive thresholds based on ATR%
- Multi-timeframe alignment detection

USAGE:
    engine = EMAFilterEngine(EMAConfig())

    # Warmup
    engine.warmup(candles_by_tf)

    # Incremental updates
    engine.on_candle_close("1m", new_candle, atr_percent=0.003)

    # Get state
    state = engine.get_state("1m")
    mtf_state = engine.get_mtf_state()
"""

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# ============================================================================
# ENUMS
# ============================================================================


class EMABias(Enum):
    """EMA directional bias."""

    BULL = "BULL"
    BEAR = "BEAR"
    NEUTRAL = "NEUTRAL"


class EMARegime(Enum):
    """Market regime based on EMA."""

    TREND = "TREND"
    RANGE = "RANGE"


class EMAAlignment(Enum):
    """EMA stacking alignment."""

    STACKED_UP = "STACKED_UP"  # 9 > 21 > 50
    STACKED_DOWN = "STACKED_DOWN"  # 9 < 21 < 50
    MIXED = "MIXED"


class MTFAlignment(Enum):
    """Multi-timeframe alignment."""

    ALIGNED = "ALIGNED"  # HTF and LTF bias match
    MIXED = "MIXED"  # HTF and LTF disagree
    RANGE_DOMINANT = "RANGE_DOMINANT"  # HTF in range


# ============================================================================
# DATA CLASSES
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
class EMAConfig:
    """Configuration for EMA filter."""

    # EMA periods
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 50])

    # Slope lookback (bars to look back per timeframe)
    slope_lookback_by_tf: Dict[str, int] = field(
        default_factory=lambda: {
            "15s": 20,
            "1m": 15,
            "5m": 10,
            "15m": 8,
            "1h": 4,
            "4h": 3,
        }
    )
    slope_lookback_default: int = 10

    # Helper periods
    rv_period: int = 20
    atr_period: int = 14

    # Adaptive threshold factors (multiplied by ATR%)
    slope_threshold_factor: float = 0.15  # Trend slope threshold
    width_threshold_factor: float = 0.10  # Ribbon width for range
    extended_factor: float = 0.60  # Extension threshold
    pullback_band_factor: float = 0.25  # Pullback zone around EMA21

    # Static fallback thresholds (when ATR% unavailable)
    slope_threshold_static_by_tf: Dict[str, float] = field(
        default_factory=lambda: {
            "15s": 0.0001,
            "1m": 0.0002,
            "5m": 0.0002,
            "15m": 0.0005,
            "1h": 0.0010,
            "4h": 0.0020,
        }
    )

    width_threshold_static_by_tf: Dict[str, float] = field(
        default_factory=lambda: {
            "15s": 0.0004,
            "1m": 0.0006,
            "5m": 0.0010,
            "15m": 0.0015,
            "1h": 0.0020,
            "4h": 0.0040,
        }
    )

    extended_static_by_tf: Dict[str, float] = field(
        default_factory=lambda: {
            "15s": 0.0010,
            "1m": 0.0015,
            "5m": 0.0030,
            "15m": 0.0050,
            "1h": 0.0080,
            "4h": 0.0150,
        }
    )

    # Weights for trend strength composite
    strength_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "slope": 0.35,
            "width": 0.25,
            "alignment": 0.25,
            "bias": 0.15,
        }
    )


@dataclass
class EMAState:
    """Complete EMA state for a timeframe."""

    # EMA values
    ema9: float
    ema21: float
    ema50: float

    # Slopes (percent change over k bars)
    slope_21: float
    slope_50: float

    # Ribbon metrics
    ribbon_width: float
    ext_21: float  # Extension from EMA21

    # Classification
    ema_bias: EMABias
    ema_regime: EMARegime
    ema_alignment: EMAAlignment

    # Flags
    pullback_zone_hit: bool
    extended: bool

    # Score
    trend_strength_0_100: float

    # Debug info
    debug: Dict = field(default_factory=dict)

    # Ready flag
    ready: bool = True


@dataclass
class EMAMultiTFState:
    """Multi-timeframe EMA state."""

    states_by_tf: Dict[str, EMAState]
    alignment_summary: MTFAlignment
    htf_bias: EMABias
    ltf_bias: EMABias


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def compute_sma(values: List[float], period: int) -> float:
    """Compute simple moving average."""
    if len(values) < period:
        return sum(values) / len(values) if values else 0.0
    return sum(values[-period:]) / period


def compute_atr(candles: List[Candle], period: int = 14) -> float:
    """Compute ATR for most recent candle."""
    if len(candles) < 2:
        return 0.0

    true_ranges = []
    for i in range(1, len(candles)):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i - 1].close

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)

    if len(true_ranges) < period:
        return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0

    # Simple average for last period
    return sum(true_ranges[-period:]) / period


def compute_atr_percent(candles: List[Candle], period: int = 14) -> float:
    """Compute ATR as percentage of price."""
    atr = compute_atr(candles, period)
    if candles:
        close = candles[-1].close
        if close > 0:
            return atr / close
    return 0.0


def compute_rv(candles: List[Candle], period: int = 20) -> float:
    """Compute Relative Volume."""
    if len(candles) < period:
        return 1.0

    volumes = [c.volume for c in candles[-period:]]
    avg_vol = sum(volumes[:-1]) / (len(volumes) - 1) if len(volumes) > 1 else volumes[-1]

    if avg_vol > 0:
        return candles[-1].volume / avg_vol
    return 1.0


# ============================================================================
# EMA FILTER ENGINE
# ============================================================================


class EMAFilterEngine:
    """
    Real-time EMA Trend Filter Engine.

    Supports:
    - Batch warmup from historical candles
    - Incremental O(1) updates on new candle close
    - Multi-timeframe analysis
    """

    def __init__(self, config: Optional[EMAConfig] = None):
        self.config = config or EMAConfig()

        # State per timeframe
        self._ema_values: Dict[str, Dict[int, float]] = {}  # tf -> {period: ema_value}
        self._ema_history: Dict[str, Dict[int, deque]] = {}  # tf -> {period: deque of ema values}
        self._states: Dict[str, EMAState] = {}
        self._ready: Dict[str, bool] = {}

        # Latest close per TF (for incremental updates)
        self._last_close: Dict[str, float] = {}

        # ATR and RV cache
        self._atr_percent: Dict[str, float] = {}
        self._rv: Dict[str, float] = {}

    def warmup(self, candles_by_tf: Dict[str, List[Candle]]) -> Dict[str, EMAState]:
        """
        Warmup from historical candles (batch initialization).

        Args:
            candles_by_tf: Dict mapping timeframe to list of candles

        Returns:
            Dict of EMAState per timeframe
        """
        states = {}

        for tf, candles in candles_by_tf.items():
            if len(candles) < max(self.config.ema_periods):
                # Not enough data
                self._ready[tf] = False
                continue

            # Initialize EMAs using SMA for first values
            self._initialize_emas(tf, candles)

            # Compute ATR% and RV
            self._atr_percent[tf] = compute_atr_percent(candles, self.config.atr_period)
            self._rv[tf] = compute_rv(candles, self.config.rv_period)

            # Compute state
            state = self._compute_state(
                tf, candles[-1], self._atr_percent.get(tf), self._rv.get(tf)
            )

            states[tf] = state
            self._states[tf] = state
            self._ready[tf] = True
            self._last_close[tf] = candles[-1].close

        return states

    def on_candle_close(
        self,
        tf: str,
        candle: Candle,
        atr_percent: Optional[float] = None,
        rv: Optional[float] = None,
    ) -> EMAState:
        """
        Incremental update on new candle close (O(1) operation).

        Args:
            tf: Timeframe
            candle: New candle
            atr_percent: Optional ATR% (if not provided, uses cached)
            rv: Optional RV (if not provided, uses cached)

        Returns:
            Updated EMAState
        """
        # Update EMAs incrementally
        self._update_emas_incremental(tf, candle.close)

        # Update cache
        if atr_percent is not None:
            self._atr_percent[tf] = atr_percent
        if rv is not None:
            self._rv[tf] = rv

        # Compute state
        state = self._compute_state(tf, candle, self._atr_percent.get(tf), self._rv.get(tf))

        self._states[tf] = state
        self._last_close[tf] = candle.close

        return state

    def update(
        self,
        candles_by_tf: Dict[str, List[Candle]],
        atr_percent_by_tf: Optional[Dict[str, float]] = None,
        rv_by_tf: Optional[Dict[str, float]] = None,
    ) -> Dict[str, EMAState]:
        """
        Convenience method for batch recompute (e.g., on restart).

        For continuous operation, prefer warmup() once + on_candle_close() per tick.
        """
        return self.warmup(candles_by_tf)

    def get_state(self, tf: str) -> Optional[EMAState]:
        """Get current EMA state for timeframe."""
        return self._states.get(tf)

    def get_mtf_state(self, htf: str = "1h", ltf: str = "1m") -> Optional[EMAMultiTFState]:
        """
        Get multi-timeframe alignment state.

        Args:
            htf: Higher timeframe key
            ltf: Lower timeframe key

        Returns:
            EMAMultiTFState or None if data missing
        """
        if htf not in self._states or ltf not in self._states:
            return None

        htf_state = self._states[htf]
        ltf_state = self._states[ltf]

        # Determine alignment
        if htf_state.ema_regime == EMARegime.RANGE:
            alignment = MTFAlignment.RANGE_DOMINANT
        elif htf_state.ema_bias == ltf_state.ema_bias and htf_state.ema_bias != EMABias.NEUTRAL:
            alignment = MTFAlignment.ALIGNED
        else:
            alignment = MTFAlignment.MIXED

        return EMAMultiTFState(
            states_by_tf={htf: htf_state, ltf: ltf_state},
            alignment_summary=alignment,
            htf_bias=htf_state.ema_bias,
            ltf_bias=ltf_state.ema_bias,
        )

    def is_ready(self, tf: str) -> bool:
        """Check if timeframe has been warmed up."""
        return self._ready.get(tf, False)

    # ========================================================================
    # INTERNAL METHODS
    # ========================================================================

    def _initialize_emas(self, tf: str, candles: List[Candle]):
        """Initialize EMAs from candle history using batch computation."""
        closes = [c.close for c in candles]

        # Initialize storage
        if tf not in self._ema_values:
            self._ema_values[tf] = {}
        if tf not in self._ema_history:
            self._ema_history[tf] = {}

        for period in self.config.ema_periods:
            # Seed with SMA
            if len(closes) >= period:
                sma = compute_sma(closes[:period], period)
            else:
                sma = compute_sma(closes, len(closes))

            # Compute EMA for all candles
            ema = sma
            alpha = 2 / (period + 1)

            for close in closes[period:]:
                ema = alpha * close + (1 - alpha) * ema

            self._ema_values[tf][period] = ema

            # Store history for slope calculation
            lookback = self._get_slope_lookback(tf)
            self._ema_history[tf][period] = deque(maxlen=lookback + 1)

            # Fill history by recomputing last N EMAs
            ema_temp = sma
            for i, close in enumerate(closes[period:]):
                ema_temp = alpha * close + (1 - alpha) * ema_temp
                # Keep last lookback+1 values
                if i >= len(closes[period:]) - lookback - 1:
                    self._ema_history[tf][period].append(ema_temp)

    def _update_emas_incremental(self, tf: str, close: float):
        """Update EMAs incrementally (O(1) operation)."""
        if tf not in self._ema_values:
            return

        for period in self.config.ema_periods:
            if period not in self._ema_values[tf]:
                continue

            alpha = 2 / (period + 1)
            old_ema = self._ema_values[tf][period]
            new_ema = alpha * close + (1 - alpha) * old_ema

            self._ema_values[tf][period] = new_ema

            # Update history
            if period in self._ema_history[tf]:
                self._ema_history[tf][period].append(new_ema)

    def _compute_state(
        self, tf: str, candle: Candle, atr_percent: Optional[float], rv: Optional[float]
    ) -> EMAState:
        """Compute complete EMA state."""

        ema_vals = self._ema_values.get(tf, {})
        ema9 = ema_vals.get(9, candle.close)
        ema21 = ema_vals.get(21, candle.close)
        ema50 = ema_vals.get(50, candle.close)

        close = candle.close

        # Slopes
        slope_21 = self._compute_slope(tf, 21)
        slope_50 = self._compute_slope(tf, 50)

        # Ribbon width
        ribbon_width = self._compute_ribbon_width(ema9, ema21, ema50, close)

        # Extension from EMA21
        ext_21 = abs(close - ema21) / (close + 1e-10)

        # Get thresholds
        thresholds = self._get_thresholds(tf, atr_percent)

        # Alignment
        alignment = self._classify_alignment(ema9, ema21, ema50)

        # Bias
        bias = self._classify_bias(close, ema50, slope_50, thresholds["slope_threshold"])

        # Regime
        regime = self._classify_regime(close, ema50, slope_50, ribbon_width, alignment, thresholds)

        # Extended flag
        extended = ext_21 > thresholds["extended_threshold"]

        # Pullback zone
        pullback_zone_hit = abs(close - ema21) <= thresholds["pullback_band"]

        # Trend strength
        trend_strength = self._compute_trend_strength(
            slope_50, ribbon_width, alignment, bias, regime, thresholds
        )

        return EMAState(
            ema9=ema9,
            ema21=ema21,
            ema50=ema50,
            slope_21=slope_21,
            slope_50=slope_50,
            ribbon_width=ribbon_width,
            ext_21=ext_21,
            ema_bias=bias,
            ema_regime=regime,
            ema_alignment=alignment,
            pullback_zone_hit=pullback_zone_hit,
            extended=extended,
            trend_strength_0_100=trend_strength,
            debug=thresholds,
            ready=True,
        )

    def _compute_slope(self, tf: str, period: int) -> float:
        """Compute EMA slope (percent change over k bars)."""
        if tf not in self._ema_history or period not in self._ema_history[tf]:
            return 0.0

        history = self._ema_history[tf][period]

        if len(history) < 2:
            return 0.0

        ema_now = history[-1]
        ema_past = history[0]

        if ema_past == 0:
            return 0.0

        return (ema_now - ema_past) / ema_past

    def _compute_ribbon_width(self, ema9: float, ema21: float, ema50: float, close: float) -> float:
        """Compute ribbon width (normalized by price)."""
        max_ema = max(ema9, ema21, ema50)
        min_ema = min(ema9, ema21, ema50)

        return (max_ema - min_ema) / (close + 1e-10)

    def _classify_alignment(self, ema9: float, ema21: float, ema50: float) -> EMAAlignment:
        """Classify EMA alignment."""
        if ema9 > ema21 > ema50:
            return EMAAlignment.STACKED_UP
        elif ema9 < ema21 < ema50:
            return EMAAlignment.STACKED_DOWN
        else:
            return EMAAlignment.MIXED

    def _classify_bias(
        self, close: float, ema50: float, slope_50: float, slope_threshold: float
    ) -> EMABias:
        """Classify EMA bias."""
        # BULL: close > ema50 AND slope_50 > +slope_threshold*0.5
        if close > ema50 and slope_50 > slope_threshold * 0.5:
            return EMABias.BULL

        # BEAR: close < ema50 AND slope_50 < -slope_threshold*0.5
        if close < ema50 and slope_50 < -slope_threshold * 0.5:
            return EMABias.BEAR

        return EMABias.NEUTRAL

    def _classify_regime(
        self,
        close: float,
        ema50: float,
        slope_50: float,
        ribbon_width: float,
        alignment: EMAAlignment,
        thresholds: Dict,
    ) -> EMARegime:
        """Classify market regime (TREND or RANGE)."""
        slope_thr = thresholds["slope_threshold"]
        width_thr = thresholds["width_threshold"]

        # RANGE conditions
        if abs(slope_50) < slope_thr * 0.5 and ribbon_width < width_thr:
            return EMARegime.RANGE

        # TREND conditions (require 2 of 3 for bullish, mirrored for bearish)
        bullish_conditions = [
            close > ema50,
            slope_50 > slope_thr,
            alignment == EMAAlignment.STACKED_UP,
        ]

        bearish_conditions = [
            close < ema50,
            slope_50 < -slope_thr,
            alignment == EMAAlignment.STACKED_DOWN,
        ]

        if sum(bullish_conditions) >= 2 or sum(bearish_conditions) >= 2:
            return EMARegime.TREND

        return EMARegime.RANGE

    def _compute_trend_strength(
        self,
        slope_50: float,
        ribbon_width: float,
        alignment: EMAAlignment,
        bias: EMABias,
        regime: EMARegime,
        thresholds: Dict,
    ) -> float:
        """Compute composite trend strength (0-100)."""

        slope_thr = thresholds["slope_threshold"]
        width_thr = thresholds["width_threshold"]

        # Components (0-1)
        slope_component = min(1.0, abs(slope_50) / (2 * slope_thr + 1e-10))
        width_component = min(1.0, ribbon_width / (2 * width_thr + 1e-10))

        alignment_component = (
            1.0 if alignment in [EMAAlignment.STACKED_UP, EMAAlignment.STACKED_DOWN] else 0.4
        )
        bias_component = 1.0 if bias != EMABias.NEUTRAL else 0.5

        # Weighted sum
        weights = self.config.strength_weights
        strength = (
            weights["slope"] * slope_component
            + weights["width"] * width_component
            + weights["alignment"] * alignment_component
            + weights["bias"] * bias_component
        ) * 100

        # Cap at 40 if RANGE
        if regime == EMARegime.RANGE:
            strength = min(strength, 40.0)

        return max(0.0, min(100.0, strength))

    def _get_thresholds(self, tf: str, atr_percent: Optional[float]) -> Dict:
        """Get adaptive thresholds (ATR-based or static fallback)."""

        if atr_percent and atr_percent > 0:
            # Adaptive thresholds
            slope_threshold = self.config.slope_threshold_factor * atr_percent
            width_threshold = self.config.width_threshold_factor * atr_percent
            extended_threshold = self.config.extended_factor * atr_percent
            pullback_band = self.config.pullback_band_factor * atr_percent
        else:
            # Static fallback
            slope_threshold = self.config.slope_threshold_static_by_tf.get(tf, 0.0002)
            width_threshold = self.config.width_threshold_static_by_tf.get(tf, 0.0010)
            extended_threshold = self.config.extended_static_by_tf.get(tf, 0.0030)
            pullback_band = max(0.0005, width_threshold * 0.5)

        return {
            "slope_threshold": slope_threshold,
            "width_threshold": width_threshold,
            "extended_threshold": extended_threshold,
            "pullback_band": pullback_band,
        }

    def _get_slope_lookback(self, tf: str) -> int:
        """Get slope lookback period for timeframe."""
        return self.config.slope_lookback_by_tf.get(tf, self.config.slope_lookback_default)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def create_candle(
    timestamp: int, o: float, h: float, l: float, c: float, v: float = 1000.0
) -> Candle:
    """Helper to create a candle."""
    return Candle(timestamp=timestamp, open=o, high=h, low=l, close=c, volume=v)


def print_ema_state(tf: str, state: EMAState, compact: bool = True):
    """Print EMA state in readable format."""
    from indicator.display.colors import Colors

    # Regime color
    regime_color = Colors.GREEN if state.ema_regime == EMARegime.TREND else Colors.YELLOW

    # Bias color
    bias_color = (
        Colors.GREEN
        if state.ema_bias == EMABias.BULL
        else Colors.RED if state.ema_bias == EMABias.BEAR else Colors.DIM
    )

    # Alignment color
    align_color = (
        Colors.GREEN
        if state.ema_alignment == EMAAlignment.STACKED_UP
        else Colors.RED if state.ema_alignment == EMAAlignment.STACKED_DOWN else Colors.YELLOW
    )

    # Strength color
    strength = state.trend_strength_0_100
    strength_color = (
        Colors.GREEN if strength >= 70 else Colors.YELLOW if strength >= 50 else Colors.RED
    )

    if compact:
        # Compact single line
        print(
            f"  {Colors.BOLD}{tf:6}{Colors.RESET} "
            f"Regime {regime_color}{state.ema_regime.value:5}{Colors.RESET} | "
            f"Bias {bias_color}{state.ema_bias.value:7}{Colors.RESET} | "
            f"Align {align_color}{state.ema_alignment.value:11}{Colors.RESET} | "
            f"Strength {strength_color}{strength:>3.0f}{Colors.RESET}"
        )

        # Details line
        slope_thr = state.debug.get("slope_threshold", 0.0)
        width_thr = state.debug.get("width_threshold", 0.0)

        extended_str = f"{Colors.YELLOW}YES{Colors.RESET}" if state.extended else "NO"
        pullback_str = f"{Colors.GREEN}YES{Colors.RESET}" if state.pullback_zone_hit else "NO"

        print(
            f"    slope50={state.slope_50:+.5f} (thr={slope_thr:.5f}) "
            f"width={state.ribbon_width:.4f} (thr={width_thr:.4f}) "
            f"ext21={state.ext_21:.4f} extended={extended_str} pullback={pullback_str}"
        )
    else:
        # Full detail
        print(f"\n{Colors.BOLD}{Colors.CYAN}EMA STATE - {tf}{Colors.RESET}")
        print(f"  EMAs: 9={state.ema9:.2f} | 21={state.ema21:.2f} | 50={state.ema50:.2f}")
        print(f"  Regime: {regime_color}{state.ema_regime.value}{Colors.RESET}")
        print(f"  Bias: {bias_color}{state.ema_bias.value}{Colors.RESET}")
        print(f"  Alignment: {align_color}{state.ema_alignment.value}{Colors.RESET}")
        print(f"  Strength: {strength_color}{strength:.0f}%{Colors.RESET}")
        print(f"  Slope21: {state.slope_21:+.5f} | Slope50: {state.slope_50:+.5f}")
        print(f"  Ribbon Width: {state.ribbon_width:.4f}")
        print(f"  Extension: {state.ext_21:.4f} ({'extended' if state.extended else 'normal'})")
        print(f"  Pullback Zone: {'HIT' if state.pullback_zone_hit else 'no'}")


def print_ema_block(engine: EMAFilterEngine, ltf: str = "1m", htf: str = "1h"):
    """Print compact EMA filter block."""
    from indicator.display.colors import Colors

    print(f"\n{Colors.BOLD}{Colors.CYAN}EMA FILTER{Colors.RESET}")

    # LTF
    ltf_state = engine.get_state(ltf)
    if ltf_state:
        print_ema_state(f"LTF({ltf})", ltf_state, compact=True)

    # HTF
    htf_state = engine.get_state(htf)
    if htf_state:
        print_ema_state(f"HTF({htf})", htf_state, compact=True)

    # Multi-TF
    mtf = engine.get_mtf_state(htf, ltf)
    if mtf:
        align_color = (
            Colors.GREEN
            if mtf.alignment_summary == MTFAlignment.ALIGNED
            else Colors.YELLOW if mtf.alignment_summary == MTFAlignment.MIXED else Colors.DIM
        )

        print(
            f"  {Colors.BOLD}Multi-TF:{Colors.RESET} {align_color}{mtf.alignment_summary.value}{Colors.RESET}"
        )
