"""
ATR Expansion Module - Volatility Regime & Timing Detection

Computes True Range, ATR (Wilder smoothing), and expansion metrics to detect
when volatility is "waking up" - used as a TIMING gate, not an entry signal.

Key Metrics:
- TR (True Range)
- ATR (Average True Range, Wilder smoothed)
- ATR% (ATR as % of price)
- ATR Expansion Ratio (ATR / SMA(ATR))
- TR Spike Ratio (TR / SMA(TR))
- Volatility State: SQUEEZE / NORMAL / EXPANSION / EXTREME / FADE_RISK

All computations are O(1) per candle using rolling windows.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ============================================================================
# CONSTANTS
# ============================================================================

EPS = 1e-12  # Epsilon for safe division


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class Candle:
    """OHLCV candle data."""

    timestamp: int  # Unix timestamp in ms
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0  # Optional


@dataclass
class ATRExpansionConfig:
    """Configuration for ATR expansion analysis."""

    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "1h"])
    atr_period: int = 14  # Wilder ATR period
    sma_period: int = 20  # SMA period for TR and ATR
    use_external_atr_percent: bool = False  # Use externally provided ATR%

    # State thresholds
    squeeze_thr: float = 0.80  # ATR_exp < 0.80 => SQUEEZE
    expansion_thr: float = 1.20  # ATR_exp > 1.20 => EXPANSION
    extreme_thr: float = 1.60  # ATR_exp > 1.60 => EXTREME
    tr_spike_thr: float = 1.50  # TR_spike > 1.50 => shock
    fade_slope_thr: float = -0.05  # Negative slope => FADE_RISK


@dataclass
class ATRExpansionState:
    """ATR expansion state for a single timeframe."""

    tr: float  # Current True Range
    atr: Optional[float] = None  # Current ATR (None until seeded)
    atr_percent: Optional[float] = None  # ATR as % of price
    sma_atr: Optional[float] = None  # SMA of ATR
    atr_exp: Optional[float] = None  # ATR expansion ratio
    sma_tr: Optional[float] = None  # SMA of TR
    tr_spike: Optional[float] = None  # TR spike ratio
    atr_exp_slope: Optional[float] = None  # Change in ATR expansion
    vol_state: str = "WARMUP"  # Volatility state
    vol_score_0_100: Optional[float] = None  # Volatility score (0-100)
    debug: Dict = field(default_factory=dict)  # Debug info


# ============================================================================
# INTERNAL STATE (PER TIMEFRAME)
# ============================================================================


class _TimeframeState:
    """Internal state for a single timeframe (O(1) updates)."""

    def __init__(self, config: ATRExpansionConfig):
        self.config = config

        # Previous close for TR calculation
        self.prev_close: Optional[float] = None

        # ATR seed accumulation (until we have atr_period samples)
        self.tr_seed_deque: deque = deque(maxlen=config.atr_period)
        self.atr: Optional[float] = None

        # Rolling windows for SMA calculations
        self.tr_deque: deque = deque(maxlen=config.sma_period)
        self.tr_sum: float = 0.0

        self.atr_deque: deque = deque(maxlen=config.sma_period)
        self.atr_sum: float = 0.0

        # Previous ATR expansion for slope calculation
        self.prev_atr_exp: Optional[float] = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _clip(x: float, lo: float, hi: float) -> float:
    """Clip value to [lo, hi]."""
    return max(lo, min(hi, x))


def _true_range(high: float, low: float, prev_close: Optional[float]) -> float:
    """
    Calculate True Range.

    TR = max(high-low, abs(high-prev_close), abs(low-prev_close))

    If prev_close is None (first candle), TR = high - low.
    """
    if prev_close is None:
        return high - low

    hl = high - low
    hc = abs(high - prev_close)
    lc = abs(low - prev_close)

    return max(hl, hc, lc)


def _update_rolling_sum(deque_obj: deque, new_value: float, rolling_sum: float) -> float:
    """
    Update rolling sum when adding to a fixed-size deque.

    Returns new rolling sum.
    """
    # If deque is at max capacity, subtract the value that will be evicted
    if len(deque_obj) == deque_obj.maxlen:
        rolling_sum -= deque_obj[0]

    # Add new value
    rolling_sum += new_value

    return rolling_sum


def _classify_vol_state(
    atr_exp: float, atr_exp_slope: Optional[float], config: ATRExpansionConfig
) -> str:
    """
    Classify volatility state based on ATR expansion.

    Returns: SQUEEZE / NORMAL / EXPANSION / EXTREME / FADE_RISK
    """
    # Check for FADE_RISK first (high expansion but falling)
    if (
        atr_exp >= config.expansion_thr
        and atr_exp_slope is not None
        and atr_exp_slope <= config.fade_slope_thr
    ):
        return "FADE_RISK"

    # Normal state classification
    if atr_exp < config.squeeze_thr:
        return "SQUEEZE"
    elif atr_exp >= config.extreme_thr:
        return "EXTREME"
    elif atr_exp >= config.expansion_thr:
        return "EXPANSION"
    else:
        return "NORMAL"


def _calculate_vol_score(
    atr_exp: float, vol_state: str, tr_spike: Optional[float], config: ATRExpansionConfig
) -> float:
    """
    Calculate volatility score (0-100) based on ATR expansion.

    Higher score = more volatility = better timing for moves.
    """
    # Base score from ATR expansion (0 to 1 range)
    base = _clip(
        (atr_exp - config.squeeze_thr) / (config.extreme_thr - config.squeeze_thr + EPS), 0.0, 1.0
    )
    score = 100 * base

    # Apply state-based caps
    if vol_state == "SQUEEZE":
        score = min(score, 30)
    elif vol_state == "NORMAL":
        score = min(score, 60)
    elif vol_state == "EXPANSION":
        score = min(score, 85)
    # EXTREME allows up to 100

    # Shock bonus
    shock_now = tr_spike is not None and tr_spike >= config.tr_spike_thr
    if shock_now:
        score = min(score + 5, 100)

    # FADE_RISK penalty
    if vol_state == "FADE_RISK":
        score = max(score - 10, 0)

    return score


# ============================================================================
# ATR EXPANSION ENGINE
# ============================================================================


class ATRExpansionEngine:
    """
    ATR Expansion Engine for volatility regime detection.

    Tracks multiple timeframes, computes TR/ATR incrementally (O(1) per candle),
    and classifies volatility states for timing gates.
    """

    def __init__(self, config: Optional[ATRExpansionConfig] = None):
        """Initialize engine with configuration."""
        self.config = config or ATRExpansionConfig()

        # Internal state per timeframe
        self._states: Dict[str, _TimeframeState] = {}
        for tf in self.config.timeframes:
            self._states[tf] = _TimeframeState(self.config)

    def warmup(self, candles_by_tf: Dict[str, List[Candle]]) -> Dict[str, ATRExpansionState]:
        """
        Warmup engine with historical candles.

        Args:
            candles_by_tf: Dict mapping timeframe -> list of candles (chronological)

        Returns:
            Dict mapping timeframe -> ATRExpansionState
        """
        results = {}

        for tf, candles in candles_by_tf.items():
            if tf not in self._states:
                self._states[tf] = _TimeframeState(self.config)

            # Process candles sequentially
            for candle in candles:
                state = self.on_candle_close(tf, candle)

            # Return final state
            results[tf] = state if candles else self._get_warmup_state()

        return results

    def on_candle_close(
        self, tf: str, candle: Candle, external_atr_percent: Optional[float] = None
    ) -> ATRExpansionState:
        """
        Process a candle close for a given timeframe.

        Args:
            tf: Timeframe (e.g., "1m", "5m", "1h")
            candle: Candle data
            external_atr_percent: Optional externally computed ATR%

        Returns:
            Updated ATRExpansionState
        """
        # Ensure timeframe state exists
        if tf not in self._states:
            self._states[tf] = _TimeframeState(self.config)

        state = self._states[tf]

        # 1. Calculate True Range
        tr = _true_range(candle.high, candle.low, state.prev_close)

        # 2. Update ATR (Wilder smoothing)
        if state.atr is None:
            # Seed ATR with SMA of first N TRs
            state.tr_seed_deque.append(tr)
            if len(state.tr_seed_deque) == self.config.atr_period:
                state.atr = sum(state.tr_seed_deque) / len(state.tr_seed_deque)
        else:
            # Wilder smoothing: ATR = (ATR_prev * (n-1) + TR) / n
            state.atr = (state.atr * (self.config.atr_period - 1) + tr) / self.config.atr_period

        # 3. Calculate ATR%
        atr_percent = None
        if state.atr is not None:
            if self.config.use_external_atr_percent and external_atr_percent is not None:
                atr_percent = external_atr_percent
            else:
                atr_percent = state.atr / (candle.close + EPS)

        # 4. Update rolling windows for SMA calculations
        state.tr_sum = _update_rolling_sum(state.tr_deque, tr, state.tr_sum)
        state.tr_deque.append(tr)

        if state.atr is not None:
            state.atr_sum = _update_rolling_sum(state.atr_deque, state.atr, state.atr_sum)
            state.atr_deque.append(state.atr)

        # 5. Calculate SMAs (only when enough samples)
        sma_tr = None
        if len(state.tr_deque) == self.config.sma_period:
            sma_tr = state.tr_sum / self.config.sma_period

        sma_atr = None
        if len(state.atr_deque) == self.config.sma_period:
            sma_atr = state.atr_sum / self.config.sma_period

        # 6. Calculate ratios
        atr_exp = None
        tr_spike = None

        if state.atr is not None and sma_atr is not None:
            atr_exp = state.atr / (sma_atr + EPS)

        if sma_tr is not None:
            tr_spike = tr / (sma_tr + EPS)

        # 7. Calculate ATR expansion slope
        atr_exp_slope = None
        if atr_exp is not None and state.prev_atr_exp is not None:
            atr_exp_slope = atr_exp - state.prev_atr_exp

        # 8. Classify volatility state
        vol_state = "WARMUP"
        vol_score = None

        if atr_exp is not None:
            vol_state = _classify_vol_state(atr_exp, atr_exp_slope, self.config)
            vol_score = _calculate_vol_score(atr_exp, vol_state, tr_spike, self.config)

        # 9. Build result
        shock_now = tr_spike is not None and tr_spike >= self.config.tr_spike_thr

        result = ATRExpansionState(
            tr=tr,
            atr=state.atr,
            atr_percent=atr_percent,
            sma_atr=sma_atr,
            atr_exp=atr_exp,
            sma_tr=sma_tr,
            tr_spike=tr_spike,
            atr_exp_slope=atr_exp_slope,
            vol_state=vol_state,
            vol_score_0_100=vol_score,
            debug={
                "shock_now": shock_now,
                "tr_samples": len(state.tr_deque),
                "atr_samples": len(state.atr_deque),
                "prev_close": state.prev_close,
            },
        )

        # 10. Update state for next iteration
        state.prev_close = candle.close
        state.prev_atr_exp = atr_exp

        return result

    def update(self, candles_by_tf: Dict[str, List[Candle]]) -> Dict[str, ATRExpansionState]:
        """
        Update engine with new candles (batch processing).

        Args:
            candles_by_tf: Dict mapping timeframe -> list of new candles

        Returns:
            Dict mapping timeframe -> latest ATRExpansionState
        """
        results = {}

        for tf, candles in candles_by_tf.items():
            for candle in candles:
                state = self.on_candle_close(tf, candle)
            results[tf] = state

        return results

    def get_state(self, tf: str) -> Optional[ATRExpansionState]:
        """
        Get current state for a timeframe.

        Returns None if no candles have been processed yet.
        """
        if tf not in self._states:
            return None

        state = self._states[tf]

        # Return warmup state if not ready
        if state.atr is None:
            return self._get_warmup_state()

        # Build state from current internal state
        atr_exp = None
        if state.atr is not None and len(state.atr_deque) == self.config.sma_period:
            sma_atr = state.atr_sum / self.config.sma_period
            atr_exp = state.atr / (sma_atr + EPS)

        sma_tr = None
        tr_spike = None
        if len(state.tr_deque) == self.config.sma_period:
            sma_tr = state.tr_sum / self.config.sma_period
            if len(state.tr_deque) > 0:
                tr_spike = state.tr_deque[-1] / (sma_tr + EPS)

        vol_state = "WARMUP"
        vol_score = None
        if atr_exp is not None:
            vol_state = _classify_vol_state(atr_exp, state.prev_atr_exp, self.config)
            vol_score = _calculate_vol_score(atr_exp, vol_state, tr_spike, self.config)

        return ATRExpansionState(
            tr=state.tr_deque[-1] if state.tr_deque else 0.0,
            atr=state.atr,
            atr_percent=(
                state.atr / (state.prev_close + EPS) if state.atr and state.prev_close else None
            ),
            sma_atr=sma_tr,
            atr_exp=atr_exp,
            sma_tr=sma_tr,
            tr_spike=tr_spike,
            atr_exp_slope=None,
            vol_state=vol_state,
            vol_score_0_100=vol_score,
            debug={},
        )

    def _get_warmup_state(self) -> ATRExpansionState:
        """Return a warmup state."""
        return ATRExpansionState(tr=0.0, vol_state="WARMUP")


# ============================================================================
# DISPLAY HELPERS
# ============================================================================


def format_atr_state(tf: str, state: ATRExpansionState) -> str:
    """
    Format ATR expansion state for compact display.

    Example:
    "1m: state=EXPANSION score=72 atrp=0.22% atr_exp=1.31 slope=+0.07 TR_spike=1.62 shock=YES"
    """
    parts = [f"{tf}:"]
    parts.append(f"state={state.vol_state}")

    if state.vol_score_0_100 is not None:
        parts.append(f"score={state.vol_score_0_100:.0f}")

    if state.atr_percent is not None:
        parts.append(f"atrp={state.atr_percent*100:.2f}%")

    if state.atr_exp is not None:
        parts.append(f"atr_exp={state.atr_exp:.2f}")

    if state.atr_exp_slope is not None:
        sign = "+" if state.atr_exp_slope >= 0 else ""
        parts.append(f"slope={sign}{state.atr_exp_slope:.2f}")

    if state.tr_spike is not None:
        parts.append(f"TR_spike={state.tr_spike:.2f}")

    shock = "YES" if state.debug.get("shock_now") else "NO"
    parts.append(f"shock={shock}")

    return " ".join(parts)


def print_atr_expansion(states: Dict[str, ATRExpansionState]) -> None:
    """
    Print compact ATR expansion summary for multiple timeframes.

    Example:
    ATR EXPANSION
    1m: state=EXPANSION score=72 atrp=0.22% atr_exp=1.31 slope=+0.07 TR_spike=1.62 shock=YES
    5m: state=SQUEEZE  score=18 atr_exp=0.74 slope=-0.02 TR_spike=0.88 shock=NO
    1h: state=FADE_RISK score=63 atr_exp=1.25 slope=-0.08
    """
    print("\nATR EXPANSION")
    for tf, state in states.items():
        print(format_atr_state(tf, state))
