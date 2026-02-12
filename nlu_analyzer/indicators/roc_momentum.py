"""
ROC Momentum & Timing Module

Computes multi-horizon Rate of Change (ROC), acceleration, normalized momentum,
and low-lag timing signals with O(1) incremental updates per candle.

Features:
- Multi-lookback ROC calculation (fast/mid/slow)
- Optional log returns
- Acceleration (ROC derivative)
- ATR-normalized ROC for volatility adjustment
- Momentum state machine: IMPULSE/PULLBACK/FADE/NOISE
- Divergence detection hooks
- O(1) incremental updates

Author: Generated for NLU trading system
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from collections import deque
import math


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Candle:
    """OHLCV candle."""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class ROCConfig:
    """Configuration for ROC Momentum Engine."""

    # Timeframes to track
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "1h"])

    # ROC lookbacks by timeframe (fast, mid, slow)
    # Format: {"tf": [fast, mid, slow]}
    roc_lookbacks_by_tf: Dict[str, List[int]] = field(default_factory=lambda: {
        "1m": [5, 20, 60],
        "5m": [3, 12, 36],
        "1h": [3, 6, 12],
    })

    # Fallback lookbacks for unlisted timeframes
    fallback_lookbacks: List[int] = field(default_factory=lambda: [5, 20, 60])

    # Use log returns instead of simple ROC
    use_log_returns: bool = False

    # ATR configuration (if atr_percent not provided externally)
    atr_period: int = 14

    # Acceleration smoothing (EMA period for ROC before computing ACC)
    # Set to 1 for no smoothing
    accel_smooth_period: int = 3

    # Normalization parameters
    norm_atrp_factor: float = 1.0  # ROC_norm = ROC / (factor * atrp)
    clip_norm: float = 3.0         # Clip normalized ROC to [-clip, +clip]

    # State thresholds (applied to ROC_norm_fast)
    noise_norm_threshold: float = 0.3     # Below this -> NOISE
    impulse_norm_threshold: float = 0.8   # Above this -> IMPULSE candidate
    blowoff_norm_threshold: float = 1.5   # Warning threshold

    # Epsilon for safe division
    eps: float = 1e-10


@dataclass
class ROCState:
    """ROC momentum state for a single timeframe."""

    # Core ROC values: lookback -> value (decimal, not percent)
    roc: Dict[int, float] = field(default_factory=dict)

    # Optional log returns: lookback -> value
    logret: Dict[int, float] = field(default_factory=dict)

    # Acceleration: lookback -> ROC(t) - ROC(t-1)
    acc: Dict[int, float] = field(default_factory=dict)

    # Normalized & clipped ROC: lookback -> value
    roc_norm: Dict[int, float] = field(default_factory=dict)

    # Momentum state classification
    momentum_state: str = "NOISE"  # IMPULSE/PULLBACK/FADE/NOISE

    # Momentum score (0-100)
    momentum_score_0_100: float = 0.0

    # Debug info and flags
    debug: Dict[str, any] = field(default_factory=dict)

    # Latest values for quick access
    latest_close: float = 0.0
    latest_atrp: float = 0.0
    timestamp: float = 0.0


# =============================================================================
# INTERNAL STATE TRACKER
# =============================================================================

class _TimeframeState:
    """Internal state for a single timeframe - O(1) updates."""

    def __init__(self, lookbacks: List[int], atr_period: int, accel_smooth_period: int, eps: float):
        self.lookbacks = sorted(lookbacks)
        self.max_lookback = max(lookbacks)
        self.atr_period = atr_period
        self.accel_smooth_period = accel_smooth_period
        self.eps = eps

        # Price history (need max_lookback + 1 for ROC computation)
        self.closes = deque(maxlen=self.max_lookback + 2)

        # Previous ROC values for acceleration
        self.prev_roc: Dict[int, float] = {lb: 0.0 for lb in lookbacks}

        # ROC smoothing state (if accel_smooth_period > 1)
        self.roc_ema: Dict[int, Optional[float]] = {lb: None for lb in lookbacks}
        self.ema_alpha = 2.0 / (accel_smooth_period + 1) if accel_smooth_period > 1 else 1.0

        # ATR Wilder state
        self.atr: Optional[float] = None
        self.prev_close: Optional[float] = None
        self.tr_history = deque(maxlen=atr_period)  # For warmup SMA

        # Swing tracking for divergence
        self.last_swing_high: Optional[Tuple[float, float]] = None  # (price, roc_norm)
        self.last_swing_low: Optional[Tuple[float, float]] = None

        # Current state
        self.current_state: Optional[ROCState] = None

    def is_warmed_up(self) -> bool:
        """Check if we have enough data."""
        return len(self.closes) >= self.max_lookback + 1

    def update_atr(self, candle: Candle) -> float:
        """Update ATR using Wilder's method. Returns atrp (ATR/close)."""
        # Calculate True Range
        if self.prev_close is not None:
            tr = max(
                candle.high - candle.low,
                abs(candle.high - self.prev_close),
                abs(candle.low - self.prev_close)
            )
        else:
            tr = candle.high - candle.low

        self.prev_close = candle.close

        # Warmup phase: collect TRs for SMA
        if self.atr is None:
            self.tr_history.append(tr)
            if len(self.tr_history) >= self.atr_period:
                self.atr = sum(self.tr_history) / len(self.tr_history)
            else:
                # Not warmed up, return rough estimate
                return tr / (candle.close + self.eps)
        else:
            # Wilder's smoothing: ATR = (ATR_prev * (n-1) + TR) / n
            self.atr = (self.atr * (self.atr_period - 1) + tr) / self.atr_period

        # Return ATR%
        return self.atr / (candle.close + self.eps)


# =============================================================================
# ROC MOMENTUM ENGINE
# =============================================================================

class ROCMomentumEngine:
    """
    Multi-timeframe ROC momentum engine with O(1) incremental updates.

    Computes:
    - Multi-horizon ROC (fast/mid/slow)
    - Acceleration (ROC derivative)
    - ATR-normalized ROC
    - Momentum state: IMPULSE/PULLBACK/FADE/NOISE
    - Divergence detection

    Usage:
        config = ROCConfig()
        engine = ROCMomentumEngine(config)

        # Warmup (optional but recommended)
        engine.warmup({"1m": candles_1m, "5m": candles_5m})

        # Process new candles
        state = engine.on_candle_close("1m", candle, atr_percent=0.015)

        print(f"State: {state.momentum_state}")
        print(f"Score: {state.momentum_score_0_100:.0f}")
    """

    def __init__(self, config: ROCConfig):
        self.config = config
        self.states: Dict[str, _TimeframeState] = {}

        # Initialize state for each timeframe
        for tf in config.timeframes:
            lookbacks = config.roc_lookbacks_by_tf.get(tf, config.fallback_lookbacks)
            self.states[tf] = _TimeframeState(
                lookbacks=lookbacks,
                atr_period=config.atr_period,
                accel_smooth_period=config.accel_smooth_period,
                eps=config.eps
            )

    def warmup(
        self,
        candles_by_tf: Dict[str, List[Candle]],
        atr_percent_by_tf: Optional[Dict[str, List[float]]] = None
    ) -> None:
        """
        Warmup engine with historical candles.

        Args:
            candles_by_tf: {"tf": [candles]} sorted oldest to newest
            atr_percent_by_tf: Optional {"tf": [atr_percent]} parallel to candles
        """
        for tf, candles in candles_by_tf.items():
            if tf not in self.states:
                continue

            atr_list = atr_percent_by_tf.get(tf) if atr_percent_by_tf else None

            for i, candle in enumerate(candles):
                atr_pct = atr_list[i] if atr_list and i < len(atr_list) else None
                self.on_candle_close(tf, candle, atr_percent=atr_pct)

    def on_candle_close(
        self,
        tf: str,
        candle: Candle,
        atr_percent: Optional[float] = None,
        bias: Optional[int] = None
    ) -> Optional[ROCState]:
        """
        Process new candle close. O(1) update.

        Args:
            tf: Timeframe identifier
            candle: New candle
            atr_percent: Optional ATR% (if None, compute internally)
            bias: Optional directional bias (+1 bull, -1 bear) for PULLBACK detection

        Returns:
            ROCState if warmed up, else None
        """
        if tf not in self.states:
            return None

        state = self.states[tf]
        eps = self.config.eps

        # Add close to history
        state.closes.append(candle.close)

        # Get or compute ATR%
        if atr_percent is None:
            atr_percent = state.update_atr(candle)
        else:
            # Still update internal ATR for consistency
            state.update_atr(candle)

        # Check warmup
        if not state.is_warmed_up():
            return None

        # Compute ROC for all lookbacks
        roc_dict: Dict[int, float] = {}
        logret_dict: Dict[int, float] = {}
        roc_smooth_dict: Dict[int, float] = {}

        current_close = candle.close

        for lb in state.lookbacks:
            # Get historical close (lb candles ago)
            # closes[-1] is current, closes[-lb-1] is lb candles ago
            idx = -(lb + 1)
            if abs(idx) > len(state.closes):
                continue

            hist_close = state.closes[idx]

            # Simple ROC (decimal)
            roc = (current_close - hist_close) / (hist_close + eps)
            roc_dict[lb] = roc

            # Log return (optional)
            if self.config.use_log_returns:
                logret = math.log((current_close + eps) / (hist_close + eps))
                logret_dict[lb] = logret

            # ROC smoothing for acceleration
            if self.config.accel_smooth_period > 1:
                if state.roc_ema[lb] is None:
                    state.roc_ema[lb] = roc
                else:
                    state.roc_ema[lb] = (
                        state.ema_alpha * roc +
                        (1 - state.ema_alpha) * state.roc_ema[lb]
                    )
                roc_smooth_dict[lb] = state.roc_ema[lb]
            else:
                roc_smooth_dict[lb] = roc

        # Compute acceleration (ROC derivative)
        acc_dict: Dict[int, float] = {}
        for lb in state.lookbacks:
            if lb not in roc_smooth_dict:
                continue

            roc_current = roc_smooth_dict[lb]
            roc_prev = state.prev_roc[lb]

            acc_dict[lb] = roc_current - roc_prev

            # Update prev ROC
            state.prev_roc[lb] = roc_current

        # Normalize ROC with ATR%
        roc_norm_dict: Dict[int, float] = {}
        for lb, roc_val in roc_dict.items():
            norm_factor = self.config.norm_atrp_factor * atr_percent + eps
            roc_norm = roc_val / norm_factor

            # Clip
            roc_norm = max(-self.config.clip_norm, min(self.config.clip_norm, roc_norm))
            roc_norm_dict[lb] = roc_norm

        # Determine momentum state
        lookbacks = state.lookbacks
        fast_lb = lookbacks[0]
        mid_lb = lookbacks[1] if len(lookbacks) > 1 else lookbacks[0]
        slow_lb = lookbacks[2] if len(lookbacks) > 2 else mid_lb

        rf = roc_norm_dict.get(fast_lb, 0.0)
        rm = roc_norm_dict.get(mid_lb, 0.0)
        rs = roc_norm_dict.get(slow_lb, 0.0)
        af = acc_dict.get(fast_lb, 0.0)

        momentum_state, debug = self._classify_momentum_state(
            rf, rm, rs, af, bias, roc_dict.get(fast_lb, 0.0)
        )

        # Compute momentum score
        momentum_score = self._compute_momentum_score(rf, rm, af, roc_dict.get(fast_lb, 0.0), momentum_state)

        # Check divergence
        self._check_divergence(state, candle.close, rf, debug)

        # Build ROCState
        roc_state = ROCState(
            roc=roc_dict,
            logret=logret_dict if self.config.use_log_returns else {},
            acc=acc_dict,
            roc_norm=roc_norm_dict,
            momentum_state=momentum_state,
            momentum_score_0_100=momentum_score,
            debug=debug,
            latest_close=current_close,
            latest_atrp=atr_percent,
            timestamp=candle.timestamp
        )

        state.current_state = roc_state
        return roc_state

    def _classify_momentum_state(
        self,
        rf: float,  # ROC_norm_fast
        rm: float,  # ROC_norm_mid
        rs: float,  # ROC_norm_slow
        af: float,  # ACC_fast
        bias: Optional[int],
        roc_fast_raw: float
    ) -> Tuple[str, Dict]:
        """Classify momentum state with direction-aware logic."""
        debug = {}

        # Thresholds
        noise_thresh = self.config.noise_norm_threshold
        impulse_thresh = self.config.impulse_norm_threshold
        blowoff_thresh = self.config.blowoff_norm_threshold

        # Blowoff warning
        if abs(rf) > blowoff_thresh:
            debug["blowoff"] = "YES"
        else:
            debug["blowoff"] = "NO"

        # 1) NOISE: low momentum on both fast and mid
        if abs(rf) < noise_thresh and abs(rm) < noise_thresh:
            return "NOISE", debug

        # 2) IMPULSE: strong directional move with acceleration
        # Bull impulse
        if rf > impulse_thresh and rm > 0 and af > 0:
            debug["direction"] = "BULL"
            return "IMPULSE", debug

        # Bear impulse
        if rf < -impulse_thresh and rm < 0 and af < 0:
            debug["direction"] = "BEAR"
            return "IMPULSE", debug

        # 3) PULLBACK: counter-trend retracement
        # Infer bias from mid if not provided
        if bias is None:
            bias = 1 if rm > 0 else -1 if rm < 0 else 0

        if bias > 0:  # Bull context
            if rf < 0 and rm > 0:
                debug["context"] = "BULL_PULLBACK"
                return "PULLBACK", debug
        elif bias < 0:  # Bear context
            if rf > 0 and rm < 0:
                debug["context"] = "BEAR_PULLBACK"
                return "PULLBACK", debug

        # 4) FADE: momentum decelerating or whipsawing
        # Strong move but acceleration reversing
        if abs(rf) >= impulse_thresh:
            # Bull fade: strong up but decelerating
            if rf > 0 and af < 0:
                debug["fade_type"] = "BULL_DECEL"
                return "FADE", debug

            # Bear fade: strong down but decelerating (af > 0)
            if rf < 0 and af > 0:
                debug["fade_type"] = "BEAR_DECEL"
                return "FADE", debug

        # Mid near zero with fast flipping (whipsaw)
        if abs(rm) < noise_thresh and abs(rf) > noise_thresh:
            debug["fade_type"] = "WHIPSAW"
            return "FADE", debug

        # Default: ambiguous state, keep as NOISE or return generic
        return "NOISE", debug

    def _compute_momentum_score(
        self,
        rf: float,
        rm: float,
        af: float,
        roc_fast_raw: float,
        momentum_state: str
    ) -> float:
        """
        Compute momentum score (0-100).

        Components:
        - Magnitude: abs(rf) / impulse_threshold
        - Follow-through: abs(rm) / impulse_threshold
        - Acceleration: abs(af) / abs(roc_fast_raw) (relative acceleration)

        Weighted: 50% mag + 30% follow + 20% accel
        """
        impulse_thresh = self.config.impulse_norm_threshold
        eps = self.config.eps

        # Magnitude (0-1)
        mag = min(abs(rf) / impulse_thresh, 1.0)

        # Follow-through (0-1)
        follow = min(abs(rm) / impulse_thresh, 1.0)

        # Relative acceleration (0-1)
        accel_rel = min(abs(af) / (abs(roc_fast_raw) + eps), 1.0)

        # Weighted score
        score = 100 * (0.5 * mag + 0.3 * follow + 0.2 * accel_rel)

        # Cap NOISE state at 25
        if momentum_state == "NOISE":
            score = min(score, 25.0)

        return score

    def _check_divergence(
        self,
        state: _TimeframeState,
        current_price: float,
        roc_norm_fast: float,
        debug: Dict
    ) -> None:
        """Check for price/momentum divergence."""
        debug["divergence"] = "NONE"

        # Bearish divergence: higher high price, lower high ROC
        if state.last_swing_high is not None:
            prev_price, prev_roc = state.last_swing_high
            if current_price > prev_price and roc_norm_fast < prev_roc:
                debug["divergence"] = "BEARISH"

        # Bullish divergence: lower low price, higher low ROC
        if state.last_swing_low is not None:
            prev_price, prev_roc = state.last_swing_low
            if current_price < prev_price and roc_norm_fast > prev_roc:
                debug["divergence"] = "BULLISH"

    def get_state(self, tf: str) -> Optional[ROCState]:
        """Get current ROC state for timeframe."""
        if tf not in self.states:
            return None
        return self.states[tf].current_state

    def record_swing_high(self, tf: str, swing_price: float, roc_norm_fast: float) -> None:
        """Record swing high for divergence detection."""
        if tf in self.states:
            self.states[tf].last_swing_high = (swing_price, roc_norm_fast)

    def record_swing_low(self, tf: str, swing_price: float, roc_norm_fast: float) -> None:
        """Record swing low for divergence detection."""
        if tf in self.states:
            self.states[tf].last_swing_low = (swing_price, roc_norm_fast)


# =============================================================================
# DISPLAY UTILITIES
# =============================================================================

def print_roc_momentum(engine: ROCMomentumEngine, timeframes: Optional[List[str]] = None) -> None:
    """
    Compact print block for ROC momentum state.

    Example output:
        ┌─────────────────────────────────────────────┐
        │  ROC MOMENTUM                               │
        └─────────────────────────────────────────────┘
        1m: state=IMPULSE score=76
          ROC: 5=+0.0008 20=+0.0022 60=+0.0060
          ACC(5)=+0.0003  norm(5)=+1.10 (clipped)
          flags: blowoff=NO divergence=NONE
        5m: state=PULLBACK score=44 norm(3)=-0.35 norm(12)=+0.60
    """
    if timeframes is None:
        timeframes = engine.config.timeframes

    print("\n┌─────────────────────────────────────────────┐")
    print("│  ROC MOMENTUM                               │")
    print("└─────────────────────────────────────────────┘")

    for tf in timeframes:
        state = engine.get_state(tf)
        if state is None:
            print(f"{tf}: [warming up]")
            continue

        # Main line: state and score
        print(f"{tf}: state={state.momentum_state:8} score={state.momentum_score_0_100:.0f}")

        # ROC values
        roc_strs = [f"{lb}={val:+.4f}" for lb, val in sorted(state.roc.items())]
        print(f"  ROC: {' '.join(roc_strs)}")

        # ACC and norm for fast lookback
        if state.acc:
            fast_lb = min(state.acc.keys())
            acc_val = state.acc[fast_lb]
            norm_val = state.roc_norm.get(fast_lb, 0.0)

            # Check if clipped
            clipped = ""
            if abs(norm_val) >= engine.config.clip_norm - 0.01:
                clipped = " (clipped)"

            print(f"  ACC({fast_lb})={acc_val:+.4f}  norm({fast_lb})={norm_val:+.2f}{clipped}")

        # Flags
        blowoff = state.debug.get("blowoff", "NO")
        divergence = state.debug.get("divergence", "NONE")
        print(f"  flags: blowoff={blowoff} divergence={divergence}")

        # Optional: direction/context from debug
        if "direction" in state.debug:
            print(f"    direction={state.debug['direction']}")
        if "context" in state.debug:
            print(f"    context={state.debug['context']}")
        if "fade_type" in state.debug:
            print(f"    fade_type={state.debug['fade_type']}")

        print()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: synthetic trending data
    config = ROCConfig(
        timeframes=["1m"],
        roc_lookbacks_by_tf={"1m": [5, 20, 60]},
        accel_smooth_period=3
    )

    engine = ROCMomentumEngine(config)

    # Generate synthetic uptrend
    base_price = 100.0
    candles = []
    for i in range(100):
        # Uptrend with noise
        close = base_price + i * 0.1 + (i % 5) * 0.05
        high = close + 0.2
        low = close - 0.2

        candle = Candle(
            timestamp=float(i),
            open=close - 0.05,
            high=high,
            low=low,
            close=close,
            volume=1000.0
        )
        candles.append(candle)

    # Warmup
    engine.warmup({"1m": candles[:70]})

    # Process remaining
    for candle in candles[70:]:
        state = engine.on_candle_close("1m", candle)
        if state and candle.timestamp % 10 == 0:
            print(f"\n[Candle {int(candle.timestamp)}] Price: {candle.close:.2f}")
            print_roc_momentum(engine, ["1m"])
