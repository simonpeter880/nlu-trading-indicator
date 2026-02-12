"""
ROC Momentum Integration Snippet

Shows how to integrate roc_momentum.py with existing trading system.

Integration points:
1. Add ROC momentum engine to continuous analyzer
2. Use ROC state for trade filtering
3. Combine with market structure for confluence
4. Display ROC momentum in analysis output.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from nlu_analyzer.indicators.roc_momentum import (
    Candle,
    ROCConfig,
    ROCMomentumEngine,
    ROCState,
    print_roc_momentum,
)

# =============================================================================
# INTEGRATION WITH CONTINUOUS ANALYZER
# =============================================================================


class ROCMomentumAdapter:
    """
    Adapter to integrate ROC momentum into continuous analysis pipeline.

    Usage in ContinuousAnalyzer:
        self.roc_adapter = ROCMomentumAdapter(config)

        # In on_candle_close:
        roc_state = self.roc_adapter.on_candle_close(
            tf="1m",
            candle=candle,
            atr_percent=volume_engine.get_atr_percent(),
            bias=structure_state.trend_bias
        )
    """

    def __init__(self, config: Optional[ROCConfig] = None):
        if config is None:
            # Default config for multi-timeframe analysis
            config = ROCConfig(
                timeframes=["1m", "5m", "15m", "1h"],
                roc_lookbacks_by_tf={
                    "1m": [5, 20, 60],  # Fast intraday
                    "5m": [3, 12, 36],  # Medium intraday
                    "15m": [4, 16, 48],  # Swing
                    "1h": [3, 6, 12],  # Position
                },
                use_log_returns=False,
                atr_period=14,
                accel_smooth_period=3,  # Smooth out noise
                noise_norm_threshold=0.3,
                impulse_norm_threshold=0.8,
                blowoff_norm_threshold=1.5,
            )

        self.engine = ROCMomentumEngine(config)
        self.config = config

    def warmup(self, candles_by_tf: Dict[str, List[Candle]]) -> None:
        """Warmup with historical candles."""
        self.engine.warmup(candles_by_tf)

    def on_candle_close(
        self,
        tf: str,
        candle: Candle,
        atr_percent: Optional[float] = None,
        bias: Optional[int] = None,
    ) -> Optional[ROCState]:
        """Process new candle and return ROC state."""
        return self.engine.on_candle_close(tf, candle, atr_percent, bias)

    def get_state(self, tf: str) -> Optional[ROCState]:
        """Get current ROC state."""
        return self.engine.get_state(tf)

    def record_swing(self, tf: str, is_high: bool, price: float) -> None:
        """Record swing point for divergence detection."""
        state = self.get_state(tf)
        if state is None:
            return

        # Get fast ROC_norm for divergence tracking
        fast_lb = min(state.roc_norm.keys()) if state.roc_norm else None
        if fast_lb is None:
            return

        roc_norm_fast = state.roc_norm[fast_lb]

        if is_high:
            self.engine.record_swing_high(tf, price, roc_norm_fast)
        else:
            self.engine.record_swing_low(tf, price, roc_norm_fast)


# =============================================================================
# TRADE FILTERING WITH ROC MOMENTUM
# =============================================================================


@dataclass
class ROCTradeFilter:
    """
    Filter trades based on ROC momentum state.

    Rules:
    1. IMPULSE: Allow trades in direction of impulse
    2. PULLBACK: Allow counter-trend entries (careful!)
    3. FADE: Reject new entries, consider exits
    4. NOISE: Reject trades (no clear direction)
    5. Divergence: Warning flag, reduce position size.
    """

    min_score_for_entry: float = 50.0
    allow_pullback_entries: bool = False
    block_on_blowoff: bool = True
    reduce_size_on_divergence: bool = True

    def should_enter_long(self, roc_state: ROCState) -> tuple[bool, str]:
        """Check if long entry is allowed based on ROC state."""
        if roc_state is None:
            return False, "no_roc_state"

        state = roc_state.momentum_state
        score = roc_state.momentum_score_0_100

        # NOISE: no entry
        if state == "NOISE":
            return False, "noise_state"

        # FADE: declining momentum, avoid entry
        if state == "FADE":
            return False, "fade_state"

        # Blowoff warning: extreme momentum, likely reversal
        if self.block_on_blowoff and roc_state.debug.get("blowoff") == "YES":
            return False, "blowoff_warning"

        # IMPULSE bull: allow if score sufficient
        if state == "IMPULSE" and roc_state.debug.get("direction") == "BULL":
            if score >= self.min_score_for_entry:
                return True, "impulse_bull"
            else:
                return False, "impulse_weak"

        # PULLBACK in bull context: allow if configured
        if state == "PULLBACK" and roc_state.debug.get("context") == "BULL_PULLBACK":
            if self.allow_pullback_entries and score >= self.min_score_for_entry * 0.7:
                return True, "pullback_bull"
            else:
                return False, "pullback_disabled"

        # Default: reject
        return False, "default_reject"

    def should_enter_short(self, roc_state: ROCState) -> tuple[bool, str]:
        """Check if short entry is allowed based on ROC state."""
        if roc_state is None:
            return False, "no_roc_state"

        state = roc_state.momentum_state
        score = roc_state.momentum_score_0_100

        if state == "NOISE":
            return False, "noise_state"

        if state == "FADE":
            return False, "fade_state"

        if self.block_on_blowoff and roc_state.debug.get("blowoff") == "YES":
            return False, "blowoff_warning"

        # IMPULSE bear
        if state == "IMPULSE" and roc_state.debug.get("direction") == "BEAR":
            if score >= self.min_score_for_entry:
                return True, "impulse_bear"
            else:
                return False, "impulse_weak"

        # PULLBACK in bear context
        if state == "PULLBACK" and roc_state.debug.get("context") == "BEAR_PULLBACK":
            if self.allow_pullback_entries and score >= self.min_score_for_entry * 0.7:
                return True, "pullback_bear"
            else:
                return False, "pullback_disabled"

        return False, "default_reject"

    def get_position_size_multiplier(self, roc_state: ROCState) -> float:
        """
        Get position size multiplier based on ROC state.

        Returns:
            Multiplier (0.0 - 1.0) for position sizing.
        """
        if roc_state is None:
            return 0.5  # Reduced size if no data

        # Divergence: reduce size
        if self.reduce_size_on_divergence:
            divergence = roc_state.debug.get("divergence", "NONE")
            if divergence != "NONE":
                return 0.5  # Half size on divergence

        # Blowoff: reduce size
        if roc_state.debug.get("blowoff") == "YES":
            return 0.3  # Very small size on extreme moves

        # Scale by momentum score
        score = roc_state.momentum_score_0_100
        if score >= 70:
            return 1.0  # Full size on strong momentum
        elif score >= 50:
            return 0.8
        elif score >= 30:
            return 0.5
        else:
            return 0.3  # Reduced size on weak momentum


# =============================================================================
# INTEGRATION WITH MARKET STRUCTURE
# =============================================================================


def combine_roc_with_structure(
    roc_state: Optional[ROCState],
    structure_trend: str,  # "BULL", "BEAR", "RANGE"
    direction: str,  # "long" or "short"
) -> tuple[bool, float, str]:
    """
    Combine ROC momentum with market structure for trade decision.

    Returns:
        (allow_trade, confidence_multiplier, reason)
    """
    if roc_state is None:
        return False, 0.0, "no_roc_data"

    # Check for confluence
    if direction == "long":
        # Need bullish structure + bullish momentum
        if structure_trend != "BULL":
            return False, 0.0, "structure_not_bull"

        # Check ROC state
        if roc_state.momentum_state == "IMPULSE" and roc_state.debug.get("direction") == "BULL":
            confidence = 1.0 + (roc_state.momentum_score_0_100 - 50) / 100
            return True, min(confidence, 1.5), "impulse_structure_confluence"

        if (
            roc_state.momentum_state == "PULLBACK"
            and roc_state.debug.get("context") == "BULL_PULLBACK"
        ):
            confidence = 0.8
            return True, confidence, "pullback_entry"

        return False, 0.0, "roc_not_bullish"

    else:  # short
        if structure_trend != "BEAR":
            return False, 0.0, "structure_not_bear"

        if roc_state.momentum_state == "IMPULSE" and roc_state.debug.get("direction") == "BEAR":
            confidence = 1.0 + (roc_state.momentum_score_0_100 - 50) / 100
            return True, min(confidence, 1.5), "impulse_structure_confluence"

        if (
            roc_state.momentum_state == "PULLBACK"
            and roc_state.debug.get("context") == "BEAR_PULLBACK"
        ):
            confidence = 0.8
            return True, confidence, "pullback_entry"

        return False, 0.0, "roc_not_bearish"


# =============================================================================
# COMPACT DISPLAY INTEGRATION
# =============================================================================


def print_roc_compact(engine: ROCMomentumEngine, primary_tf: str = "1m") -> None:
    """
    Compact single-line ROC display for continuous runner status line.

    Example output:
        ROC: IMPULSE(↑76) 5m=+0.8% div=NONE.
    """
    state = engine.get_state(primary_tf)
    if state is None:
        print("ROC: [warming up]", end="")
        return

    # State with direction emoji
    state_str = state.momentum_state
    if state.momentum_state == "IMPULSE":
        direction = state.debug.get("direction", "")
        arrow = "↑" if direction == "BULL" else "↓" if direction == "BEAR" else "·"
        state_str = f"{state.momentum_state}({arrow}{state.momentum_score_0_100:.0f})"
    else:
        state_str = f"{state.momentum_state}({state.momentum_score_0_100:.0f})"

    # Fast ROC as percentage
    if state.roc:
        fast_lb = min(state.roc.keys())
        fast_roc_pct = state.roc[fast_lb] * 100
        roc_str = f"{primary_tf}={fast_roc_pct:+.1f}%"
    else:
        roc_str = f"{primary_tf}=N/A"

    # Divergence
    div = state.debug.get("divergence", "NONE")

    print(f"ROC: {state_str} {roc_str} div={div}", end="")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ROC MOMENTUM INTEGRATION EXAMPLE")
    print("=" * 60)

    # 1. Create adapter with default config
    adapter = ROCMomentumAdapter()

    # 2. Generate sample data
    sample_candles = []
    base_price = 100.0
    for i in range(100):
        close = base_price + i * 0.3 + (i % 10) * 0.1
        candle = Candle(
            timestamp=float(i * 60),  # 1-minute candles
            open=close - 0.05,
            high=close + 0.15,
            low=close - 0.15,
            close=close,
            volume=1000.0 + i * 10,
        )
        sample_candles.append(candle)

    # 3. Warmup
    print("\n[1] Warming up with historical data...")
    adapter.warmup({"1m": sample_candles[:80]})

    # 4. Process new candles
    print("[2] Processing new candles...\n")
    for candle in sample_candles[80:90]:
        state = adapter.on_candle_close("1m", candle, atr_percent=0.012, bias=1)

        if state:
            print(f"\nCandle @ {candle.close:.2f}")
            print(f"  State: {state.momentum_state}")
            print(f"  Score: {state.momentum_score_0_100:.0f}")

            # Test trade filter
            filter = ROCTradeFilter(min_score_for_entry=50.0)
            allow_long, reason_long = filter.should_enter_long(state)
            allow_short, reason_short = filter.should_enter_short(state)

            print(f"  Long entry: {allow_long} ({reason_long})")
            print(f"  Short entry: {allow_short} ({reason_short})")

            # Position sizing
            size_mult = filter.get_position_size_multiplier(state)
            print(f"  Position size: {size_mult:.0%}")

    # 5. Display full ROC state
    print("\n" + "=" * 60)
    print("FULL ROC MOMENTUM STATE")
    print("=" * 60)
    print_roc_momentum(adapter.engine, ["1m"])

    # 6. Test compact display
    print("\n" + "=" * 60)
    print("COMPACT DISPLAY (for status line)")
    print("=" * 60)
    print_roc_compact(adapter.engine, "1m")
    print("\n")

    print("\n" + "=" * 60)
    print("INTEGRATION COMPLETE")
    print("=" * 60)
    print("\nKey integration points:")
    print("1. Add ROCMomentumAdapter to ContinuousAnalyzer")
    print("2. Call on_candle_close() for each timeframe")
    print("3. Use ROCTradeFilter for entry/exit decisions")
    print("4. Combine with market structure for confluence")
    print("5. Display with print_roc_momentum() or print_roc_compact()")
