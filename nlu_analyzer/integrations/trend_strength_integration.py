"""
Trend Strength Integration Example

Demonstrates how to integrate the Trend Strength composite module
with EMA Ribbon, Supertrend, and other systems for lag-free trend measurement.
"""

from nlu_analyzer.indicators.trend_strength import (
    TrendStrengthEngine,
    TrendStrengthConfig,
    Candle,
    format_trend_strength_output
)
from typing import Dict


def example_basic_integration():
    """
    Example: Basic integration of Trend Strength Engine.

    Workflow:
    1. Initialize engine with config
    2. Warmup with historical data
    3. Process new candles incrementally
    4. Provide external indicators (EMA slope, ribbon, RV, OI)
    5. Get strength score (0-100) and bucket (WEAK/EMERGING/STRONG)
    """

    # ========== CONFIGURATION ==========
    config = TrendStrengthConfig(
        # Timeframes
        timeframes=["1m", "5m", "1h"],

        # Smoothing and periods
        smooth_period=5,
        atr_period=14,
        rv_period=20,

        # EMA slope normalization
        ema_slope_strong_factor=0.20,

        # Ribbon normalization
        ribbon_wr_low=-0.10,
        ribbon_wr_high=0.20,

        # RV normalization
        rv_low=0.8,
        rv_high=2.0,

        # Component weights
        w_ema_slope=0.35,
        w_ribbon=0.25,
        w_rv=0.20,
        w_oi=0.20,

        # Safety caps
        cap_when_structure_range=50,
        cap_when_supertrend_chop=50,
        cap_when_rv_dead=25,

        # Bucketing
        bucket_weak_max=30.0,
        bucket_emerging_max=60.0
    )

    # ========== INITIALIZATION ==========
    engine = TrendStrengthEngine(config)

    # ========== WARMUP ==========
    print("Warming up Trend Strength Engine...")

    # Generate synthetic historical candles
    historical_candles_by_tf = {
        "1m": generate_test_candles(100.0, 100, 0.002),
        "5m": generate_test_candles(100.0, 80, 0.003),
        "1h": generate_test_candles(100.0, 60, 0.004)
    }

    warmup_results = engine.warmup(historical_candles_by_tf)
    print(f"Warmed up {len(warmup_results)} timeframes.\n")

    # ========== STREAMING LOOP ==========
    print("=" * 80)
    print("STREAMING MODE - Processing new candles")
    print("=" * 80)

    # Example: New 1m candle closes
    new_candle = Candle(
        timestamp=100.0,
        open=110.5,
        high=110.8,
        low=110.3,
        close=110.7,
        volume=50000.0
    )

    # Get external indicators from other systems
    # (EMA Ribbon, Supertrend, Volume analyzer, etc.)
    slope_50 = 0.0025              # From EMA system (slope magnitude)
    ribbon_width_rate = 0.15       # From EMA Ribbon
    rv = 1.5                       # From volume analyzer
    oi_now = 100000.0              # Current OI
    oi_prev = 97000.0              # Previous OI (for change rate calculation)
    bias = "BULL"                  # From EMA or structure analysis

    # Update Trend Strength with directional bias
    result = engine.on_candle_close(
        "1m",
        new_candle,
        slope_50=slope_50,
        ribbon_width_rate=ribbon_width_rate,
        rv=rv,
        oi_now=oi_now,
        oi_prev=oi_prev,
        bias=bias  # NEW: Directional bias
    )

    print(f"\n[1m CANDLE CLOSED] Price: {new_candle.close:.2f}")
    print(f"Trend Strength: {result.strength_smooth:.1f}/100")
    print(f"Signed Strength: {result.strength_signed:+.1f}/100")
    print(f"Direction Bias: {result.direction_bias:+d} ({bias})")
    print(f"Bucket: {result.bucket.value}")
    print(f"Components used: {list(result.weights.keys())}")

    # ========== MULTI-TIMEFRAME UPDATE ==========
    print("\n" + "=" * 80)
    print("MULTI-TIMEFRAME UPDATE")
    print("=" * 80)

    latest_candles = {
        "1m": [new_candle],
        "5m": [Candle(100.0, 110.5, 111.5, 110.0, 111.0, 250000.0)],
        "1h": [Candle(100.0, 109.0, 112.0, 108.5, 111.5, 3000000.0)]
    }

    # External indicators per timeframe (with directional bias)
    external_by_tf = {
        "1m": {
            "slope_50": 0.0025,
            "ribbon_width_rate": 0.15,
            "rv": 1.5,
            "oi_now": 100000.0,
            "oi_prev": 97000.0,
            "bias": "BULL"  # NEW
        },
        "5m": {
            "slope_50": 0.0030,
            "ribbon_width_rate": 0.18,
            "rv": 1.7,
            "oi_now": 100000.0,
            "oi_prev": 96000.0,
            "bias": "BULL"  # NEW
        },
        "1h": {
            "slope_50": 0.0035,
            "ribbon_width_rate": 0.20,
            "rv": 1.8,
            "oi_now": 100000.0,
            "oi_prev": 95000.0,
            "direction_bias": 1  # NEW: Can also use int directly
        }
    }

    all_results = {}
    for tf, candles in latest_candles.items():
        for candle in candles:
            all_results[tf] = engine.on_candle_close(
                tf,
                candle,
                **external_by_tf[tf]
            )

    print(format_trend_strength_output(all_results, compact=True))

    # ========== DETAILED METRICS ACCESS ==========
    print("\n" + "=" * 80)
    print("DETAILED METRICS ACCESS")
    print("=" * 80)

    state = result
    print(f"\nTimeframe: 1m")
    print(f"  Strength (final): {state.strength_smooth:.1f}/100")
    print(f"  Strength (raw): {state.strength_raw:.1f}/100")
    print(f"  Bucket: {state.bucket.value}")
    print(f"  Components normalized:")
    for comp, val in state.components_norm.items():
        if val is not None:
            print(f"    {comp}: {val:.3f}")
    print(f"  Weights used: {state.weights}")

    # ========== TRADING LOGIC EXAMPLE ==========
    print("\n" + "=" * 80)
    print("TRADING LOGIC EXAMPLE")
    print("=" * 80)

    def analyze_trend_strength(ts_state):
        """Example: Use Trend Strength for position sizing and risk management"""

        signals = []

        if ts_state.bucket.value == "STRONG":
            signals.append(f"✓ STRONG TREND ({ts_state.strength_smooth:.0f}/100)")
            signals.append("  → Full position size allowed")
            signals.append("  → Trailing stops farther out")
            signals.append("  → Add to winners on pullbacks")

        elif ts_state.bucket.value == "EMERGING":
            signals.append(f"⚠ EMERGING TREND ({ts_state.strength_smooth:.0f}/100)")
            signals.append("  → Reduced position size (50-70%)")
            signals.append("  → Tighter stops")
            signals.append("  → Wait for confirmation before scaling")

        elif ts_state.bucket.value == "WEAK":
            signals.append(f"✗ WEAK TREND ({ts_state.strength_smooth:.0f}/100)")
            signals.append("  → Minimal position size (20-30%)")
            signals.append("  → Very tight stops")
            signals.append("  → Avoid adding to positions")

        # Component-specific insights
        if ts_state.components_norm.get('ema_slope') is not None:
            ema_comp = ts_state.components_norm['ema_slope']
            if ema_comp > 0.8:
                signals.append(f"  EMA slope very strong ({ema_comp:.2f})")
            elif ema_comp < 0.3:
                signals.append(f"  EMA slope weak ({ema_comp:.2f})")

        if ts_state.components_norm.get('ribbon') is not None:
            ribbon_comp = ts_state.components_norm['ribbon']
            if ribbon_comp > 0.7:
                signals.append(f"  Ribbon expanding well ({ribbon_comp:.2f})")
            elif ribbon_comp < 0.2:
                signals.append(f"  Ribbon contracting ({ribbon_comp:.2f})")

        if ts_state.components_norm.get('rv') is not None:
            rv_comp = ts_state.components_norm['rv']
            if rv_comp > 0.7:
                signals.append(f"  Volume strong ({rv_comp:.2f})")
            elif rv_comp < 0.3:
                signals.append(f"  Volume weak ({rv_comp:.2f})")

        # Note: Safety caps are applied via flags parameter
        # (not visible in state output)

        return signals

    for tf in ["1m", "5m", "1h"]:
        if tf in all_results:
            signals = analyze_trend_strength(all_results[tf])
            print(f"\n{tf} Analysis:")
            for signal in signals:
                print(f"  {signal}")


def example_directional_signing():
    """
    Example: Directional signing with bias parameter.

    NEW FEATURE: Trend Strength now outputs signed strength (-100..+100)
    based on directional bias from EMA or structure analysis.
    """
    print("\n" + "=" * 80)
    print("DIRECTIONAL SIGNING EXAMPLE")
    print("=" * 80)

    config = TrendStrengthConfig()
    engine = TrendStrengthEngine(config)

    # Warmup
    candles = generate_test_candles(100.0, 50, 0.002)
    engine.warmup({"1m": candles})

    # Example 1: BULL bias
    print("\n1. BULL BIAS (uptrend):")
    result_bull = engine.on_candle_close(
        "1m",
        Candle(50.0, 110.0, 110.5, 109.8, 110.3, 50000.0),
        slope_50=0.0030,
        ribbon_width_rate=0.15,
        rv=1.5,
        bias="BULL"  # or direction_bias=+1
    )
    print(f"  Strength (unsigned): {result_bull.strength_smooth:.1f}")
    print(f"  Direction bias: {result_bull.direction_bias:+d}")
    print(f"  Signed strength: {result_bull.strength_signed:+.1f}")
    print(f"  → Positive signed strength = bullish momentum")

    # Example 2: BEAR bias
    print("\n2. BEAR BIAS (downtrend):")
    result_bear = engine.on_candle_close(
        "1m",
        Candle(51.0, 110.0, 110.2, 109.5, 109.7, 50000.0),
        slope_50=0.0030,
        ribbon_width_rate=0.15,
        rv=1.5,
        bias="BEAR"  # or direction_bias=-1
    )
    print(f"  Strength (unsigned): {result_bear.strength_smooth:.1f}")
    print(f"  Direction bias: {result_bear.direction_bias:+d}")
    print(f"  Signed strength: {result_bear.strength_signed:+.1f}")
    print(f"  → Negative signed strength = bearish momentum")

    # Example 3: NEUTRAL bias
    print("\n3. NEUTRAL BIAS (no directional conviction):")
    result_neutral = engine.on_candle_close(
        "1m",
        Candle(52.0, 110.0, 110.3, 109.8, 110.1, 50000.0),
        slope_50=0.0030,
        ribbon_width_rate=0.15,
        rv=1.5,
        bias="NEUTRAL"  # or direction_bias=0
    )
    print(f"  Strength (unsigned): {result_neutral.strength_smooth:.1f}")
    print(f"  Direction bias: {result_neutral.direction_bias:+d}")
    print(f"  Signed strength: {result_neutral.strength_signed:+.1f}")
    print(f"  → Zero signed strength = no directional conviction")

    # Example 4: Using direction_bias int directly (takes precedence)
    print("\n4. USING direction_bias INT (overrides bias string):")
    result_int = engine.on_candle_close(
        "1m",
        Candle(53.0, 110.0, 110.4, 109.9, 110.2, 50000.0),
        slope_50=0.0030,
        ribbon_width_rate=0.15,
        rv=1.5,
        bias="BULL",  # This will be ignored
        direction_bias=-1  # This takes precedence
    )
    print(f"  bias='BULL' but direction_bias=-1")
    print(f"  Direction bias used: {result_int.direction_bias:+d}")
    print(f"  Signed strength: {result_int.strength_signed:+.1f}")
    print(f"  → direction_bias parameter takes precedence")

    print("\nUSAGE NOTES:")
    print("  - Get bias from EMA system: ema_state.ema_bias ('BULL'/'BEAR'/'NEUTRAL')")
    print("  - Or from structure: structure_bias based on swing highs/lows")
    print("  - Use signed strength for unified long/short scoring")
    print("  - Filter entries: only longs if signed > 0, shorts if signed < 0")


def generate_test_candles(start_price: float, count: int, trend: float):
    """Generate synthetic test candles"""
    candles = []
    price = start_price

    for i in range(count):
        high = price * 1.002
        low = price * 0.998
        close = price * (1 + trend)

        candles.append(Candle(
            timestamp=float(i),
            open=price,
            high=high,
            low=low,
            close=close,
            volume=100000.0
        ))

        price = close

    return candles


# ========== INTEGRATION WITH OTHER MODULES ==========

def example_with_ema_ribbon_and_supertrend():
    """
    Example: Integrate Trend Strength with EMA Ribbon and Supertrend.

    This shows the complete picture:
    - Supertrend: Regime label (TREND/CHOP) + Direction (UP/DOWN)
    - EMA Ribbon: Stack health + ribbon metrics
    - Trend Strength: Composite strength score (0-100)
    """

    # Assume you have initialized these engines:
    # - supertrend_engine (from supertrend_filter.py)
    # - ema_ribbon_engine (from ema_ribbon.py)
    # - trend_strength_engine (from trend_strength.py)

    print("\n" + "=" * 80)
    print("COMPLETE INTEGRATION EXAMPLE")
    print("=" * 80)

    # On each candle close:
    # 1. Update Supertrend
    # st_state = supertrend_engine.on_candle_close(tf, candle)

    # 2. Update EMA Ribbon
    # ribbon_state = ema_ribbon_engine.on_candle_close(tf, candle)

    # 3. Extract metrics from EMA Ribbon
    # ema_slope = ribbon_state.center_ema_slope
    # ribbon_wr = ribbon_state.ribbon_width_rate

    # 4. Get RV from volume analyzer
    # rv = volume_analyzer.get_rv(tf)

    # 5. Get OI change from OI tracker (optional)
    # oi_change = oi_tracker.get_change_rate(tf)

    # 6. Update Trend Strength with all inputs
    # ts_state = trend_strength_engine.on_candle_close(
    #     tf, candle,
    #     ema_slope_magnitude=abs(ema_slope),
    #     ribbon_width_rate=ribbon_wr,
    #     rv=rv,
    #     oi_change_rate=oi_change
    # )

    # 7. Make trading decisions based on complete context
    # if st_state.regime == Regime.TREND:
    #     if st_state.st_direction == Direction.UP:
    #         if ribbon_state.state == RibbonState.HEALTHY:
    #             if ts_state.bucket == TrendBucket.STRONG:
    #                 # BEST SETUP: Trend confirmed, stack healthy, strength strong
    #                 # → Full position size, aggressive entries
    #                 pass
    #             elif ts_state.bucket == TrendBucket.EMERGING:
    #                 # GOOD SETUP: Trend forming
    #                 # → Moderate position size
    #                 pass
    #         elif ribbon_state.state == RibbonState.WEAKENING:
    #             if ts_state.bucket == TrendBucket.STRONG:
    #                 # MIXED SIGNALS: Ribbon weakening but strength still strong
    #                 # → Reduced size, watch for exit
    #                 pass

    print("See code comments above for integration pattern.")
    print("\nKey principle:")
    print("  - Supertrend: Filter regime and direction")
    print("  - EMA Ribbon: Check stack health")
    print("  - Trend Strength: Measure momentum strength")
    print("  - All three together → Highest conviction setups")


if __name__ == "__main__":
    print("TREND STRENGTH - INTEGRATION EXAMPLE")
    print("=" * 80)
    example_directional_signing()
    example_basic_integration()
    example_with_ema_ribbon_and_supertrend()
