#!/usr/bin/env python3
"""
ATR Expansion Demo

Demonstrates the ATR expansion module with synthetic data showing
different volatility regimes.
"""

from indicator.engines.atr_expansion import (
    ATRExpansionConfig,
    ATRExpansionEngine,
    Candle,
    format_atr_state,
    print_atr_expansion,
)


def generate_squeeze_candles(base_price=100, count=20):
    """Generate candles with low volatility (squeeze)."""
    candles = []
    price = base_price

    for i in range(count):
        # Tiny moves
        high = price + 0.5
        low = price - 0.5
        close = price + (i % 2) * 0.1  # Tiny fluctuation
        candle = Candle(
            timestamp=1000 * i, open=price, high=high, low=low, close=close, volume=1000
        )
        candles.append(candle)
        price = close

    return candles


def generate_expansion_candles(base_price=100, count=15):
    """Generate candles with increasing volatility (expansion)."""
    candles = []
    price = base_price

    for i in range(count):
        # Progressively larger moves
        volatility = 1 + i * 0.5
        high = price + volatility
        low = price - volatility
        close = price + volatility * 0.5  # Trending up
        candle = Candle(
            timestamp=1000 * (20 + i), open=price, high=high, low=low, close=close, volume=2000
        )
        candles.append(candle)
        price = close

    return candles


def generate_extreme_candles(base_price=110, count=10):
    """Generate candles with extreme volatility."""
    candles = []
    price = base_price

    for i in range(count):
        # Large moves
        high = price + 10
        low = price - 8
        close = price + 5  # Big moves
        candle = Candle(
            timestamp=1000 * (35 + i), open=price, high=high, low=low, close=close, volume=5000
        )
        candles.append(candle)
        price = close

    return candles


def demo_basic_usage():
    """Demo 1: Basic usage showing state transitions."""
    print("=" * 70)
    print("DEMO 1: Basic Usage - State Transitions")
    print("=" * 70)

    config = ATRExpansionConfig(timeframes=["demo"], atr_period=5, sma_period=10)
    engine = ATRExpansionEngine(config)

    # Phase 1: Squeeze
    print("\n📊 Phase 1: SQUEEZE (Low Volatility)")
    print("-" * 70)
    squeeze_candles = generate_squeeze_candles(base_price=100, count=15)

    for i, candle in enumerate(squeeze_candles):
        state = engine.on_candle_close("demo", candle)

        if i % 5 == 0 or i == len(squeeze_candles) - 1:
            print(f"Candle {i:2d}: {format_atr_state('demo', state)}")

    print(f"\n➡️  Final state: {state.vol_state} (score={state.vol_score_0_100:.0f})")

    # Phase 2: Expansion
    print("\n📊 Phase 2: EXPANSION (Volatility Waking Up)")
    print("-" * 70)
    expansion_candles = generate_expansion_candles(
        base_price=state.debug.get("prev_close", 100), count=15
    )

    for i, candle in enumerate(expansion_candles):
        state = engine.on_candle_close("demo", candle)

        if i % 5 == 0 or i == len(expansion_candles) - 1:
            print(f"Candle {i:2d}: {format_atr_state('demo', state)}")

    print(f"\n➡️  Final state: {state.vol_state} (score={state.vol_score_0_100:.0f})")

    # Phase 3: Extreme
    print("\n📊 Phase 3: EXTREME (Move is ON)")
    print("-" * 70)
    extreme_candles = generate_extreme_candles(
        base_price=state.debug.get("prev_close", 120), count=10
    )

    for i, candle in enumerate(extreme_candles):
        state = engine.on_candle_close("demo", candle)

        if i % 3 == 0 or i == len(extreme_candles) - 1:
            print(f"Candle {i:2d}: {format_atr_state('demo', state)}")

    print(f"\n➡️  Final state: {state.vol_state} (score={state.vol_score_0_100:.0f})")


def demo_multi_timeframe():
    """Demo 2: Multi-timeframe analysis."""
    print("\n\n" + "=" * 70)
    print("DEMO 2: Multi-Timeframe Analysis")
    print("=" * 70)

    config = ATRExpansionConfig(timeframes=["1m", "5m", "15m"], atr_period=5, sma_period=10)
    engine = ATRExpansionEngine(config)

    # Generate different scenarios for each TF
    all_candles = (
        generate_squeeze_candles(100, 10)
        + generate_expansion_candles(100, 10)
        + generate_extreme_candles(110, 10)
    )

    # Simulate different timeframes
    candles_1m = all_candles
    candles_5m = all_candles[::5]  # Every 5th candle
    candles_15m = all_candles[::15]  # Every 15th candle

    # Warmup
    states = engine.warmup(
        {
            "1m": candles_1m,
            "5m": candles_5m,
            "15m": candles_15m,
        }
    )

    print("\n📊 Multi-Timeframe States:")
    print("-" * 70)
    print_atr_expansion(states)

    print("\n💡 Interpretation:")
    if states["1m"].vol_state == "EXTREME" and states["5m"].vol_state == "EXPANSION":
        print("   ✅ 1m showing extreme vol, 5m confirming expansion")
        print("   → Good timing for fast scalps with tight stops")
    elif all(s.vol_state == "SQUEEZE" for s in states.values() if s):
        print("   ⏸️  All timeframes squeezed")
        print("   → Wait for expansion before taking positions")


def demo_shock_detection():
    """Demo 3: TR shock detection."""
    print("\n\n" + "=" * 70)
    print("DEMO 3: TR Shock Detection")
    print("=" * 70)

    config = ATRExpansionConfig(timeframes=["demo"], atr_period=5, sma_period=10)
    engine = ATRExpansionEngine(config)

    # Normal candles
    normal_candles = generate_squeeze_candles(100, 15)
    for candle in normal_candles:
        engine.on_candle_close("demo", candle)

    print("\n📊 Baseline (normal volatility):")
    state = engine.get_state("demo")
    if state:
        tr_spike_str = f"{state.tr_spike:.2f}" if state.tr_spike is not None else "N/A"
        print(f"   TR: {state.tr:.2f}, TR_spike: {tr_spike_str}")

    # Shock candle (sudden large move)
    print("\n⚡ SHOCK CANDLE:")
    prev_close = normal_candles[-1].close
    shock_candle = Candle(
        timestamp=16000,
        open=prev_close,
        high=prev_close + 15,  # HUGE move
        low=prev_close - 2,
        close=prev_close + 12,
        volume=10000,
    )

    state = engine.on_candle_close("demo", shock_candle)
    tr_spike_str = f"{state.tr_spike:.2f}" if state.tr_spike is not None else "N/A"
    print(f"   TR: {state.tr:.2f}, TR_spike: {tr_spike_str}")

    if state.debug.get("shock_now"):
        print("   🔥 SHOCK DETECTED - immediate volatility spike!")
        print("   → Fast entry opportunity with tight stops")


def demo_fade_risk():
    """Demo 4: Fade risk detection."""
    print("\n\n" + "=" * 70)
    print("DEMO 4: Fade Risk Detection")
    print("=" * 70)

    config = ATRExpansionConfig(timeframes=["demo"], atr_period=5, sma_period=10)
    engine = ATRExpansionEngine(config)

    # Build up expansion
    candles = generate_squeeze_candles(100, 10) + generate_expansion_candles(100, 15)

    for candle in candles:
        engine.on_candle_close("demo", candle)

    print("\n📊 During Expansion:")
    state = engine.get_state("demo")
    if state:
        atr_exp_str = f"{state.atr_exp:.2f}" if state.atr_exp is not None else "N/A"
        print(f"   State: {state.vol_state}, ATR_exp: {atr_exp_str}")

    # Now volatility starts to contract (fade)
    print("\n📊 Fading (contraction after expansion):")
    fade_start_price = candles[-1].close

    for i in range(10):
        # Progressively smaller moves
        volatility = 8 - i * 0.7  # Decreasing
        candle = Candle(
            timestamp=26000 + i * 1000,
            open=fade_start_price + i,
            high=fade_start_price + i + volatility,
            low=fade_start_price + i - volatility * 0.5,
            close=fade_start_price + i + 1,
            volume=1500,
        )
        state = engine.on_candle_close("demo", candle)

        if i % 3 == 0:
            slope_str = f"{state.atr_exp_slope:+.3f}" if state.atr_exp_slope is not None else "N/A"
            atr_exp_str = f"{state.atr_exp:.2f}" if state.atr_exp is not None else "N/A"
            print(
                f"   Candle {i}: ATR_exp={atr_exp_str}, slope={slope_str}, state={state.vol_state}"
            )

    if state.vol_state == "FADE_RISK":
        print("\n⚠️  FADE_RISK detected - expansion losing steam!")
        print("   → Consider tightening stops or taking profits")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "ATR EXPANSION DEMONSTRATION" + " " * 23 + "║")
    print("╚" + "=" * 68 + "╝")

    demo_basic_usage()
    demo_multi_timeframe()
    demo_shock_detection()
    demo_fade_risk()

    print("\n\n" + "=" * 70)
    print("✅ Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. ATR expansion detects volatility regime changes")
    print("2. SQUEEZE → EXPANSION → EXTREME → FADE_RISK lifecycle")
    print("3. Use as timing gate, NOT entry signal")
    print("4. Multi-TF confirmation increases conviction")
    print("5. TR shock detects immediate move starts")
    print("\n")
