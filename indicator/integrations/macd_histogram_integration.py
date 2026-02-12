"""
MACD Histogram Integration Examples

Shows how to integrate MACD Histogram (momentum shift detection)
into continuous runner and batch analysis as a momentum confirmation tool.

CRITICAL: MACD Histogram is a MOMENTUM SHIFT tool, NOT an entry signal.
Always combine with price action, volume, and market structure.
"""

from indicator.engines.macd_histogram import (
    MACDHistogramEngine,
    MACDHistConfig,
    Candle,
    print_macd_histogram,
    format_macd_state,
    interpret_macd,
)


# ============================================================================
# EXAMPLE 1: Continuous Runner Integration
# ============================================================================

def example_continuous_integration():
    """
    Example: Integrate MACD Histogram into continuous/orchestrator.py

    Use MACD for:
    - Momentum shift warnings (BULL_SHIFT/BEAR_SHIFT)
    - Weakening signals (BULL_WEAKEN/BEAR_WEAKEN)
    - Phase confirmation (BULL/BEAR)
    """

    # In continuous/orchestrator.py __init__:
    # self.macd_engine = MACDHistogramEngine(MACDHistConfig(
    #     timeframes=["15s", "1m", "5m"],
    #     fast_period=12,
    #     slow_period=26,
    #     signal_period=9,
    #     normalize="atr",  # or "price" or "none"
    #     normalize_clip=3.0,
    # ))

    # In _on_candle_close or compute_signals:
    # # Update MACD (optionally pass ATR% for normalization)
    # macd_state = self.macd_engine.on_candle_close(tf, candle, atr_percent=atrp)

    # # Use as momentum confirmation:
    # if (macd_state.event == "BULL_SHIFT" and
    #     price_breaking_resistance and
    #     volume_increasing):
    #     # Strong momentum shift up - prepare for continuation
    #     momentum_shift_up = True
    #
    # elif (macd_state.event == "BULL_WEAKEN" and
    #       macd_state.phase == "BULL" and
    #       price_at_resistance):
    #     # Weakening momentum in uptrend - potential reversal
    #     weakening_warning = True
    #
    # elif (macd_state.event == "BEAR_SHIFT" and
    #       price_breaking_support):
    #     # Strong momentum shift down - prepare for breakdown
    #     momentum_shift_down = True

    pass


# ============================================================================
# EXAMPLE 2: Batch Analysis Integration (runner.py)
# ============================================================================

def example_batch_integration():
    """
    Example: Integrate MACD Histogram into runner.py analyze_pair()

    Add after RSI/momentum indicators or before unified score.
    """

    # In runner.py after RSI/MACD section:

    # print_section("MACD HISTOGRAM - Momentum Shifts", "📊")
    #
    # # Convert klines to Candle objects (reuse from ATR/CHOP if present)
    # candles = [
    #     Candle(k.timestamp, k.open, k.high, k.low, k.close, k.volume)
    #     for k in klines
    # ]
    #
    # # Initialize MACD engine
    # macd_config = MACDHistConfig(
    #     timeframes=[timeframe],
    #     fast_period=12,
    #     slow_period=26,
    #     signal_period=9,
    #     normalize="atr",
    # )
    # macd_engine = MACDHistogramEngine(macd_config)
    #
    # # Warmup with historical candles
    # macd_states = macd_engine.warmup({timeframe: candles})
    #
    # # Display
    # print_macd_histogram(macd_states)
    #
    # # Interpret
    # if timeframe in macd_states:
    #     macd_state = macd_states[timeframe]
    #     interpretation = interpret_macd(macd_state)
    #
    #     if macd_state.event.endswith("_SHIFT"):
    #         print(f"  {Colors.YELLOW}⚠️  MOMENTUM SHIFT: {interpretation}{Colors.RESET}")
    #     elif macd_state.event.endswith("_WEAKEN"):
    #         print(f"  {Colors.CYAN}⚡ WEAKENING: {interpretation}{Colors.RESET}")
    #     else:
    #         print(f"  {Colors.CYAN}ℹ️  {interpretation}{Colors.RESET}")
    #
    # print()

    pass


# ============================================================================
# EXAMPLE 3: Combined with RSI for Divergence Confirmation
# ============================================================================

def example_combined_rsi_macd():
    """
    Example: Combine MACD with RSI for stronger reversal signals.

    Best practice: RSI divergence + MACD weakening = high probability reversal.
    """

    # Pseudo-code assuming you have RSI timing engine:

    # rsi_state = rsi_engine.get_state("1m")
    # macd_state = macd_engine.get_state("1m")
    #
    # # Strong reversal up signal
    # if (rsi_state.divergence == "REG_BULL" and
    #     macd_state.event == "BEAR_WEAKEN" and
    #     price_at_support):
    #     print("🔥 STRONG REVERSAL UP: RSI bull div + MACD bear weaken")
    #     # High conviction long setup
    #
    # # Strong reversal down signal
    # elif (rsi_state.divergence == "REG_BEAR" and
    #       macd_state.event == "BULL_WEAKEN" and
    #       price_at_resistance):
    #     print("🔥 STRONG REVERSAL DOWN: RSI bear div + MACD bull weaken")
    #     # High conviction short setup

    pass


# ============================================================================
# EXAMPLE 4: Multi-Timeframe Momentum Alignment
# ============================================================================

def example_multi_timeframe_momentum():
    """
    Example: Use multi-timeframe MACD for stronger momentum signals.

    Strongest signals when momentum shifts align across timeframes.
    """

    # Get states for multiple timeframes
    # macd_1m = macd_engine.get_state("1m")
    # macd_5m = macd_engine.get_state("5m")
    # macd_1h = macd_engine.get_state("1h")
    #
    # # Check alignment
    # if (macd_1m.event == "BULL_SHIFT" and
    #     macd_5m.event == "BULL_SHIFT"):
    #     print("🚀 STRONG MOMENTUM UP: Multi-TF bullish shift")
    #     print("→ High probability trend continuation up")
    #
    # elif (macd_1m.event == "BEAR_SHIFT" and
    #       macd_5m.event == "BEAR_SHIFT" and
    #       macd_1h.phase == "BEAR"):
    #     print("📉 STRONG MOMENTUM DOWN: Multi-TF bearish shift")
    #     print("→ High conviction downtrend")
    #
    # # Divergence warning (lower TF weakening while higher TF strong)
    # elif (macd_1m.event == "BULL_WEAKEN" and
    #       macd_5m.phase == "BULL" and
    #       macd_1h.phase == "BULL"):
    #     print("⚠️ DIVERGENCE: 1m weakening but higher TFs still bullish")
    #     print("→ Potential short-term pullback")

    pass


# ============================================================================
# EXAMPLE 5: MACD Phase as Bias Filter
# ============================================================================

def example_phase_bias():
    """
    Example: Use MACD phase (BULL/BEAR) as directional bias.

    NOT for entries, but for filtering which setups to take.
    """

    # macd_state = macd_engine.get_state("5m")
    #
    # if macd_state.phase == "BULL":
    #     # Histogram > 0: Favor long setups
    #     print("📈 Bullish momentum phase - favor long entries")
    #     allow_long_setups = True
    #     allow_short_setups = False  # Avoid counter-trend
    #
    # elif macd_state.phase == "BEAR":
    #     # Histogram < 0: Favor short setups
    #     print("📉 Bearish momentum phase - favor short entries")
    #     allow_long_setups = False
    #     allow_short_setups = True

    pass


# ============================================================================
# EXAMPLE 6: Event Confidence for Position Sizing
# ============================================================================

def example_event_confidence():
    """
    Example: Use event confidence to scale positions.

    Higher confidence = higher conviction.
    """

    # macd_state = macd_engine.get_state("1m")
    #
    # if macd_state.event != "NONE":
    #     confidence = macd_state.event_confidence_0_100 or 50
    #
    #     if confidence > 80:
    #         print(f"🔥 High confidence {macd_state.event} ({confidence:.0f}%)")
    #         position_multiplier = 1.5  # Aggressive
    #     elif confidence > 50:
    #         print(f"✅ Moderate {macd_state.event} ({confidence:.0f}%)")
    #         position_multiplier = 1.0  # Standard
    #     else:
    #         print(f"⚠️ Low confidence {macd_state.event} ({confidence:.0f}%)")
    #         position_multiplier = 0.5  # Reduced

    pass


# ============================================================================
# EXAMPLE 7: Compact Print Block for Display
# ============================================================================

def example_compact_display():
    """
    Example: Display MACD histogram in deep-dive or batch analysis.

    Output format:
    MACD HISTOGRAM
    1m: hist=0.23  phase=BULL  event=BULL_WEAKEN  conf=78
    5m: hist=-0.15 phase=BEAR  event=NONE         conf=--
    1h: hist=0.45  phase=BULL  event=BULL_SHIFT   conf=92
    """

    # Create engine
    config = MACDHistConfig(
        timeframes=["1m", "5m", "1h"],
        fast_period=12,
        slow_period=26,
        signal_period=9,
        normalize="none",
    )
    engine = MACDHistogramEngine(config)

    # Example candles
    example_candles = []
    price = 100.0
    for i in range(50):
        change = 2 if i % 4 == 0 else -0.5
        price += change
        example_candles.append(Candle(
            i * 60000, price - 1, price + 2, price - 2, price, 1000
        ))

    # Warmup
    states = engine.warmup({
        "1m": example_candles,
        "5m": example_candles[::5],
        "1h": example_candles[::60] if len(example_candles) >= 60 else example_candles[:10],
    })

    # Display
    print("\n" + "=" * 70)
    print_macd_histogram(states)
    print("=" * 70)

    # Interpret each
    for tf, state in states.items():
        if state.hist is not None:
            print(f"{tf}: {interpret_macd(state)}")


# ============================================================================
# EXAMPLE 8: Real-time Continuous Updates
# ============================================================================

def example_realtime_updates():
    """
    Example: Process real-time candle closes with O(1) MACD updates.
    """

    config = MACDHistConfig(timeframes=["1m"], fast_period=12, slow_period=26, signal_period=9)
    engine = MACDHistogramEngine(config)

    # Simulate real-time candle closes
    print("\nSimulating real-time MACD histogram updates:\n")

    price = 100.0
    for i in range(40):
        # Simulate varying price action
        if i < 15:
            change = 1.5  # Uptrend
        elif i < 25:
            change = -0.5  # Pullback
        else:
            change = 2.0  # Resume uptrend

        price += change
        candle = Candle(
            timestamp=i * 60000,
            open=price - change,
            high=price + 1,
            low=price - 2,
            close=price,
            volume=1000 + i * 10
        )

        # O(1) update
        state = engine.on_candle_close("1m", candle)

        # Print every 5th candle
        if i % 5 == 0:
            if state.hist is not None:
                print(f"Candle {i:2d}: {format_macd_state('1m', state)}")
                if state.event != "NONE":
                    print(f"          ⚠️  {interpret_macd(state)}")
            else:
                print(f"Candle {i:2d}: WARMUP (MACD not ready)")


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == "__main__":
    print("\n╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "MACD HISTOGRAM INTEGRATION" + " " * 26 + "║")
    print("╚" + "=" * 68 + "╝\n")

    print("\n📊 EXAMPLE 7: Compact Display")
    example_compact_display()

    print("\n⏱️  EXAMPLE 8: Real-time Updates")
    example_realtime_updates()

    print("\n\n" + "=" * 70)
    print("✅ Integration Examples Complete!")
    print("=" * 70)
    print("\nKey Integration Points:")
    print("1. Continuous runner: Add MACD engine to orchestrator")
    print("2. Batch analysis: Add MACD section after momentum indicators")
    print("3. RSI confirmation: Combine divergence + MACD weakening")
    print("4. Multi-TF confirmation: Align momentum shifts across timeframes")
    print("5. Phase filter: Use BULL/BEAR phase for directional bias")
    print("6. Position sizing: Scale by event confidence")
    print("\n⚠️  REMEMBER: MACD Histogram is a MOMENTUM SHIFT tool, NOT an entry signal!")
    print("Always combine with price action, volume, and market structure.")
    print()
