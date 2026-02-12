"""
RSI Timing Integration Examples

Shows how to integrate RSI timing (divergence detection + regime labeling)
into continuous runner and batch analysis as a timing confirmation tool.

CRITICAL: RSI is a TIMING tool, NOT an entry signal.
Always combine with price action, volume, and market structure.
"""

from indicator.engines.rsi_timing import (
    Candle,
    RSITimingConfig,
    RSITimingEngine,
    format_rsi_state,
    interpret_rsi,
    print_rsi_timing,
)

# ============================================================================
# EXAMPLE 1: Continuous Runner Integration
# ============================================================================


def example_continuous_integration():
    """
    Example: Integrate RSI Timing into continuous/orchestrator.py

    Use RSI for:
    - Divergence warnings (exhaustion)
    - Regime confirmation (timing)
    - Structure confluence
    """

    # In continuous/orchestrator.py __init__:
    # self.rsi_engine = RSITimingEngine(RSITimingConfig(
    #     timeframes=["15s", "1m", "5m"],
    #     rsi_period=14,
    #     regime_high=55.0,
    #     regime_low=45.0,
    # ))

    # In _on_candle_close or compute_signals:
    # # Update RSI (optionally pass ATR% for better divergence threshold)
    # rsi_state = self.rsi_engine.on_candle_close(tf, candle, atr_percent=atrp)

    # # When market structure confirms a swing:
    # self.rsi_engine.on_swing_high(tf, swing.timestamp_ms, swing.price)
    # self.rsi_engine.on_swing_low(tf, swing.timestamp_ms, swing.price)

    # # Use as timing confirmation:
    # if (rsi_state.divergence == "REG_BEAR" and
    #     price_at_resistance and
    #     volume_declining):
    #     # Strong exhaustion signal - prepare for reversal
    #     reversal_warning = True
    #
    # elif (rsi_state.divergence == "HID_BULL" and
    #       price_at_support and
    #       rsi_state.rsi_regime == "BULLISH"):
    #     # Trend continuation setup - higher low forming
    #     continuation_signal = True

    pass


# ============================================================================
# EXAMPLE 2: Batch Analysis Integration (runner.py)
# ============================================================================


def example_batch_integration():
    """
    Example: Integrate RSI Timing into runner.py analyze_pair()

    Add after momentum indicators or before unified score.
    """

    # In runner.py after RSI/MACD:

    # print_section("RSI TIMING - Divergence & Regime", "⚡")
    #
    # # Convert klines to Candle objects (reuse from ATR/CHOP if present)
    # candles = [
    #     Candle(k.timestamp, k.open, k.high, k.low, k.close, k.volume)
    #     for k in klines
    # ]
    #
    # # Initialize RSI timing engine
    # rsi_config = RSITimingConfig(
    #     timeframes=[timeframe],
    #     rsi_period=14,
    # )
    # rsi_engine = RSITimingEngine(rsi_config)
    #
    # # Warmup with historical candles
    # rsi_states = rsi_engine.warmup({timeframe: candles})
    #
    # # Display
    # print_rsi_timing(rsi_states)
    #
    # # Interpret
    # if timeframe in rsi_states:
    #     rsi_state = rsi_states[timeframe]
    #     interpretation = interpret_rsi(rsi_state)
    #
    #     if rsi_state.divergence.startswith("REG_"):
    #         print(f"  {Colors.YELLOW}⚠️  DIVERGENCE: {interpretation}{Colors.RESET}")
    #     elif rsi_state.divergence.startswith("HID_"):
    #         print(f"  {Colors.GREEN}✅ CONTINUATION: {interpretation}{Colors.RESET}")
    #     else:
    #         print(f"  {Colors.CYAN}ℹ️  {interpretation}{Colors.RESET}")
    #
    #     if rsi_state.failure_swing != "NONE":
    #         print(f"  {Colors.RED}🔥 FAILURE SWING: Strong {rsi_state.failure_swing} signal{Colors.RESET}")
    #
    # print()

    pass


# ============================================================================
# EXAMPLE 3: Combined with Market Structure for Swing Detection
# ============================================================================


def example_combined_market_structure():
    """
    Example: Combine RSI with market structure module for swing-based divergence.

    Best practice: Let market structure identify swings, then RSI confirms divergence.
    """

    # Pseudo-code assuming you have a market structure module:

    # # When market structure confirms a swing high:
    # def on_structure_swing_high(tf, swing_high):
    #     # Notify RSI engine
    #     rsi_engine.on_swing_high(tf, swing_high.timestamp_ms, swing_high.price)
    #
    #     # Check for divergence
    #     rsi_state = rsi_engine.get_state(tf)
    #
    #     if rsi_state.divergence == "REG_BEAR":
    #         print(f"⚠️  Regular bearish divergence at {swing_high.price}")
    #         # Look for reversal confirmation (volume, price action)
    #
    #     elif rsi_state.divergence == "HID_BEAR":
    #         print(f"✅ Hidden bearish divergence - downtrend continuation")
    #         # Look for trend continuation entry
    #
    # # Similar for swing lows
    # def on_structure_swing_low(tf, swing_low):
    #     rsi_engine.on_swing_low(tf, swing_low.timestamp_ms, swing_low.price)
    #
    #     rsi_state = rsi_engine.get_state(tf)
    #
    #     if rsi_state.divergence == "REG_BULL":
    #         print(f"⚠️  Regular bullish divergence at {swing_low.price}")
    #         # Potential reversal up
    #
    #     elif rsi_state.divergence == "HID_BULL":
    #         print(f"✅ Hidden bullish divergence - uptrend continuation")
    #         # Trend continuation long

    pass


# ============================================================================
# EXAMPLE 4: Multi-Timeframe Divergence Confirmation
# ============================================================================


def example_multi_timeframe_divergence():
    """
    Example: Use multi-timeframe RSI for stronger divergence signals.

    Strongest signals when divergence aligns across timeframes.
    """

    # Get states for multiple timeframes
    # rsi_1m = rsi_engine.get_state("1m")
    # rsi_5m = rsi_engine.get_state("5m")
    # rsi_1h = rsi_engine.get_state("1h")
    #
    # # Check alignment
    # if (rsi_1m.divergence == "REG_BEAR" and
    #     rsi_5m.divergence == "REG_BEAR"):
    #     print("⚠️⚠️ STRONG EXHAUSTION: Multi-TF bearish divergence")
    #     print("→ High probability reversal DOWN")
    #
    # elif (rsi_1m.divergence == "HID_BULL" and
    #       rsi_5m.divergence == "HID_BULL" and
    #       rsi_1h.rsi_regime == "BULLISH"):
    #     print("✅✅ STRONG CONTINUATION: Multi-TF hidden bull div + bullish regime")
    #     print("→ High conviction uptrend continuation")

    pass


# ============================================================================
# EXAMPLE 5: RSI Regime as Bias Filter
# ============================================================================


def example_regime_bias():
    """
    Example: Use RSI regime (55/45 bands) as directional bias.

    NOT for entries, but for filtering which setups to take.
    """

    # rsi_state = rsi_engine.get_state("5m")
    #
    # if rsi_state.rsi_regime == "BULLISH":
    #     # RSI > 55: Favor long setups
    #     print("📈 Bullish regime - favor long entries at pullbacks")
    #     allow_long_setups = True
    #     allow_short_setups = False  # Avoid counter-trend
    #
    # elif rsi_state.rsi_regime == "BEARISH":
    #     # RSI < 45: Favor short setups
    #     print("📉 Bearish regime - favor short entries at bounces")
    #     allow_long_setups = False
    #     allow_short_setups = True
    #
    # else:  # RANGE
    #     # 45-55: No clear bias
    #     print("⚖️ Range regime - no directional bias, reduce size")
    #     allow_long_setups = True
    #     allow_short_setups = True
    #     position_multiplier = 0.5  # Reduced conviction

    pass


# ============================================================================
# EXAMPLE 6: Divergence Strength for Position Sizing
# ============================================================================


def example_divergence_strength():
    """
    Example: Use divergence strength to scale positions.

    Stronger divergence = higher conviction.
    """

    # rsi_state = rsi_engine.get_state("1m")
    #
    # if rsi_state.divergence != "NONE":
    #     strength = rsi_state.div_strength_0_100 or 50
    #
    #     if strength > 70:
    #         print(f"🔥 Strong {rsi_state.divergence} (strength={strength:.0f}%)")
    #         position_multiplier = 1.5  # Aggressive
    #     elif strength > 40:
    #         print(f"✅ Moderate {rsi_state.divergence} (strength={strength:.0f}%)")
    #         position_multiplier = 1.0  # Standard
    #     else:
    #         print(f"⚠️ Weak {rsi_state.divergence} (strength={strength:.0f}%)")
    #         position_multiplier = 0.5  # Reduced

    pass


# ============================================================================
# EXAMPLE 7: Compact Print Block for Display
# ============================================================================


def example_compact_display():
    """
    Example: Display RSI timing in deep-dive or batch analysis.

    Output format:
    RSI TIMING
    1m: rsi=58.4 regime=BULLISH div=HID_BULL strength=63 fs=NONE
    5m: rsi=47.2 regime=RANGE   div=NONE     strength=-- fs=NONE
    1h: rsi=41.3 regime=BEARISH div=REG_BULL strength=71 fs=BULL
    """

    # Create engine
    config = RSITimingConfig(
        timeframes=["1m", "5m", "1h"],
        rsi_period=14,
    )
    engine = RSITimingEngine(config)

    # Example candles
    example_candles = []
    price = 100.0
    for i in range(30):
        change = 2 if i % 3 == 0 else -1
        price += change
        example_candles.append(Candle(i * 60000, price - 1, price + 2, price - 2, price, 1000))

    # Warmup
    states = engine.warmup(
        {
            "1m": example_candles,
            "5m": example_candles[::5],
            "1h": example_candles[::60] if len(example_candles) >= 60 else example_candles[:5],
        }
    )

    # Add some swing events for divergence
    engine.on_swing_high("1m", 10000, 105.0)
    for i in range(5):
        candle = example_candles[10 + i]
        engine.on_candle_close("1m", candle)
    engine.on_swing_high("1m", 15000, 108.0)

    # Get updated states
    states = {tf: engine.get_state(tf) for tf in ["1m", "5m", "1h"]}

    # Display
    print("\n" + "=" * 70)
    print_rsi_timing(states)
    print("=" * 70)

    # Interpret each
    for tf, state in states.items():
        if state.rsi is not None:
            print(f"{tf}: {interpret_rsi(state)}")


# ============================================================================
# EXAMPLE 8: Real-time Continuous Updates
# ============================================================================


def example_realtime_updates():
    """
    Example: Process real-time candle closes with O(1) RSI updates.
    """

    config = RSITimingConfig(timeframes=["1m"], rsi_period=14)
    engine = RSITimingEngine(config)

    # Simulate real-time candle closes
    print("\nSimulating real-time RSI timing updates:\n")

    price = 100.0
    for i in range(25):
        # Simulate varying price action
        if i < 10:
            change = 1.5  # Uptrend
        elif i < 15:
            change = -1.0  # Pullback
        else:
            change = 2.0  # Resume uptrend

        price += change
        candle = Candle(
            timestamp=i * 60000,
            open=price - change,
            high=price + 1,
            low=price - 2,
            close=price,
            volume=1000 + i * 10,
        )

        # O(1) update
        state = engine.on_candle_close("1m", candle)

        # Simulate swing detection (simple: every 5th bar)
        if i % 5 == 0 and i > 0:
            if change > 0:
                engine.on_swing_low("1m", i * 60000, price - 3)
            else:
                engine.on_swing_high("1m", i * 60000, price + 2)

            # Refresh state after swing
            state = engine.get_state("1m")

        # Print every 5th candle
        if i % 5 == 0:
            if state.rsi is not None:
                print(f"Candle {i:2d}: {format_rsi_state('1m', state)}")
                if state.divergence != "NONE":
                    print(f"          ⚠️  {interpret_rsi(state)}")
            else:
                print(f"Candle {i:2d}: WARMUP (RSI not ready)")


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == "__main__":
    print("\n╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "RSI TIMING INTEGRATION" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝\n")

    print("\n📊 EXAMPLE 7: Compact Display")
    example_compact_display()

    print("\n⏱️  EXAMPLE 8: Real-time Updates")
    example_realtime_updates()

    print("\n\n" + "=" * 70)
    print("✅ Integration Examples Complete!")
    print("=" * 70)
    print("\nKey Integration Points:")
    print("1. Continuous runner: Add RSI engine to orchestrator")
    print("2. Batch analysis: Add RSI timing section after momentum indicators")
    print("3. Market structure: Connect swings to RSI for divergence detection")
    print("4. Multi-TF confirmation: Align divergence across timeframes")
    print("5. Regime filter: Use 55/45 bands for directional bias")
    print("6. Position sizing: Scale by divergence strength")
    print("\n⚠️  REMEMBER: RSI is a TIMING tool, NOT an entry signal!")
    print("Always combine with price action, volume, and market structure.")
    print()
