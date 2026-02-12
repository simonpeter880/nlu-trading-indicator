"""
Choppiness Index Integration Examples

Shows how to integrate CHOP into continuous runner and batch analysis
as a regime filter for trend vs range-bound detection.
"""

from indicator.engines.choppiness import (
    Candle,
    ChoppinessConfig,
    ChoppinessEngine,
    format_chop_state,
    interpret_chop,
    print_choppiness,
)

# ============================================================================
# EXAMPLE 1: Continuous Runner Integration
# ============================================================================


def example_continuous_integration():
    """
    Example: Integrate CHOP into continuous/orchestrator.py

    Use CHOP as a regime filter to determine trading strategy:
    - CHOP state => prefer mean reversion (VWAP fades, range trading)
    - TREND state => allow breakout attempts, trend continuation
    - TRANSITION => reduce position size, wait for clarity
    """

    # In continuous/orchestrator.py __init__:
    # self.chop_engine = ChoppinessEngine(ChoppinessConfig(
    #     timeframes=["15s", "1m", "5m"],
    #     period_by_tf={"15s": 14, "1m": 14, "5m": 14}
    # ))

    # In _on_candle_close or compute_signals:
    # chop_state = self.chop_engine.on_candle_close(tf, candle)

    # Use as regime gate:
    # if chop_state.chop_state == "CHOP":
    #     # Range-bound: prefer mean reversion
    #     allow_vwap_fades = True
    #     allow_breakouts = False
    # elif chop_state.chop_state == "TREND":
    #     # Trending: allow trend continuation
    #     allow_vwap_fades = False
    #     allow_breakouts = True
    # else:  # TRANSITION
    #     # Mixed: reduce position size
    #     position_multiplier = 0.5

    pass


# ============================================================================
# EXAMPLE 2: Batch Analysis Integration (runner.py)
# ============================================================================


def example_batch_integration():
    """
    Example: Integrate CHOP into runner.py analyze_pair()

    Add after volatility indicators section.
    """

    # In runner.py after ATR/Bollinger Bands:

    # print_section("CHOPPINESS INDEX - Regime Filter", "📊")
    #
    # # Convert klines to Candle objects (reuse from ATR section if present)
    # candles = [
    #     Candle(k.timestamp, k.open, k.high, k.low, k.close, k.volume)
    #     for k in klines
    # ]
    #
    # # Initialize CHOP engine
    # chop_config = ChoppinessConfig(
    #     timeframes=[timeframe],
    #     default_period=14,
    # )
    # chop_engine = ChoppinessEngine(chop_config)
    #
    # # Warmup with historical candles
    # chop_states = chop_engine.warmup({timeframe: candles})
    #
    # # Display
    # print_choppiness(chop_states)
    #
    # # Interpret
    # if timeframe in chop_states:
    #     chop_state = chop_states[timeframe]
    #     interpretation = interpret_chop(chop_state)
    #
    #     if chop_state.chop_state == "CHOP":
    #         print(f"  {Colors.YELLOW}📊 REGIME: Range-bound - {interpretation}{Colors.RESET}")
    #     elif chop_state.chop_state == "TREND":
    #         print(f"  {Colors.GREEN}📈 REGIME: Trending - {interpretation}{Colors.RESET}")
    #     else:
    #         print(f"  {Colors.CYAN}⚖️  REGIME: Transition - {interpretation}{Colors.RESET}")
    #
    # print()

    pass


# ============================================================================
# EXAMPLE 3: Combined with ATR Expansion for Entry Timing
# ============================================================================


def example_combined_atr_chop():
    """
    Example: Combine CHOP with ATR Expansion for precise entry timing.

    Strategy:
    1. High CHOP + ATR squeeze => Range compression, prepare for breakout
    2. CHOP starts falling + TR spike + ATR expansion => Breakout confirmed
    3. Low CHOP + ATR expansion => Trend continuation mode
    4. CHOP rising + ATR fade => Trend exhausting, prepare for range
    """

    # Pseudo-code for combined logic:

    # if (chop_state.chop_state == "CHOP" and
    #     chop_state.chop_slope < 0 and  # CHOP starting to fall
    #     atr_state.vol_state == "EXPANSION" and
    #     atr_state.debug.get("shock_now")):
    #
    #     print("🚀 BREAKOUT WINDOW: Range compression breaking into trend")
    #     # High conviction breakout entry
    #     allow_breakout_entry = True
    #     position_size_multiplier = 1.5
    #
    # elif (chop_state.chop_state == "TREND" and
    #       atr_state.vol_state in ["EXPANSION", "EXTREME"]):
    #
    #     print("📈 TREND CONTINUATION: Trend confirmed by both CHOP and ATR")
    #     # Trend following mode
    #     allow_trend_entries = True
    #     use_trailing_stops = True
    #
    # elif (chop_state.chop_state == "CHOP" and
    #       atr_state.vol_state == "SQUEEZE"):
    #
    #     print("⏸️ RANGE COMPRESSION: Wait for breakout setup")
    #     # Wait for expansion
    #     allow_entries = False
    #
    # elif (chop_state.chop_slope > 0 and  # CHOP rising (trending -> range)
    #       atr_state.vol_state == "FADE_RISK"):
    #
    #     print("⚠️ TREND EXHAUSTION: Prepare for range-bound regime")
    #     # Tighten stops, consider exits
    #     tighten_stops = True
    #     reduce_position_size = True

    pass


# ============================================================================
# EXAMPLE 4: Multi-Timeframe CHOP Confirmation
# ============================================================================


def example_multi_timeframe_chop():
    """
    Example: Use multi-timeframe CHOP for regime confirmation.

    Strongest signals when multiple timeframes align:
    - All CHOP => Strong range, avoid breakouts
    - All TREND => Strong trend, aggressive continuation
    - Mixed => Transitional, cautious
    """

    # Get states for multiple timeframes
    # chop_15s = chop_engine.get_state("15s")
    # chop_1m = chop_engine.get_state("1m")
    # chop_5m = chop_engine.get_state("5m")
    #
    # # Check alignment
    # all_chop = all(s.chop_state == "CHOP" for s in [chop_15s, chop_1m, chop_5m] if s and s.chop)
    # all_trend = all(s.chop_state == "TREND" for s in [chop_15s, chop_1m, chop_5m] if s and s.chop)
    #
    # if all_chop:
    #     print("📊 STRONG RANGE: All timeframes choppy - strict mean reversion only")
    #     strategy_mode = "MEAN_REVERSION"
    # elif all_trend:
    #     print("📈 STRONG TREND: All timeframes trending - aggressive continuation")
    #     strategy_mode = "TREND_FOLLOWING"
    # else:
    #     print("⚖️ MIXED REGIME: Reduce position size, wait for alignment")
    #     strategy_mode = "CAUTIOUS"

    pass


# ============================================================================
# EXAMPLE 5: CHOP as Position Sizing Input
# ============================================================================


def example_position_sizing():
    """
    Example: Use CHOP to adjust position sizing.

    Higher CHOP => smaller positions (range-bound, lower conviction)
    Lower CHOP => larger positions (trending, higher conviction)
    """

    # chop_state = chop_engine.get_state("1m")
    #
    # if chop_state and chop_state.chop is not None:
    #     # Trendiness score = 100 - CHOP
    #     trendiness = chop_state.debug.get("trendiness_score", 50)
    #
    #     # Scale position size by trendiness
    #     # Range: 0.5x (high CHOP) to 1.5x (low CHOP)
    #     position_multiplier = 0.5 + (trendiness / 100.0)
    #
    #     base_position = 1000  # USDT
    #     adjusted_position = base_position * position_multiplier
    #
    #     print(f"Position sizing: {adjusted_position:.2f} USDT (trendiness: {trendiness:.0f}%)")

    pass


# ============================================================================
# EXAMPLE 6: Compact Print Block for Display
# ============================================================================


def example_compact_display():
    """
    Example: Display CHOP in deep-dive or batch analysis.

    Output format:
    CHOPPINESS
    1m: chop=64.2 state=CHOP slope=+1.8 sumTR=123.4 range=56.7
    5m: chop=41.7 state=TRANSITION slope=-0.9 sumTR=245.1 range=89.3
    1h: chop=33.5 state=TREND slope=-0.4 sumTR=512.7 range=234.5
    """

    # Create engine
    config = ChoppinessConfig(
        timeframes=["1m", "5m", "1h"],
        default_period=14,
    )
    engine = ChoppinessEngine(config)

    # Example candles (would come from real data)
    example_candles = [
        Candle(i * 60000, 100 + i * 0.5, 105 + i * 0.5, 95 + i * 0.5, 102 + i * 0.5, 1000)
        for i in range(20)
    ]

    # Warmup
    states = engine.warmup(
        {
            "1m": example_candles,
            "5m": example_candles[::5],
            "1h": example_candles[::60] if len(example_candles) >= 60 else example_candles,
        }
    )

    # Display
    print("\n" + "=" * 70)
    print_choppiness(states)
    print("=" * 70)

    # Interpret each
    for tf, state in states.items():
        if state.chop is not None:
            print(f"{tf}: {interpret_chop(state)}")


# ============================================================================
# EXAMPLE 7: Real-time Continuous Updates
# ============================================================================


def example_realtime_updates():
    """
    Example: Process real-time candle closes with O(1) updates.
    """

    config = ChoppinessConfig(timeframes=["1m"], default_period=14)
    engine = ChoppinessEngine(config)

    # Simulate real-time candle closes
    print("\nSimulating real-time CHOP updates:\n")

    for i in range(20):
        # Simulate candle
        price = 100 + i * 0.5
        candle = Candle(
            timestamp=i * 60000,
            open=price,
            high=price + 2,
            low=price - 2,
            close=price + 1,
            volume=1000 + i * 10,
        )

        # O(1) update
        state = engine.on_candle_close("1m", candle)

        # Print every 5th candle
        if i % 5 == 0:
            if state.chop is not None:
                print(f"Candle {i:2d}: {format_chop_state('1m', state)}")
            else:
                print(
                    f"Candle {i:2d}: WARMUP ({state.debug.get('bars_collected', 0)}/{state.debug.get('bars_needed', 14)} bars)"
                )


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == "__main__":
    print("\n╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "CHOPPINESS INDEX INTEGRATION" + " " * 22 + "║")
    print("╚" + "=" * 68 + "╝\n")

    print("\n📊 EXAMPLE 6: Compact Display")
    example_compact_display()

    print("\n⏱️  EXAMPLE 7: Real-time Updates")
    example_realtime_updates()

    print("\n\n" + "=" * 70)
    print("✅ Integration Examples Complete!")
    print("=" * 70)
    print("\nKey Integration Points:")
    print("1. Continuous runner: Add CHOP engine to orchestrator")
    print("2. Batch analysis: Add CHOP section after volatility indicators")
    print("3. Combine with ATR: Use both for precise entry timing")
    print("4. Multi-TF confirmation: Align regimes across timeframes")
    print("5. Position sizing: Scale by trendiness score")
    print("6. Strategy selection: CHOP => mean reversion, TREND => continuation")
    print()
