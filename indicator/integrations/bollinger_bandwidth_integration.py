"""
Bollinger Bandwidth Integration Examples

Shows how to integrate Bollinger Bandwidth (volatility compression/expansion)
into continuous runner and batch analysis for squeeze detection and breakout timing.

CRITICAL: Bollinger Bandwidth is a VOLATILITY REGIME detector, NOT an entry signal.
Combine with price action, volume, and momentum for complete analysis.
"""

from indicator.engines.bollinger_bandwidth import (
    BollingerBandwidthConfig,
    BollingerBandwidthEngine,
    Candle,
    format_bandwidth_state,
    interpret_bandwidth,
    print_bollinger_bandwidth,
)

# ============================================================================
# EXAMPLE 1: Continuous Runner Integration
# ============================================================================


def example_continuous_integration():
    """
    Example: Integrate Bollinger Bandwidth into continuous/orchestrator.py

    Use for:
    - Squeeze detection (COMPRESSED state)
    - Breakout confirmation (EXPANDING state)
    - Exhaustion warnings (EXTREME/FADE_RISK states)
    """

    # In continuous/orchestrator.py __init__:
    # self.bb_bw_engine = BollingerBandwidthEngine(BollingerBandwidthConfig(
    #     timeframes=["15s", "1m", "5m"],
    #     bb_period=20,
    #     bb_k=2.0,
    #     bw_sma_period=50,
    #     smooth_bw_ema_period=3,
    # ))

    # In _on_candle_close or compute_signals:
    # bw_state = self.bb_bw_engine.on_candle_close(tf, candle)

    # # Use for timing decisions:
    # if bw_state.bw_state == "COMPRESSED":
    #     # Squeeze building - arm breakout setup
    #     if chop_state == "CHOP" and atr_exp < 0.8:
    #         self.arm_squeeze_breakout(tf)
    #         print(f"🔒 SQUEEZE: {tf} bandwidth compressed (ratio={bw_state.bw_ratio:.2f})")
    #
    # elif bw_state.bw_state == "EXPANDING":
    #     # Breakout in progress - confirm expansion
    #     if roc_surge and volume_increasing:
    #         print(f"⚡ BREAKOUT: {tf} bandwidth expanding (ratio={bw_state.bw_ratio:.2f})")
    #         self.execute_breakout_entry(tf)
    #
    # elif bw_state.bw_state in ["EXTREME", "FADE_RISK"]:
    #     # Exhaustion risk - be cautious
    #     print(f"⚠️  EXHAUSTION: {tf} bandwidth {bw_state.bw_state} (ratio={bw_state.bw_ratio:.2f})")
    #     self.reduce_position_size(tf)

    pass


# ============================================================================
# EXAMPLE 2: Batch Analysis Integration (runner.py)
# ============================================================================


def example_batch_integration():
    """
    Example: Integrate Bollinger Bandwidth into runner.py analyze_pair()

    Add after CHOP or volatility sections.
    """

    # In runner.py after CHOP/ATR sections:

    # print_section("BOLLINGER BANDWIDTH - Volatility Regime", "📏")
    #
    # # Convert klines to Candle objects
    # candles = [
    #     Candle(k.timestamp, k.open, k.high, k.low, k.close, k.volume)
    #     for k in klines
    # ]
    #
    # # Initialize BB Bandwidth engine
    # bb_bw_config = BollingerBandwidthConfig(
    #     timeframes=[timeframe],
    #     bb_period=20,
    #     bb_k=2.0,
    #     bw_sma_period=50,
    # )
    # bb_bw_engine = BollingerBandwidthEngine(bb_bw_config)
    #
    # # Warmup with historical candles
    # bw_states = bb_bw_engine.warmup({timeframe: candles})
    #
    # # Display
    # print_bollinger_bandwidth(bw_states)
    #
    # # Interpret
    # if timeframe in bw_states:
    #     bw_state = bw_states[timeframe]
    #     interpretation = interpret_bandwidth(bw_state)
    #
    #     if bw_state.bw_state == "COMPRESSED":
    #         print(f"  {Colors.YELLOW}🔒 SQUEEZE: {interpretation}{Colors.RESET}")
    #     elif bw_state.bw_state == "EXPANDING":
    #         print(f"  {Colors.GREEN}⚡ BREAKOUT: {interpretation}{Colors.RESET}")
    #     elif bw_state.bw_state in ["EXTREME", "FADE_RISK"]:
    #         print(f"  {Colors.RED}⚠️  EXHAUSTION: {interpretation}{Colors.RESET}")
    #     else:
    #         print(f"  {Colors.CYAN}ℹ️  {interpretation}{Colors.RESET}")
    #
    # print()

    pass


# ============================================================================
# EXAMPLE 3: Combined with CHOP and ATR Expansion
# ============================================================================


def example_combined_chop_atr():
    """
    Example: Combine BB Bandwidth with CHOP and ATR for squeeze-break detection.

    Best practice: COMPRESSED + CHOP high + ATR low = high-quality squeeze setup.
    """

    # Pseudo-code assuming you have CHOP and ATR expansion engines:

    # bw_state = bb_bw_engine.get_state("1m")
    # chop_state = chop_engine.get_state("1m")
    # atr_state = atr_exp_engine.get_state("1m")
    #
    # # Triple confirmation squeeze
    # if (bw_state.bw_state == "COMPRESSED" and
    #     chop_state.chop_state == "CHOP" and
    #     atr_state.atr_exp_state == "SQUEEZE"):
    #     print("🔥 HIGH-QUALITY SQUEEZE: BB compressed + CHOP high + ATR squeeze")
    #     print(f"   BB ratio={bw_state.bw_ratio:.2f}, CHOP={chop_state.chop_value:.1f}, ATR exp={atr_state.atr_exp:.2f}")
    #     # Very high probability breakout setup
    #
    # # Breakout confirmation
    # elif (bw_state.bw_state == "EXPANDING" and
    #       chop_state.chop_state == "TREND" and
    #       atr_state.atr_exp_state == "EXPANDING"):
    #     print("⚡ CONFIRMED BREAKOUT: All volatility indicators aligned")
    #     # High conviction entry

    pass


# ============================================================================
# EXAMPLE 4: Multi-Timeframe Volatility Alignment
# ============================================================================


def example_multi_timeframe_volatility():
    """
    Example: Use multi-TF bandwidth for stronger squeeze/breakout signals.

    Strongest setups when volatility regimes align across timeframes.
    """

    # Get states for multiple timeframes
    # bw_1m = bb_bw_engine.get_state("1m")
    # bw_5m = bb_bw_engine.get_state("5m")
    # bw_1h = bb_bw_engine.get_state("1h")
    #
    # # Multi-TF squeeze (very rare, very powerful)
    # if (bw_1m.bw_state == "COMPRESSED" and
    #     bw_5m.bw_state == "COMPRESSED" and
    #     bw_1h.bw_state == "COMPRESSED"):
    #     print("🔥 RARE: Multi-TF squeeze across 1m/5m/1h")
    #     print("→ Expect explosive breakout when compression releases")
    #
    # # Divergence warning (lower TF expanding, higher TF compressed)
    # elif (bw_1m.bw_state == "EXPANDING" and
    #       bw_5m.bw_state == "COMPRESSED"):
    #     print("⚠️  DIVERGENCE: 1m expanding but 5m still compressed")
    #     print("→ Potential false breakout or 5m about to follow")
    #
    # # Multi-TF exhaustion
    # elif (bw_1m.bw_state in ["EXTREME", "FADE_RISK"] and
    #       bw_5m.bw_state in ["EXTREME", "FADE_RISK"]):
    #     print("🛑 EXHAUSTION: Multi-TF extreme volatility")
    #     print("→ High reversal risk, avoid chasing")

    pass


# ============================================================================
# EXAMPLE 5: Bollinger Bands as Dynamic Support/Resistance
# ============================================================================


def example_bollinger_bands_sr():
    """
    Example: Use Bollinger Bands (upper/lower) as dynamic S/R levels.

    NOT for entries, but for understanding price boundaries.
    """

    # bw_state = bb_bw_engine.get_state("5m")
    # current_price = latest_candle.close
    #
    # if bw_state.upper and bw_state.lower and bw_state.mid:
    #     # Distance from bands
    #     dist_to_upper = (bw_state.upper - current_price) / current_price * 100
    #     dist_to_lower = (current_price - bw_state.lower) / current_price * 100
    #
    #     print(f"Bollinger Bands:")
    #     print(f"  Upper: ${bw_state.upper:.2f} ({dist_to_upper:+.2f}% away)")
    #     print(f"  Mid:   ${bw_state.mid:.2f}")
    #     print(f"  Lower: ${bw_state.lower:.2f} ({dist_to_lower:+.2f}% away)")
    #
    #     # Price at band extremes
    #     if current_price >= bw_state.upper * 0.99:
    #         print("  → Price at upper band (potential resistance)")
    #     elif current_price <= bw_state.lower * 1.01:
    #         print("  → Price at lower band (potential support)")

    pass


# ============================================================================
# EXAMPLE 6: Bandwidth Score for Ranking Pairs
# ============================================================================


def example_bandwidth_score_ranking():
    """
    Example: Use bw_score to rank trading pairs by setup quality.

    Higher score = better squeeze/breakout opportunity.
    """

    # # Collect scores for multiple pairs
    # pair_scores = {}
    # for pair in trading_pairs:
    #     bw_state = get_bandwidth_state(pair, "5m")
    #     if bw_state.bw_score_0_100 is not None:
    #         pair_scores[pair] = (bw_state.bw_score_0_100, bw_state.bw_state)
    #
    # # Sort by score
    # ranked = sorted(pair_scores.items(), key=lambda x: x[1][0], reverse=True)
    #
    # print("Top squeeze/breakout opportunities:")
    # for i, (pair, (score, state)) in enumerate(ranked[:5], 1):
    #     print(f"{i}. {pair}: score={score:.0f} state={state}")
    #
    # # Focus on low scores (squeeze building)
    # print("\nBest squeeze setups (lowest scores):")
    # for i, (pair, (score, state)) in enumerate(sorted(pair_scores.items(), key=lambda x: x[1][0])[:5], 1):
    #     if state == "COMPRESSED":
    #         print(f"{i}. {pair}: score={score:.0f} (tight squeeze)")

    pass


# ============================================================================
# EXAMPLE 7: Compact Display
# ============================================================================


def example_compact_display():
    """
    Example: Display Bollinger Bandwidth for multiple timeframes.

    Output format:
    BOLLINGER BW
    1m: bw=0.0123 ratio=0.74 state=COMPRESSED slope=-0.0012 score=18
    5m: bw=0.0198 ratio=1.28 state=EXPANDING  slope=+0.0021 score=73
    1h: bw=0.0410 ratio=1.67 state=EXTREME    slope=-0.0040 score=90
    """

    # Create engine
    config = BollingerBandwidthConfig(
        timeframes=["1m", "5m", "1h"],
        bb_period=20,
        bb_k=2.0,
        bw_sma_period=50,
    )
    engine = BollingerBandwidthEngine(config)

    # Simulate different volatility regimes

    # 1m: Compressed (tight range)
    price_1m = 100.0
    for i in range(80):
        price_1m += 0.05 if i % 2 == 0 else -0.03
        candle = Candle(i * 60000, price_1m, price_1m + 0.1, price_1m - 0.1, price_1m, 1000)
        engine.on_candle_close("1m", candle)

    # 5m: Expanding (breakout)
    price_5m = 100.0
    for i in range(80):
        if i < 50:
            price_5m += 0.1 if i % 2 == 0 else -0.05  # Start tight
        else:
            price_5m += 2.0 if i % 2 == 0 else -1.5  # Then expand
        candle = Candle(i * 300000, price_5m, price_5m + 1, price_5m - 1, price_5m, 1500)
        engine.on_candle_close("5m", candle)

    # 1h: Extreme (high volatility)
    price_1h = 100.0
    for i in range(80):
        if i < 50:
            price_1h += 0.1
        else:
            price_1h += 5.0 if i % 2 == 0 else -4.0  # Extreme moves
        candle = Candle(i * 3600000, price_1h, price_1h + 3, price_1h - 3, price_1h, 2000)
        engine.on_candle_close("1h", candle)

    # Get states
    states = {
        "1m": engine.get_state("1m"),
        "5m": engine.get_state("5m"),
        "1h": engine.get_state("1h"),
    }

    # Display
    print("\n" + "=" * 70)
    print_bollinger_bandwidth(states)
    print("=" * 70)

    # Interpret each
    for tf, state in states.items():
        if state is not None:
            print(f"\n{tf}: {interpret_bandwidth(state)}")


# ============================================================================
# EXAMPLE 8: Real-time Updates
# ============================================================================


def example_realtime_updates():
    """
    Example: Process real-time candle closes with O(1) bandwidth updates.
    """

    config = BollingerBandwidthConfig(
        timeframes=["1m"],
        bb_period=20,
        bw_sma_period=50,
    )
    engine = BollingerBandwidthEngine(config)

    print("\nSimulating real-time Bollinger Bandwidth updates:\n")

    price = 100.0
    for i in range(100):
        # Simulate volatility cycle: compress → expand → extreme → fade
        if i < 30:
            # Compression phase
            price += 0.05 if i % 2 == 0 else -0.03
        elif i < 50:
            # Expansion phase
            price += 1.5 if i % 2 == 0 else -1.0
        elif i < 70:
            # Extreme phase
            price += 3.0 if i % 2 == 0 else -2.5
        else:
            # Fade phase
            price += 0.8 if i % 2 == 0 else -0.5

        candle = Candle(i * 60000, price, price + 1, price - 1, price, 1000 + i * 10)

        # O(1) update
        state = engine.on_candle_close("1m", candle)

        # Print every 10th candle or when state changes
        if i % 10 == 0 or (i > 0 and state.bw_state != prev_state):
            print(f"Bar {i:3d}: {format_bandwidth_state('1m', state)}")
            if state.bw_state != "WARMUP" and i > 0 and state.bw_state != prev_state:
                print(f"         → State changed to {state.bw_state}")

        prev_state = state.bw_state if i > 0 else "WARMUP"


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == "__main__":
    print("\n╔" + "=" * 68 + "╗")
    print("║" + " " * 14 + "BOLLINGER BANDWIDTH INTEGRATION" + " " * 21 + "║")
    print("╚" + "=" * 68 + "╝\n")

    print("\n📊 EXAMPLE 7: Compact Display")
    example_compact_display()

    print("\n\n⏱️  EXAMPLE 8: Real-time Updates")
    example_realtime_updates()

    print("\n\n" + "=" * 70)
    print("✅ Integration Examples Complete!")
    print("=" * 70)
    print("\nKey Integration Points:")
    print("1. Continuous runner: Add BB Bandwidth engine for volatility regime")
    print("2. Batch analysis: Add bandwidth section after CHOP/ATR")
    print("3. Squeeze detection: COMPRESSED + CHOP + ATR squeeze = setup")
    print("4. Breakout confirmation: EXPANDING + trend + volume = entry")
    print("5. Multi-TF alignment: Squeeze/expansion across timeframes")
    print("6. Exhaustion warnings: EXTREME/FADE_RISK states")
    print("\n⚠️  REMEMBER: Bollinger Bandwidth is a VOLATILITY REGIME detector!")
    print("Combine with price action, momentum, and volume for complete analysis.")
    print()
