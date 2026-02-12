"""
ATR Expansion Integration Snippet

Shows how to integrate ATR expansion timing gates into the continuous runner
and batch analysis workflows.
"""

# ============================================================================
# INTEGRATION 1: CONTINUOUS RUNNER
# ============================================================================


def integrate_continuous_runner_example():
    """
    Example integration with continuous runner.

    Add this to continuous/orchestrator.py or create a new adapter.
    """

    from indicator.engines.atr_expansion import ATRExpansionConfig, ATRExpansionEngine, Candle

    # Initialize engine
    config = ATRExpansionConfig(
        timeframes=["15s", "1m", "5m"],  # Match your trading timeframes
        atr_period=14,
        sma_period=20,
    )
    atr_engine = ATRExpansionEngine(config)

    # === ON CANDLE CLOSE ===
    def on_candle_close(timeframe: str, ohlcv_data: dict):
        """Called when a candle closes for a given timeframe."""

        # Convert to Candle object
        candle = Candle(
            timestamp=ohlcv_data["timestamp"],
            open=ohlcv_data["open"],
            high=ohlcv_data["high"],
            low=ohlcv_data["low"],
            close=ohlcv_data["close"],
            volume=ohlcv_data["volume"],
        )

        # Update ATR expansion state
        atr_state = atr_engine.on_candle_close(timeframe, candle)

        # === TIMING GATES ===

        # Gate 1: Only allow breakout attempts during expansion
        if atr_state.vol_state in ["EXPANSION", "EXTREME"]:
            print(f"[{timeframe}] Volatility expanding - breakout window open")
            # Enable breakout validation here

        # Gate 2: Require shock for immediate entries
        if atr_state.debug.get("shock_now"):
            print(f"[{timeframe}] TR SHOCK detected - move starting NOW")
            # Fast entry signal

        # Gate 3: Avoid entries during SQUEEZE (choppy, low conviction)
        if atr_state.vol_state == "SQUEEZE":
            print(f"[{timeframe}] SQUEEZE - low volatility, avoid trades")
            # Disable entries

        # Gate 4: FADE_RISK warning (expansion losing steam)
        if atr_state.vol_state == "FADE_RISK":
            print(f"[{timeframe}] FADE_RISK - expansion slowing, tighten stops")
            # Consider taking profits or tightening stops

        # Gate 5: Use vol_score as confidence multiplier
        if atr_state.vol_score_0_100 is not None:
            vol_confidence = atr_state.vol_score_0_100 / 100.0
            # Scale position size or stop distance by vol_confidence
            print(
                f"[{timeframe}] Vol score: {atr_state.vol_score_0_100:.0f}/100 (conf={vol_confidence:.2f})"
            )

        return atr_state

    # === WARMUP ON STARTUP ===
    def warmup_on_startup(historical_candles: dict):
        """Warmup engine with historical data."""
        # historical_candles = {"1m": [candle1, candle2, ...], "5m": [...]}

        states = atr_engine.warmup(historical_candles)

        print("\n=== ATR EXPANSION WARMUP COMPLETE ===")
        for tf, state in states.items():
            print(f"{tf}: {state.vol_state} (score={state.vol_score_0_100})")

        return states


# ============================================================================
# INTEGRATION 2: BATCH ANALYSIS (runner.py / analyze.py)
# ============================================================================


def integrate_batch_analysis_example():
    """
    Example integration with batch analysis (analyze.py).

    Add this after fetching klines in runner.py analyze_pair().
    """

    from indicator.engines.atr_expansion import (
        ATRExpansionConfig,
        ATRExpansionEngine,
        Candle,
        print_atr_expansion,
    )

    # After fetching klines in analyze_pair():
    # klines: List[OHLCVData] = data['klines']
    # Convert to Candle objects
    candles = []
    for k in klines:
        candle = Candle(
            timestamp=k.timestamp,
            open=k.open,
            high=k.high,
            low=k.low,
            close=k.close,
            volume=k.volume,
        )
        candles.append(candle)

    # Initialize and warmup engine
    config = ATRExpansionConfig(
        timeframes=["1h"],  # Match analysis timeframe
        atr_period=14,
        sma_period=20,
    )
    atr_engine = ATRExpansionEngine(config)

    states = atr_engine.warmup({"1h": candles})

    # Display ATR expansion analysis
    print_atr_expansion(states)

    # Use state for decision logic
    atr_state = states["1h"]

    if atr_state.vol_state == "EXPANSION":
        print("\n⚡ TIMING: Volatility expanding - good time for breakout attempts")
    elif atr_state.vol_state == "SQUEEZE":
        print("\n⏸️  TIMING: Low volatility squeeze - wait for expansion")
    elif atr_state.vol_state == "EXTREME":
        print("\n🔥 TIMING: Extreme volatility - move is ON, tight risk management required")
    elif atr_state.vol_state == "FADE_RISK":
        print("\n⚠️  TIMING: Expansion fading - consider taking profits")


# ============================================================================
# INTEGRATION 3: MULTI-TIMEFRAME CONFIRMATION
# ============================================================================


def multi_timeframe_timing_gate_example():
    """
    Use ATR expansion across multiple timeframes for confirmation.

    Example: Only enter if BOTH 1m and 5m are expanding.
    """

    from indicator.engines.atr_expansion import ATRExpansionConfig, ATRExpansionEngine

    config = ATRExpansionConfig(timeframes=["1m", "5m", "15m"])
    atr_engine = ATRExpansionEngine(config)

    # ... after warmup or on_candle_close ...

    state_1m = atr_engine.get_state("1m")
    state_5m = atr_engine.get_state("5m")
    state_15m = atr_engine.get_state("15m")

    # === CONFIRMATION LOGIC ===

    # Require alignment across timeframes
    expansion_states = ["EXPANSION", "EXTREME"]

    all_expanding = (
        state_1m
        and state_1m.vol_state in expansion_states
        and state_5m
        and state_5m.vol_state in expansion_states
    )

    if all_expanding:
        print("✅ MULTI-TF CONFIRMATION: 1m + 5m both expanding - STRONG timing")
        # High-confidence entry

    # Divergence warning
    if state_1m and state_5m:
        if state_1m.vol_state == "EXPANSION" and state_5m.vol_state == "SQUEEZE":
            print("⚠️  TF DIVERGENCE: 1m expanding but 5m squeezed - wait for 5m confirmation")

        # Progressive expansion (ideal)
        if (
            state_15m
            and state_15m.vol_state == "EXPANSION"
            and state_5m.vol_state == "EXPANSION"
            and state_1m.vol_state == "EXTREME"
        ):
            print("🚀 PROGRESSIVE EXPANSION: 15m→5m→1m cascade - OPTIMAL timing")


# ============================================================================
# INTEGRATION 4: DISPLAY IN DEEP-DIVE MODE
# ============================================================================


def integrate_deep_dive_display():
    """
    Add ATR expansion display to continuous_runner.py deep-dive mode.

    Insert this in DeepDiveDisplay.print_deep_dive() after unified score.
    """

    code_snippet = """
    # In continuous_runner.py DeepDiveDisplay.print_deep_dive():

    # After printing unified score...

    # Get ATR expansion states
    if hasattr(analyzer, 'atr_engine'):
        atr_states = {
            "15s": analyzer.atr_engine.get_state("15s"),
            "1m": analyzer.atr_engine.get_state("1m"),
            "5m": analyzer.atr_engine.get_state("5m"),
        }

        # Filter out None states
        atr_states = {tf: state for tf, state in atr_states.items() if state}

        if atr_states:
            from indicator.engines.atr_expansion import print_atr_expansion
            print_atr_expansion(atr_states)
    """

    print(code_snippet)


# ============================================================================
# INTEGRATION 5: COMPACT PRINT EXAMPLE
# ============================================================================


def example_compact_output():
    """
    Example of what the compact print output looks like.
    """

    print(
        """
Example Output:

ATR EXPANSION
1m: state=EXPANSION score=72 atrp=0.22% atr_exp=1.31 slope=+0.07 TR_spike=1.62 shock=YES
5m: state=SQUEEZE  score=18 atrp=0.15% atr_exp=0.74 slope=-0.02 TR_spike=0.88 shock=NO
1h: state=FADE_RISK score=63 atr_exp=1.25 slope=-0.08 TR_spike=1.12 shock=NO

Interpretation:
- 1m: Volatility expanding with shock - move is starting NOW
- 5m: Still in squeeze - wait for 5m confirmation before large position
- 1h: Expansion losing momentum - consider scaling out

Trading Decision:
- Timing: ✅ Good (1m expansion + shock)
- Confirmation: ⚠️ Partial (5m not yet expanded)
- Action: Small entry on 1m, add to position when 5m confirms
"""
    )


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=== ATR EXPANSION INTEGRATION EXAMPLES ===\n")

    print("1. Continuous Runner Integration:")
    print("-" * 60)
    integrate_continuous_runner_example()

    print("\n2. Batch Analysis Integration:")
    print("-" * 60)
    integrate_batch_analysis_example()

    print("\n3. Multi-Timeframe Confirmation:")
    print("-" * 60)
    multi_timeframe_timing_gate_example()

    print("\n4. Deep-Dive Display Integration:")
    print("-" * 60)
    integrate_deep_dive_display()

    print("\n5. Example Output:")
    print("-" * 60)
    example_compact_output()

    print("\n" + "=" * 60)
    print("Integration complete! See examples above for usage patterns.")
    print("=" * 60)
