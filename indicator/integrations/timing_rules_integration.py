"""
Timing Rules Integration Examples

Shows how to integrate the Timing Rules Engine as the final decision layer
that consolidates all indicator outputs into ONE timing decision per candle.

CRITICAL: Timing Rules is the DECISION LAYER, NOT a standalone indicator.
It consumes outputs from trend, VWAP, momentum, ROC, ATR, CHOP, RSI, and MACD.
"""

from indicator.engines.timing_rules import (
    TimingRulesEngine,
    TimingRulesConfig,
    TimingDecision,
    print_timing_decisions,
    format_timing_decision,
    interpret_timing,
)


# ============================================================================
# EXAMPLE 1: Continuous Runner Integration
# ============================================================================

def example_continuous_integration():
    """
    Example: Integrate Timing Rules into continuous/orchestrator.py

    This is the FINAL decision layer that consolidates all indicators.
    """

    # In continuous/orchestrator.py __init__:
    # self.timing_engine = TimingRulesEngine(TimingRulesConfig(
    #     timeframes=["15s", "1m", "5m"],
    #     continuation_confirm_bars=2,
    #     exhaustion_confirm_bars=2,
    #     cooldown_bars_after_exhaustion=3,
    # ))

    # In _on_candle_close or compute_signals (LAST step):
    # # Gather all indicator outputs
    # timing_inputs = {
    #     # Trend (from EMA/ribbon/supertrend)
    #     "trend_ok": self.trend_ok[tf],
    #     "trend_bias": self.trend_bias[tf],
    #     "ribbon_width_rate": self.ribbon_width_rate[tf],
    #
    #     # VWAP
    #     "vwap_state": self.vwap_state[tf],
    #     "vwap_position": self.vwap_position[tf],
    #
    #     # ROC
    #     "roc_fast_norm": self.roc_fast_norm[tf],
    #     "roc_fast_norm_prev": self.roc_fast_norm_prev[tf],
    #     "acc_fast": self.roc_acc_fast[tf],
    #
    #     # ATR expansion
    #     "atr_exp": self.atr_exp[tf],
    #     "atr_exp_slope": self.atr_exp_slope[tf],
    #     "tr_spike": self.tr_spike[tf],
    #
    #     # CHOP
    #     "chop_value": self.chop_value[tf],
    #     "chop_state": self.chop_state[tf],
    #
    #     # RSI timing
    #     "rsi_divergence": self.rsi_divergence[tf],
    #     "structure_pivot_type": self.structure_pivot_type[tf],
    #
    #     # MACD histogram (optional)
    #     "macd_hist_slope": self.macd_hist_slope[tf],
    #     "macd_event": self.macd_event[tf],
    # }
    #
    # # Get timing decision
    # timing_decision = self.timing_engine.on_candle_close(tf, timing_inputs)
    #
    # # Use for execution decisions
    # if timing_decision.timing_state == "CONTINUATION_READY":
    #     if timing_decision.direction == 1 and timing_decision.confidence_0_100 >= 70:
    #         # High conviction continuation long
    #         self.prepare_long_entry(tf, size_multiplier=1.2)
    #
    # elif timing_decision.timing_state == "BREAKOUT_WINDOW":
    #     if timing_decision.confidence_0_100 >= 75:
    #         # Breakout opportunity (use wider stop)
    #         self.prepare_breakout_entry(tf, timing_decision.direction)
    #
    # elif timing_decision.timing_state == "EXHAUSTION_WARNING":
    #     # Don't chase, prepare for reversal
    #     self.close_positions(tf, reason="exhaustion_warning")
    #     self.prepare_counter_trend_setup(tf, -timing_decision.direction)

    pass


# ============================================================================
# EXAMPLE 2: Batch Analysis Integration (runner.py)
# ============================================================================

def example_batch_integration():
    """
    Example: Integrate Timing Rules into runner.py analyze_pair()

    Add as FINAL section after all indicators.
    """

    # In runner.py after all indicator sections:

    # print_section("TIMING DECISION - Consolidated Rules", "⚡")
    #
    # # Initialize timing engine
    # timing_config = TimingRulesConfig(
    #     timeframes=[timeframe],
    #     continuation_confirm_bars=2,
    #     exhaustion_confirm_bars=2,
    # )
    # timing_engine = TimingRulesEngine(timing_config)
    #
    # # Gather all indicator outputs (from previous sections)
    # # NOTE: This requires all indicators to store their outputs
    # timing_inputs = {
    #     "trend_ok": trend_ok,  # from trend analysis
    #     "trend_bias": trend_bias,
    #     "vwap_state": vwap_state,  # from VWAP section
    #     "vwap_position": vwap_position,
    #     "roc_fast_norm": roc_fast_norm,  # from ROC section
    #     "roc_fast_norm_prev": roc_fast_norm_prev,
    #     "atr_exp": atr_exp,  # from ATR expansion
    #     "chop_state": chop_state,  # from CHOP
    #     "rsi_divergence": rsi_divergence,  # from RSI timing
    #     "macd_hist_slope": macd_hist_slope,  # from MACD
    # }
    #
    # # Get timing decision (no warmup needed for batch analysis)
    # decision = timing_engine.on_candle_close(timeframe, timing_inputs)
    #
    # # Display
    # print(format_timing_decision(timeframe, decision))
    # print()
    #
    # # Interpret
    # interpretation = interpret_timing(decision)
    # if decision.timing_state == "CONTINUATION_READY":
    #     print(f"  {Colors.GREEN}✅ {interpretation}{Colors.RESET}")
    # elif decision.timing_state == "BREAKOUT_WINDOW":
    #     print(f"  {Colors.YELLOW}⚡ {interpretation}{Colors.RESET}")
    # elif decision.timing_state == "EXHAUSTION_WARNING":
    #     print(f"  {Colors.RED}⚠️  {interpretation}{Colors.RESET}")
    # else:
    #     print(f"  {Colors.CYAN}ℹ️  {interpretation}{Colors.RESET}")
    #
    # # Show detailed checklist
    # print(f"\n  Checklist:")
    # for key, val in decision.checklist.items():
    #     status = "✓" if val else "✗"
    #     print(f"    {status} {key}")
    #
    # print()

    pass


# ============================================================================
# EXAMPLE 3: Multi-Timeframe Alignment
# ============================================================================

def example_multi_timeframe_alignment():
    """
    Example: Use timing decisions across multiple timeframes for confluence.

    Strongest signals when timing states align.
    """

    # Get timing decisions for all timeframes
    # timing_1m = timing_engine.get_state("1m")
    # timing_5m = timing_engine.get_state("5m")
    # timing_1h = timing_engine.get_state("1h")
    #
    # # Check for confluence
    # if (timing_1m.timing_state == "CONTINUATION_READY" and
    #     timing_5m.timing_state == "CONTINUATION_READY" and
    #     timing_1m.direction == timing_5m.direction == 1):
    #     print("🚀 STRONG CONFLUENCE: Multi-TF continuation ready LONG")
    #     print(f"   1m conf={timing_1m.confidence_0_100:.0f}%, 5m conf={timing_5m.confidence_0_100:.0f}%")
    #     # High conviction entry
    #
    # elif (timing_1m.timing_state == "BREAKOUT_WINDOW" and
    #       timing_5m.timing_state == "CONTINUATION_READY" and
    #       timing_1m.direction == timing_5m.direction):
    #     print("⚡ CONFLUENCE: 1m breakout + 5m continuation")
    #     # Breakout with trend support
    #
    # elif (timing_1m.timing_state == "EXHAUSTION_WARNING" or
    #       timing_5m.timing_state == "EXHAUSTION_WARNING"):
    #     print("⚠️  WARNING: Exhaustion detected on at least one timeframe")
    #     # Avoid entries, consider exits

    pass


# ============================================================================
# EXAMPLE 4: Position Sizing Based on Confidence
# ============================================================================

def example_position_sizing():
    """
    Example: Scale position size based on timing confidence and score.
    """

    # decision = timing_engine.get_state("1m")
    #
    # if decision.timing_state in ["CONTINUATION_READY", "BREAKOUT_WINDOW"]:
    #     confidence = decision.confidence_0_100
    #     score = decision.timing_score_0_100 or 50
    #
    #     # Base size
    #     base_size = 100.0  # USD
    #
    #     # Scale by confidence
    #     if confidence >= 85:
    #         size_multiplier = 1.5  # High conviction
    #     elif confidence >= 70:
    #         size_multiplier = 1.2  # Good confidence
    #     elif confidence >= 60:
    #         size_multiplier = 1.0  # Standard
    #     else:
    #         size_multiplier = 0.7  # Reduced
    #
    #     # Scale by timing score
    #     if score >= 80:
    #         size_multiplier *= 1.2  # Excellent timing
    #     elif score < 50:
    #         size_multiplier *= 0.8  # Marginal timing
    #
    #     final_size = base_size * size_multiplier
    #     print(f"Position size: ${final_size:.2f} (conf={confidence:.0f}%, score={score:.0f})")

    pass


# ============================================================================
# EXAMPLE 5: Compact Display
# ============================================================================

def example_compact_display():
    """
    Example: Display timing decisions for multiple timeframes.

    Output format:
    TIMING
    1m: state=CONTINUATION_READY dir=+1 conf=78 score=72 reasons=[trend_ok,vwap_reclaim,roc_turn_up,atr_exp>=1.0]
    5m: state=EXHAUSTION_WARNING dir=+1 conf=81 score=40 reasons=[roc_decelerating,blowoff,rsi_reg_div]
    1h: state=BREAKOUT_WINDOW dir=-1 conf=88 score=84 reasons=[squeeze_armed,atr_exp>=1.2,roc_surge]
    """

    # Create engine
    config = TimingRulesConfig(
        timeframes=["1m", "5m", "1h"],
        continuation_confirm_bars=2,
        exhaustion_confirm_bars=2,
        breakout_confirm_bars=2,
    )
    engine = TimingRulesEngine(config)

    # Simulate different scenarios per timeframe

    # 1m: Continuation ready
    inputs_1m = {
        "trend_ok": True,
        "trend_bias": 1,
        "vwap_state": "RECLAIM",
        "vwap_position": "ABOVE",
        "roc_fast_norm": 0.45,
        "roc_fast_norm_prev": 0.30,
        "atr_exp": 1.10,
    }

    # 5m: Exhaustion warning
    inputs_5m = {
        "trend_bias": 1,
        "roc_fast_norm": 0.50,
        "acc_fast": -0.20,
        "atr_exp": 1.70,
        "ribbon_width_rate": -0.05,
    }

    # 1h: Breakout window
    # First arm the setup
    inputs_1h_setup = {
        "atr_exp": 0.75,
        "chop_state": "CHOP",
    }

    # Then ignition
    inputs_1h_ignite = {
        "atr_exp": 1.25,
        "roc_fast_norm": -0.85,
    }

    # Feed inputs
    decisions = {}

    # 1m
    for _ in range(2):
        decisions["1m"] = engine.on_candle_close("1m", inputs_1m)

    # 5m
    for _ in range(2):
        decisions["5m"] = engine.on_candle_close("5m", inputs_5m)

    # 1h (setup + ignite)
    engine.on_candle_close("1h", inputs_1h_setup)
    for _ in range(2):
        decisions["1h"] = engine.on_candle_close("1h", inputs_1h_ignite)

    # Display
    print("\n" + "=" * 70)
    print_timing_decisions(decisions)
    print("=" * 70)

    # Interpret each
    for tf, decision in decisions.items():
        print(f"\n{tf}: {interpret_timing(decision)}")
        if decision.timing_score_0_100 is not None:
            score_level = decision.debug.get("score_level", "N/A")
            print(f"   → Timing score: {decision.timing_score_0_100:.0f}/100 ({score_level})")


# ============================================================================
# EXAMPLE 6: Real-time Decision Updates
# ============================================================================

def example_realtime_updates():
    """
    Example: Process real-time candle closes with O(1) timing decisions.
    """

    config = TimingRulesConfig(
        timeframes=["1m"],
        continuation_confirm_bars=2,
    )
    engine = TimingRulesEngine(config)

    # Simulate real-time updates
    print("\nSimulating real-time timing decisions:\n")

    # Start with no setup
    for i in range(3):
        inputs = {"trend_ok": False}
        decision = engine.on_candle_close("1m", inputs)
        print(f"Bar {i:2d}: {format_timing_decision('1m', decision)}")

    # Trend emerges
    print("\n[Trend emerges]")
    for i in range(3, 6):
        inputs = {
            "trend_ok": True,
            "trend_bias": 1,
            "vwap_state": "ACCEPT",
            "vwap_position": "ABOVE",
            "roc_fast_norm": 0.40,
            "roc_fast_norm_prev": 0.35,
            "atr_exp": 1.05,
        }
        decision = engine.on_candle_close("1m", inputs)
        print(f"Bar {i:2d}: {format_timing_decision('1m', decision)}")
        if decision.timing_state == "CONTINUATION_READY":
            print(f"       → {interpret_timing(decision)}")

    # Exhaustion forms
    print("\n[Exhaustion forms]")
    for i in range(6, 9):
        inputs = {
            "trend_bias": 1,
            "roc_fast_norm": 0.50,
            "acc_fast": -0.25,
            "atr_exp": 1.65,
            "ribbon_width_rate": -0.08,
        }
        decision = engine.on_candle_close("1m", inputs)
        print(f"Bar {i:2d}: {format_timing_decision('1m', decision)}")
        if decision.timing_state == "EXHAUSTION_WARNING":
            print(f"       → {interpret_timing(decision)}")

    # Cooldown period
    print("\n[Cooldown - no continuation allowed]")
    for i in range(9, 12):
        inputs = {
            "trend_ok": True,
            "trend_bias": 1,
            "vwap_state": "ACCEPT",
            "vwap_position": "ABOVE",
            "roc_fast_norm": 0.40,
            "roc_fast_norm_prev": 0.30,
            "atr_exp": 1.05,
        }
        decision = engine.on_candle_close("1m", inputs)
        print(f"Bar {i:2d}: {format_timing_decision('1m', decision)}")


# ============================================================================
# Run Examples
# ============================================================================

if __name__ == "__main__":
    print("\n╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "TIMING RULES INTEGRATION" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝\n")

    print("\n📊 EXAMPLE 5: Compact Display")
    example_compact_display()

    print("\n\n⏱️  EXAMPLE 6: Real-time Updates")
    example_realtime_updates()

    print("\n\n" + "=" * 70)
    print("✅ Integration Examples Complete!")
    print("=" * 70)
    print("\nKey Integration Points:")
    print("1. Continuous runner: Add timing engine as FINAL decision layer")
    print("2. Batch analysis: Add timing section AFTER all indicators")
    print("3. Multi-TF confluence: Align timing states across timeframes")
    print("4. Position sizing: Scale by confidence and timing score")
    print("5. Real-time updates: O(1) decisions with confirmation counters")
    print("\n⚠️  REMEMBER: Timing Rules is the DECISION LAYER!")
    print("It consolidates ALL indicator outputs into ONE timing decision.")
    print("Priority: EXHAUSTION > BREAKOUT > CONTINUATION > TRANSITION/NO_TRADE")
    print()
