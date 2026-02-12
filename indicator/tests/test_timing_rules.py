"""
Tests for Timing Rules Engine

Verifies:
- 2-of-3 momentum triggers for continuation
- 2-of-3 exhaustion conditions
- Cooldown blocking
- Squeeze-break two-step logic
- Priority ordering (exhaustion > breakout > continuation)
- Missing data robustness
- Confirmation counters preventing flips
"""

import pytest

from timing_rules import TimingDecision, TimingRulesConfig, TimingRulesEngine

# ============================================================================
# TEST 1: CONTINUATION TRIGGERS (2-of-3)
# ============================================================================


def test_continuation_2of3_roc_atr():
    """ROC restart + ATR support (2 of 3) triggers continuation."""
    config = TimingRulesConfig(
        timeframes=["1m"],
        continuation_confirm_bars=2,
        use_macd_optional=False,  # Disable MACD to test 2-of-2
    )
    engine = TimingRulesEngine(config)

    inputs = {
        "trend_ok": True,
        "trend_bias": 1,
        "vwap_state": "ACCEPT",
        "vwap_position": "ABOVE",
        "roc_fast_norm": 0.45,
        "roc_fast_norm_prev": 0.30,
        "atr_exp": 1.10,
        "atr_exp_slope": 0.05,
    }

    # First bar
    decision = engine.on_candle_close("1m", inputs)
    assert decision.timing_state != "CONTINUATION_READY"  # Not confirmed yet

    # Second bar (confirmation)
    decision = engine.on_candle_close("1m", inputs)
    assert decision.timing_state == "CONTINUATION_READY"
    assert decision.direction == 1
    assert decision.confidence_0_100 >= 60  # 50 + 15*2
    assert "roc_turn_up" in decision.reasons
    assert "atr_exp>=1.0" in decision.reasons


def test_continuation_2of3_with_macd():
    """ROC restart + MACD support (2 of 3) triggers continuation."""
    config = TimingRulesConfig(
        timeframes=["1m"],
        continuation_confirm_bars=2,
        use_macd_optional=True,
    )
    engine = TimingRulesEngine(config)

    inputs = {
        "trend_ok": True,
        "trend_bias": 1,
        "vwap_state": "RECLAIM",
        "vwap_position": "ABOVE",
        "roc_fast_norm": 0.45,
        "roc_fast_norm_prev": 0.30,
        "atr_exp": 0.85,  # Below threshold
        "macd_hist_slope": 0.05,  # Positive for bull
    }

    # Confirm
    for _ in range(2):
        decision = engine.on_candle_close("1m", inputs)

    assert decision.timing_state == "CONTINUATION_READY"
    assert decision.direction == 1
    assert "roc_turn_up" in decision.reasons
    assert "macd_aligned" in decision.reasons
    assert "vwap_reclaim" in decision.reasons


def test_continuation_requires_all_gates():
    """Continuation fails if any gate is false."""
    config = TimingRulesConfig(timeframes=["1m"], continuation_confirm_bars=1)
    engine = TimingRulesEngine(config)

    # Missing trend_ok
    inputs = {
        "trend_ok": False,
        "trend_bias": 1,
        "vwap_state": "ACCEPT",
        "vwap_position": "ABOVE",
        "roc_fast_norm": 0.45,
        "roc_fast_norm_prev": 0.30,
        "atr_exp": 1.10,
    }
    decision = engine.on_candle_close("1m", inputs)
    assert decision.timing_state != "CONTINUATION_READY"

    # Missing vwap_state
    inputs["trend_ok"] = True
    inputs["vwap_state"] = "REJECT"
    decision = engine.on_candle_close("1m", inputs)
    assert decision.timing_state != "CONTINUATION_READY"

    # Misaligned vwap_position
    inputs["vwap_state"] = "ACCEPT"
    inputs["vwap_position"] = "BELOW"  # Conflict with bull bias
    decision = engine.on_candle_close("1m", inputs)
    assert decision.timing_state != "CONTINUATION_READY"


def test_continuation_bear_bias():
    """Continuation works for bearish bias."""
    config = TimingRulesConfig(timeframes=["1m"], continuation_confirm_bars=2)
    engine = TimingRulesEngine(config)

    inputs = {
        "trend_ok": True,
        "trend_bias": -1,
        "vwap_state": "ACCEPT",
        "vwap_position": "BELOW",
        "roc_fast_norm": -0.45,
        "roc_fast_norm_prev": -0.30,
        "atr_exp": 1.10,
    }

    for _ in range(2):
        decision = engine.on_candle_close("1m", inputs)

    assert decision.timing_state == "CONTINUATION_READY"
    assert decision.direction == -1
    assert "roc_turn_down" in decision.reasons


# ============================================================================
# TEST 2: EXHAUSTION WARNING (2-of-3)
# ============================================================================


def test_exhaustion_2of3_roc_blowoff():
    """ROC decelerating + blowoff triggers exhaustion."""
    config = TimingRulesConfig(
        timeframes=["1m"],
        exhaustion_confirm_bars=2,
    )
    engine = TimingRulesEngine(config)

    inputs = {
        "trend_bias": 1,
        "roc_fast_norm": 0.50,
        "acc_fast": -0.20,  # Decelerating
        "atr_exp": 1.70,  # Blowoff
        "ribbon_width_rate": -0.05,  # Compressing
    }

    # First bar
    decision = engine.on_candle_close("1m", inputs)
    assert decision.timing_state != "EXHAUSTION_WARNING"

    # Second bar (confirmation)
    decision = engine.on_candle_close("1m", inputs)
    assert decision.timing_state == "EXHAUSTION_WARNING"
    assert decision.direction == 1
    assert decision.confidence_0_100 >= 70  # 60 + 20*(2-2) + 10
    assert "roc_decelerating" in decision.reasons
    assert "blowoff" in decision.reasons


def test_exhaustion_2of3_rsi_div():
    """Blowoff + RSI divergence triggers exhaustion."""
    config = TimingRulesConfig(timeframes=["1m"], exhaustion_confirm_bars=2)
    engine = TimingRulesEngine(config)

    inputs = {
        "trend_bias": 1,
        "atr_exp": 1.25,  # Hot
        "tr_spike": 1.60,  # Hot
        "ribbon_width_rate": -0.10,  # Compressing
        "rsi_divergence": "REG_BEAR",
        "structure_pivot_type": "SWING_HIGH",
    }

    for _ in range(2):
        decision = engine.on_candle_close("1m", inputs)

    assert decision.timing_state == "EXHAUSTION_WARNING"
    assert "blowoff" in decision.reasons
    assert "rsi_reg_div" in decision.reasons


def test_exhaustion_bear_context():
    """Exhaustion works for bearish context."""
    config = TimingRulesConfig(timeframes=["1m"], exhaustion_confirm_bars=2)
    engine = TimingRulesEngine(config)

    inputs = {
        "trend_bias": -1,
        "roc_fast_norm": -0.50,
        "acc_fast": 0.20,  # Decelerating (acc positive in downtrend)
        "atr_exp": 1.70,
        "ribbon_width_rate": -0.05,
    }

    for _ in range(2):
        decision = engine.on_candle_close("1m", inputs)

    assert decision.timing_state == "EXHAUSTION_WARNING"
    assert decision.direction == -1


def test_exhaustion_requires_trend_context():
    """Exhaustion not evaluated when trend_bias == 0."""
    config = TimingRulesConfig(timeframes=["1m"], exhaustion_confirm_bars=1)
    engine = TimingRulesEngine(config)

    inputs = {
        "trend_bias": 0,  # No trend
        "roc_fast_norm": 0.50,
        "acc_fast": -0.20,
        "atr_exp": 1.70,
        "ribbon_width_rate": -0.05,
    }

    decision = engine.on_candle_close("1m", inputs)
    assert decision.timing_state != "EXHAUSTION_WARNING"


# ============================================================================
# TEST 3: COOLDOWN BLOCKS CONTINUATION
# ============================================================================


def test_cooldown_blocks_continuation():
    """After exhaustion fires, continuation blocked until cooldown expires."""
    config = TimingRulesConfig(
        timeframes=["1m"],
        exhaustion_confirm_bars=2,
        continuation_confirm_bars=2,
        cooldown_bars_after_exhaustion=3,
    )
    engine = TimingRulesEngine(config)

    # Fire exhaustion
    ex_inputs = {
        "trend_bias": 1,
        "roc_fast_norm": 0.50,
        "acc_fast": -0.20,
        "atr_exp": 1.70,
        "ribbon_width_rate": -0.05,
    }

    for _ in range(2):
        decision = engine.on_candle_close("1m", ex_inputs)
    assert decision.timing_state == "EXHAUSTION_WARNING"

    # Try continuation immediately (should fail due to cooldown)
    cont_inputs = {
        "trend_ok": True,
        "trend_bias": 1,
        "vwap_state": "ACCEPT",
        "vwap_position": "ABOVE",
        "roc_fast_norm": 0.45,
        "roc_fast_norm_prev": 0.30,
        "atr_exp": 1.10,
    }

    for i in range(5):
        decision = engine.on_candle_close("1m", cont_inputs)
        if i < 3:  # Cooldown bars
            assert decision.timing_state != "CONTINUATION_READY"
        else:  # After cooldown
            if i >= 4:  # After confirmation bars
                assert decision.timing_state == "CONTINUATION_READY"


# ============================================================================
# TEST 4: SQUEEZE-BREAK TWO-STEP
# ============================================================================


def test_breakout_two_step():
    """Squeeze setup arms, then ignition fires."""
    config = TimingRulesConfig(
        timeframes=["1m"],
        breakout_confirm_bars=2,
        breakout_requires_roc_surge=True,
    )
    engine = TimingRulesEngine(config)

    # Step 1: Setup (compression)
    setup_inputs = {
        "atr_exp": 0.75,
        "chop_state": "CHOP",
    }

    decision = engine.on_candle_close("1m", setup_inputs)
    assert decision.timing_state != "BREAKOUT_WINDOW"
    assert decision.checklist.get("brk_squeeze_setup") is True
    assert "squeeze_armed" in decision.reasons

    # Step 2: Ignition
    ignite_inputs = {
        "atr_exp": 1.25,
        "roc_fast_norm": 0.85,
    }

    # First ignition bar
    decision = engine.on_candle_close("1m", ignite_inputs)
    assert decision.timing_state != "BREAKOUT_WINDOW"  # Not confirmed yet

    # Second ignition bar (confirmation)
    decision = engine.on_candle_close("1m", ignite_inputs)
    assert decision.timing_state == "BREAKOUT_WINDOW"
    assert decision.direction == 1  # Positive ROC
    assert decision.confidence_0_100 >= 75  # 55 + 20 + 20
    assert "roc_surge" in decision.reasons


def test_breakout_without_setup_fails():
    """Ignition without prior setup does not fire."""
    config = TimingRulesConfig(timeframes=["1m"], breakout_confirm_bars=1)
    engine = TimingRulesEngine(config)

    # No setup, just ignition
    ignite_inputs = {
        "atr_exp": 1.25,
        "roc_fast_norm": 0.85,
    }

    decision = engine.on_candle_close("1m", ignite_inputs)
    assert decision.timing_state != "BREAKOUT_WINDOW"


def test_breakout_chop_value_threshold():
    """Setup can use chop_value threshold instead of chop_state."""
    config = TimingRulesConfig(timeframes=["1m"], breakout_confirm_bars=2)
    engine = TimingRulesEngine(config)

    # Setup with chop_value
    setup_inputs = {
        "atr_exp": 0.75,
        "chop_value": 65.0,  # Above chop_high_thr (61.8)
    }

    decision = engine.on_candle_close("1m", setup_inputs)
    assert decision.checklist.get("brk_squeeze_setup") is True

    # Ignition
    ignite_inputs = {
        "atr_exp": 1.25,
        "roc_fast_norm": 0.85,
    }

    for _ in range(2):
        decision = engine.on_candle_close("1m", ignite_inputs)

    assert decision.timing_state == "BREAKOUT_WINDOW"


def test_breakout_tr_spike_alternative():
    """TR spike can substitute for atr_exp in ignition."""
    config = TimingRulesConfig(timeframes=["1m"], breakout_confirm_bars=2)
    engine = TimingRulesEngine(config)

    # Setup
    setup_inputs = {"atr_exp": 0.75, "chop_state": "CHOP"}
    engine.on_candle_close("1m", setup_inputs)

    # Ignition with TR spike
    ignite_inputs = {
        "tr_spike": 1.60,  # Above tr_spike_break_thr (1.50)
        "roc_fast_norm": 0.85,
    }

    for _ in range(2):
        decision = engine.on_candle_close("1m", ignite_inputs)

    assert decision.timing_state == "BREAKOUT_WINDOW"
    assert decision.checklist.get("brk_vol_ignite") is True


# ============================================================================
# TEST 5: PRIORITY ORDERING
# ============================================================================


def test_priority_exhaustion_over_continuation():
    """Exhaustion fires even if continuation conditions met."""
    config = TimingRulesConfig(
        timeframes=["1m"],
        exhaustion_confirm_bars=2,
        continuation_confirm_bars=2,
    )
    engine = TimingRulesEngine(config)

    # Inputs satisfying both exhaustion and continuation
    inputs = {
        # Continuation gates
        "trend_ok": True,
        "trend_bias": 1,
        "vwap_state": "ACCEPT",
        "vwap_position": "ABOVE",
        "roc_fast_norm": 0.50,
        "roc_fast_norm_prev": 0.30,
        "atr_exp": 1.70,  # Also triggers blowoff
        # Exhaustion conditions
        "acc_fast": -0.20,  # Decelerating
        "ribbon_width_rate": -0.05,  # Compressing
    }

    for _ in range(2):
        decision = engine.on_candle_close("1m", inputs)

    # Exhaustion should win
    assert decision.timing_state == "EXHAUSTION_WARNING"
    assert decision.timing_state != "CONTINUATION_READY"


def test_priority_exhaustion_over_breakout():
    """Exhaustion fires even if breakout conditions met."""
    config = TimingRulesConfig(timeframes=["1m"], exhaustion_confirm_bars=2)
    engine = TimingRulesEngine(config)

    # Arm breakout
    engine.on_candle_close("1m", {"atr_exp": 0.75, "chop_state": "CHOP"})

    # Inputs satisfying both exhaustion and breakout ignition
    inputs = {
        "trend_bias": 1,
        "roc_fast_norm": 0.85,  # Surge
        "acc_fast": -0.20,  # Decelerating
        "atr_exp": 1.70,  # Blowoff + ignition
        "ribbon_width_rate": -0.05,
    }

    for _ in range(2):
        decision = engine.on_candle_close("1m", inputs)

    assert decision.timing_state == "EXHAUSTION_WARNING"


def test_priority_breakout_over_continuation():
    """Breakout fires if armed, even if continuation met."""
    config = TimingRulesConfig(
        timeframes=["1m"],
        breakout_confirm_bars=2,
        continuation_confirm_bars=2,
    )
    engine = TimingRulesEngine(config)

    # Arm breakout
    engine.on_candle_close("1m", {"atr_exp": 0.75, "chop_state": "CHOP"})

    # Inputs satisfying both breakout and continuation
    inputs = {
        "trend_ok": True,
        "trend_bias": 1,
        "vwap_state": "ACCEPT",
        "vwap_position": "ABOVE",
        "roc_fast_norm": 0.85,
        "roc_fast_norm_prev": 0.30,
        "atr_exp": 1.25,
    }

    for _ in range(2):
        decision = engine.on_candle_close("1m", inputs)

    assert decision.timing_state == "BREAKOUT_WINDOW"


# ============================================================================
# TEST 6: MISSING DATA ROBUSTNESS
# ============================================================================


def test_continuation_without_macd():
    """Continuation works with MACD missing (ROC + ATR only)."""
    config = TimingRulesConfig(
        timeframes=["1m"],
        continuation_confirm_bars=2,
        use_macd_optional=True,  # Enabled but MACD data missing
    )
    engine = TimingRulesEngine(config)

    inputs = {
        "trend_ok": True,
        "trend_bias": 1,
        "vwap_state": "ACCEPT",
        "vwap_position": "ABOVE",
        "roc_fast_norm": 0.45,
        "roc_fast_norm_prev": 0.30,
        "atr_exp": 1.10,
        # No macd_hist_slope
    }

    for _ in range(2):
        decision = engine.on_candle_close("1m", inputs)

    assert decision.timing_state == "CONTINUATION_READY"
    assert decision.checklist.get("cont_macd_available") is False


def test_exhaustion_with_partial_data():
    """Exhaustion works with some conditions missing."""
    config = TimingRulesConfig(timeframes=["1m"], exhaustion_confirm_bars=2)
    engine = TimingRulesEngine(config)

    # Only 2 of 3 conditions (RSI divergence missing)
    inputs = {
        "trend_bias": 1,
        "roc_fast_norm": 0.50,
        "acc_fast": -0.20,
        "atr_exp": 1.70,
        "ribbon_width_rate": -0.05,
        # No rsi_divergence or structure_pivot_type
    }

    for _ in range(2):
        decision = engine.on_candle_close("1m", inputs)

    assert decision.timing_state == "EXHAUSTION_WARNING"
    assert decision.checklist.get("ex_rsi_div") is False


def test_breakout_without_roc():
    """Breakout fires without ROC if roc_surge not required."""
    config = TimingRulesConfig(
        timeframes=["1m"],
        breakout_confirm_bars=2,
        breakout_requires_roc_surge=False,
    )
    engine = TimingRulesEngine(config)

    # Setup
    engine.on_candle_close("1m", {"atr_exp": 0.75, "chop_state": "CHOP"})

    # Ignition without ROC
    ignite_inputs = {"atr_exp": 1.25}

    for _ in range(2):
        decision = engine.on_candle_close("1m", ignite_inputs)

    assert decision.timing_state == "BREAKOUT_WINDOW"
    assert decision.direction == 0  # No direction without ROC


# ============================================================================
# TEST 7: CONFIRMATION COUNTERS (NO FLIP)
# ============================================================================


def test_confirmation_prevents_single_bar_trigger():
    """Single bar meeting conditions does not trigger (requires confirmation)."""
    config = TimingRulesConfig(
        timeframes=["1m"],
        continuation_confirm_bars=3,  # Require 3 bars
    )
    engine = TimingRulesEngine(config)

    inputs = {
        "trend_ok": True,
        "trend_bias": 1,
        "vwap_state": "ACCEPT",
        "vwap_position": "ABOVE",
        "roc_fast_norm": 0.45,
        "roc_fast_norm_prev": 0.30,
        "atr_exp": 1.10,
    }

    # First bar
    decision = engine.on_candle_close("1m", inputs)
    assert decision.timing_state != "CONTINUATION_READY"

    # Second bar
    decision = engine.on_candle_close("1m", inputs)
    assert decision.timing_state != "CONTINUATION_READY"

    # Third bar (confirmation)
    decision = engine.on_candle_close("1m", inputs)
    assert decision.timing_state == "CONTINUATION_READY"


def test_counter_resets_on_condition_loss():
    """Counter resets if conditions not met on subsequent bar."""
    config = TimingRulesConfig(
        timeframes=["1m"],
        continuation_confirm_bars=3,
    )
    engine = TimingRulesEngine(config)

    good_inputs = {
        "trend_ok": True,
        "trend_bias": 1,
        "vwap_state": "ACCEPT",
        "vwap_position": "ABOVE",
        "roc_fast_norm": 0.45,
        "roc_fast_norm_prev": 0.30,
        "atr_exp": 1.10,
    }

    bad_inputs = {
        "trend_ok": False,  # Gate fails
    }

    # Two good bars
    engine.on_candle_close("1m", good_inputs)
    engine.on_candle_close("1m", good_inputs)

    # One bad bar (resets)
    engine.on_candle_close("1m", bad_inputs)

    # Two more good bars
    engine.on_candle_close("1m", good_inputs)
    decision = engine.on_candle_close("1m", good_inputs)

    # Should not trigger yet (counter reset)
    assert decision.timing_state != "CONTINUATION_READY"

    # One more good bar to complete
    decision = engine.on_candle_close("1m", good_inputs)
    assert decision.timing_state == "CONTINUATION_READY"


# ============================================================================
# TEST 8: DEFAULT STATES
# ============================================================================


def test_default_transition():
    """Default to TRANSITION when trend_ok but no signals."""
    config = TimingRulesConfig(timeframes=["1m"])
    engine = TimingRulesEngine(config)

    inputs = {"trend_ok": True}

    decision = engine.on_candle_close("1m", inputs)
    assert decision.timing_state == "TRANSITION"
    assert decision.confidence_0_100 == 30.0


def test_default_no_trade():
    """Default to NO_TRADE when no trend."""
    config = TimingRulesConfig(timeframes=["1m"])
    engine = TimingRulesEngine(config)

    inputs = {"trend_ok": False}

    decision = engine.on_candle_close("1m", inputs)
    assert decision.timing_state == "NO_TRADE"
    assert decision.confidence_0_100 == 10.0


# ============================================================================
# TEST 9: MULTI-TIMEFRAME
# ============================================================================


def test_multi_timeframe_independence():
    """Each timeframe maintains independent state."""
    config = TimingRulesConfig(timeframes=["1m", "5m"])
    engine = TimingRulesEngine(config)

    inputs_1m = {
        "trend_ok": True,
        "trend_bias": 1,
        "vwap_state": "ACCEPT",
        "vwap_position": "ABOVE",
        "roc_fast_norm": 0.45,
        "roc_fast_norm_prev": 0.30,
        "atr_exp": 1.10,
    }

    inputs_5m = {
        "trend_bias": 1,
        "roc_fast_norm": 0.50,
        "acc_fast": -0.20,
        "atr_exp": 1.70,
        "ribbon_width_rate": -0.05,
    }

    # Feed 1m
    for _ in range(2):
        decision_1m = engine.on_candle_close("1m", inputs_1m)

    # Feed 5m
    for _ in range(2):
        decision_5m = engine.on_candle_close("5m", inputs_5m)

    assert decision_1m.timing_state == "CONTINUATION_READY"
    assert decision_5m.timing_state == "EXHAUSTION_WARNING"


# ============================================================================
# TEST 10: TIMING SCORE
# ============================================================================


def test_timing_score_computation():
    """Timing score computes correctly."""
    config = TimingRulesConfig(
        timeframes=["1m"],
        enable_timing_score=True,
        continuation_confirm_bars=2,
    )
    engine = TimingRulesEngine(config)

    inputs = {
        "trend_ok": True,  # +25
        "trend_bias": 1,
        "vwap_state": "ACCEPT",  # +25
        "vwap_position": "ABOVE",
        "roc_fast_norm": 0.45,  # +15
        "roc_fast_norm_prev": 0.30,
        "atr_exp": 1.10,  # +15
        "macd_hist_slope": 0.05,  # +10
    }

    for _ in range(2):
        decision = engine.on_candle_close("1m", inputs)

    assert decision.timing_score_0_100 is not None
    # 25 + 25 + 15 + 15 + 10 = 90
    assert decision.timing_score_0_100 >= 85  # Allow some tolerance


def test_timing_score_penalties():
    """Timing score applies penalties correctly."""
    config = TimingRulesConfig(
        timeframes=["1m"],
        enable_timing_score=True,
        exhaustion_confirm_bars=2,
    )
    engine = TimingRulesEngine(config)

    # Trigger exhaustion (penalty -40)
    inputs = {
        "trend_bias": 1,
        "roc_fast_norm": 0.50,
        "acc_fast": -0.20,
        "atr_exp": 1.70,
        "ribbon_width_rate": -0.05,
    }

    for _ in range(2):
        decision = engine.on_candle_close("1m", inputs)

    assert decision.timing_state == "EXHAUSTION_WARNING"
    assert decision.timing_score_0_100 is not None
    # Score should be low due to exhaustion penalty
    assert decision.timing_score_0_100 < 50


# ============================================================================
# TEST 11: HELPER FUNCTIONS
# ============================================================================


def test_print_timing_decisions():
    """Print helper does not crash."""
    from timing_rules import format_timing_decision, interpret_timing, print_timing_decisions

    decision = TimingDecision(
        timing_state="CONTINUATION_READY",
        direction=1,
        confidence_0_100=78.0,
        timing_score_0_100=72.0,
        checklist={},
        reasons=["trend_ok", "vwap_reclaim"],
        debug={},
    )

    # Should not raise
    formatted = format_timing_decision("1m", decision)
    assert "CONTINUATION_READY" in formatted
    assert "dir=+1" in formatted

    interpretation = interpret_timing(decision)
    assert "Continuation ready" in interpretation

    # Print full
    print_timing_decisions({"1m": decision})


# ============================================================================
# TEST 12: RESET
# ============================================================================


def test_reset():
    """Reset clears state for timeframe."""
    config = TimingRulesConfig(timeframes=["1m"], continuation_confirm_bars=1)
    engine = TimingRulesEngine(config)

    inputs = {
        "trend_ok": True,
        "trend_bias": 1,
        "vwap_state": "ACCEPT",
        "vwap_position": "ABOVE",
        "roc_fast_norm": 0.45,
        "roc_fast_norm_prev": 0.30,
        "atr_exp": 1.10,
    }

    decision = engine.on_candle_close("1m", inputs)
    assert decision.timing_state == "CONTINUATION_READY"

    # Reset
    engine.reset("1m")

    # State should be cleared
    state = engine.get_state("1m")
    assert state is None or state.debug.get("bar_index") == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
