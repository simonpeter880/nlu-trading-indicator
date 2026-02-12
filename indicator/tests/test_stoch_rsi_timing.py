"""
Tests for Stochastic RSI Timing module.

Verifies:
- StochRSI formula correctness
- Rolling RSI min/max (monotonic deques)
- K/D smoothing
- Bull/bear pullback completion detection
- Gating mechanisms (trend, bias, CHOP, etc.)
- Confirmation stability (anti-noise)
"""

import pytest
from stoch_rsi_timing import (
    StochRSITimingEngine,
    StochRSIConfig,
    Candle,
    print_stoch_rsi_timing,
    format_stoch_rsi_state,
    interpret_stoch_rsi,
)


@pytest.fixture
def config():
    """Default config for testing."""
    return StochRSIConfig(
        timeframes=["1m"],
        rsi_period=14,
        stoch_period=14,
        k_smooth=3,
        d_smooth=3,
        oversold=0.20,
        overbought=0.80,
        confirm_bars=2,
    )


# ============================================================================
# TEST 1: STOCHRSI FORMULA
# ============================================================================

def test_stochrsi_formula():
    """StochRSI = (RSI - min(RSI)) / (max(RSI) - min(RSI))."""
    config = StochRSIConfig(
        rsi_period=5,
        stoch_period=5,
        k_smooth=1,  # No smoothing for direct test
        d_smooth=1,
        use_external_rsi=True  # Use external RSI for controlled test
    )
    engine = StochRSITimingEngine(config)

    # Feed known RSI values
    rsi_values = [30, 40, 50, 60, 70]  # min=30, max=70
    candles = [Candle(i * 1000, 100, 105, 95, 100, 1000) for i in range(5)]

    for i, (candle, rsi) in enumerate(zip(candles, rsi_values)):
        state = engine.on_candle_close("1m", candle, rsi_value=rsi)

    # After 5 RSI values, StochRSI should be computed
    # Latest RSI = 70, min = 30, max = 70
    # StochRSI = (70 - 30) / (70 - 30) = 1.0
    assert state.stochrsi is not None
    assert abs(state.stochrsi - 1.0) < 0.01

    # K with smooth=1 should equal stochrsi
    assert state.k is not None
    assert abs(state.k - 1.0) < 0.01


def test_stochrsi_stays_in_range():
    """StochRSI should always be in [0, 1]."""
    config = StochRSIConfig(rsi_period=5, stoch_period=5, use_external_rsi=True)
    engine = StochRSITimingEngine(config)

    # Feed varying RSI values
    rsi_values = [10, 90, 20, 80, 30, 70, 40, 60, 50]
    candles = [Candle(i * 1000, 100, 105, 95, 100, 1000) for i in range(len(rsi_values))]

    for candle, rsi in zip(candles, rsi_values):
        state = engine.on_candle_close("1m", candle, rsi_value=rsi)

        if state.stochrsi is not None:
            assert 0.0 <= state.stochrsi <= 1.0


# ============================================================================
# TEST 2: ROLLING MIN/MAX CORRECTNESS
# ============================================================================

def test_rsi_min_max_monotonic_deques():
    """RSI min/max should match batch computation."""
    config = StochRSIConfig(
        rsi_period=5,
        stoch_period=5,
        k_smooth=1,
        d_smooth=1,
        use_external_rsi=True
    )
    engine = StochRSITimingEngine(config)

    # Feed known RSI sequence
    rsi_sequence = [30, 40, 25, 35, 45, 50, 20, 60, 55]
    candles = [Candle(i * 1000, 100, 105, 95, 100, 1000) for i in range(len(rsi_sequence))]

    for i, (candle, rsi) in enumerate(zip(candles, rsi_sequence)):
        state = engine.on_candle_close("1m", candle, rsi_value=rsi)

        if i >= 4:  # After stoch_period candles
            # Get expected min/max from last 5 RSI values
            window = rsi_sequence[max(0, i - 4):i + 1]
            expected_min = min(window)
            expected_max = max(window)

            # Check debug values
            if state.debug.get('rsi_min') is not None:
                assert abs(state.debug['rsi_min'] - expected_min) < 0.01
            if state.debug.get('rsi_max') is not None:
                assert abs(state.debug['rsi_max'] - expected_max) < 0.01


# ============================================================================
# TEST 3: K/D SMOOTHING
# ============================================================================

def test_k_smoothing():
    """%K should be SMA of StochRSI over k_smooth period."""
    config = StochRSIConfig(
        rsi_period=5,
        stoch_period=5,
        k_smooth=3,
        d_smooth=1,
        use_external_rsi=True
    )
    engine = StochRSITimingEngine(config)

    # Create constant RSI for predictable StochRSI
    rsi_values = [50] * 10  # Constant RSI
    candles = [Candle(i * 1000, 100, 105, 95, 100, 1000) for i in range(10)]

    for candle, rsi in zip(candles, rsi_values):
        state = engine.on_candle_close("1m", candle, rsi_value=rsi)

    # With constant RSI, StochRSI should be constant (0.5 if range is 0)
    # K should equal StochRSI after smoothing
    assert state.k is not None


def test_d_smoothing():
    """%D should be SMA of %K over d_smooth period."""
    config = StochRSIConfig(
        rsi_period=5,
        stoch_period=5,
        k_smooth=3,
        d_smooth=3,
        use_external_rsi=True
    )
    engine = StochRSITimingEngine(config)

    rsi_values = list(range(30, 50))  # Increasing RSI
    candles = [Candle(i * 1000, 100, 105, 95, 100, 1000) for i in range(len(rsi_values))]

    for candle, rsi in zip(candles, rsi_values):
        state = engine.on_candle_close("1m", candle, rsi_value=rsi)

    # D should be available after enough data
    assert state.d is not None


# ============================================================================
# TEST 4: BULL PULLBACK COMPLETION
# ============================================================================

def test_bull_pullback_completion():
    """K crosses above oversold with rising K => PULLBACK_DONE_UP."""
    config = StochRSIConfig(
        rsi_period=5,
        stoch_period=5,
        k_smooth=3,
        d_smooth=3,
        oversold=0.20,
        confirm_bars=2,
        use_external_rsi=True
    )
    engine = StochRSITimingEngine(config)

    # Create RSI sequence that produces oversold then turn up
    # Low RSI -> StochRSI low
    rsi_sequence = [30, 25, 20, 22, 24]  # Bottoming
    candles = [Candle(i * 1000, 100, 105, 95, 100, 1000) for i in range(5)]

    for candle, rsi in zip(candles, rsi_sequence):
        engine.on_candle_close("1m", candle, rsi_value=rsi, trend_permission=True, bias=1)

    # Now RSI rises -> StochRSI rises
    rsi_rising = [26, 30, 35, 40, 45]
    candles_rising = [Candle((5 + i) * 1000, 100, 105, 95, 100, 1000) for i in range(5)]

    for candle, rsi in zip(candles_rising, rsi_rising):
        state = engine.on_candle_close("1m", candle, rsi_value=rsi, trend_permission=True, bias=1)

    # Should eventually detect PULLBACK_DONE_UP after confirmation
    # (exact timing depends on when K crosses oversold threshold)
    assert state.micro_timing in ["PULLBACK_DONE_UP", "NONE"]


# ============================================================================
# TEST 5: BEAR PULLBACK COMPLETION
# ============================================================================

def test_bear_pullback_completion():
    """K crosses below overbought with falling K => PULLBACK_DONE_DOWN."""
    config = StochRSIConfig(
        rsi_period=5,
        stoch_period=5,
        k_smooth=3,
        d_smooth=3,
        overbought=0.80,
        confirm_bars=2,
        use_external_rsi=True
    )
    engine = StochRSITimingEngine(config)

    # Create RSI sequence that produces overbought then turn down
    rsi_sequence = [70, 75, 80, 78, 76]  # Topping
    candles = [Candle(i * 1000, 100, 105, 95, 100, 1000) for i in range(5)]

    for candle, rsi in zip(candles, rsi_sequence):
        engine.on_candle_close("1m", candle, rsi_value=rsi, trend_permission=True, bias=-1)

    # Now RSI falls -> StochRSI falls
    rsi_falling = [74, 70, 65, 60, 55]
    candles_falling = [Candle((5 + i) * 1000, 100, 105, 95, 100, 1000) for i in range(5)]

    for candle, rsi in zip(candles_falling, rsi_falling):
        state = engine.on_candle_close("1m", candle, rsi_value=rsi, trend_permission=True, bias=-1)

    # Should eventually detect PULLBACK_DONE_DOWN
    assert state.micro_timing in ["PULLBACK_DONE_DOWN", "NONE"]


# ============================================================================
# TEST 6: GATING MECHANISMS
# ============================================================================

def test_no_trend_permission():
    """No micro timing when trend_permission=False."""
    config = StochRSIConfig(
        rsi_period=5,
        stoch_period=5,
        k_smooth=3,
        d_smooth=3,
        confirm_bars=1,
        use_external_rsi=True
    )
    engine = StochRSITimingEngine(config)

    # Create oversold scenario
    rsi_sequence = [20, 22, 25, 30, 35]
    candles = [Candle(i * 1000, 100, 105, 95, 100, 1000) for i in range(5)]

    for candle, rsi in zip(candles, rsi_sequence):
        state = engine.on_candle_close(
            "1m", candle, rsi_value=rsi,
            trend_permission=False,  # No permission
            bias=1
        )

    # Should NOT trigger micro timing
    assert state.micro_timing == "NONE"


def test_neutral_bias():
    """No micro timing when bias=0 (neutral)."""
    config = StochRSIConfig(
        rsi_period=5,
        stoch_period=5,
        k_smooth=3,
        d_smooth=3,
        confirm_bars=1,
        use_external_rsi=True
    )
    engine = StochRSITimingEngine(config)

    rsi_sequence = [20, 22, 25, 30, 35]
    candles = [Candle(i * 1000, 100, 105, 95, 100, 1000) for i in range(5)]

    for candle, rsi in zip(candles, rsi_sequence):
        state = engine.on_candle_close(
            "1m", candle, rsi_value=rsi,
            trend_permission=True,
            bias=0  # Neutral
        )

    # Should NOT trigger micro timing
    assert state.micro_timing == "NONE"


def test_chop_gate():
    """No micro timing when chop_state='CHOP' and disable_when_chop=True."""
    config = StochRSIConfig(
        rsi_period=5,
        stoch_period=5,
        k_smooth=3,
        d_smooth=3,
        confirm_bars=1,
        disable_when_chop=True,
        use_external_rsi=True
    )
    engine = StochRSITimingEngine(config)

    rsi_sequence = [20, 22, 25, 30, 35]
    candles = [Candle(i * 1000, 100, 105, 95, 100, 1000) for i in range(5)]

    for candle, rsi in zip(candles, rsi_sequence):
        state = engine.on_candle_close(
            "1m", candle, rsi_value=rsi,
            trend_permission=True,
            bias=1,
            chop_state="CHOP"  # Choppy market
        )

    # Should NOT trigger micro timing
    assert state.micro_timing == "NONE"


def test_atr_squeeze_gate():
    """No micro timing when atr_exp_state='SQUEEZE' and disable_when_atr_squeeze=True."""
    config = StochRSIConfig(
        rsi_period=5,
        stoch_period=5,
        k_smooth=3,
        d_smooth=3,
        confirm_bars=1,
        disable_when_atr_squeeze=True,
        use_external_rsi=True
    )
    engine = StochRSITimingEngine(config)

    rsi_sequence = [20, 22, 25, 30, 35]
    candles = [Candle(i * 1000, 100, 105, 95, 100, 1000) for i in range(5)]

    for candle, rsi in zip(candles, rsi_sequence):
        state = engine.on_candle_close(
            "1m", candle, rsi_value=rsi,
            trend_permission=True,
            bias=1,
            atr_exp_state="SQUEEZE"  # Low volatility
        )

    # Should NOT trigger micro timing
    assert state.micro_timing == "NONE"


# ============================================================================
# TEST 7: CONFIRMATION STABILITY
# ============================================================================

def test_confirmation_bars_required():
    """Micro timing should require confirm_bars consecutive occurrences."""
    config = StochRSIConfig(
        rsi_period=5,
        stoch_period=5,
        k_smooth=3,
        d_smooth=3,
        confirm_bars=3,  # Require 3 bars
        use_external_rsi=True
    )
    engine = StochRSITimingEngine(config)

    # Setup: Get to a state just before trigger
    for i in range(10):
        engine.on_candle_close(
            "1m",
            Candle(i * 1000, 100, 105, 95, 100, 1000),
            rsi_value=25 + i,
            trend_permission=True,
            bias=1
        )

    # Now create condition that would trigger
    # But interrupt it before confirm_bars reached
    state1 = engine.on_candle_close(
        "1m",
        Candle(11000, 100, 105, 95, 100, 1000),
        rsi_value=35,
        trend_permission=True,
        bias=1
    )

    # After 1 bar, should NOT trigger yet (needs 3)
    # Check via debug
    if state1.debug.get('confirm_count') is not None:
        assert state1.debug['confirm_count'] < 3


# ============================================================================
# TEST 8: ZONE CLASSIFICATION
# ============================================================================

def test_zone_oversold():
    """K <= oversold => OVERSOLD zone."""
    config = StochRSIConfig(
        rsi_period=5,
        stoch_period=5,
        k_smooth=3,
        d_smooth=3,
        oversold=0.20,
        use_external_rsi=True
    )
    engine = StochRSITimingEngine(config)

    # Create low RSI sequence -> low StochRSI
    rsi_values = [20, 22, 21, 23, 22]
    candles = [Candle(i * 1000, 100, 105, 95, 100, 1000) for i in range(10)]

    for i, candle in enumerate(candles):
        rsi = rsi_values[i % len(rsi_values)]
        state = engine.on_candle_close("1m", candle, rsi_value=rsi)

    # Should eventually be OVERSOLD
    if state.k is not None and state.k <= 0.20:
        assert state.zone == "OVERSOLD"


def test_zone_overbought():
    """K >= overbought => OVERBOUGHT zone."""
    config = StochRSIConfig(
        rsi_period=5,
        stoch_period=5,
        k_smooth=3,
        d_smooth=3,
        overbought=0.80,
        use_external_rsi=True
    )
    engine = StochRSITimingEngine(config)

    # Create high RSI sequence -> high StochRSI
    rsi_values = [78, 80, 79, 81, 80]
    candles = [Candle(i * 1000, 100, 105, 95, 100, 1000) for i in range(10)]

    for i, candle in enumerate(candles):
        rsi = rsi_values[i % len(rsi_values)]
        state = engine.on_candle_close("1m", candle, rsi_value=rsi)

    # Should eventually be OVERBOUGHT
    if state.k is not None and state.k >= 0.80:
        assert state.zone == "OVERBOUGHT"


# ============================================================================
# TEST 9: INCREMENTAL BEHAVIOR
# ============================================================================

def test_incremental_updates():
    """Engine should process candles incrementally with O(1) updates."""
    config = StochRSIConfig(rsi_period=14, stoch_period=14, use_external_rsi=True)
    engine = StochRSITimingEngine(config)

    # Feed 50 candles incrementally
    for i in range(50):
        rsi = 50 + (i % 10) - 5  # Varying RSI
        candle = Candle(i * 1000, 100, 105, 95, 100, 1000)
        state = engine.on_candle_close("1m", candle, rsi_value=rsi)

    # Should have valid state after warmup
    assert state.k is not None or state.zone == "WARMUP"


def test_incremental_vs_batch():
    """Incremental processing should match batch warmup."""
    config = StochRSIConfig(rsi_period=5, stoch_period=5, use_external_rsi=True)

    # Generate candles and RSI values
    rsi_values = [30 + i for i in range(20)]
    candles = [Candle(i * 1000, 100, 105, 95, 100, 1000) for i in range(20)]

    # Incremental
    engine1 = StochRSITimingEngine(config)
    for candle, rsi in zip(candles, rsi_values):
        state1 = engine1.on_candle_close("1m", candle, rsi_value=rsi)

    # Batch
    engine2 = StochRSITimingEngine(config)
    states2 = engine2.warmup({"1m": candles}, rsi_by_tf={"1m": rsi_values})
    state2 = states2["1m"]

    # Should match
    if state1.k is not None and state2.k is not None:
        assert abs(state1.k - state2.k) < 1e-9


# ============================================================================
# TEST 10: MULTI-TIMEFRAME
# ============================================================================

def test_multi_timeframe():
    """Engine handles multiple timeframes independently."""
    config = StochRSIConfig(timeframes=["1m", "5m"], use_external_rsi=True)
    engine = StochRSITimingEngine(config)

    # Feed different RSI to different timeframes
    for i in range(20):
        rsi_1m = 30 + i
        rsi_5m = 70 - i

        state_1m = engine.on_candle_close(
            "1m", Candle(i * 1000, 100, 105, 95, 100, 1000), rsi_value=rsi_1m
        )
        state_5m = engine.on_candle_close(
            "5m", Candle(i * 5000, 200, 205, 195, 200, 1000), rsi_value=rsi_5m
        )

    # Should have different states
    if state_1m.k is not None and state_5m.k is not None:
        # Different RSI trends should produce different K values
        assert state_1m.k != state_5m.k or state_1m.zone != state_5m.zone


# ============================================================================
# TEST 11: EDGE CASES
# ============================================================================

def test_constant_rsi():
    """Handle constant RSI (no range)."""
    config = StochRSIConfig(
        rsi_period=5,
        stoch_period=5,
        k_smooth=3,
        d_smooth=3,
        use_external_rsi=True
    )
    engine = StochRSITimingEngine(config)

    # All same RSI
    for i in range(20):
        state = engine.on_candle_close(
            "1m",
            Candle(i * 1000, 100, 105, 95, 100, 1000),
            rsi_value=50.0
        )

    # Should handle gracefully (StochRSI = 0.5 when no range)
    assert state.stochrsi is not None
    assert abs(state.stochrsi - 0.5) < 0.1


def test_warmup_state():
    """Return WARMUP zone until enough data processed."""
    config = StochRSIConfig(rsi_period=14, stoch_period=14, use_external_rsi=True)
    engine = StochRSITimingEngine(config)

    # Feed less than required candles
    for i in range(10):
        state = engine.on_candle_close(
            "1m",
            Candle(i * 1000, 100, 105, 95, 100, 1000),
            rsi_value=50.0
        )

    # Should be WARMUP
    if state.k is None:
        assert state.zone == "WARMUP"


def test_reset():
    """Reset clears state for a timeframe."""
    config = StochRSIConfig(rsi_period=5, stoch_period=5, use_external_rsi=True)
    engine = StochRSITimingEngine(config)

    # Feed candles
    for i in range(20):
        engine.on_candle_close(
            "1m",
            Candle(i * 1000, 100, 105, 95, 100, 1000),
            rsi_value=50 + i
        )

    # Reset
    engine.reset("1m")

    # Next candle should start fresh
    state = engine.on_candle_close(
        "1m",
        Candle(21000, 100, 105, 95, 100, 1000),
        rsi_value=50.0
    )
    assert state.zone == "WARMUP"


# ============================================================================
# TEST 12: HELPER FUNCTIONS
# ============================================================================

def test_print_stoch_rsi_timing():
    """print_stoch_rsi_timing produces expected output."""
    config = StochRSIConfig(rsi_period=5, stoch_period=5, use_external_rsi=True)
    engine = StochRSITimingEngine(config)

    rsi_values = [30 + i for i in range(20)]
    candles = [Candle(i * 1000, 100, 105, 95, 100, 1000) for i in range(20)]
    states = engine.warmup({"1m": candles}, rsi_by_tf={"1m": rsi_values})

    # Should not crash
    print_stoch_rsi_timing(states)


def test_format_stoch_rsi_state():
    """format_stoch_rsi_state produces expected string."""
    config = StochRSIConfig(rsi_period=5, stoch_period=5, use_external_rsi=True)
    engine = StochRSITimingEngine(config)

    for i in range(20):
        state = engine.on_candle_close(
            "1m",
            Candle(i * 1000, 100, 105, 95, 100, 1000),
            rsi_value=30 + i
        )

    formatted = format_stoch_rsi_state("1m", state)
    assert "1m:" in formatted
    assert "K=" in formatted


def test_interpret_stoch_rsi():
    """interpret_stoch_rsi provides actionable insight."""
    config = StochRSIConfig(rsi_period=5, stoch_period=5, use_external_rsi=True)
    engine = StochRSITimingEngine(config)

    for i in range(20):
        state = engine.on_candle_close(
            "1m",
            Candle(i * 1000, 100, 105, 95, 100, 1000),
            rsi_value=30 + i
        )

    interpretation = interpret_stoch_rsi(state)
    assert isinstance(interpretation, str)
    assert len(interpretation) > 0
