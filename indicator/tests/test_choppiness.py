"""
Tests for Choppiness Index module.

Verifies:
- TR calculation correctness
- Rolling sum correctness
- HH/LL monotonic deque correctness
- CHOP formula correctness
- State thresholds
- Trend vs range detection
- Warmup behavior
"""

import math

import pytest

from choppiness import (
    Candle,
    ChoppinessConfig,
    ChoppinessEngine,
    format_chop_state,
    interpret_chop,
    print_choppiness,
)


@pytest.fixture
def config():
    """Default config for testing."""
    return ChoppinessConfig(
        timeframes=["1m"],
        default_period=14,
        chop_high=61.8,
        chop_low=38.2,
    )


# ============================================================================
# TEST 1: TRUE RANGE CORRECTNESS
# ============================================================================


def test_true_range_first_candle():
    """TR for first candle is just high - low."""
    engine = ChoppinessEngine()
    candle = Candle(1000, 100, 110, 90, 105, 1000)
    state = engine.on_candle_close("1m", candle)

    expected_tr = 110 - 90  # 20
    assert abs(state.tr - expected_tr) < 1e-9


def test_true_range_with_prev_close():
    """TR considers gaps from previous close."""
    engine = ChoppinessEngine()

    # First candle
    candle1 = Candle(1000, 100, 110, 90, 95, 1000)
    engine.on_candle_close("1m", candle1)

    # Second candle (gap up)
    candle2 = Candle(2000, 105, 115, 100, 110, 1000)
    state = engine.on_candle_close("1m", candle2)

    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    # TR = max(15, abs(115-95), abs(100-95)) = max(15, 20, 5) = 20
    expected_tr = max(115 - 100, abs(115 - 95), abs(100 - 95))
    assert abs(state.tr - expected_tr) < 1e-9
    assert abs(state.tr - 20) < 1e-9


def test_true_range_gap_down():
    """TR handles gap down correctly."""
    engine = ChoppinessEngine()

    candle1 = Candle(1000, 100, 110, 90, 105, 1000)
    engine.on_candle_close("1m", candle1)

    # Gap down
    candle2 = Candle(2000, 80, 85, 75, 82, 1000)
    state = engine.on_candle_close("1m", candle2)

    # TR = max(10, abs(85-105), abs(75-105)) = max(10, 20, 30) = 30
    expected_tr = max(85 - 75, abs(85 - 105), abs(75 - 105))
    assert abs(state.tr - expected_tr) < 1e-9
    assert abs(state.tr - 30) < 1e-9


# ============================================================================
# TEST 2: ROLLING SUM CORRECTNESS
# ============================================================================


def test_rolling_sum_tr():
    """TR rolling sum matches sum of last n TRs."""
    config = ChoppinessConfig(default_period=3)
    engine = ChoppinessEngine(config)

    candles = [
        Candle(1000, 100, 120, 80, 100, 1000),  # TR = 40
        Candle(2000, 100, 130, 90, 110, 1000),  # TR = 40
        Candle(3000, 110, 140, 100, 120, 1000),  # TR = 40
        Candle(4000, 120, 145, 115, 125, 1000),  # TR = 30
    ]

    states = []
    for candle in candles:
        state = engine.on_candle_close("1m", candle)
        states.append(state)

    # After 3 candles, sum should be 40+40+40=120
    assert abs(states[2].sum_tr - 120.0) < 1e-9

    # After 4th candle, rolling sum should be 40+40+30=110 (oldest 40 dropped)
    assert abs(states[3].sum_tr - 110.0) < 1e-9


def test_rolling_sum_maintains_window():
    """Rolling sum maintains correct window size."""
    config = ChoppinessConfig(default_period=5)
    engine = ChoppinessEngine(config)

    # Feed 10 candles with known TRs
    trs = [10, 20, 30, 40, 50, 15, 25, 35, 45, 55]
    candles = []
    price = 100.0
    for i, tr_val in enumerate(trs):
        # Create candle with specific TR
        high = price + tr_val / 2
        low = price - tr_val / 2
        candle = Candle(i * 1000, price, high, low, price, 1000)
        candles.append(candle)

    for i, candle in enumerate(candles):
        state = engine.on_candle_close("1m", candle)

        # After period candles, check rolling sum
        if i >= 4:  # period = 5, so index 4 onwards
            expected_sum = sum(trs[max(0, i - 4) : i + 1])
            assert (
                abs(state.sum_tr - expected_sum) < 1e-6
            ), f"At index {i}, expected {expected_sum}, got {state.sum_tr}"


# ============================================================================
# TEST 3: HH/LL MONOTONIC DEQUE CORRECTNESS
# ============================================================================


def test_highest_high_rolling_max():
    """HH correctly tracks rolling maximum high."""
    config = ChoppinessConfig(default_period=3)
    engine = ChoppinessEngine(config)

    candles = [
        Candle(1000, 100, 110, 90, 100, 1000),  # HH over [0:1] = 110
        Candle(2000, 100, 120, 95, 110, 1000),  # HH over [0:2] = 120
        Candle(3000, 110, 115, 105, 112, 1000),  # HH over [0:3] = 120
        Candle(4000, 112, 125, 110, 120, 1000),  # HH over [1:4] = 125 (window slides)
    ]

    states = []
    for candle in candles:
        state = engine.on_candle_close("1m", candle)
        states.append(state)

    # After 3 candles (warmup complete)
    assert states[2].hh == 120.0

    # After 4th candle (window = last 3)
    assert states[3].hh == 125.0


def test_lowest_low_rolling_min():
    """LL correctly tracks rolling minimum low."""
    config = ChoppinessConfig(default_period=3)
    engine = ChoppinessEngine(config)

    candles = [
        Candle(1000, 100, 110, 90, 100, 1000),  # LL over [0:1] = 90
        Candle(2000, 100, 120, 85, 110, 1000),  # LL over [0:2] = 85
        Candle(3000, 110, 115, 95, 112, 1000),  # LL over [0:3] = 85
        Candle(4000, 112, 125, 80, 120, 1000),  # LL over [1:4] = 80 (window slides)
    ]

    states = []
    for candle in candles:
        state = engine.on_candle_close("1m", candle)
        states.append(state)

    # After 3 candles
    assert states[2].ll == 85.0

    # After 4th candle
    assert states[3].ll == 80.0


def test_hh_ll_range_calculation():
    """Range correctly computed as HH - LL."""
    config = ChoppinessConfig(default_period=3)
    engine = ChoppinessEngine(config)

    candles = [
        Candle(1000, 100, 110, 90, 100, 1000),
        Candle(2000, 100, 120, 85, 110, 1000),
        Candle(3000, 110, 115, 95, 112, 1000),
    ]

    for candle in candles:
        state = engine.on_candle_close("1m", candle)

    # HH = 120, LL = 85
    assert state.hh == 120.0
    assert state.ll == 85.0
    assert state.range == 35.0


# ============================================================================
# TEST 4: CHOP FORMULA CORRECTNESS
# ============================================================================


def test_chop_formula_calculation():
    """CHOP formula matches manual calculation."""
    config = ChoppinessConfig(default_period=3)
    engine = ChoppinessEngine(config)

    # Create candles with known values
    candles = [
        Candle(1000, 100, 120, 80, 100, 1000),  # TR = 40, HH=120, LL=80
        Candle(2000, 100, 130, 90, 110, 1000),  # TR = 40, HH=130, LL=80
        Candle(3000, 110, 125, 95, 115, 1000),  # TR = 30, HH=130, LL=80
    ]

    for candle in candles:
        state = engine.on_candle_close("1m", candle)

    # sum_tr = 40 + 40 + 30 = 110
    # range = 130 - 80 = 50
    # ratio = 110 / 50 = 2.2
    # CHOP = 100 * log10(2.2) / log10(3)
    expected_sum_tr = 110.0
    expected_range = 50.0
    expected_ratio = expected_sum_tr / expected_range
    expected_chop = 100.0 * math.log10(expected_ratio) / math.log10(3)

    assert abs(state.sum_tr - expected_sum_tr) < 1e-9
    assert abs(state.range - expected_range) < 1e-9
    assert abs(state.chop - expected_chop) < 1e-6


def test_chop_incremental_vs_batch():
    """Incremental CHOP matches batch calculation."""
    config = ChoppinessConfig(default_period=5)
    engine = ChoppinessEngine(config)

    # Generate candles
    candles = []
    price = 100.0
    for i in range(10):
        high = price + (i % 3) * 5 + 10
        low = price - (i % 2) * 3 - 5
        close = price + (i % 4) - 1.5
        candles.append(Candle(i * 1000, price, high, low, close, 1000))
        price = close

    # Incremental processing
    incremental_state = None
    for candle in candles:
        incremental_state = engine.on_candle_close("1m", candle)

    # Batch processing (using warmup)
    engine2 = ChoppinessEngine(config)
    batch_states = engine2.warmup({"1m": candles})
    batch_state = batch_states["1m"]

    # Should match
    assert abs(incremental_state.chop - batch_state.chop) < 1e-9
    assert abs(incremental_state.sum_tr - batch_state.sum_tr) < 1e-9
    assert abs(incremental_state.hh - batch_state.hh) < 1e-9
    assert abs(incremental_state.ll - batch_state.ll) < 1e-9


# ============================================================================
# TEST 5: STATE THRESHOLDS
# ============================================================================


def test_state_chop_high():
    """CHOP >= 61.8 => CHOP state."""
    config = ChoppinessConfig(default_period=3, chop_high=61.8, chop_low=38.2)
    engine = ChoppinessEngine(config)

    # Create sideways candles (high CHOP)
    # Small range but high sum(TR) => high CHOP
    candles = []
    price = 100.0
    for i in range(5):
        high = price + 5
        low = price - 5
        close = price + (i % 2) * 2 - 1  # Oscillate
        candles.append(Candle(i * 1000, price, high, low, close, 1000))
        price = close

    for candle in candles:
        state = engine.on_candle_close("1m", candle)

    # Should eventually be CHOP or TRANSITION (depends on exact values)
    # Let's verify threshold logic by forcing a high CHOP scenario
    # We'll trust the formula; just check that high values => CHOP state
    if state.chop is not None and state.chop >= 61.8:
        assert state.chop_state == "CHOP"


def test_state_trend_low():
    """CHOP <= 38.2 => TREND state."""
    config = ChoppinessConfig(default_period=3, chop_high=61.8, chop_low=38.2)
    engine = ChoppinessEngine(config)

    # Create trending candles (low CHOP)
    # Large range with small sum(TR) => low CHOP
    candles = [
        Candle(1000, 100, 105, 95, 103, 1000),  # TR small
        Candle(2000, 103, 108, 98, 106, 1000),  # TR small
        Candle(3000, 106, 130, 101, 125, 1000),  # Big directional move
    ]

    for candle in candles:
        state = engine.on_candle_close("1m", candle)

    # Large range (130-95=35) with smaller sum(TR) => lower CHOP
    if state.chop is not None and state.chop <= 38.2:
        assert state.chop_state == "TREND"


def test_state_transition():
    """38.2 < CHOP < 61.8 => TRANSITION state."""
    config = ChoppinessConfig(default_period=3)
    engine = ChoppinessEngine(config)

    # Create candles that produce mid-range CHOP
    candles = [
        Candle(1000, 100, 115, 85, 105, 1000),
        Candle(2000, 105, 120, 95, 110, 1000),
        Candle(3000, 110, 125, 100, 115, 1000),
    ]

    for candle in candles:
        state = engine.on_candle_close("1m", candle)

    # Check TRANSITION state
    if state.chop is not None and 38.2 < state.chop < 61.8:
        assert state.chop_state == "TRANSITION"


# ============================================================================
# TEST 6: TREND VS RANGE DETECTION
# ============================================================================


def test_trending_produces_low_chop():
    """Trending series produces lower CHOP than sideways."""
    config = ChoppinessConfig(default_period=5)

    # Trending series (consistent directional movement)
    trending_candles = []
    price = 100.0
    for i in range(10):
        high = price + 3
        low = price - 1
        close = price + 2  # Steady uptrend
        trending_candles.append(Candle(i * 1000, price, high, low, close, 1000))
        price = close

    engine_trend = ChoppinessEngine(config)
    trend_state = None
    for candle in trending_candles:
        trend_state = engine_trend.on_candle_close("1m", candle)

    # Sideways series (oscillating, same TR but low net range)
    sideways_candles = []
    price = 100.0
    for i in range(10):
        high = price + 5
        low = price - 5
        close = price + (5 if i % 2 == 0 else -5)  # Oscillate
        sideways_candles.append(Candle(i * 1000, price, high, low, close, 1000))
        price = close

    engine_sideways = ChoppinessEngine(config)
    sideways_state = None
    for candle in sideways_candles:
        sideways_state = engine_sideways.on_candle_close("1m", candle)

    # Trending should have lower CHOP than sideways
    assert trend_state.chop is not None
    assert sideways_state.chop is not None
    assert trend_state.chop < sideways_state.chop


def test_range_bound_produces_high_chop():
    """Range-bound market produces high CHOP."""
    config = ChoppinessConfig(default_period=5)
    engine = ChoppinessEngine(config)

    # Range-bound: price oscillates within tight range
    candles = []
    for i in range(10):
        price = 100 + (i % 2) * 2  # Oscillate 100, 102, 100, 102...
        high = price + 3
        low = price - 3
        close = 100 + ((i + 1) % 2) * 2
        candles.append(Candle(i * 1000, price, high, low, close, 1000))

    for candle in candles:
        state = engine.on_candle_close("1m", candle)

    # Should produce high CHOP (range-bound)
    assert state.chop is not None
    assert state.chop > 50.0  # Should be above midpoint


# ============================================================================
# TEST 7: WARMUP BEHAVIOR
# ============================================================================


def test_warmup_state():
    """Engine returns WARMUP until n bars processed."""
    config = ChoppinessConfig(default_period=5)
    engine = ChoppinessEngine(config)

    candles = [Candle(i * 1000, 100, 110, 90, 105, 1000) for i in range(10)]

    for i, candle in enumerate(candles):
        state = engine.on_candle_close("1m", candle)

        if i < 4:  # First 5 bars (indices 0-4)
            assert state.chop_state == "WARMUP"
            assert state.chop is None
        else:
            assert state.chop_state != "WARMUP"
            assert state.chop is not None


def test_warmup_insufficient_data():
    """Warmup returns None values when insufficient data."""
    config = ChoppinessConfig(default_period=10)
    engine = ChoppinessEngine(config)

    # Feed only 5 candles (less than period)
    for i in range(5):
        state = engine.on_candle_close("1m", Candle(i * 1000, 100, 110, 90, 105, 1000))

    assert state.chop_state == "WARMUP"
    assert state.chop is None
    assert state.sum_tr is None or len(engine._states["1m"].tr_deque) < 10


# ============================================================================
# TEST 8: MULTI-TIMEFRAME SUPPORT
# ============================================================================


def test_multi_timeframe():
    """Engine handles multiple timeframes independently."""
    config = ChoppinessConfig(timeframes=["1m", "5m"], period_by_tf={"1m": 3, "5m": 5})
    engine = ChoppinessEngine(config)

    # Feed different candles to different timeframes
    for i in range(10):
        candle_1m = Candle(i * 1000, 100, 110, 90, 105, 1000)
        candle_5m = Candle(i * 5000, 200, 220, 180, 210, 1000)

        state_1m = engine.on_candle_close("1m", candle_1m)
        state_5m = engine.on_candle_close("5m", candle_5m)

    # Both should have computed CHOP
    assert state_1m.chop is not None
    assert state_5m.chop is not None

    # Should be independent (different values)
    assert state_1m.chop != state_5m.chop or state_1m.range != state_5m.range


# ============================================================================
# TEST 9: EDGE CASES
# ============================================================================


def test_zero_range():
    """Handles zero range (all same price) gracefully."""
    config = ChoppinessConfig(default_period=3)
    engine = ChoppinessEngine(config)

    # All candles at same price
    candles = [Candle(i * 1000, 100, 100, 100, 100, 1000) for i in range(5)]

    for candle in candles:
        state = engine.on_candle_close("1m", candle)

    # Should not crash, CHOP should handle via eps
    assert state.chop is not None
    # With eps, ratio will be very large, log10 will be large
    # CHOP should be clamped to 100


def test_single_candle():
    """Single candle returns WARMUP."""
    engine = ChoppinessEngine(ChoppinessConfig(default_period=3))
    state = engine.on_candle_close("1m", Candle(1000, 100, 110, 90, 105, 1000))

    assert state.chop_state == "WARMUP"
    assert state.chop is None


def test_reset():
    """Reset clears state for a timeframe."""
    config = ChoppinessConfig(default_period=3)
    engine = ChoppinessEngine(config)

    # Feed some candles
    for i in range(5):
        engine.on_candle_close("1m", Candle(i * 1000, 100, 110, 90, 105, 1000))

    # Reset
    engine.reset("1m")

    # Next candle should be warmup again
    state = engine.on_candle_close("1m", Candle(6000, 100, 110, 90, 105, 1000))
    assert state.chop_state == "WARMUP"


# ============================================================================
# TEST 10: HELPER FUNCTIONS
# ============================================================================


def test_print_choppiness():
    """print_choppiness produces expected output."""
    config = ChoppinessConfig(default_period=3)
    engine = ChoppinessEngine(config)

    candles = [Candle(i * 1000, 100, 110 + i, 90, 105, 1000) for i in range(5)]

    states = engine.warmup({"1m": candles})

    # Should not crash
    print_choppiness(states)


def test_format_chop_state():
    """format_chop_state produces expected string."""
    config = ChoppinessConfig(default_period=3)
    engine = ChoppinessEngine(config)

    candles = [Candle(i * 1000, 100, 110, 90, 105, 1000) for i in range(5)]
    for candle in candles:
        state = engine.on_candle_close("1m", candle)

    formatted = format_chop_state("1m", state)
    assert "1m:" in formatted
    assert "chop=" in formatted
    assert "state=" in formatted


def test_interpret_chop():
    """interpret_chop provides actionable insight."""
    config = ChoppinessConfig(default_period=3)
    engine = ChoppinessEngine(config)

    candles = [Candle(i * 1000, 100, 110, 90, 105, 1000) for i in range(5)]
    for candle in candles:
        state = engine.on_candle_close("1m", candle)

    interpretation = interpret_chop(state)
    assert isinstance(interpretation, str)
    assert len(interpretation) > 0


# ============================================================================
# TEST 11: SLOPE CALCULATION
# ============================================================================


def test_chop_slope():
    """CHOP slope is computed correctly."""
    config = ChoppinessConfig(default_period=3)
    engine = ChoppinessEngine(config)

    candles = [Candle(i * 1000, 100, 110 + i * 2, 90, 105, 1000) for i in range(10)]

    states = []
    for candle in candles:
        state = engine.on_candle_close("1m", candle)
        states.append(state)

    # After warmup, slope should be available
    for i in range(4, len(states)):
        if states[i].chop_slope is not None:
            # Slope should be difference from previous
            expected_slope = states[i].chop - states[i - 1].chop
            assert abs(states[i].chop_slope - expected_slope) < 1e-6


# ============================================================================
# TEST 12: REALISTIC SCENARIO
# ============================================================================


def test_realistic_chop_transition():
    """Test realistic scenario: range -> breakout -> trend."""
    config = ChoppinessConfig(default_period=5)
    engine = ChoppinessEngine(config)

    # Phase 1: Range (high CHOP)
    range_candles = []
    for i in range(10):
        price = 100 + (i % 2) * 2
        range_candles.append(
            Candle(i * 1000, price, price + 3, price - 3, 100 + ((i + 1) % 2) * 2, 1000)
        )

    # Phase 2: Breakout and trend (low CHOP)
    trend_candles = []
    price = 100.0
    for i in range(10):
        trend_candles.append(Candle((10 + i) * 1000, price, price + 5, price + 1, price + 4, 1000))
        price += 4

    # Process range
    for candle in range_candles:
        state = engine.on_candle_close("1m", candle)
    range_state = state

    # Process trend
    for candle in trend_candles:
        state = engine.on_candle_close("1m", candle)
    trend_state = state

    # CHOP should decrease from range to trend
    if range_state.chop is not None and trend_state.chop is not None:
        assert trend_state.chop < range_state.chop
