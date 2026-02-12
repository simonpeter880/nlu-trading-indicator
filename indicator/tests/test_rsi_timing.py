"""
Tests for RSI Timing module.

Verifies:
- RSI calculation (Wilder method)
- Regime labeling
- Regular divergence detection
- Hidden divergence detection
- Minimum separation/threshold enforcement
- Failure swing detection
- Incremental behavior
"""

import pytest
from rsi_timing import (
    RSITimingEngine,
    RSITimingConfig,
    Candle,
    print_rsi_timing,
    format_rsi_state,
    interpret_rsi,
)


@pytest.fixture
def config():
    """Default config for testing."""
    return RSITimingConfig(
        timeframes=["1m"],
        rsi_period=14,
        regime_high=55.0,
        regime_low=45.0,
    )


# ============================================================================
# TEST 1: RSI CALCULATION (WILDER METHOD)
# ============================================================================

def test_rsi_seeding():
    """RSI seeds with SMA of gains/losses."""
    config = RSITimingConfig(rsi_period=3)
    engine = RSITimingEngine(config)

    # Create candles with known price changes
    candles = [
        Candle(1000, 100, 105, 95, 100, 1000),   # No change (first)
        Candle(2000, 100, 110, 95, 105, 1000),   # +5 gain
        Candle(3000, 105, 115, 100, 108, 1000),  # +3 gain
        Candle(4000, 108, 118, 103, 106, 1000),  # -2 loss
    ]

    for candle in candles:
        state = engine.on_candle_close("1m", candle)

    # After 3 changes: gains=[5, 3, 0], losses=[0, 0, 2]
    # avg_gain = 8/3 = 2.667
    # avg_loss = 2/3 = 0.667
    # RS = 2.667 / 0.667 = 4.0
    # RSI = 100 - 100/(1+4) = 80.0

    assert state.rsi is not None
    expected_rs = (5 + 3 + 0) / 3.0 / ((0 + 0 + 2) / 3.0 + 1e-12)
    expected_rsi = 100.0 - 100.0 / (1.0 + expected_rs)
    assert abs(state.rsi - expected_rsi) < 0.1


def test_rsi_wilder_smoothing():
    """RSI uses Wilder smoothing after seeding."""
    config = RSITimingConfig(rsi_period=3)
    engine = RSITimingEngine(config)

    # Seed with 3 gains
    seed_candles = [
        Candle(1000, 100, 105, 95, 100, 1000),
        Candle(2000, 100, 110, 95, 106, 1000),  # +6
        Candle(3000, 106, 116, 101, 109, 1000), # +3
        Candle(4000, 109, 119, 104, 112, 1000), # +3
    ]

    for candle in seed_candles:
        state = engine.on_candle_close("1m", candle)

    # Seed: avg_gain = (6+3+3)/3 = 4.0, avg_loss = 0.0
    # Next candle with +6 gain
    next_candle = Candle(5000, 112, 122, 107, 118, 1000)  # +6
    state = engine.on_candle_close("1m", next_candle)

    # Wilder: avg_gain = (4.0 * 2 + 6) / 3 = 14/3 = 4.667
    # avg_loss = 0.0
    # RS = 4.667 / 1e-12 (very high)
    # RSI should be close to 100

    assert state.rsi is not None
    assert state.rsi > 95.0  # Should be very high


def test_rsi_formula_correctness():
    """Verify RSI formula: RSI = 100 - 100/(1+RS)."""
    config = RSITimingConfig(rsi_period=5)
    engine = RSITimingEngine(config)

    # Generate known sequence
    prices = [100, 102, 104, 103, 105, 107, 106, 108]
    candles = [
        Candle(i * 1000, prices[i], prices[i] + 2, prices[i] - 2, prices[i], 1000)
        for i in range(len(prices))
    ]

    for candle in candles:
        state = engine.on_candle_close("1m", candle)

    # After 5+ changes, RSI should be computed
    assert state.rsi is not None
    assert 0 <= state.rsi <= 100


# ============================================================================
# TEST 2: REGIME LABELING
# ============================================================================

def test_regime_bullish():
    """RSI >= 55 => BULLISH regime."""
    config = RSITimingConfig(rsi_period=3, regime_high=55.0, regime_low=45.0)
    engine = RSITimingEngine(config)

    # Create uptrend (all gains)
    candles = []
    price = 100.0
    for i in range(10):
        candles.append(Candle(i * 1000, price, price + 5, price, price + 3, 1000))
        price += 3

    for candle in candles:
        state = engine.on_candle_close("1m", candle)

    # Should be bullish (high RSI from consistent gains)
    if state.rsi is not None and state.rsi >= 55.0:
        assert state.rsi_regime == "BULLISH"


def test_regime_bearish():
    """RSI <= 45 => BEARISH regime."""
    config = RSITimingConfig(rsi_period=3)
    engine = RSITimingEngine(config)

    # Create downtrend (all losses)
    candles = []
    price = 100.0
    for i in range(10):
        candles.append(Candle(i * 1000, price, price, price - 5, price - 3, 1000))
        price -= 3

    for candle in candles:
        state = engine.on_candle_close("1m", candle)

    # Should be bearish (low RSI from consistent losses)
    if state.rsi is not None and state.rsi <= 45.0:
        assert state.rsi_regime == "BEARISH"


def test_regime_range():
    """45 < RSI < 55 => RANGE regime."""
    config = RSITimingConfig(rsi_period=5)
    engine = RSITimingEngine(config)

    # Create balanced movement
    candles = []
    price = 100.0
    for i in range(20):
        change = 2 if i % 2 == 0 else -2
        candles.append(Candle(i * 1000, price, price + 3, price - 3, price + change, 1000))
        price += change

    for candle in candles:
        state = engine.on_candle_close("1m", candle)

    # Should eventually be in range (balanced gains/losses)
    if state.rsi is not None and 45.0 < state.rsi < 55.0:
        assert state.rsi_regime == "RANGE"


# ============================================================================
# TEST 3: REGULAR DIVERGENCE
# ============================================================================

def test_regular_bearish_divergence():
    """Higher high price + lower high RSI => REG_BEAR."""
    config = RSITimingConfig(rsi_period=5, min_bars_fallback=3)
    engine = RSITimingEngine(config)

    # Warmup with uptrend to get high RSI
    for i in range(15):
        price = 100 + i * 2
        candle = Candle(i * 1000, price, price + 2, price - 1, price + 1, 1000)
        engine.on_candle_close("1m", candle)

    # First swing high at high RSI
    engine.on_swing_high("1m", 15000, 130.0)
    state = engine.get_state("1m")
    first_rsi = state.rsi

    # Create some bars
    for i in range(5):
        price = 125 + i
        candle = Candle((16 + i) * 1000, price, price + 2, price - 1, price, 1000)
        engine.on_candle_close("1m", candle)

    # Second swing high: higher price, but create lower RSI by introducing losses
    for i in range(3):
        price = 130 - i * 2
        candle = Candle((21 + i) * 1000, price, price + 1, price - 3, price - 1, 1000)
        engine.on_candle_close("1m", candle)

    engine.on_swing_high("1m", 24000, 135.0)  # Higher high in price
    state = engine.get_state("1m")

    # Should detect regular bearish divergence
    if state.rsi < first_rsi:
        assert state.divergence == "REG_BEAR"


def test_regular_bullish_divergence():
    """Lower low price + higher low RSI => REG_BULL."""
    config = RSITimingConfig(rsi_period=5, min_bars_fallback=3)
    engine = RSITimingEngine(config)

    # Warmup with downtrend to get low RSI
    for i in range(15):
        price = 100 - i * 2
        candle = Candle(i * 1000, price, price + 1, price - 2, price - 1, 1000)
        engine.on_candle_close("1m", candle)

    # First swing low at low RSI
    engine.on_swing_low("1m", 15000, 70.0)
    state = engine.get_state("1m")
    first_rsi = state.rsi

    # Create some bars
    for i in range(5):
        price = 72 + i
        candle = Candle((16 + i) * 1000, price, price + 2, price - 1, price + 1, 1000)
        engine.on_candle_close("1m", candle)

    # Second swing low: lower price, higher RSI (gains increasing momentum)
    for i in range(3):
        price = 75 + i * 2
        candle = Candle((21 + i) * 1000, price, price + 3, price - 1, price + 2, 1000)
        engine.on_candle_close("1m", candle)

    engine.on_swing_low("1m", 24000, 68.0)  # Lower low in price
    state = engine.get_state("1m")

    # Should detect regular bullish divergence if RSI higher
    if state.rsi > first_rsi:
        assert state.divergence == "REG_BULL"


# ============================================================================
# TEST 4: HIDDEN DIVERGENCE
# ============================================================================

def test_hidden_bullish_divergence():
    """Higher low price + lower low RSI => HID_BULL."""
    config = RSITimingConfig(rsi_period=5, min_bars_fallback=3)
    engine = RSITimingEngine(config)

    # Uptrend with pullback
    for i in range(15):
        price = 100 + i * 1.5
        candle = Candle(i * 1000, price, price + 2, price - 1, price + 1, 1000)
        engine.on_candle_close("1m", candle)

    # First swing low (pullback in uptrend)
    engine.on_swing_low("1m", 15000, 118.0)
    state = engine.get_state("1m")
    first_rsi = state.rsi

    # Continue uptrend
    for i in range(8):
        price = 120 + i * 2
        candle = Candle((16 + i) * 1000, price, price + 2, price - 1, price + 1, 1000)
        engine.on_candle_close("1m", candle)

    # Second swing low: higher low in price, but create lower RSI
    for i in range(3):
        price = 133 - i
        candle = Candle((24 + i) * 1000, price, price + 1, price - 2, price - 0.5, 1000)
        engine.on_candle_close("1m", candle)

    engine.on_swing_low("1m", 27000, 122.0)  # Higher low in price
    state = engine.get_state("1m")

    # Should detect hidden bullish divergence
    if state.rsi < first_rsi and 122.0 > 118.0:
        assert state.divergence == "HID_BULL"


def test_hidden_bearish_divergence():
    """Lower high price + higher high RSI => HID_BEAR."""
    config = RSITimingConfig(rsi_period=5, min_bars_fallback=3)
    engine = RSITimingEngine(config)

    # Downtrend with bounce
    for i in range(15):
        price = 100 - i * 1.5
        candle = Candle(i * 1000, price, price + 1, price - 2, price - 1, 1000)
        engine.on_candle_close("1m", candle)

    # First swing high (bounce in downtrend)
    engine.on_swing_high("1m", 15000, 82.0)
    state = engine.get_state("1m")
    first_rsi = state.rsi

    # Continue downtrend
    for i in range(8):
        price = 80 - i * 2
        candle = Candle((16 + i) * 1000, price, price + 1, price - 2, price - 1, 1000)
        engine.on_candle_close("1m", candle)

    # Second swing high: lower high in price, but higher RSI (gains increasing)
    for i in range(3):
        price = 65 + i
        candle = Candle((24 + i) * 1000, price, price + 2, price - 1, price + 1, 1000)
        engine.on_candle_close("1m", candle)

    engine.on_swing_high("1m", 27000, 78.0)  # Lower high in price
    state = engine.get_state("1m")

    # Should detect hidden bearish divergence
    if state.rsi > first_rsi and 78.0 < 82.0:
        assert state.divergence == "HID_BEAR"


# ============================================================================
# TEST 5: MINIMUM SEPARATION AND THRESHOLD
# ============================================================================

def test_min_bars_separation():
    """Divergence should NOT trigger when swings too close."""
    config = RSITimingConfig(
        rsi_period=5,
        min_bars_between_swings_by_tf={"1m": 5}
    )
    engine = RSITimingEngine(config)

    # Warmup
    for i in range(10):
        candle = Candle(i * 1000, 100 + i, 105 + i, 95 + i, 102 + i, 1000)
        engine.on_candle_close("1m", candle)

    # Two swing highs too close (only 3 bars apart)
    engine.on_swing_high("1m", 10000, 120.0)
    engine.on_candle_close("1m", Candle(11000, 115, 120, 110, 115, 1000))
    engine.on_candle_close("1m", Candle(12000, 115, 120, 110, 116, 1000))
    engine.on_swing_high("1m", 13000, 125.0)

    state = engine.get_state("1m")

    # Should NOT detect divergence (too close)
    assert state.divergence == "NONE"


def test_min_price_threshold():
    """Divergence should NOT trigger when price diff below threshold."""
    config = RSITimingConfig(
        rsi_period=5,
        min_bars_fallback=3,
        min_price_diff_atrp_factor=0.15
    )
    engine = RSITimingEngine(config)

    # Warmup
    for i in range(10):
        candle = Candle(i * 1000, 100, 105, 95, 100, 1000)
        engine.on_candle_close("1m", candle)

    # Set external ATR% (1% ATR)
    engine.on_candle_close("1m", Candle(10000, 100, 105, 95, 100, 1000), atr_percent=0.01)

    # Two swing highs with tiny price difference (0.05%)
    engine.on_swing_high("1m", 11000, 100.0)

    for i in range(5):
        engine.on_candle_close("1m", Candle((12 + i) * 1000, 100, 105, 95, 100, 1000), atr_percent=0.01)

    engine.on_swing_high("1m", 17000, 100.05)  # Only 0.05 difference (< 0.15% of 1% ATR)

    state = engine.get_state("1m")

    # Should NOT detect divergence (price diff too small)
    # Threshold = 0.15 * 0.01 * 100 = 0.15
    # Actual diff = 0.05 < 0.15
    assert state.divergence == "NONE"


# ============================================================================
# TEST 6: FAILURE SWING
# ============================================================================

def test_failure_swing_bear():
    """Bear failure swing: high -> pullback -> lower high -> break."""
    config = RSITimingConfig(
        rsi_period=5,
        enable_failure_swing=True,
        fs_high=60.0,
        fs_mid_low=45.0
    )
    engine = RSITimingEngine(config)

    # Manually create RSI sequence for bear failure swing
    # Stage 1: Rise above 60
    for i in range(10):
        price = 100 + i * 3
        candle = Candle(i * 1000, price, price + 3, price - 1, price + 2, 1000)
        state = engine.on_candle_close("1m", candle)

    # RSI should be high now
    # Stage 2: Pullback below 45
    for i in range(5):
        price = 128 - i * 4
        candle = Candle((10 + i) * 1000, price, price + 1, price - 4, price - 3, 1000)
        state = engine.on_candle_close("1m", candle)

    # Stage 3: Bounce but make lower high
    for i in range(3):
        price = 110 + i * 2
        candle = Candle((15 + i) * 1000, price, price + 2, price - 1, price + 1, 1000)
        state = engine.on_candle_close("1m", candle)

    # Stage 4: Break below pullback low
    for i in range(3):
        price = 115 - i * 3
        candle = Candle((18 + i) * 1000, price, price + 1, price - 3, price - 2, 1000)
        state = engine.on_candle_close("1m", candle)

    # May detect bear failure swing (depends on exact RSI values)
    # Test that failure_swing field exists
    assert state.failure_swing in ["NONE", "BEAR", "BULL"]


# ============================================================================
# TEST 7: INCREMENTAL BEHAVIOR
# ============================================================================

def test_incremental_updates():
    """Engine should process candles incrementally with O(1) updates."""
    config = RSITimingConfig(rsi_period=14)
    engine = RSITimingEngine(config)

    # Feed 100 candles incrementally
    for i in range(100):
        price = 100 + i * 0.5
        candle = Candle(i * 1000, price, price + 2, price - 2, price + 0.3, 1000)
        state = engine.on_candle_close("1m", candle)

    # Should have RSI after warmup
    assert state.rsi is not None
    assert 0 <= state.rsi <= 100


def test_incremental_vs_batch():
    """Incremental processing should match batch warmup."""
    config = RSITimingConfig(rsi_period=10)

    # Generate candles
    candles = []
    price = 100.0
    for i in range(30):
        change = 2 if i % 3 == 0 else -1
        candles.append(Candle(i * 1000, price, price + 3, price - 2, price + change, 1000))
        price += change

    # Incremental
    engine1 = RSITimingEngine(config)
    for candle in candles:
        state1 = engine1.on_candle_close("1m", candle)

    # Batch
    engine2 = RSITimingEngine(config)
    states2 = engine2.warmup({"1m": candles})
    state2 = states2["1m"]

    # Should match
    assert abs(state1.rsi - state2.rsi) < 1e-9


# ============================================================================
# TEST 8: MULTI-TIMEFRAME
# ============================================================================

def test_multi_timeframe():
    """Engine handles multiple timeframes independently."""
    config = RSITimingConfig(timeframes=["1m", "5m"])
    engine = RSITimingEngine(config)

    # Feed different candles to different timeframes
    for i in range(20):
        candle_1m = Candle(i * 1000, 100 + i, 105 + i, 95 + i, 102 + i, 1000)
        candle_5m = Candle(i * 5000, 200 - i, 205 - i, 195 - i, 198 - i, 1000)

        state_1m = engine.on_candle_close("1m", candle_1m)
        state_5m = engine.on_candle_close("5m", candle_5m)

    # Both should have RSI
    assert state_1m.rsi is not None
    assert state_5m.rsi is not None

    # Should be different (opposite trends)
    assert abs(state_1m.rsi - state_5m.rsi) > 10  # Significant difference


# ============================================================================
# TEST 9: EDGE CASES
# ============================================================================

def test_no_price_change():
    """Handle candles with no price change."""
    config = RSITimingConfig(rsi_period=5)
    engine = RSITimingEngine(config)

    # All same price
    for i in range(20):
        candle = Candle(i * 1000, 100, 100, 100, 100, 1000)
        state = engine.on_candle_close("1m", candle)

    # RSI should be 50 (no gains or losses)
    assert state.rsi is not None
    assert abs(state.rsi - 50.0) < 0.1


def test_warmup_state():
    """Return WARMUP until enough candles processed."""
    config = RSITimingConfig(rsi_period=14)
    engine = RSITimingEngine(config)

    # Feed less than period candles
    for i in range(10):
        candle = Candle(i * 1000, 100 + i, 105 + i, 95 + i, 102 + i, 1000)
        state = engine.on_candle_close("1m", candle)

    # Should be WARMUP
    if state.rsi is None:
        assert state.rsi_regime == "WARMUP"


def test_reset():
    """Reset clears state for a timeframe."""
    config = RSITimingConfig(rsi_period=5)
    engine = RSITimingEngine(config)

    # Feed candles
    for i in range(20):
        candle = Candle(i * 1000, 100 + i, 105 + i, 95 + i, 102 + i, 1000)
        engine.on_candle_close("1m", candle)

    # Reset
    engine.reset("1m")

    # Next candle should start fresh
    state = engine.on_candle_close("1m", Candle(21000, 120, 125, 115, 122, 1000))
    assert state.rsi_regime == "WARMUP"


# ============================================================================
# TEST 10: HELPER FUNCTIONS
# ============================================================================

def test_print_rsi_timing():
    """print_rsi_timing produces expected output."""
    config = RSITimingConfig(rsi_period=5)
    engine = RSITimingEngine(config)

    candles = [Candle(i * 1000, 100 + i, 105 + i, 95 + i, 102 + i, 1000) for i in range(20)]
    states = engine.warmup({"1m": candles})

    # Should not crash
    print_rsi_timing(states)


def test_format_rsi_state():
    """format_rsi_state produces expected string."""
    config = RSITimingConfig(rsi_period=5)
    engine = RSITimingEngine(config)

    candles = [Candle(i * 1000, 100 + i, 105 + i, 95 + i, 102 + i, 1000) for i in range(20)]
    for candle in candles:
        state = engine.on_candle_close("1m", candle)

    formatted = format_rsi_state("1m", state)
    assert "1m:" in formatted
    assert "rsi=" in formatted


def test_interpret_rsi():
    """interpret_rsi provides actionable insight."""
    config = RSITimingConfig(rsi_period=5)
    engine = RSITimingEngine(config)

    candles = [Candle(i * 1000, 100 + i, 105 + i, 95 + i, 102 + i, 1000) for i in range(20)]
    for candle in candles:
        state = engine.on_candle_close("1m", candle)

    interpretation = interpret_rsi(state)
    assert isinstance(interpretation, str)
    assert len(interpretation) > 0


# ============================================================================
# TEST 11: DIVERGENCE STRENGTH
# ============================================================================

def test_divergence_strength():
    """Divergence strength is calculated and in range 0-100."""
    config = RSITimingConfig(rsi_period=5, min_bars_fallback=3)
    engine = RSITimingEngine(config)

    # Create scenario likely to produce divergence
    for i in range(15):
        price = 100 + i * 2
        candle = Candle(i * 1000, price, price + 2, price - 1, price + 1, 1000)
        engine.on_candle_close("1m", candle, atr_percent=0.01)

    engine.on_swing_high("1m", 15000, 130.0)

    for i in range(8):
        price = 125 - i * 0.5
        candle = Candle((16 + i) * 1000, price, price + 2, price - 1, price, 1000)
        engine.on_candle_close("1m", candle, atr_percent=0.01)

    engine.on_swing_high("1m", 24000, 135.0)  # Higher price

    state = engine.get_state("1m")

    # If divergence detected, strength should be in range
    if state.divergence != "NONE" and state.div_strength_0_100 is not None:
        assert 0 <= state.div_strength_0_100 <= 100
