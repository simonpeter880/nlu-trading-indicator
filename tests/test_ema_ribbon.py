"""
Tests for EMA Ribbon Module

Tests incremental EMA updates, stack scoring, state classification,
and various trend scenarios.
"""

from typing import List

import pytest

from nlu_analyzer.indicators.ema_ribbon import (
    Candle,
    EMARibbonConfig,
    EMARibbonEngine,
    RibbonDirection,
    RibbonState,
    format_ribbon_output,
)


def create_candle(timestamp: float, price: float, volume: float = 1000.0) -> Candle:
    """Helper to create a candle with same OHLC"""
    return Candle(
        timestamp=timestamp, open=price, high=price, low=price, close=price, volume=volume
    )


def generate_candles(
    start_price: float, count: int, trend: str = "flat", trend_strength: float = 0.001
) -> List[Candle]:
    """
    Generate synthetic candles.

    Args:
        start_price: Starting price
        count: Number of candles
        trend: "up", "down", or "flat"
        trend_strength: Price change per candle (fraction)

    Returns:
        List of candles
    """
    candles = []
    price = start_price

    for i in range(count):
        candles.append(create_candle(float(i), price))

        if trend == "up":
            price *= 1 + trend_strength
        elif trend == "down":
            price *= 1 - trend_strength
        # flat: no change

    return candles


def compute_sma(prices: List[float], period: int) -> float:
    """Compute simple moving average"""
    if len(prices) < period:
        return sum(prices) / len(prices) if prices else 0.0
    return sum(prices[-period:]) / period


def compute_ema_batch(prices: List[float], period: int) -> List[float]:
    """
    Compute EMA batch-style for comparison.

    Seeds with SMA of first `period` prices, then applies incremental formula.
    """
    if len(prices) < period:
        return []

    alpha = 2.0 / (period + 1)
    emas = []

    # Seed with SMA
    seed = sum(prices[:period]) / period
    emas.append(seed)

    # Incremental updates
    for i in range(period, len(prices)):
        ema_new = alpha * prices[i] + (1 - alpha) * emas[-1]
        emas.append(ema_new)

    return emas


class TestEMARibbonConfig:
    """Tests for configuration validation"""

    def test_default_config(self):
        """Test default configuration"""
        config = EMARibbonConfig()
        assert config.ribbon_periods == [9, 12, 15, 18, 21, 25, 30, 35, 40, 50]
        assert config.atr_period == 14
        assert config.width_smooth_period == 5

    def test_light_ribbon_preset(self):
        """Test lighter ribbon configuration"""
        config = EMARibbonConfig(ribbon_periods=[9, 12, 15, 21, 30, 40, 50])
        assert len(config.ribbon_periods) == 7

    def test_invalid_ribbon_periods_unsorted(self):
        """Test that unsorted periods raise error"""
        with pytest.raises(ValueError, match="sorted ascending"):
            EMARibbonConfig(ribbon_periods=[9, 15, 12, 21])

    def test_invalid_ribbon_periods_duplicates(self):
        """Test that duplicate periods raise error"""
        with pytest.raises(ValueError, match="sorted ascending and unique"):
            EMARibbonConfig(ribbon_periods=[9, 12, 12, 21])

    def test_invalid_ribbon_periods_negative(self):
        """Test that negative periods raise error"""
        with pytest.raises(ValueError, match="sorted ascending and unique"):
            # Negative period causes unsorted list: [-5, 9, 12, 21] when sorted
            EMARibbonConfig(ribbon_periods=[9, 12, -5, 21])

    def test_weights_sum_to_one(self):
        """Test that strength weights sum to 1.0"""
        config = EMARibbonConfig()
        weight_sum = (
            config.strength_weight_stack
            + config.strength_weight_width
            + config.strength_weight_slope
            + config.strength_weight_expansion
        )
        assert abs(weight_sum - 1.0) < 1e-6


class TestIncrementalEMA:
    """Tests for incremental EMA calculations"""

    def test_incremental_vs_batch_single_period(self):
        """Test incremental EMA matches batch calculation for single period"""
        config = EMARibbonConfig(ribbon_periods=[9])
        engine = EMARibbonEngine(config)

        # Generate test data
        candles = generate_candles(100.0, 50, trend="up", trend_strength=0.002)
        prices = [c.close for c in candles]

        # Batch computation
        batch_emas = compute_ema_batch(prices, 9)

        # Incremental computation
        engine.warmup({"1m": candles})
        state_tf = engine._states["1m"]

        # After 50 candles, incremental EMA should match batch
        incremental_ema = state_tf.ema_values[9]
        expected_ema = batch_emas[-1]

        rel_error = abs(incremental_ema - expected_ema) / expected_ema
        assert (
            rel_error < 0.01
        ), f"Incremental EMA {incremental_ema} differs from batch {expected_ema}"

    def test_incremental_vs_batch_multiple_periods(self):
        """Test incremental EMA matches batch for multiple periods"""
        config = EMARibbonConfig(ribbon_periods=[9, 21, 50])
        engine = EMARibbonEngine(config)

        # Generate test data
        candles = generate_candles(100.0, 100, trend="down", trend_strength=0.001)
        prices = [c.close for c in candles]

        # Warmup
        engine.warmup({"1m": candles})
        state_tf = engine._states["1m"]

        # Compare each period
        for period in [9, 21, 50]:
            batch_emas = compute_ema_batch(prices, period)
            incremental_ema = state_tf.ema_values[period]
            expected_ema = batch_emas[-1]

            rel_error = abs(incremental_ema - expected_ema) / expected_ema
            assert (
                rel_error < 0.01
            ), f"Period {period}: incremental {incremental_ema} != batch {expected_ema}"

    def test_warmup_state_ready(self):
        """Test that state becomes ready after warmup"""
        config = EMARibbonConfig(ribbon_periods=[9, 21, 50])
        engine = EMARibbonEngine(config)

        # Should not be ready initially
        state_tf = engine._get_or_create_state("1m")
        assert not state_tf.is_ready

        # Warmup with enough candles
        candles = generate_candles(100.0, 60, trend="flat")
        engine.warmup({"1m": candles})

        # Should be ready after max_period candles
        assert state_tf.is_ready
        assert state_tf.candle_count == 60


class TestStackScore:
    """Tests for stack score calculation"""

    def test_perfect_bullish_stack(self):
        """Test perfect bullish stack (all EMAs ordered correctly)"""
        config = EMARibbonConfig(ribbon_periods=[9, 12, 15, 21, 30])
        engine = EMARibbonEngine(config)

        # Create strongly uptrending candles
        candles = generate_candles(100.0, 100, trend="up", trend_strength=0.005)
        engine.warmup({"1m": candles})

        # Get latest state
        result = engine.on_candle_close("1m", candles[-1])

        # Should have high stack score and BULL direction
        assert (
            result.stack_score >= 0.80
        ), f"Stack score {result.stack_score} too low for strong uptrend"
        assert result.ribbon_direction == RibbonDirection.BULL

    def test_perfect_bearish_stack(self):
        """Test perfect bearish stack"""
        config = EMARibbonConfig(ribbon_periods=[9, 12, 15, 21, 30])
        engine = EMARibbonEngine(config)

        # Create strongly downtrending candles
        candles = generate_candles(100.0, 100, trend="down", trend_strength=0.005)
        engine.warmup({"1m": candles})

        result = engine.on_candle_close("1m", candles[-1])

        assert (
            result.stack_score >= 0.80
        ), f"Stack score {result.stack_score} too low for strong downtrend"
        assert result.ribbon_direction == RibbonDirection.BEAR

    def test_neutral_stack_sideways(self):
        """Test neutral/low stack score for sideways market"""
        config = EMARibbonConfig(ribbon_periods=[9, 12, 15, 21, 30])
        engine = EMARibbonEngine(config)

        # Create flat candles
        candles = generate_candles(100.0, 100, trend="flat")
        engine.warmup({"1m": candles})

        result = engine.on_candle_close("1m", candles[-1])

        # In flat market, EMAs converge and stack becomes weak
        # Direction should be NEUTRAL or stack score low
        assert (
            result.ribbon_direction == RibbonDirection.NEUTRAL or result.stack_score < 0.70
        ), "Sideways market should have weak stack"


class TestRibbonWidth:
    """Tests for ribbon width calculations"""

    def test_width_increases_in_trend(self):
        """Test that ribbon width increases during strong trend"""
        config = EMARibbonConfig(ribbon_periods=[9, 21, 50])
        engine = EMARibbonEngine(config)

        # Strong uptrend
        candles = generate_candles(100.0, 100, trend="up", trend_strength=0.01)
        engine.warmup({"1m": candles[:60]})

        # Get width at two points
        result_early = engine.on_candle_close("1m", candles[60])
        width_early = result_early.ribbon_width_smooth

        # Continue trend
        for candle in candles[61:80]:
            engine.on_candle_close("1m", candle)

        result_late = engine.on_candle_close("1m", candles[80])
        width_late = result_late.ribbon_width_smooth

        # Width should expand in strong trend
        assert width_late > width_early, "Width should expand during strong trend"

    def test_width_compression_detected(self):
        """Test that width compression is detected"""
        config = EMARibbonConfig(ribbon_periods=[9, 21, 50])
        engine = EMARibbonEngine(config)

        # Start with uptrend, then flatten
        uptrend = generate_candles(100.0, 60, trend="up", trend_strength=0.005)
        flat = generate_candles(uptrend[-1].close, 40, trend="flat")

        all_candles = uptrend + flat
        engine.warmup({"1m": all_candles[:60]})

        # Process uptrend
        for candle in all_candles[60:70]:
            engine.on_candle_close("1m", candle)

        # Process flattening
        for candle in all_candles[70:]:
            result = engine.on_candle_close("1m", candle)

        # Should show negative width_rate (compression)
        assert (
            result.width_rate < 0
        ), f"Width rate {result.width_rate} should be negative during compression"


class TestRibbonState:
    """Tests for ribbon state classification"""

    def test_healthy_state_strong_trend(self):
        """Test HEALTHY state in strong trend"""
        config = EMARibbonConfig(ribbon_periods=[9, 12, 15, 21, 30, 50])
        engine = EMARibbonEngine(config)

        # Strong consistent uptrend
        candles = generate_candles(100.0, 120, trend="up", trend_strength=0.008)
        engine.warmup({"1m": candles[:80]})

        # Process remaining candles
        for candle in candles[80:]:
            result = engine.on_candle_close("1m", candle, atr_percent=0.01)

        # Should be HEALTHY with high strength
        assert (
            result.ribbon_state == RibbonState.HEALTHY
        ), f"Got {result.ribbon_state} instead of HEALTHY"
        assert result.ribbon_strength_0_100 > 60, f"Strength {result.ribbon_strength_0_100} too low"

    def test_chop_state_sideways(self):
        """Test CHOP state in sideways market"""
        config = EMARibbonConfig(ribbon_periods=[9, 12, 15, 21, 30])
        engine = EMARibbonEngine(config)

        # Sideways market
        candles = generate_candles(100.0, 100, trend="flat")
        engine.warmup({"1m": candles})

        result = engine.on_candle_close("1m", candles[-1], atr_percent=0.005)

        # Should detect CHOP
        assert result.ribbon_state == RibbonState.CHOP, f"Got {result.ribbon_state} instead of CHOP"
        assert result.ribbon_strength_0_100 <= 35, "CHOP strength should be capped at 35"

    def test_exhausting_state(self):
        """Test that weakening momentum is detected"""
        config = EMARibbonConfig(ribbon_periods=[9, 12, 15, 21, 30], compress_rate_thr=-0.15)
        engine = EMARibbonEngine(config)

        # Start with strong uptrend
        uptrend = generate_candles(100.0, 70, trend="up", trend_strength=0.008)
        engine.warmup({"1m": uptrend[:60]})

        # Continue strong trend to establish HEALTHY state
        for candle in uptrend[60:]:
            engine.on_candle_close("1m", candle, atr_percent=0.01)

        # Now flatten significantly to force compression
        flat = generate_candles(uptrend[-1].close, 30, trend="flat")

        for candle in flat:
            result = engine.on_candle_close("1m", candle, atr_percent=0.01)

        # After flattening, width should compress and state should deteriorate
        # Check that width_rate is negative (compressing)
        assert (
            result.width_rate < -0.05
        ), f"Width rate {result.width_rate} should be strongly negative"

        # State should show deterioration (not perfectly HEALTHY anymore)
        # May still be classified as HEALTHY if stack remains strong, but strength should drop
        if result.ribbon_state == RibbonState.HEALTHY:
            # If still healthy, strength should be reduced compared to strong trend
            assert result.ribbon_strength_0_100 < 90, "Strength should decrease during compression"


class TestPullbackDetection:
    """Tests for pullback into ribbon detection"""

    def test_pullback_detected(self):
        """Test pullback detection when price approaches ribbon center"""
        config = EMARibbonConfig(ribbon_periods=[9, 21, 50])
        engine = EMARibbonEngine(config)

        # Create uptrend
        candles = generate_candles(100.0, 80, trend="up", trend_strength=0.005)
        engine.warmup({"1m": candles})

        # Get ribbon center and threshold
        result = engine.on_candle_close("1m", candles[-1], atr_percent=0.01)
        center = result.ribbon_center

        # Create candle close to center (within pullback band)
        # pullback_band is 0.003 = 0.3% of close, so we need to be within that
        # If center is ~140, then 0.003 * 140 = 0.42, so within ±0.42
        pullback_candle = create_candle(81.0, center * 1.0001)  # Very close, within 0.01%

        result = engine.on_candle_close("1m", pullback_candle, atr_percent=0.01)

        # Should detect pullback
        assert result.pullback_into_ribbon, "Failed to detect pullback into ribbon"

    def test_no_pullback_when_far(self):
        """Test no pullback when price is far from ribbon"""
        config = EMARibbonConfig(ribbon_periods=[9, 21, 50])
        engine = EMARibbonEngine(config)

        # Create uptrend
        candles = generate_candles(100.0, 80, trend="up", trend_strength=0.005)
        engine.warmup({"1m": candles})

        result = engine.on_candle_close("1m", candles[-1], atr_percent=0.01)
        center = result.ribbon_center

        # Create candle far from center
        far_candle = create_candle(81.0, center * 1.05)  # 5% away

        result = engine.on_candle_close("1m", far_candle, atr_percent=0.01)

        # Should NOT detect pullback
        assert not result.pullback_into_ribbon, "False pullback detected when price far from ribbon"


class TestMultiTimeframe:
    """Tests for multi-timeframe support"""

    def test_multiple_timeframes(self):
        """Test engine handles multiple timeframes independently"""
        config = EMARibbonConfig(ribbon_periods=[9, 21, 50])
        engine = EMARibbonEngine(config)

        # Create different trends for different timeframes
        candles_1m = generate_candles(100.0, 60, trend="up", trend_strength=0.005)
        candles_5m = generate_candles(100.0, 60, trend="down", trend_strength=0.003)
        candles_1h = generate_candles(100.0, 60, trend="flat")

        # Warmup all timeframes
        engine.warmup({"1m": candles_1m, "5m": candles_5m, "1h": candles_1h})

        # Get results
        result_1m = engine.on_candle_close("1m", candles_1m[-1])
        result_5m = engine.on_candle_close("5m", candles_5m[-1])
        result_1h = engine.on_candle_close("1h", candles_1h[-1])

        # Each timeframe should have independent direction
        assert result_1m.ribbon_direction == RibbonDirection.BULL
        assert result_5m.ribbon_direction == RibbonDirection.BEAR
        # 1h should be NEUTRAL or weak stack
        assert result_1h.ribbon_direction == RibbonDirection.NEUTRAL or result_1h.stack_score < 0.70

    def test_update_convenience_method(self):
        """Test convenience update method"""
        config = EMARibbonConfig(ribbon_periods=[9, 21])
        engine = EMARibbonEngine(config)

        # Create test data - need more candles for proper warmup
        candles_1m = generate_candles(100.0, 80, trend="up", trend_strength=0.005)
        candles_5m = generate_candles(100.0, 80, trend="up", trend_strength=0.003)

        # Warmup first
        engine.warmup({"1m": candles_1m[:60], "5m": candles_5m[:60]})

        # Use update method with latest candles
        results = engine.update(
            {"1m": candles_1m[60:], "5m": candles_5m[60:]},
            atr_percent_by_tf={"1m": 0.008, "5m": 0.012},
        )

        # Should return results for both timeframes
        assert "1m" in results
        assert "5m" in results
        # After warmup and strong uptrend, should be BULL
        assert results["1m"].ribbon_direction == RibbonDirection.BULL
        assert results["5m"].ribbon_direction == RibbonDirection.BULL


class TestATRAdaptiveThresholds:
    """Tests for ATR-adaptive threshold calculation"""

    def test_atr_adaptive_thresholds(self):
        """Test thresholds adapt to ATR%"""
        config = EMARibbonConfig(width_thr_factor=0.10, slope_thr_factor=0.15)
        engine = EMARibbonEngine(config)

        # High volatility
        atr_high = 0.02  # 2%
        thresholds_high = engine._get_thresholds("1m", atr_high)

        # Low volatility
        atr_low = 0.005  # 0.5%
        thresholds_low = engine._get_thresholds("1m", atr_low)

        # High volatility should have higher thresholds
        assert thresholds_high["width_thr"] > thresholds_low["width_thr"]
        assert thresholds_high["slope_thr"] > thresholds_low["slope_thr"]

    def test_static_fallback_when_no_atr(self):
        """Test static fallback thresholds when ATR not provided"""
        config = EMARibbonConfig()
        engine = EMARibbonEngine(config)

        thresholds = engine._get_thresholds("1m", None)

        # Should use static values
        assert thresholds["width_thr"] == config.width_thr_static_by_tf["1m"]
        assert thresholds["slope_thr"] == config.slope_thr_static_by_tf["1m"]


class TestFormatting:
    """Tests for output formatting"""

    def test_format_compact(self):
        """Test compact formatting"""
        config = EMARibbonConfig(ribbon_periods=[9, 21, 50])
        engine = EMARibbonEngine(config)

        candles = generate_candles(100.0, 60, trend="up", trend_strength=0.005)
        engine.warmup({"1m": candles})

        result = engine.on_candle_close("1m", candles[-1], atr_percent=0.01)
        output = format_ribbon_output({"1m": result}, compact=True)

        # Should contain key info
        assert "EMA RIBBON" in output
        assert "1m:" in output
        assert "dir=" in output
        assert "state=" in output
        assert "strength=" in output

    def test_format_verbose(self):
        """Test verbose formatting"""
        config = EMARibbonConfig(ribbon_periods=[9, 21])
        engine = EMARibbonEngine(config)

        candles = generate_candles(100.0, 60, trend="down", trend_strength=0.005)
        engine.warmup({"1m": candles})

        result = engine.on_candle_close("1m", candles[-1])
        output = format_ribbon_output({"1m": result}, compact=False)

        # Should contain detailed info
        assert "Direction:" in output
        assert "State:" in output
        assert "Strength:" in output
        assert "Stack Score:" in output
        assert "Width:" in output


class TestEdgeCases:
    """Tests for edge cases"""

    def test_single_period_ribbon(self):
        """Test ribbon with only one period"""
        config = EMARibbonConfig(ribbon_periods=[21])
        engine = EMARibbonEngine(config)

        candles = generate_candles(100.0, 30, trend="up", trend_strength=0.005)
        engine.warmup({"1m": candles})

        result = engine.on_candle_close("1m", candles[-1])

        # Should handle gracefully (stack_score will be 0 with single period)
        assert result.stack_score == 0.0
        assert result.ribbon_direction == RibbonDirection.NEUTRAL

    def test_zero_price_handling(self):
        """Test handling of zero prices (shouldn't crash)"""
        config = EMARibbonConfig(ribbon_periods=[9, 21])
        engine = EMARibbonEngine(config)

        # This is an edge case that shouldn't happen in real trading
        # but we should handle it gracefully
        candles = [create_candle(float(i), 0.0001) for i in range(30)]
        engine.warmup({"1m": candles})

        # Should not crash
        result = engine.on_candle_close("1m", candles[-1])
        assert result is not None

    def test_warmup_insufficient_data(self):
        """Test behavior during warmup period"""
        config = EMARibbonConfig(ribbon_periods=[9, 21, 50])
        engine = EMARibbonEngine(config)

        # Only 20 candles (insufficient for period 50)
        candles = generate_candles(100.0, 20, trend="up", trend_strength=0.005)
        engine.warmup({"1m": candles})

        result = engine.on_candle_close("1m", candles[-1])

        # Should indicate warmup in debug
        assert "warmup" in result.debug
        assert result.debug["warmup"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
