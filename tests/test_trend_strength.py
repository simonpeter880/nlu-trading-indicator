"""
Tests for Trend Strength Module

Tests component normalization, weighting, smoothing, and bucketing.
"""

import pytest

from nlu_analyzer.indicators.trend_strength import (
    Bucket,
    Candle,
    TrendStrengthConfig,
    TrendStrengthEngine,
    format_trend_strength_output,
)


def create_candle(timestamp: float, high: float, low: float, close: float, volume: float) -> Candle:
    """Helper to create a candle"""
    return Candle(timestamp=timestamp, open=close, high=high, low=low, close=close, volume=volume)


def create_simple_candle(timestamp: float, price: float, volume: float = 1000.0) -> Candle:
    """Helper to create candle with same OHLC"""
    return create_candle(timestamp, price, price, price, volume)


class TestComponentNormalization:
    """Tests for individual component normalization"""

    def test_ema_slope_normalization(self):
        """Test EMA slope hits ~1.0 when slope == factor × ATR%"""
        config = TrendStrengthConfig(atr_period=5, ema_slope_strong_factor=0.20)
        engine = TrendStrengthEngine(config)

        # Warmup to establish ATR
        candles = []
        for i in range(10):
            price = 100.0 + i * 0.5
            candles.append(create_simple_candle(float(i), price, 1000.0))

        engine.warmup({"1m": candles})

        # Simulate slope matching strong factor
        # If ATR% = 0.01, strong slope = 0.20 × 0.01 = 0.002
        result = engine.on_candle_close(
            "1m",
            create_simple_candle(10.0, 105.0, 1000.0),
            slope_50=0.002,  # Matches strong threshold
            atr_percent=0.01,
        )

        # Should normalize to ~1.0
        assert "ema_slope" in result.components_norm
        assert 0.8 < result.components_norm["ema_slope"] <= 1.0

    def test_ribbon_normalization_range(self):
        """Test ribbon WR maps correctly to 0..1"""
        config = TrendStrengthConfig(ribbon_wr_low=-0.10, ribbon_wr_high=0.20)
        engine = TrendStrengthEngine(config)

        # Warmup
        candles = [create_simple_candle(float(i), 100.0, 1000.0) for i in range(15)]
        engine.warmup({"1m": candles})

        # Test at low boundary
        result_low = engine.on_candle_close(
            "1m", create_simple_candle(15.0, 100.0, 1000.0), ribbon_width_rate=-0.10
        )
        assert result_low.components_norm["ribbon"] == 0.0

        # Test at high boundary
        result_high = engine.on_candle_close(
            "1m", create_simple_candle(16.0, 100.0, 1000.0), ribbon_width_rate=0.20
        )
        assert abs(result_high.components_norm["ribbon"] - 1.0) < 1e-6

        # Test at mid-point
        result_mid = engine.on_candle_close(
            "1m",
            create_simple_candle(17.0, 100.0, 1000.0),
            ribbon_width_rate=0.05,  # Mid between -0.10 and 0.20
        )
        assert 0.4 < result_mid.components_norm["ribbon"] < 0.6

    def test_rv_normalization_saturation(self):
        """Test RV normalization with saturation"""
        config = TrendStrengthConfig(rv_period=10, rv_low=0.8, rv_high=2.0)
        engine = TrendStrengthEngine(config)

        # Warmup with consistent volume
        candles = [create_simple_candle(float(i), 100.0, 1000.0) for i in range(15)]
        engine.warmup({"1m": candles})

        # Test RV at low boundary (should -> 0)
        result_low = engine.on_candle_close("1m", create_simple_candle(15.0, 100.0, 1000.0), rv=0.8)
        assert result_low.components_norm["rv"] == 0.0

        # Test RV at high boundary (should -> 1)
        result_high = engine.on_candle_close(
            "1m", create_simple_candle(16.0, 100.0, 1000.0), rv=2.0
        )
        assert abs(result_high.components_norm["rv"] - 1.0) < 1e-6

        # Test RV above high (saturates at 1)
        result_saturated = engine.on_candle_close(
            "1m", create_simple_candle(17.0, 100.0, 1000.0), rv=3.0
        )
        assert result_saturated.components_norm["rv"] == 1.0

    def test_oi_normalization(self):
        """Test OI expansion normalization"""
        config = TrendStrengthConfig(oi_ref_by_tf={"1m": 0.003})
        engine = TrendStrengthEngine(config)

        # Warmup
        candles = [create_simple_candle(float(i), 100.0, 1000.0) for i in range(10)]
        engine.warmup({"1m": candles})

        # dOI matching reference should normalize to ~1.0
        result = engine.on_candle_close(
            "1m",
            create_simple_candle(10.0, 100.0, 1000.0),
            oi_now=1000.0,
            oi_prev=997.0,  # dOI = 3/997 ≈ 0.003
        )

        assert "oi" in result.components_norm
        assert 0.8 < result.components_norm["oi"] <= 1.0


class TestWeightingAndComposite:
    """Tests for weight normalization and composite calculation"""

    def test_all_components_present(self):
        """Test composite with all components present"""
        config = TrendStrengthConfig(w_ema_slope=0.35, w_ribbon=0.25, w_rv=0.20, w_oi=0.20)
        engine = TrendStrengthEngine(config)

        # Warmup
        candles = [create_simple_candle(float(i), 100.0, 1000.0) for i in range(15)]
        engine.warmup({"1m": candles})

        # All components at 0.5
        result = engine.on_candle_close(
            "1m",
            create_simple_candle(15.0, 100.0, 1000.0),
            slope_50=0.001,
            ribbon_width_rate=0.05,
            rv=1.4,
            oi_now=1000.0,
            oi_prev=997.0,
            atr_percent=0.01,
        )

        # Weights should be original
        assert len(result.weights) == 4
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_missing_oi_renormalizes_weights(self):
        """Test weight renormalization when OI missing"""
        config = TrendStrengthConfig(w_ema_slope=0.35, w_ribbon=0.25, w_rv=0.20, w_oi=0.20)
        engine = TrendStrengthEngine(config)

        # Warmup
        candles = [create_simple_candle(float(i), 100.0, 1000.0) for i in range(15)]
        engine.warmup({"1m": candles})

        # No OI provided
        result = engine.on_candle_close(
            "1m",
            create_simple_candle(15.0, 100.0, 1000.0),
            slope_50=0.001,
            ribbon_width_rate=0.05,
            rv=1.4,
            atr_percent=0.01,
        )

        # Should have 3 components
        assert len(result.weights) == 3
        assert "oi" not in result.weights

        # Weights should sum to 1.0
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

        # Weights should be renormalized
        # Original: ema=0.35, ribbon=0.25, rv=0.20, sum=0.80
        # Renormalized: ema=0.35/0.80, ribbon=0.25/0.80, rv=0.20/0.80
        expected_ema = 0.35 / 0.80
        assert abs(result.weights["ema_slope"] - expected_ema) < 0.01


class TestSmoothing:
    """Tests for EMA smoothing of strength"""

    def test_smoothing_reduces_volatility(self):
        """Test that smoothing reduces score volatility"""
        config = TrendStrengthConfig(smooth_period=5)
        engine = TrendStrengthEngine(config)

        # Warmup
        candles = [create_simple_candle(float(i), 100.0, 1000.0) for i in range(15)]
        engine.warmup({"1m": candles})

        # Feed alternating strong/weak inputs
        results = []
        for i in range(20):
            slope = 0.002 if i % 2 == 0 else 0.0001
            result = engine.on_candle_close(
                "1m",
                create_simple_candle(15.0 + i, 100.0, 1000.0),
                slope_50=slope,
                atr_percent=0.01,
            )
            results.append(result)

        # Smooth should be less volatile than raw
        raw_changes = [
            abs(results[i].strength_raw - results[i - 1].strength_raw)
            for i in range(1, len(results))
        ]
        smooth_changes = [
            abs(results[i].strength_smooth - results[i - 1].strength_smooth)
            for i in range(1, len(results))
        ]

        avg_raw_change = sum(raw_changes) / len(raw_changes)
        avg_smooth_change = sum(smooth_changes) / len(smooth_changes)

        assert avg_smooth_change < avg_raw_change


class TestBucketing:
    """Tests for bucket classification"""

    def test_weak_bucket(self):
        """Test WEAK bucket classification"""
        config = TrendStrengthConfig()
        engine = TrendStrengthEngine(config)

        # Warmup
        candles = [create_simple_candle(float(i), 100.0, 1000.0) for i in range(15)]
        engine.warmup({"1m": candles})

        # Very weak components -> WEAK
        result = engine.on_candle_close(
            "1m",
            create_simple_candle(15.0, 100.0, 1000.0),
            slope_50=0.0001,
            ribbon_width_rate=-0.05,
            rv=0.9,
            atr_percent=0.01,
        )

        assert result.bucket == Bucket.WEAK
        assert result.strength_smooth <= 30.0

    def test_strong_bucket(self):
        """Test STRONG bucket classification"""
        config = TrendStrengthConfig()
        engine = TrendStrengthEngine(config)

        # Warmup
        candles = [create_simple_candle(float(i), 100.0, 1000.0) for i in range(15)]
        engine.warmup({"1m": candles})

        # Strong components -> STRONG
        result = engine.on_candle_close(
            "1m",
            create_simple_candle(15.0, 100.0, 1000.0),
            slope_50=0.003,
            ribbon_width_rate=0.15,
            rv=1.8,
            oi_now=1000.0,
            oi_prev=990.0,
            atr_percent=0.01,
        )

        assert result.bucket == Bucket.STRONG
        assert result.strength_smooth > 60.0

    def test_emerging_bucket(self):
        """Test EMERGING bucket classification"""
        config = TrendStrengthConfig()
        engine = TrendStrengthEngine(config)

        # Warmup
        candles = [create_simple_candle(float(i), 100.0, 1000.0) for i in range(15)]
        engine.warmup({"1m": candles})

        # Moderate components -> EMERGING
        result = engine.on_candle_close(
            "1m",
            create_simple_candle(15.0, 100.0, 1000.0),
            slope_50=0.0015,
            ribbon_width_rate=0.05,
            rv=1.2,
            atr_percent=0.01,
        )

        assert result.bucket == Bucket.EMERGING
        assert 30.0 < result.strength_smooth <= 60.0


class TestSafetyCaps:
    """Tests for safety cap application"""

    def test_structure_range_cap(self):
        """Test structure range cap limits strength"""
        config = TrendStrengthConfig(cap_when_structure_range=50)
        engine = TrendStrengthEngine(config)

        # Warmup
        candles = [create_simple_candle(float(i), 100.0, 1000.0) for i in range(15)]
        engine.warmup({"1m": candles})

        # Strong components but structure is range
        result = engine.on_candle_close(
            "1m",
            create_simple_candle(15.0, 100.0, 1000.0),
            slope_50=0.003,
            ribbon_width_rate=0.15,
            rv=1.8,
            atr_percent=0.01,
            flags={"structure_is_range": True},
        )

        assert result.strength_smooth <= 50.0

    def test_supertrend_chop_cap(self):
        """Test supertrend chop cap limits strength"""
        config = TrendStrengthConfig(cap_when_supertrend_chop=50)
        engine = TrendStrengthEngine(config)

        # Warmup
        candles = [create_simple_candle(float(i), 100.0, 1000.0) for i in range(15)]
        engine.warmup({"1m": candles})

        # Strong components but supertrend is chop
        result = engine.on_candle_close(
            "1m",
            create_simple_candle(15.0, 100.0, 1000.0),
            slope_50=0.003,
            ribbon_width_rate=0.15,
            rv=1.8,
            atr_percent=0.01,
            flags={"supertrend_is_chop": True},
        )

        assert result.strength_smooth <= 50.0

    def test_rv_dead_cap(self):
        """Test RV dead cap limits strength"""
        config = TrendStrengthConfig(cap_when_rv_dead=25)
        engine = TrendStrengthEngine(config)

        # Warmup
        candles = [create_simple_candle(float(i), 100.0, 1000.0) for i in range(15)]
        engine.warmup({"1m": candles})

        # Strong components but RV is dead
        result = engine.on_candle_close(
            "1m",
            create_simple_candle(15.0, 100.0, 1000.0),
            slope_50=0.003,
            ribbon_width_rate=0.15,
            rv=0.2,  # Very low RV
            atr_percent=0.01,
        )

        assert result.strength_smooth <= 25.0


class TestEndToEnd:
    """End-to-end integration tests"""

    def test_strong_trend_scenario(self):
        """Test strong trend produces STRONG bucket"""
        config = TrendStrengthConfig()
        engine = TrendStrengthEngine(config)

        # Build up strong trend
        candles = []
        for i in range(30):
            price = 100.0 + i * 0.5
            volume = 1000.0 * (1.5 + i * 0.02)  # Increasing volume
            candles.append(create_simple_candle(float(i), price, volume))

        engine.warmup({"1m": candles[:20]})

        # Process remaining with strong signals
        for i, candle in enumerate(candles[20:], start=20):
            result = engine.on_candle_close(
                "1m",
                candle,
                slope_50=0.003,  # Strong slope
                ribbon_width_rate=0.15,  # Expanding
                rv=1.7,  # High volume
                oi_now=1000.0 + i * 5,
                oi_prev=1000.0 + (i - 1) * 5,
                atr_percent=0.01,
            )

        # Final result should be STRONG
        assert result.bucket == Bucket.STRONG
        assert result.strength_smooth > 60.0

    def test_chop_scenario(self):
        """Test choppy market produces WEAK bucket"""
        config = TrendStrengthConfig()
        engine = TrendStrengthEngine(config)

        # Build up choppy market
        candles = []
        price = 100.0
        for i in range(30):
            price = 100.0 + (i % 4 - 2) * 0.2  # Oscillate
            volume = 1000.0
            candles.append(create_simple_candle(float(i), price, volume))

        engine.warmup({"1m": candles[:20]})

        # Process remaining with weak signals
        for candle in candles[20:]:
            result = engine.on_candle_close(
                "1m",
                candle,
                slope_50=0.0001,  # Minimal slope
                ribbon_width_rate=-0.05,  # Compressing
                rv=0.9,  # Low volume
                atr_percent=0.01,
            )

        # Final result should be WEAK
        assert result.bucket == Bucket.WEAK
        assert result.strength_smooth < 30.0


class TestDirectionalSigning:
    """Tests for directional bias and signed strength"""

    def test_bull_bias_positive_strength(self):
        """Test BULL bias produces positive signed strength"""
        config = TrendStrengthConfig()
        engine = TrendStrengthEngine(config)

        # Warmup
        candles = [create_simple_candle(float(i), 100.0, 1000.0) for i in range(15)]
        engine.warmup({"1m": candles})

        # Strong trend with BULL bias
        result = engine.on_candle_close(
            "1m",
            create_simple_candle(15.0, 100.0, 1000.0),
            slope_50=0.0030,
            ribbon_width_rate=0.15,
            rv=1.5,
            oi_now=100000.0,
            oi_prev=97000.0,
            bias="BULL",
        )

        # Verify direction_bias is +1
        assert result.direction_bias == 1

        # Verify strength_signed is positive and equals strength_smooth
        assert result.strength_signed > 0
        assert abs(result.strength_signed - result.strength_smooth) < 0.01

    def test_bear_bias_negative_strength(self):
        """Test BEAR bias produces negative signed strength"""
        config = TrendStrengthConfig()
        engine = TrendStrengthEngine(config)

        # Warmup
        candles = [create_simple_candle(float(i), 100.0, 1000.0) for i in range(15)]
        engine.warmup({"1m": candles})

        # Strong trend with BEAR bias
        result = engine.on_candle_close(
            "1m",
            create_simple_candle(15.0, 100.0, 1000.0),
            slope_50=0.0030,
            ribbon_width_rate=0.15,
            rv=1.5,
            oi_now=100000.0,
            oi_prev=97000.0,
            bias="BEAR",
        )

        # Verify direction_bias is -1
        assert result.direction_bias == -1

        # Verify strength_signed is negative
        assert result.strength_signed < 0
        assert abs(result.strength_signed + result.strength_smooth) < 0.01

    def test_neutral_bias_zero_strength(self):
        """Test NEUTRAL bias produces zero signed strength"""
        config = TrendStrengthConfig()
        engine = TrendStrengthEngine(config)

        # Warmup
        candles = [create_simple_candle(float(i), 100.0, 1000.0) for i in range(15)]
        engine.warmup({"1m": candles})

        # Trend with NEUTRAL bias
        result = engine.on_candle_close(
            "1m",
            create_simple_candle(15.0, 100.0, 1000.0),
            slope_50=0.0030,
            ribbon_width_rate=0.15,
            rv=1.5,
            bias="NEUTRAL",
        )

        # Verify direction_bias is 0
        assert result.direction_bias == 0

        # Verify strength_signed is 0
        assert result.strength_signed == 0.0

    def test_direction_bias_param_overrides_bias_string(self):
        """Test direction_bias parameter takes precedence over bias string"""
        config = TrendStrengthConfig()
        engine = TrendStrengthEngine(config)

        # Warmup
        candles = [create_simple_candle(float(i), 100.0, 1000.0) for i in range(15)]
        engine.warmup({"1m": candles})

        # Provide both bias="BULL" and direction_bias=-1
        # direction_bias should win
        result = engine.on_candle_close(
            "1m",
            create_simple_candle(15.0, 100.0, 1000.0),
            slope_50=0.0030,
            ribbon_width_rate=0.15,
            rv=1.5,
            bias="BULL",  # This should be ignored
            direction_bias=-1,  # This should be used
        )

        # Verify direction_bias parameter was used
        assert result.direction_bias == -1
        assert result.strength_signed < 0

    def test_no_bias_defaults_to_neutral(self):
        """Test no bias provided defaults to neutral (0)"""
        config = TrendStrengthConfig()
        engine = TrendStrengthEngine(config)

        # Warmup
        candles = [create_simple_candle(float(i), 100.0, 1000.0) for i in range(15)]
        engine.warmup({"1m": candles})

        # No bias parameter
        result = engine.on_candle_close(
            "1m",
            create_simple_candle(15.0, 100.0, 1000.0),
            slope_50=0.0030,
            ribbon_width_rate=0.15,
            rv=1.5,
        )

        # Verify defaults to neutral
        assert result.direction_bias == 0
        assert result.strength_signed == 0.0

    def test_invalid_bias_string_defaults_to_neutral(self):
        """Test invalid bias string defaults to neutral"""
        config = TrendStrengthConfig()
        engine = TrendStrengthEngine(config)

        # Warmup
        candles = [create_simple_candle(float(i), 100.0, 1000.0) for i in range(15)]
        engine.warmup({"1m": candles})

        # Invalid bias string
        result = engine.on_candle_close(
            "1m",
            create_simple_candle(15.0, 100.0, 1000.0),
            slope_50=0.0030,
            ribbon_width_rate=0.15,
            rv=1.5,
            bias="INVALID",
        )

        # Verify defaults to neutral
        assert result.direction_bias == 0
        assert result.strength_signed == 0.0

    def test_direction_bias_int_validation(self):
        """Test direction_bias validates to -1, 0, or +1"""
        config = TrendStrengthConfig()
        engine = TrendStrengthEngine(config)

        # Warmup
        candles = [create_simple_candle(float(i), 100.0, 1000.0) for i in range(15)]
        engine.warmup({"1m": candles})

        # Invalid direction_bias value (out of range)
        result = engine.on_candle_close(
            "1m",
            create_simple_candle(15.0, 100.0, 1000.0),
            slope_50=0.0030,
            ribbon_width_rate=0.15,
            rv=1.5,
            direction_bias=5,  # Invalid
        )

        # Should default to 0
        assert result.direction_bias == 0
        assert result.strength_signed == 0.0


class TestFormatting:
    """Tests for output formatting"""

    def test_format_compact(self):
        """Test compact formatting"""
        config = TrendStrengthConfig()
        engine = TrendStrengthEngine(config)

        candles = [create_simple_candle(float(i), 100.0, 1000.0) for i in range(15)]
        engine.warmup({"1m": candles})

        result = engine.on_candle_close(
            "1m",
            create_simple_candle(15.0, 100.0, 1000.0),
            slope_50=0.002,
            ribbon_width_rate=0.10,
            rv=1.5,
            atr_percent=0.01,
            bias="BULL",
        )

        output = format_trend_strength_output({"1m": result}, compact=True)

        assert "TREND STRENGTH" in output
        assert "1m:" in output
        assert "bias=" in output  # Check for directional bias
        assert "comps_norm:" in output
        assert "comps_raw" in output


class TestEdgeCases:
    """Tests for edge cases"""

    def test_no_components_available(self):
        """Test handling when no components available"""
        config = TrendStrengthConfig()
        engine = TrendStrengthEngine(config)

        # No warmup, no components
        result = engine.on_candle_close("1m", create_simple_candle(1.0, 100.0, 1000.0))

        # Should return 0 strength
        assert result.strength_raw == 0.0
        assert result.strength_smooth == 0.0

    def test_weight_validation(self):
        """Test config validates weights sum to 1.0"""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            TrendStrengthConfig(w_ema_slope=0.3, w_ribbon=0.3, w_rv=0.3, w_oi=0.3)  # Sums to 1.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
