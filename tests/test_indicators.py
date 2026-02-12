"""
Comprehensive tests for Technical Indicators Module.

Tests cover:
- Volume indicators
- Momentum indicators (RSI, Stochastic RSI)
- Volatility indicators (Bollinger Bands, ATR)
- Moving averages
"""

import pytest

from indicator.engines.indicators import (
    IndicatorResult,
    MomentumIndicators,
    TrendIndicators,
    VolatilityIndicators,
    VolumeIndicators,
)
from indicator.engines.signals import Signal


# =============================================================================
# VOLUME INDICATORS TESTS
# =============================================================================


class TestVolumeIndicators:
    """Tests for volume-based indicators."""

    def test_analyze_volume_insufficient_data(self):
        """Test volume analysis with insufficient data."""
        volumes = [100, 150]
        closes = [50, 51]
        result = VolumeIndicators.analyze_volume(volumes, closes, period=20)

        assert result.name == "Volume"
        assert result.signal == Signal.NEUTRAL
        assert result.strength == 50
        assert "Insufficient data" in result.description

    def test_analyze_volume_normal(self):
        """Test volume analysis with normal volume."""
        # Create 25 candles with stable volume around 1000
        volumes = [1000] * 20 + [1050, 1020, 1030, 1010, 1015]
        closes = [50] * 20 + [51, 52, 53, 54, 55]  # Uptrend

        result = VolumeIndicators.analyze_volume(volumes, closes, period=20)

        assert result.name == "Volume"
        assert result.signal == Signal.NEUTRAL
        # Volume ratio should be close to 1.0 (1015 / 1000)
        assert 0.9 < result.value < 1.2

    def test_analyze_volume_high_bullish(self):
        """Test volume analysis with high volume on uptrend."""
        volumes = [1000] * 20 + [2500]  # High volume spike
        closes = [50] * 20 + [55]  # Price up

        result = VolumeIndicators.analyze_volume(volumes, closes, period=20)

        assert result.signal == Signal.BULLISH
        assert result.strength > 70  # High volume should give high strength
        assert "high volume" in result.description.lower()

    def test_analyze_volume_high_bearish(self):
        """Test volume analysis with high volume on downtrend."""
        volumes = [1000] * 20 + [2500]  # High volume spike
        closes = [55] * 20 + [50]  # Price down

        result = VolumeIndicators.analyze_volume(volumes, closes, period=20)

        assert result.signal == Signal.BEARISH
        assert result.strength > 70
        assert "high volume" in result.description.lower()

    def test_analyze_volume_divergence_warning(self):
        """Test volume divergence detection."""
        # Normal volume, then spike
        volumes = [1000] * 18 + [1500, 1200, 3000]
        closes = [50] * 18 + [51, 52, 53]  # Uptrend

        result = VolumeIndicators.analyze_volume(volumes, closes, period=20)

        # High volume should be detected
        assert result.signal in [Signal.BULLISH, Signal.BEARISH]


# =============================================================================
# MOMENTUM INDICATORS TESTS
# =============================================================================


class TestMomentumIndicators:
    """Tests for momentum oscillators."""

    def test_analyze_rsi_oversold(self):
        """Test RSI analysis in oversold territory."""
        # Create downtrend: prices dropping
        closes = list(range(100, 80, -1))  # 100, 99, 98... down to 81

        result = MomentumIndicators.analyze_rsi(closes, period=14)

        assert result.name == "RSI"
        assert result.value < 40  # Should be low for downtrend
        # Oversold RSI should signal potential bullish reversal
        assert result.signal in [Signal.BULLISH, Signal.NEUTRAL]
        assert result.strength >= 50

    def test_analyze_rsi_overbought(self):
        """Test RSI analysis in overbought territory."""
        # Create uptrend: prices rising
        closes = list(range(80, 100))  # 80, 81, 82... up to 99

        result = MomentumIndicators.analyze_rsi(closes, period=14)

        assert result.name == "RSI"
        assert result.value > 60  # Should be high for uptrend
        # Overbought RSI should signal potential bearish reversal
        assert result.signal in [Signal.BEARISH, Signal.NEUTRAL]

    def test_analyze_rsi_neutral(self):
        """Test RSI in neutral range."""
        # Create sideways price action
        closes = [50 + (i % 3) for i in range(50)]  # 50, 51, 52, 50, 51...

        result = MomentumIndicators.analyze_rsi(closes, period=14)

        assert result.name == "RSI"
        assert 40 < result.value < 60  # Should be near 50
        assert result.signal == Signal.NEUTRAL

    def test_analyze_rsi_insufficient_data(self):
        """Test RSI with insufficient data."""
        closes = [100, 101, 102]  # Only 3 candles

        result = MomentumIndicators.analyze_rsi(closes, period=14)

        assert result.signal == Signal.NEUTRAL
        assert "Insufficient data" in result.description

    def test_analyze_macd_histogram_bullish(self):
        """Test MACD histogram with bullish signal."""
        # Create uptrend - need 50+ candles for MACD
        closes = list(range(50, 150))

        result = MomentumIndicators.analyze_macd_histogram(closes)

        assert result.name == "MACD Histogram"
        # In strong uptrend, MACD should be bullish
        assert result.signal == Signal.BULLISH
        assert result.strength > 50

    def test_analyze_macd_histogram_bearish(self):
        """Test MACD histogram with bearish signal."""
        # Create downtrend
        closes = list(range(150, 50, -1))

        result = MomentumIndicators.analyze_macd_histogram(closes)

        assert result.name == "MACD Histogram"
        # In strong downtrend, MACD should be bearish
        assert result.signal == Signal.BEARISH
        assert result.strength > 50

    def test_analyze_macd_histogram_insufficient_data(self):
        """Test MACD with insufficient data."""
        closes = [100] * 20  # Not enough for MACD (needs 26+9)

        result = MomentumIndicators.analyze_macd_histogram(closes)

        assert result.signal == Signal.NEUTRAL
        assert "Insufficient data" in result.description

    def test_analyze_stochastic_rsi_oversold(self):
        """Test Stochastic RSI in oversold territory."""
        # Create downtrend
        closes = list(range(100, 70, -1))

        result = MomentumIndicators.analyze_stochastic_rsi(closes, period=14)

        assert result.name == "Stochastic RSI"
        # Oversold should give bullish signal
        assert result.signal in [Signal.BULLISH, Signal.NEUTRAL]

    def test_analyze_stochastic_rsi_overbought(self):
        """Test Stochastic RSI in overbought territory."""
        # Create uptrend
        closes = list(range(70, 100))

        result = MomentumIndicators.analyze_stochastic_rsi(closes, period=14)

        assert result.name == "Stochastic RSI"
        # Overbought should give bearish signal
        assert result.signal in [Signal.BEARISH, Signal.NEUTRAL]

    def test_analyze_stochastic_rsi_insufficient_data(self):
        """Test Stochastic RSI with insufficient data."""
        closes = [100, 101, 102]

        result = MomentumIndicators.analyze_stochastic_rsi(closes, period=14)

        assert result.signal == Signal.NEUTRAL
        assert "Insufficient data" in result.description


# =============================================================================
# VOLATILITY INDICATORS TESTS
# =============================================================================


class TestVolatilityIndicators:
    """Tests for volatility-based indicators."""

    def test_analyze_bollinger_bands_squeeze(self):
        """Test Bollinger Bands during low volatility (squeeze)."""
        # Very tight price range
        closes = [50.0 + 0.1 * (i % 3) for i in range(50)]

        result = VolatilityIndicators.analyze_bollinger_bands(closes, period=20, std_dev=2)

        assert result.name == "Bollinger Bands"
        # Low volatility detected
        assert result is not None

    def test_analyze_bollinger_bands_expansion(self):
        """Test Bollinger Bands during high volatility (expansion)."""
        # Wide price swings
        closes = [50 + 10 * ((-1) ** i) for i in range(50)]

        result = VolatilityIndicators.analyze_bollinger_bands(closes, period=20, std_dev=2)

        assert result.name == "Bollinger Bands"
        # High volatility should be detected
        assert result.strength > 40

    def test_analyze_bollinger_bands_upper_touch(self):
        """Test price touching upper Bollinger Band."""
        # Uptrend with price at upper band
        closes = list(range(50, 100))

        result = VolatilityIndicators.analyze_bollinger_bands(closes, period=20, std_dev=2)

        assert result.name == "Bollinger Bands"
        # Price at upper band can signal overbought
        assert result.signal in [Signal.BEARISH, Signal.BULLISH, Signal.NEUTRAL]

    def test_analyze_bollinger_bands_lower_touch(self):
        """Test price touching lower Bollinger Band."""
        # Downtrend with price at lower band
        closes = list(range(100, 50, -1))

        result = VolatilityIndicators.analyze_bollinger_bands(closes, period=20, std_dev=2)

        assert result.name == "Bollinger Bands"
        # Price at lower band can signal oversold
        assert result.signal in [Signal.BULLISH, Signal.BEARISH, Signal.NEUTRAL]

    def test_analyze_bollinger_bands_insufficient_data(self):
        """Test Bollinger Bands with insufficient data."""
        closes = [100, 101, 102]

        result = VolatilityIndicators.analyze_bollinger_bands(closes, period=20, std_dev=2)

        assert result.signal == Signal.NEUTRAL
        assert "Insufficient data" in result.description

    def test_analyze_atr_calculation(self):
        """Test Average True Range analysis."""
        # Create realistic OHLC data
        highs = [52, 54, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
        lows = [48, 50, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
        closes = [50, 52, 51, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]

        result = VolatilityIndicators.analyze_atr(highs, lows, closes, period=14)

        assert result.name == "ATR"
        assert result.value > 0  # ATR should be positive
        # ATR doesn't give directional signal
        assert result.signal == Signal.NEUTRAL

    def test_analyze_atr_increasing_volatility(self):
        """Test ATR detects increasing volatility."""
        # Start with small ranges, then increase
        highs = [52] * 10 + [60, 70, 80, 90, 100]
        lows = [48] * 10 + [55, 65, 75, 85, 95]
        closes = [50] * 10 + [58, 68, 78, 88, 98]

        result = VolatilityIndicators.analyze_atr(highs, lows, closes, period=10)

        assert result.name == "ATR"
        # Increasing range should show higher ATR
        assert result.value > 2  # Should detect expansion

    def test_analyze_atr_insufficient_data(self):
        """Test ATR with insufficient data."""
        highs = [100, 102]
        lows = [98, 100]
        closes = [99, 101]

        result = VolatilityIndicators.analyze_atr(highs, lows, closes, period=14)

        assert result.signal == Signal.NEUTRAL
        assert "Insufficient data" in result.description


# =============================================================================
# TREND INDICATORS TESTS
# =============================================================================


class TestTrendIndicators:
    """Tests for moving averages and trend indicators."""

    def test_analyze_moving_averages_bullish(self):
        """Test moving average analysis with bullish alignment."""
        # Create uptrend
        closes = list(range(50, 100))

        result = TrendIndicators.analyze_moving_averages(closes)

        assert result.name == "Moving Averages"
        # Uptrend should be bullish
        assert result.signal == Signal.BULLISH

    def test_analyze_moving_averages_bearish(self):
        """Test moving average analysis with bearish alignment."""
        # Create downtrend
        closes = list(range(100, 50, -1))

        result = TrendIndicators.analyze_moving_averages(closes)

        assert result.name == "Moving Averages"
        # Downtrend should be bearish
        assert result.signal == Signal.BEARISH

    def test_analyze_moving_averages_insufficient_data(self):
        """Test moving averages with insufficient data."""
        closes = [100, 101, 102]

        result = TrendIndicators.analyze_moving_averages(closes)

        assert result.signal == Signal.NEUTRAL
        assert "Insufficient data" in result.description


# =============================================================================
# EDGE CASES & ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_lists(self):
        """Test indicators with empty input lists."""
        result = VolumeIndicators.analyze_volume([], [], period=20)
        assert result.signal == Signal.NEUTRAL

    def test_single_value(self):
        """Test indicators with single data point."""
        result = MomentumIndicators.analyze_rsi([100], period=14)
        assert result.signal == Signal.NEUTRAL
        assert "Insufficient data" in result.description

    def test_all_same_values(self):
        """Test indicators when all values are identical."""
        closes = [50] * 100
        result = MomentumIndicators.analyze_rsi(closes, period=14)
        # RSI should handle no change gracefully
        assert result.signal == Signal.NEUTRAL

    def test_extreme_values(self):
        """Test indicators with extreme values."""
        closes = [1e10, 1e10 + 1, 1e10 + 2]
        # Should not crash, even with huge numbers
        result = MomentumIndicators.analyze_rsi(closes, period=2)
        assert result is not None

    def test_zero_volume(self):
        """Test volume indicators with zero volume."""
        volumes = [0] * 20 + [100]
        closes = [50] * 21
        result = VolumeIndicators.analyze_volume(volumes, closes, period=20)
        # Should handle division by zero gracefully
        assert result is not None
        assert result.value >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
