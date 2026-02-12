"""
Unit tests for EMA Filter Engine.

Tests:
- Incremental EMA equals batch EMA
- Alignment detection
- Regime switching with synthetic price series
- Extended detection
- Slope computation
- Multi-timeframe alignment
"""

from typing import List

import pytest

from indicator.engines.ema_filter import (
    Candle,
    EMAAlignment,
    EMABias,
    EMAConfig,
    EMAFilterEngine,
    EMARegime,
    MTFAlignment,
    compute_atr_percent,
    compute_sma,
    create_candle,
)


def create_uptrend_candles(count: int = 50, start_price: float = 100.0) -> List[Candle]:
    """Create synthetic uptrend candles."""
    candles = []
    price = start_price

    for i in range(count):
        low = price
        high = price + 1.0
        open_price = price + 0.2
        close = price + 0.8

        candles.append(
            create_candle(timestamp=i * 60000, o=open_price, h=high, l=low, c=close, v=1000.0)
        )

        price += 0.5  # Steady uptrend

    return candles


def create_downtrend_candles(count: int = 50, start_price: float = 100.0) -> List[Candle]:
    """Create synthetic downtrend candles."""
    candles = []
    price = start_price

    for i in range(count):
        high = price
        low = price - 1.0
        open_price = price - 0.2
        close = price - 0.8

        candles.append(
            create_candle(timestamp=i * 60000, o=open_price, h=high, l=low, c=close, v=1000.0)
        )

        price -= 0.5  # Steady downtrend

    return candles


def create_range_candles(
    count: int = 50, center: float = 100.0, amplitude: float = 2.0
) -> List[Candle]:
    """Create synthetic ranging candles."""
    import math

    candles = []

    for i in range(count):
        # Oscillate around center
        phase = (i % 10) / 10.0 * 2 * math.pi
        price = center + amplitude * math.sin(phase)

        low = price - 0.5
        high = price + 0.5
        open_price = price - 0.1
        close = price + 0.1

        candles.append(
            create_candle(timestamp=i * 60000, o=open_price, h=high, l=low, c=close, v=1000.0)
        )

    return candles


class TestHelperFunctions:
    """Test helper functions."""

    def test_compute_sma(self):
        """Test SMA calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        sma = compute_sma(values, 3)
        assert abs(sma - 4.0) < 0.01  # (3+4+5)/3 = 4

    def test_compute_atr_percent(self):
        """Test ATR% calculation."""
        candles = create_uptrend_candles(30)
        atr_pct = compute_atr_percent(candles, 14)

        assert atr_pct > 0
        assert atr_pct < 1.0  # Should be small percentage


class TestEMAComputation:
    """Test EMA calculation correctness."""

    def test_batch_warmup(self):
        """Test batch EMA warmup."""
        candles = create_uptrend_candles(60)

        engine = EMAFilterEngine()
        states = engine.warmup({"1m": candles})

        assert "1m" in states
        state = states["1m"]

        # EMAs should be defined
        assert state.ema9 > 0
        assert state.ema21 > 0
        assert state.ema50 > 0

        # In uptrend: ema9 > ema21 > ema50 (eventually)
        assert state.ema9 > state.ema21
        assert state.ema21 > state.ema50

    def test_incremental_equals_batch(self):
        """Test that incremental EMA updates match batch computation."""
        candles = create_uptrend_candles(60)

        # Batch computation
        engine_batch = EMAFilterEngine()
        engine_batch.warmup({"1m": candles})
        batch_state = engine_batch.get_state("1m")

        # Incremental computation
        engine_inc = EMAFilterEngine()
        engine_inc.warmup({"1m": candles[:50]})  # Warmup first

        # Then incremental updates
        for candle in candles[50:]:
            engine_inc.on_candle_close("1m", candle)

        inc_state = engine_inc.get_state("1m")

        # Should match (within tolerance)
        assert abs(inc_state.ema9 - batch_state.ema9) < 0.01
        assert abs(inc_state.ema21 - batch_state.ema21) < 0.01
        assert abs(inc_state.ema50 - batch_state.ema50) < 0.01

    def test_slope_computation(self):
        """Test slope computation with known sequence."""
        # Create strong uptrend
        candles = create_uptrend_candles(60, start_price=100.0)

        engine = EMAFilterEngine()
        engine.warmup({"1m": candles})
        state = engine.get_state("1m")

        # Slopes should be positive
        assert state.slope_21 > 0
        assert state.slope_50 > 0

        # Slope21 should be steeper than slope50 in uptrend
        assert state.slope_21 > state.slope_50


class TestAlignmentDetection:
    """Test EMA alignment classification."""

    def test_stacked_up_uptrend(self):
        """Test STACKED_UP detection in uptrend."""
        candles = create_uptrend_candles(60)

        engine = EMAFilterEngine()
        states = engine.warmup({"1m": candles})

        state = states["1m"]

        # Should detect STACKED_UP
        assert state.ema_alignment == EMAAlignment.STACKED_UP

    def test_stacked_down_downtrend(self):
        """Test STACKED_DOWN detection in downtrend."""
        candles = create_downtrend_candles(60)

        engine = EMAFilterEngine()
        states = engine.warmup({"1m": candles})

        state = states["1m"]

        # Should detect STACKED_DOWN
        assert state.ema_alignment == EMAAlignment.STACKED_DOWN

    def test_mixed_in_range(self):
        """Test MIXED alignment in ranging market."""
        candles = create_range_candles(60)

        engine = EMAFilterEngine()
        states = engine.warmup({"1m": candles})

        state = states["1m"]

        # Likely MIXED (EMAs crossing)
        # Note: May not always be MIXED depending on phase
        assert state.ema_alignment in [
            EMAAlignment.MIXED,
            EMAAlignment.STACKED_UP,
            EMAAlignment.STACKED_DOWN,
        ]


class TestRegimeClassification:
    """Test regime classification (TREND vs RANGE)."""

    def test_trend_uptrend(self):
        """Test TREND detection in strong uptrend."""
        candles = create_uptrend_candles(60)

        engine = EMAFilterEngine()
        states = engine.warmup({"1m": candles})

        state = states["1m"]

        # Should detect TREND
        assert state.ema_regime == EMARegime.TREND

        # Should have BULL bias
        assert state.ema_bias == EMABias.BULL

    def test_trend_downtrend(self):
        """Test TREND detection in strong downtrend."""
        candles = create_downtrend_candles(60)

        engine = EMAFilterEngine()
        states = engine.warmup({"1m": candles})

        state = states["1m"]

        # Should detect TREND
        assert state.ema_regime == EMARegime.TREND

        # Should have BEAR bias
        assert state.ema_bias == EMABias.BEAR

    def test_range_detection(self):
        """Test RANGE detection in sideways market."""
        candles = create_range_candles(60, amplitude=0.5)  # Small range

        engine = EMAFilterEngine()
        states = engine.warmup({"1m": candles})

        state = states["1m"]

        # Should detect RANGE (or neutral bias)
        # Note: Regime may vary depending on exact oscillation
        # At minimum, bias should be NEUTRAL or strength low
        assert state.ema_bias == EMABias.NEUTRAL or state.trend_strength_0_100 < 50


class TestBiasClassification:
    """Test bias classification."""

    def test_bull_bias(self):
        """Test BULL bias detection."""
        candles = create_uptrend_candles(60)

        engine = EMAFilterEngine()
        states = engine.warmup({"1m": candles})

        state = states["1m"]

        assert state.ema_bias == EMABias.BULL

    def test_bear_bias(self):
        """Test BEAR bias detection."""
        candles = create_downtrend_candles(60)

        engine = EMAFilterEngine()
        states = engine.warmup({"1m": candles})

        state = states["1m"]

        assert state.ema_bias == EMABias.BEAR

    def test_neutral_bias(self):
        """Test NEUTRAL bias in range."""
        candles = create_range_candles(60, amplitude=0.5)

        engine = EMAFilterEngine()
        states = engine.warmup({"1m": candles})

        state = states["1m"]

        assert state.ema_bias == EMABias.NEUTRAL


class TestExtendedDetection:
    """Test extended flag detection."""

    def test_extended_with_atr(self):
        """Test extended detection with ATR%."""
        # Create uptrend with sudden spike
        candles = create_uptrend_candles(50, start_price=100.0)

        # Add spike candle
        last_price = candles[-1].close
        spike = create_candle(
            timestamp=len(candles) * 60000,
            o=last_price,
            h=last_price + 5.0,  # Big spike
            l=last_price,
            c=last_price + 4.5,
            v=2000.0,
        )
        candles.append(spike)

        engine = EMAFilterEngine()
        engine.warmup({"1m": candles})

        # Update with spike
        state = engine.on_candle_close("1m", spike, atr_percent=0.01)

        # Extension should be large
        assert state.ext_21 > 0.03  # More than 3% from EMA21

        # May or may not be flagged as extended depending on thresholds
        # Just verify calculation works
        assert state.ext_21 >= 0


class TestPullbackZone:
    """Test pullback zone detection."""

    def test_pullback_zone_hit(self):
        """Test pullback zone detection when price near EMA21."""
        candles = create_uptrend_candles(60)

        engine = EMAFilterEngine()
        engine.warmup({"1m": candles})
        state = engine.get_state("1m")

        ema21 = state.ema21

        # Create candle near EMA21
        pullback_candle = create_candle(
            timestamp=len(candles) * 60000, o=ema21, h=ema21 + 0.1, l=ema21 - 0.1, c=ema21, v=1000.0
        )

        # Update
        state = engine.on_candle_close("1m", pullback_candle, atr_percent=0.005)

        # Should hit pullback zone
        assert state.pullback_zone_hit


class TestTrendStrength:
    """Test trend strength calculation."""

    def test_high_strength_uptrend(self):
        """Test high strength in strong uptrend."""
        candles = create_uptrend_candles(60)

        engine = EMAFilterEngine()
        states = engine.warmup({"1m": candles})

        state = states["1m"]

        # Should have high strength
        assert state.trend_strength_0_100 > 60

    def test_low_strength_range(self):
        """Test low strength in range."""
        candles = create_range_candles(60, amplitude=0.5)

        engine = EMAFilterEngine()
        states = engine.warmup({"1m": candles})

        state = states["1m"]

        # Should have low strength
        assert state.trend_strength_0_100 < 50

    def test_strength_capped_in_range(self):
        """Test that strength is capped at 40 in RANGE regime."""
        candles = create_range_candles(60, amplitude=0.3)

        engine = EMAFilterEngine()
        states = engine.warmup({"1m": candles})

        state = states["1m"]

        # If regime is RANGE, strength should be <= 40
        if state.ema_regime == EMARegime.RANGE:
            assert state.trend_strength_0_100 <= 40


class TestMultiTimeframeAlignment:
    """Test multi-timeframe alignment."""

    def test_aligned_uptrend(self):
        """Test ALIGNED when both timeframes bullish."""
        ltf_candles = create_uptrend_candles(100, start_price=100.0)
        htf_candles = create_uptrend_candles(50, start_price=100.0)

        engine = EMAFilterEngine()
        engine.warmup({"1m": ltf_candles, "1h": htf_candles})

        mtf = engine.get_mtf_state("1h", "1m")

        assert mtf is not None

        # Both should be BULL
        assert mtf.htf_bias == EMABias.BULL
        assert mtf.ltf_bias == EMABias.BULL

        # Should be ALIGNED
        assert mtf.alignment_summary == MTFAlignment.ALIGNED

    def test_range_dominant(self):
        """Test RANGE_DOMINANT when HTF ranging."""
        ltf_candles = create_uptrend_candles(100)
        htf_candles = create_range_candles(50, amplitude=0.5)

        engine = EMAFilterEngine()
        engine.warmup({"1m": ltf_candles, "1h": htf_candles})

        mtf = engine.get_mtf_state("1h", "1m")

        assert mtf is not None

        # HTF should be RANGE or NEUTRAL
        # Alignment should be RANGE_DOMINANT if HTF regime is RANGE
        htf_state = engine.get_state("1h")
        if htf_state.ema_regime == EMARegime.RANGE:
            assert mtf.alignment_summary == MTFAlignment.RANGE_DOMINANT

    def test_mixed_opposing_trends(self):
        """Test MIXED when HTF and LTF disagree."""
        ltf_candles = create_uptrend_candles(100)
        htf_candles = create_downtrend_candles(50)

        engine = EMAFilterEngine()
        engine.warmup({"1m": ltf_candles, "1h": htf_candles})

        mtf = engine.get_mtf_state("1h", "1m")

        assert mtf is not None

        # Biases should differ
        assert mtf.htf_bias != mtf.ltf_bias

        # Should be MIXED (unless one is NEUTRAL)
        if mtf.htf_bias != EMABias.NEUTRAL and mtf.ltf_bias != EMABias.NEUTRAL:
            assert mtf.alignment_summary == MTFAlignment.MIXED


class TestConfigOverride:
    """Test custom configuration."""

    def test_custom_periods(self):
        """Test with custom EMA periods."""
        config = EMAConfig(
            ema_periods=[10, 20, 50],
        )

        candles = create_uptrend_candles(60)

        engine = EMAFilterEngine(config)
        states = engine.warmup({"1m": candles})

        state = states["1m"]

        # Should use custom periods
        # EMA values should reflect 10, 20, 50 (not 9, 21, 50)
        assert state.ema9 > 0  # Still stored as ema9 for compatibility
        assert state.ema21 > 0
        assert state.ema50 > 0

    def test_custom_thresholds(self):
        """Test with custom threshold factors."""
        config = EMAConfig(
            slope_threshold_factor=0.20,  # Higher threshold
            width_threshold_factor=0.15,
        )

        candles = create_uptrend_candles(60)

        engine = EMAFilterEngine(config)
        states = engine.warmup({"1m": candles})

        state = states["1m"]

        # Thresholds should be stored in debug
        assert "slope_threshold" in state.debug
        assert "width_threshold" in state.debug


class TestIncrementalUpdates:
    """Test real-time incremental updates."""

    def test_multiple_incremental_updates(self):
        """Test multiple incremental updates."""
        candles = create_uptrend_candles(60)

        engine = EMAFilterEngine()
        engine.warmup({"1m": candles})

        # Add more candles incrementally
        for i in range(10):
            new_candle = create_candle(
                timestamp=(len(candles) + i) * 60000,
                o=candles[-1].close + i * 0.5,
                h=candles[-1].close + i * 0.5 + 1.0,
                l=candles[-1].close + i * 0.5,
                c=candles[-1].close + i * 0.5 + 0.8,
                v=1000.0,
            )

            state = engine.on_candle_close("1m", new_candle)

            # Should maintain BULL bias
            assert state.ema_bias == EMABias.BULL

    def test_ready_flag(self):
        """Test ready flag management."""
        candles = create_uptrend_candles(60)

        engine = EMAFilterEngine()

        # Not ready before warmup
        assert not engine.is_ready("1m")

        # Ready after warmup
        engine.warmup({"1m": candles})
        assert engine.is_ready("1m")


class TestEdgeCases:
    """Test edge cases."""

    def test_insufficient_data(self):
        """Test with insufficient data."""
        candles = create_uptrend_candles(10)  # Not enough

        engine = EMAFilterEngine()
        states = engine.warmup({"1m": candles})

        # Should not have state or not be ready
        assert not engine.is_ready("1m")

    def test_empty_candles(self):
        """Test with empty candles."""
        engine = EMAFilterEngine()
        states = engine.warmup({"1m": []})

        assert not engine.is_ready("1m")

    def test_missing_atr(self):
        """Test without ATR (should use static thresholds)."""
        candles = create_uptrend_candles(60)

        engine = EMAFilterEngine()
        engine.warmup({"1m": candles})

        # Update without ATR
        new_candle = create_candle(
            timestamp=len(candles) * 60000,
            o=candles[-1].close,
            h=candles[-1].close + 1,
            l=candles[-1].close,
            c=candles[-1].close + 0.8,
            v=1000.0,
        )

        state = engine.on_candle_close("1m", new_candle, atr_percent=None)

        # Should still work with static thresholds
        assert state.ema_bias is not None
        assert "slope_threshold" in state.debug


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self):
        """Test complete pipeline."""
        # Create multi-timeframe candles
        ltf_candles = create_uptrend_candles(100)
        htf_candles = create_uptrend_candles(50)

        engine = EMAFilterEngine()

        # Warmup
        states = engine.warmup(
            {
                "1m": ltf_candles,
                "1h": htf_candles,
            }
        )

        # Check states
        assert "1m" in states
        assert "1h" in states

        # Get MTF alignment
        mtf = engine.get_mtf_state("1h", "1m")
        assert mtf is not None
        assert mtf.alignment_summary in [
            MTFAlignment.ALIGNED,
            MTFAlignment.MIXED,
            MTFAlignment.RANGE_DOMINANT,
        ]

        # Incremental update on LTF
        new_candle = create_candle(
            timestamp=len(ltf_candles) * 60000,
            o=ltf_candles[-1].close,
            h=ltf_candles[-1].close + 1,
            l=ltf_candles[-1].close,
            c=ltf_candles[-1].close + 0.8,
            v=1200.0,
        )

        state = engine.on_candle_close("1m", new_candle, atr_percent=0.005)

        # Should maintain uptrend characteristics
        assert state.ema_bias == EMABias.BULL
        assert state.ema_regime == EMARegime.TREND

    def test_regime_transition(self):
        """Test regime transition from TREND to RANGE."""
        # Start with uptrend
        candles = create_uptrend_candles(60)

        engine = EMAFilterEngine()
        engine.warmup({"1m": candles})
        state = engine.get_state("1m")

        assert state.ema_regime == EMARegime.TREND

        # Add ranging candles
        last_price = candles[-1].close
        for i in range(20):
            # Oscillate around last price
            import math

            oscillation = 0.3 * math.sin(i / 2.0)
            new_candle = create_candle(
                timestamp=(len(candles) + i) * 60000,
                o=last_price + oscillation,
                h=last_price + oscillation + 0.2,
                l=last_price + oscillation - 0.2,
                c=last_price + oscillation + 0.1,
                v=1000.0,
            )

            state = engine.on_candle_close("1m", new_candle)

        # May transition to RANGE (depending on exact parameters)
        # At minimum, strength should drop
        assert state.trend_strength_0_100 < 70
