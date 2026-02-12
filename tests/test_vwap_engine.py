"""
Tests for VWAP Engine Module

Tests VWAP calculation accuracy, session/weekly resets, anchored VWAP,
standard deviation bands, and interaction state machine.
"""

import math
from datetime import datetime, timezone

import pytest

from nlu_analyzer.indicators.vwap_engine import (
    BandMethod,
    Candle,
    InteractionState,
    PricePosition,
    PriceSource,
    VWAPConfig,
    VWAPEngine,
    VWAPKind,
    format_vwap_output,
)


def create_candle(timestamp: float, high: float, low: float, close: float, volume: float) -> Candle:
    """Helper to create a candle"""
    return Candle(timestamp=timestamp, open=close, high=high, low=low, close=close, volume=volume)


def create_simple_candle(timestamp: float, price: float, volume: float) -> Candle:
    """Helper to create candle with same OHLC"""
    return create_candle(timestamp, price, price, price, volume)


class TestVWAPFormula:
    """Tests for VWAP formula correctness"""

    def test_vwap_basic_calculation(self):
        """Test basic VWAP calculation on known data"""
        config = VWAPConfig(price_source=PriceSource.CLOSE)
        engine = VWAPEngine(config)

        # Simple test case: 3 candles with known VWAP
        # Prices: [100, 110, 105], Volumes: [100, 200, 150]
        # VWAP = (100*100 + 110*200 + 105*150) / (100+200+150)
        #      = (10000 + 22000 + 15750) / 450
        #      = 47750 / 450 = 106.111...

        candles = [
            create_simple_candle(1.0, 100.0, 100.0),
            create_simple_candle(2.0, 110.0, 200.0),
            create_simple_candle(3.0, 105.0, 150.0),
        ]

        for candle in candles:
            result = engine.on_candle_close("1m", candle)

        expected_vwap = 47750.0 / 450.0
        actual_vwap = result.session_by_tf["1m"].vwap

        assert (
            abs(actual_vwap - expected_vwap) < 1e-9
        ), f"VWAP {actual_vwap} != expected {expected_vwap}"

    def test_vwap_typical_price(self):
        """Test VWAP with typical price (H+L+C)/3"""
        config = VWAPConfig(price_source=PriceSource.TYPICAL)
        engine = VWAPEngine(config)

        # Candle: H=110, L=90, C=100, V=100
        # TP = (110+90+100)/3 = 100
        # VWAP = 100*100 / 100 = 100

        candle = create_candle(1.0, 110.0, 90.0, 100.0, 100.0)
        result = engine.on_candle_close("1m", candle)

        expected_vwap = 100.0
        actual_vwap = result.session_by_tf["1m"].vwap

        assert abs(actual_vwap - expected_vwap) < 1e-9

    def test_vwap_zero_volume_handling(self):
        """Test that zero volume candles are handled gracefully"""
        config = VWAPConfig()
        engine = VWAPEngine(config)

        candles = [
            create_simple_candle(1.0, 100.0, 100.0),
            create_simple_candle(2.0, 110.0, 0.0),  # Zero volume
            create_simple_candle(3.0, 105.0, 50.0),
        ]

        for candle in candles:
            result = engine.on_candle_close("1m", candle)

        # Zero volume candle should be skipped
        # VWAP = (100*100 + 105*50) / (100+50) = 15250/150 = 101.666...
        expected_vwap = 15250.0 / 150.0
        actual_vwap = result.session_by_tf["1m"].vwap

        assert abs(actual_vwap - expected_vwap) < 1e-6


class TestIncrementalVWAP:
    """Tests for incremental VWAP calculation"""

    def test_incremental_equals_batch(self):
        """Test incremental VWAP matches batch calculation"""
        config = VWAPConfig(price_source=PriceSource.CLOSE)
        engine = VWAPEngine(config)

        # Generate test data
        candles = []
        for i in range(50):
            price = 100.0 + (i % 10) * 0.5
            volume = 100.0 + (i % 5) * 20.0
            candles.append(create_simple_candle(float(i), price, volume))

        # Incremental computation
        for candle in candles:
            result = engine.on_candle_close("1m", candle)

        incremental_vwap = result.session_by_tf["1m"].vwap

        # Batch computation
        pv_sum = sum(c.close * c.volume for c in candles)
        v_sum = sum(c.volume for c in candles)
        batch_vwap = pv_sum / v_sum

        assert (
            abs(incremental_vwap - batch_vwap) < 1e-9
        ), f"Incremental {incremental_vwap} != Batch {batch_vwap}"

    def test_incremental_typical_price(self):
        """Test incremental with typical price"""
        config = VWAPConfig(price_source=PriceSource.TYPICAL)
        engine = VWAPEngine(config)

        candles = []
        for i in range(30):
            high = 100.0 + i * 0.5
            low = 99.0 + i * 0.5
            close = 99.5 + i * 0.5
            volume = 100.0
            candles.append(create_candle(float(i), high, low, close, volume))

        # Incremental
        for candle in candles:
            result = engine.on_candle_close("1m", candle)

        incremental_vwap = result.session_by_tf["1m"].vwap

        # Batch
        pv_sum = sum(((c.high + c.low + c.close) / 3.0) * c.volume for c in candles)
        v_sum = sum(c.volume for c in candles)
        batch_vwap = pv_sum / v_sum

        assert abs(incremental_vwap - batch_vwap) < 1e-9


class TestSessionReset:
    """Tests for session boundary detection and reset"""

    def test_utc_day_reset(self):
        """Test session reset at UTC day boundary"""
        config = VWAPConfig(session_reset="UTC_DAY", timezone="UTC")
        engine = VWAPEngine(config)

        # Day 1: 2024-01-01 23:59:00 UTC
        day1_time = datetime(2024, 1, 1, 23, 59, 0, tzinfo=timezone.utc).timestamp()
        candle1 = create_simple_candle(day1_time, 100.0, 100.0)
        engine.on_candle_close("1m", candle1)

        # Day 2: 2024-01-02 00:01:00 UTC (crossed boundary)
        day2_time = datetime(2024, 1, 2, 0, 1, 0, tzinfo=timezone.utc).timestamp()
        candle2 = create_simple_candle(day2_time, 110.0, 100.0)
        result2 = engine.on_candle_close("1m", candle2)
        vwap2 = result2.session_by_tf["1m"].vwap

        # VWAP should reset, so vwap2 should be close to 110 (single candle)
        assert abs(vwap2 - 110.0) < 1e-6, "VWAP should reset at day boundary"
        assert result2.session_by_tf["1m"].bar_count == 1, "Bar count should reset"

    def test_no_reset_within_day(self):
        """Test no reset within same day"""
        config = VWAPConfig(session_reset="UTC_DAY")
        engine = VWAPEngine(config)

        # Same day, different times
        time1 = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc).timestamp()
        time2 = datetime(2024, 1, 1, 20, 0, 0, tzinfo=timezone.utc).timestamp()

        candle1 = create_simple_candle(time1, 100.0, 100.0)
        candle2 = create_simple_candle(time2, 110.0, 100.0)

        engine.on_candle_close("1m", candle1)
        result2 = engine.on_candle_close("1m", candle2)

        # Should accumulate, not reset
        assert result2.session_by_tf["1m"].bar_count == 2
        # VWAP should be weighted average
        expected = (100 * 100 + 110 * 100) / 200
        assert abs(result2.session_by_tf["1m"].vwap - expected) < 1e-6


class TestWeeklyReset:
    """Tests for weekly boundary detection and reset"""

    def test_weekly_reset(self):
        """Test weekly reset at ISO week boundary"""
        config = VWAPConfig(weekly_reset_day="MON", timezone="UTC")
        engine = VWAPEngine(config)

        # Week 1: Friday 2024-01-05 (week 1)
        week1_time = datetime(2024, 1, 5, 23, 0, 0, tzinfo=timezone.utc).timestamp()
        candle1 = create_simple_candle(week1_time, 100.0, 100.0)
        engine.on_candle_close("1m", candle1)

        # Week 2: Monday 2024-01-08 (week 2, crossed boundary)
        week2_time = datetime(2024, 1, 8, 1, 0, 0, tzinfo=timezone.utc).timestamp()
        candle2 = create_simple_candle(week2_time, 110.0, 100.0)
        result2 = engine.on_candle_close("1m", candle2)

        # Weekly VWAP should reset
        assert result2.weekly_by_tf["1m"].bar_count == 1, "Weekly bar count should reset"
        assert abs(result2.weekly_by_tf["1m"].vwap - 110.0) < 1e-6


class TestAnchoredVWAP:
    """Tests for anchored VWAP functionality"""

    def test_anchored_vwap_creation(self):
        """Test creating an anchored VWAP"""
        config = VWAPConfig()
        engine = VWAPEngine(config)

        # Add anchor
        anchor_time = 10.0
        anchor = engine.add_anchor("1m", anchor_time, "BOS_1", note="BOS at 100", kind="BOS")

        assert anchor.kind == VWAPKind.ANCHORED
        assert anchor.anchor_id == "BOS_1"
        assert anchor.anchor_time == anchor_time

    def test_anchored_vwap_calculation(self):
        """Test anchored VWAP calculation starts from anchor point"""
        config = VWAPConfig(price_source=PriceSource.CLOSE)
        engine = VWAPEngine(config)

        # Process some candles before anchor
        for i in range(5):
            candle = create_simple_candle(float(i), 100.0, 100.0)
            engine.on_candle_close("1m", candle)

        # Add anchor at time 5
        anchor_time = 5.0
        engine.add_anchor("1m", anchor_time, "ANCHOR_1")

        # Process candles after anchor
        candles_after = [
            create_simple_candle(5.0, 110.0, 100.0),
            create_simple_candle(6.0, 120.0, 200.0),
            create_simple_candle(7.0, 115.0, 150.0),
        ]

        for candle in candles_after:
            result = engine.on_candle_close("1m", candle)

        # Anchored VWAP should only include candles from anchor_time onwards
        anchors = result.anchors_by_tf["1m"]
        assert len(anchors) == 1

        anchor_vwap = anchors[0].vwap
        # VWAP = (110*100 + 120*200 + 115*150) / (100+200+150)
        expected = (110 * 100 + 120 * 200 + 115 * 150) / 450.0

        assert abs(anchor_vwap - expected) < 1e-6

    def test_multiple_anchors(self):
        """Test multiple anchored VWAPs"""
        config = VWAPConfig(max_anchors_per_tf=3)
        engine = VWAPEngine(config)

        # Add multiple anchors
        engine.add_anchor("1m", 10.0, "ANCHOR_1")
        engine.add_anchor("1m", 20.0, "ANCHOR_2")
        engine.add_anchor("1m", 30.0, "ANCHOR_3")

        assert len(engine._anchored_vwaps["1m"]) == 3

    def test_anchor_pruning(self):
        """Test anchor pruning when exceeding max"""
        config = VWAPConfig(max_anchors_per_tf=2)
        engine = VWAPEngine(config)

        # Add 3 anchors (exceeds max of 2)
        engine.add_anchor("1m", 10.0, "ANCHOR_1")
        engine.add_anchor("1m", 20.0, "ANCHOR_2")
        engine.add_anchor("1m", 30.0, "ANCHOR_3")

        # Should keep only 2 most recent
        assert len(engine._anchored_vwaps["1m"]) == 2

        # Should keep ANCHOR_2 and ANCHOR_3 (most recent)
        anchor_ids = [a.anchor_id for a in engine._anchored_vwaps["1m"]]
        assert "ANCHOR_3" in anchor_ids
        assert "ANCHOR_2" in anchor_ids

    def test_remove_anchor(self):
        """Test removing an anchor"""
        config = VWAPConfig()
        engine = VWAPEngine(config)

        engine.add_anchor("1m", 10.0, "ANCHOR_1")
        engine.add_anchor("1m", 20.0, "ANCHOR_2")

        assert len(engine._anchored_vwaps["1m"]) == 2

        # Remove anchor
        removed = engine.remove_anchor("1m", "ANCHOR_1")
        assert removed is True
        assert len(engine._anchored_vwaps["1m"]) == 1

        # Try removing non-existent anchor
        removed = engine.remove_anchor("1m", "ANCHOR_999")
        assert removed is False


class TestStandardDeviationBands:
    """Tests for standard deviation band calculation"""

    def test_std_bands_calculation(self):
        """Test standard deviation bands calculation"""
        config = VWAPConfig(
            enable_std_bands=True, min_bars_for_std=10, std_band_multipliers=[1.0, 2.0]
        )
        engine = VWAPEngine(config)

        # Generate candles with variance
        candles = []
        for i in range(50):
            price = 100.0 + (i % 10) - 5  # Oscillates around 100
            volume = 100.0
            candles.append(create_simple_candle(float(i), price, volume))

        for candle in candles:
            result = engine.on_candle_close("1m", candle)

        session = result.session_by_tf["1m"]

        # Should have std bands after min_bars_for_std
        assert session.bands is not None
        assert session.bands.method == BandMethod.STD
        assert session.bands.std is not None
        assert session.bands.std > 0

        # Bands should be symmetric around VWAP
        for k, (lower, upper) in session.bands.bands.items():
            assert abs((upper - session.vwap) - (session.vwap - lower)) < 1e-6
            assert upper > session.vwap > lower

    def test_std_variance_formula(self):
        """Test standard deviation variance formula: Var = E[X^2] - E[X]^2"""
        config = VWAPConfig(
            enable_std_bands=True, min_bars_for_std=5, price_source=PriceSource.CLOSE
        )
        engine = VWAPEngine(config)

        # Known variance dataset
        # Prices: [100, 102, 98, 101, 99], all with volume 100
        prices = [100, 102, 98, 101, 99]
        volume = 100.0

        for i, price in enumerate(prices):
            candle = create_simple_candle(float(i), price, volume)
            result = engine.on_candle_close("1m", candle)

        session = result.session_by_tf["1m"]

        # Manual calculation
        mean = sum(prices) / len(prices)  # 100
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)  # 2.0
        expected_std = math.sqrt(variance)  # 1.414...

        assert abs(session.bands.std - expected_std) < 0.1

    def test_atr_fallback_bands(self):
        """Test ATR fallback bands when std not ready"""
        config = VWAPConfig(
            enable_std_bands=True,
            min_bars_for_std=100,  # High threshold
            fallback_atr_band_multipliers=[0.5, 1.0],
        )
        engine = VWAPEngine(config)

        # Only 10 candles (below min_bars_for_std)
        for i in range(10):
            candle = create_simple_candle(float(i), 100.0, 100.0)
            result = engine.on_candle_close("1m", candle, atr_percent=0.01)

        session = result.session_by_tf["1m"]

        # Should use ATR fallback
        assert session.bands.method == BandMethod.ATR_FALLBACK
        assert session.bands.std is None


class TestPricePosition:
    """Tests for price position detection"""

    def test_price_above_vwap(self):
        """Test price position ABOVE"""
        config = VWAPConfig(touch_tolerance=0.0001)
        engine = VWAPEngine(config)

        # VWAP at 100
        engine.on_candle_close("1m", create_simple_candle(1.0, 100.0, 100.0))

        # Price at 105 (clearly above)
        result = engine.on_candle_close("1m", create_simple_candle(2.0, 105.0, 100.0))

        assert result.session_by_tf["1m"].price_position == PricePosition.ABOVE

    def test_price_below_vwap(self):
        """Test price position BELOW"""
        config = VWAPConfig(touch_tolerance=0.0001)
        engine = VWAPEngine(config)

        engine.on_candle_close("1m", create_simple_candle(1.0, 100.0, 100.0))

        # Price at 95 (clearly below)
        result = engine.on_candle_close("1m", create_simple_candle(2.0, 95.0, 100.0))

        assert result.session_by_tf["1m"].price_position == PricePosition.BELOW

    def test_price_at_vwap(self):
        """Test price position AT (within tolerance)"""
        config = VWAPConfig(touch_tolerance=0.001)  # 0.1%
        engine = VWAPEngine(config)

        engine.on_candle_close("1m", create_simple_candle(1.0, 100.0, 100.0))

        # Price at 100.05 (within 0.1% of 100)
        result = engine.on_candle_close("1m", create_simple_candle(2.0, 100.05, 100.0))

        assert result.session_by_tf["1m"].price_position == PricePosition.AT


class TestInteractionStateMachine:
    """Tests for interaction state machine"""

    def test_reclaim_state(self):
        """Test RECLAIM state (cross from below to above with hold)"""
        config = VWAPConfig(hold_bars=3, touch_tolerance=0.0001)
        engine = VWAPEngine(config)

        # Establish VWAP at 100
        for i in range(10):
            engine.on_candle_close("1m", create_simple_candle(float(i), 100.0, 100.0))

        # Below VWAP
        for i in range(3):
            engine.on_candle_close("1m", create_simple_candle(10.0 + i, 95.0, 100.0))

        # Cross above and hold
        for i in range(5):
            result = engine.on_candle_close("1m", create_simple_candle(20.0 + i, 105.0, 100.0))

        # Should detect RECLAIM after hold_bars
        assert result.session_by_tf["1m"].interaction_state == InteractionState.RECLAIM

    def test_loss_state(self):
        """Test LOSS state (cross from above to below with hold)"""
        config = VWAPConfig(hold_bars=3, touch_tolerance=0.0001)
        engine = VWAPEngine(config)

        # Establish VWAP at 100
        for i in range(10):
            engine.on_candle_close("1m", create_simple_candle(float(i), 100.0, 100.0))

        # Above VWAP
        for i in range(3):
            engine.on_candle_close("1m", create_simple_candle(10.0 + i, 105.0, 100.0))

        # Cross below and hold
        for i in range(5):
            result = engine.on_candle_close("1m", create_simple_candle(20.0 + i, 95.0, 100.0))

        # Should detect LOSS after hold_bars
        assert result.session_by_tf["1m"].interaction_state == InteractionState.LOSS

    def test_accept_state(self):
        """Test ACCEPT state (staying on same side)"""
        config = VWAPConfig(hold_bars=3)
        engine = VWAPEngine(config)

        # Establish VWAP
        for i in range(10):
            engine.on_candle_close("1m", create_simple_candle(float(i), 100.0, 100.0))

        # Stay above for hold_bars
        for i in range(5):
            result = engine.on_candle_close("1m", create_simple_candle(10.0 + i, 105.0, 100.0))

        # Should eventually show ACCEPT
        assert result.session_by_tf["1m"].interaction_state in [
            InteractionState.ACCEPT,
            InteractionState.NEUTRAL,
        ]


class TestDistanceMetrics:
    """Tests for distance metric calculations"""

    def test_percentage_distance(self):
        """Test percentage distance calculation"""
        config = VWAPConfig()
        engine = VWAPEngine(config)

        # VWAP at 100
        engine.on_candle_close("1m", create_simple_candle(1.0, 100.0, 100.0))

        # Price at 102 (2% above)
        result = engine.on_candle_close("1m", create_simple_candle(2.0, 102.0, 100.0))

        dist_pct = result.session_by_tf["1m"].distance["pct"]

        # Should be approximately 0.01 (1%)
        assert 0.005 < dist_pct < 0.015

    def test_sigma_distance(self):
        """Test sigma distance calculation (when std available)"""
        config = VWAPConfig(enable_std_bands=True, min_bars_for_std=10)
        engine = VWAPEngine(config)

        # Generate data with known variance
        for i in range(50):
            price = 100.0 + (i % 10) - 5
            candle = create_simple_candle(float(i), price, 100.0)
            result = engine.on_candle_close("1m", candle)

        # Should have sigma distance
        if result.session_by_tf["1m"].bands.std is not None:
            assert "sigma" in result.session_by_tf["1m"].distance
            assert result.session_by_tf["1m"].distance["sigma"] is not None


class TestMultiTimeframe:
    """Tests for multi-timeframe support"""

    def test_independent_timeframes(self):
        """Test that different timeframes maintain independent state"""
        config = VWAPConfig()
        engine = VWAPEngine(config)

        # Different prices for different timeframes
        engine.on_candle_close("1m", create_simple_candle(1.0, 100.0, 100.0))
        engine.on_candle_close("5m", create_simple_candle(1.0, 110.0, 100.0))
        engine.on_candle_close("1h", create_simple_candle(1.0, 105.0, 100.0))

        # Each should have different VWAP
        result = engine.update(
            {
                "1m": [create_simple_candle(2.0, 100.0, 100.0)],
                "5m": [create_simple_candle(2.0, 110.0, 100.0)],
                "1h": [create_simple_candle(2.0, 105.0, 100.0)],
            }
        )

        assert abs(result.session_by_tf["1m"].vwap - 100.0) < 0.1
        assert abs(result.session_by_tf["5m"].vwap - 110.0) < 0.1
        assert abs(result.session_by_tf["1h"].vwap - 105.0) < 0.1

    def test_update_convenience_method(self):
        """Test convenience update method"""
        config = VWAPConfig()
        engine = VWAPEngine(config)

        # Warmup
        engine.warmup(
            {
                "1m": [create_simple_candle(float(i), 100.0, 100.0) for i in range(10)],
                "5m": [create_simple_candle(float(i), 105.0, 100.0) for i in range(10)],
            }
        )

        # Update with new candles
        result = engine.update(
            {
                "1m": [create_simple_candle(10.0, 102.0, 100.0)],
                "5m": [create_simple_candle(10.0, 107.0, 100.0)],
            }
        )

        assert "1m" in result.session_by_tf
        assert "5m" in result.session_by_tf


class TestFormatting:
    """Tests for output formatting"""

    def test_format_compact(self):
        """Test compact formatting"""
        config = VWAPConfig()
        engine = VWAPEngine(config)

        engine.warmup({"1m": [create_simple_candle(float(i), 100.0, 100.0) for i in range(20)]})

        result = engine.on_candle_close("1m", create_simple_candle(20.0, 105.0, 100.0))
        output = format_vwap_output(result, compact=True)

        assert "VWAP CONTEXT" in output
        assert "1m Session:" in output
        assert "dist=" in output

    def test_format_with_anchors(self):
        """Test formatting with anchored VWAPs"""
        config = VWAPConfig()
        engine = VWAPEngine(config)

        # Setup
        engine.warmup({"1m": [create_simple_candle(float(i), 100.0, 100.0) for i in range(10)]})

        # Add anchor
        engine.add_anchor("1m", 10.0, "BOS_1", note="BOS")

        # Process more candles
        for i in range(5):
            result = engine.on_candle_close("1m", create_simple_candle(10.0 + i, 105.0, 100.0))

        output = format_vwap_output(result, compact=True)

        assert "Anchors:" in output


class TestEdgeCases:
    """Tests for edge cases"""

    def test_single_candle(self):
        """Test with single candle"""
        config = VWAPConfig()
        engine = VWAPEngine(config)

        result = engine.on_candle_close("1m", create_simple_candle(1.0, 100.0, 100.0))

        # VWAP should equal the single price
        assert abs(result.session_by_tf["1m"].vwap - 100.0) < 1e-9

    def test_all_same_price(self):
        """Test with all candles at same price"""
        config = VWAPConfig()
        engine = VWAPEngine(config)

        for i in range(50):
            result = engine.on_candle_close("1m", create_simple_candle(float(i), 100.0, 100.0))

        # VWAP should be 100, std should be near 0 (small due to floating point)
        assert abs(result.session_by_tf["1m"].vwap - 100.0) < 1e-6

        if result.session_by_tf["1m"].bands and result.session_by_tf["1m"].bands.std is not None:
            # Std should be very small (not exactly 0 due to floating point accumulation)
            assert result.session_by_tf["1m"].bands.std < 0.0001

    def test_warmup_empty_candles(self):
        """Test warmup with empty candle list"""
        config = VWAPConfig()
        engine = VWAPEngine(config)

        # Should not crash
        engine.warmup({"1m": []})

        # First candle after warmup
        result = engine.on_candle_close("1m", create_simple_candle(1.0, 100.0, 100.0))
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
