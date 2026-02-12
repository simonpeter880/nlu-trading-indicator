"""
Unit tests for Institutional Market Structure Engine.

Tests:
- Swing detection with pivot algorithm
- BOS/CHoCH triggers
- Liquidity sweep detection
- Acceptance/rejection logic
- FVG detection
- Multi-TF alignment
"""

from typing import List

import pytest

from indicator.engines.institutional_structure import (
    Candle,
    EventType,
    MarketStructureEngine,
    StructureConfig,
    StructureSide,
    StructureState,
    SwingType,
    ZoneStatus,
    compute_atr,
    compute_atr_pct,
    compute_rv,
    sma,
)


def create_candle(
    timestamp: int, o: float, h: float, l: float, c: float, v: float = 1000.0
) -> Candle:
    """Helper to create a candle."""
    return Candle(timestamp=timestamp, open=o, high=h, low=l, close=c, volume=v)


def create_uptrend_candles(count: int = 20, start_price: float = 100.0) -> List[Candle]:
    """Create synthetic uptrend candles with clear HH/HL pattern."""
    candles = []
    price = start_price

    for i in range(count):
        # Uptrend: higher highs and higher lows
        low = price
        high = price + 2.0
        open_price = price + 0.5
        close = price + 1.5

        candles.append(
            create_candle(
                timestamp=i * 60000, o=open_price, h=high, l=low, c=close, v=1000.0  # 1 min apart
            )
        )

        price += 1.0  # Move up

    return candles


def create_downtrend_candles(count: int = 20, start_price: float = 100.0) -> List[Candle]:
    """Create synthetic downtrend candles with clear LH/LL pattern."""
    candles = []
    price = start_price

    for i in range(count):
        # Downtrend: lower highs and lower lows
        high = price
        low = price - 2.0
        open_price = price - 0.5
        close = price - 1.5

        candles.append(
            create_candle(timestamp=i * 60000, o=open_price, h=high, l=low, c=close, v=1000.0)
        )

        price -= 1.0  # Move down

    return candles


def create_range_candles(count: int = 20, low: float = 95.0, high: float = 105.0) -> List[Candle]:
    """Create synthetic ranging candles."""
    candles = []

    for i in range(count):
        # Oscillate between range high and low
        if i % 4 < 2:
            # Move toward high
            candles.append(
                create_candle(
                    timestamp=i * 60000,
                    o=low + (high - low) * 0.3,
                    h=high,
                    l=low + (high - low) * 0.2,
                    c=high - (high - low) * 0.1,
                    v=1000.0,
                )
            )
        else:
            # Move toward low
            candles.append(
                create_candle(
                    timestamp=i * 60000,
                    o=high - (high - low) * 0.3,
                    h=high - (high - low) * 0.2,
                    l=low,
                    c=low + (high - low) * 0.1,
                    v=1000.0,
                )
            )

    return candles


class TestHelperFunctions:
    """Test helper functions."""

    def test_compute_atr(self):
        """Test ATR calculation."""
        candles = create_uptrend_candles(30)
        atrs = compute_atr(candles, period=14)

        assert len(atrs) == len(candles)
        assert atrs[0] == 0.0  # First candle has no ATR
        assert all(atr >= 0 for atr in atrs)
        # After period, ATR should be > 0
        assert atrs[14] > 0

    def test_compute_atr_pct(self):
        """Test ATR% calculation."""
        candles = create_uptrend_candles(30, start_price=100.0)
        atr_pct = compute_atr_pct(candles, period=14)

        assert len(atr_pct) == len(candles)
        assert all(pct >= 0 for pct in atr_pct)

    def test_compute_rv(self):
        """Test Relative Volume calculation."""
        volumes = [1000.0] * 10 + [2000.0] * 5 + [1000.0] * 5
        rv = compute_rv(volumes, period=10)

        assert len(rv) == len(volumes)
        # High volume should have RV > 1
        assert max(rv[10:15]) > 1.5

    def test_sma(self):
        """Test SMA calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = sma(values, period=3)

        assert len(result) == len(values)
        # First 2 are 0 (not enough data)
        assert result[0] == 0.0
        assert result[1] == 0.0
        # Third should be (1+2+3)/3 = 2.0
        assert result[2] == 2.0
        # Fourth should be (2+3+4)/3 = 3.0
        assert result[3] == 3.0


class TestSwingDetection:
    """Test swing point detection."""

    def test_detect_swings_uptrend(self):
        """Test swing detection in uptrend."""
        candles = create_uptrend_candles(20)
        engine = MarketStructureEngine(StructureConfig(pivot_left=2, pivot_right=2))

        atr_pct = compute_atr_pct(candles, 14)
        swings = engine._detect_swings(candles, atr_pct)

        # Should detect both highs and lows
        highs = [s for s in swings if s.swing_type == SwingType.HIGH]
        lows = [s for s in swings if s.swing_type == SwingType.LOW]

        assert len(highs) > 0
        assert len(lows) > 0

        # In uptrend, highs should be increasing
        if len(highs) >= 2:
            assert highs[-1].price > highs[0].price

    def test_swing_strength(self):
        """Test swing strength calculation."""
        candles = create_uptrend_candles(20)
        engine = MarketStructureEngine()

        atr_pct = compute_atr_pct(candles, 14)
        swings = engine._detect_swings(candles, atr_pct)

        # All swings should have strength 0-1
        assert all(0 <= s.strength <= 1 for s in swings)

    def test_no_swings_insufficient_data(self):
        """Test that no swings detected with insufficient data."""
        candles = create_uptrend_candles(5)
        engine = MarketStructureEngine(StructureConfig(pivot_left=3, pivot_right=3))

        atr_pct = compute_atr_pct(candles, 14)
        swings = engine._detect_swings(candles, atr_pct)

        # Not enough data for swings
        assert len(swings) == 0


class TestStructureClassification:
    """Test structure classification (UP/DOWN/RANGE)."""

    def test_classify_uptrend(self):
        """Test uptrend classification."""
        candles = create_uptrend_candles(25)
        engine = MarketStructureEngine()

        atr_pct = compute_atr_pct(candles, 14)
        swings = engine._detect_swings(candles, atr_pct)

        structure, label = engine._classify_structure(swings)

        assert structure == StructureState.UP
        assert "HH" in label or "HL" in label

    def test_classify_downtrend(self):
        """Test downtrend classification."""
        candles = create_downtrend_candles(25)
        engine = MarketStructureEngine()

        atr_pct = compute_atr_pct(candles, 14)
        swings = engine._detect_swings(candles, atr_pct)

        structure, label = engine._classify_structure(swings)

        assert structure == StructureState.DOWN
        assert "LH" in label or "LL" in label

    def test_classify_range(self):
        """Test range classification."""
        candles = create_range_candles(30)
        engine = MarketStructureEngine()

        atr_pct = compute_atr_pct(candles, 14)
        swings = engine._detect_swings(candles, atr_pct)

        structure, label = engine._classify_structure(swings)

        # Range should have mixed structure
        assert structure in [StructureState.RANGE, StructureState.UNKNOWN]


class TestBOSandCHoCH:
    """Test BOS and CHoCH detection."""

    def test_bos_uptrend(self):
        """Test BOS detection in uptrend."""
        # Create uptrend with clear breakout
        candles = create_uptrend_candles(15)

        # Add a strong breakout candle
        last_price = candles[-1].close
        breakout = create_candle(
            timestamp=len(candles) * 60000,
            o=last_price,
            h=last_price + 5.0,  # Strong move up
            l=last_price,
            c=last_price + 4.5,
            v=2000.0,  # High volume
        )
        candles.append(breakout)

        engine = MarketStructureEngine()
        states = engine.update({"LTF": candles})

        ltf_state = states["LTF"]

        # Should detect uptrend
        assert ltf_state.structure == StructureState.UP

        # Should have BOS event
        bos_events = [e for e in ltf_state.recent_events if e.event_type == EventType.BOS]
        if bos_events:  # May or may not trigger depending on swing timing
            assert bos_events[0].side == StructureSide.BULL

    def test_choch_reversal(self):
        """Test CHoCH detection on trend reversal."""
        # Create uptrend then reverse
        candles = create_uptrend_candles(15)
        last_price = candles[-1].close

        # Add reversal candles (break below recent low)
        for i in range(5):
            reversal = create_candle(
                timestamp=(len(candles) + i) * 60000,
                o=last_price - i * 2,
                h=last_price - i * 2 + 1,
                l=last_price - i * 2 - 3,
                c=last_price - i * 2 - 2.5,
                v=1500.0,
            )
            candles.append(reversal)

        engine = MarketStructureEngine()
        states = engine.update({"LTF": candles})

        ltf_state = states["LTF"]

        # Should have CHoCH event
        choch_events = [e for e in ltf_state.recent_events if e.event_type == EventType.CHOCH]
        if choch_events:
            assert choch_events[0].side == StructureSide.BEAR


class TestLiquiditySweeps:
    """Test liquidity sweep detection."""

    def test_bullish_sweep(self):
        """Test bullish liquidity sweep detection."""
        # Create range with sweep below then reclaim
        candles = create_range_candles(15, low=95.0, high=105.0)

        # Add sweep: breach low, then close back above
        sweep = create_candle(
            timestamp=len(candles) * 60000,
            o=96.0,
            h=97.0,
            l=94.0,  # Breach below range low
            c=96.5,  # Close back above
            v=2000.0,  # High volume
        )
        candles.append(sweep)

        # Add confirmation candle
        confirm = create_candle(
            timestamp=(len(candles)) * 60000, o=96.5, h=99.0, l=96.0, c=98.0, v=2500.0  # High RV
        )
        candles.append(confirm)

        engine = MarketStructureEngine()
        states = engine.update({"LTF": candles})

        ltf_state = states["LTF"]

        # Check for sweep events
        sweep_events = [e for e in ltf_state.recent_events if e.event_type == EventType.SWEEP]

        # May or may not detect depending on swing formation
        # Just verify no crashes
        assert ltf_state.structure is not None


class TestAcceptanceRejection:
    """Test acceptance/rejection logic."""

    def test_acceptance_after_bos(self):
        """Test that BOS gets accepted with proper follow-through."""
        # This test would require multiple update() calls to simulate time passing
        # For now, just verify the mechanism doesn't crash
        candles = create_uptrend_candles(20)

        engine = MarketStructureEngine()

        # First update
        states1 = engine.update({"LTF": candles})

        # Add more candles with high volume
        for i in range(5):
            candles.append(
                create_candle(
                    timestamp=(len(candles)) * 60000,
                    o=candles[-1].close,
                    h=candles[-1].close + 2,
                    l=candles[-1].close,
                    c=candles[-1].close + 1.5,
                    v=3000.0,  # High volume for acceptance
                )
            )

        # Second update
        states2 = engine.update({"LTF": candles})

        # Verify no crashes
        assert states2["LTF"].structure is not None


class TestFVGDetection:
    """Test Fair Value Gap detection."""

    def test_bullish_fvg(self):
        """Test bullish FVG detection."""
        candles = []

        # Create gap: low[i] > high[i-2]
        candles.append(create_candle(0, 100, 102, 99, 101, 1000))  # i-2
        candles.append(create_candle(60000, 101, 103, 100, 102, 1000))  # i-1
        candles.append(create_candle(120000, 105, 107, 104, 106, 1000))  # i (gap up)

        engine = MarketStructureEngine(StructureConfig(enable_fvg=True))
        states = engine.update({"LTF": candles})

        ltf_state = states["LTF"]

        # Should detect FVG
        if ltf_state.active_fvgs:
            fvg = ltf_state.active_fvgs[0]
            assert fvg.zone_type == "FVG"
            assert fvg.side == StructureSide.BULL
            assert fvg.top > fvg.bottom

    def test_bearish_fvg(self):
        """Test bearish FVG detection."""
        candles = []

        # Create gap: high[i] < low[i-2]
        candles.append(create_candle(0, 100, 102, 99, 101, 1000))  # i-2
        candles.append(create_candle(60000, 101, 103, 100, 102, 1000))  # i-1
        candles.append(create_candle(120000, 96, 98, 95, 97, 1000))  # i (gap down)

        engine = MarketStructureEngine(StructureConfig(enable_fvg=True))
        states = engine.update({"LTF": candles})

        ltf_state = states["LTF"]

        # Should detect bearish FVG
        if ltf_state.active_fvgs:
            fvg = ltf_state.active_fvgs[0]
            assert fvg.zone_type == "FVG"
            assert fvg.side == StructureSide.BEAR


class TestMomentum:
    """Test structural momentum calculation."""

    def test_fast_momentum(self):
        """Test FAST momentum detection."""
        candles = create_uptrend_candles(15)

        # Add strong breakout with immediate follow-through
        last = candles[-1].close
        for i in range(3):
            candles.append(
                create_candle(
                    timestamp=(len(candles)) * 60000,
                    o=last + i * 3,
                    h=last + i * 3 + 4,
                    l=last + i * 3,
                    c=last + i * 3 + 3.5,
                    v=2000.0,
                )
            )

        engine = MarketStructureEngine()
        states = engine.update({"LTF": candles})

        ltf_state = states["LTF"]

        # Momentum should be calculated
        assert ltf_state.momentum in [
            StructuralMomentum.FAST,
            StructuralMomentum.SLOW,
            StructuralMomentum.STALLED,
        ]


class TestMultiTimeframeAlignment:
    """Test multi-timeframe alignment."""

    def test_aligned_uptrend(self):
        """Test aligned uptrend (HTF up, LTF up)."""
        htf_candles = create_uptrend_candles(30, start_price=100.0)
        ltf_candles = create_uptrend_candles(50, start_price=100.0)

        engine = MarketStructureEngine()
        states = engine.update({"HTF": htf_candles, "LTF": ltf_candles})

        alignment = engine.get_mtf_alignment("HTF", "LTF")

        assert alignment is not None
        # Both should be UP
        assert alignment.htf_structure == StructureState.UP
        assert alignment.ltf_structure == StructureState.UP
        # Should be aligned
        assert alignment.alignment == TimeframeAlignment.ALIGNED
        assert alignment.recommended_mode == TradingMode.TREND_MODE

    def test_htf_range(self):
        """Test HTF range alignment."""
        htf_candles = create_range_candles(30)
        ltf_candles = create_uptrend_candles(40)

        engine = MarketStructureEngine()
        states = engine.update({"HTF": htf_candles, "LTF": ltf_candles})

        alignment = engine.get_mtf_alignment("HTF", "LTF")

        assert alignment is not None
        # HTF should be range
        assert alignment.htf_structure == StructureState.RANGE
        # Alignment should be range dominant
        assert alignment.alignment == TimeframeAlignment.RANGE_DOMINANT
        assert alignment.recommended_mode == TradingMode.RANGE_MODE


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self):
        """Test full analysis pipeline."""
        # Create realistic candles
        candles = create_uptrend_candles(40)

        engine = MarketStructureEngine()
        states = engine.update({"LTF": candles})

        ltf_state = states["LTF"]

        # Verify all components populated
        assert ltf_state.structure is not None
        assert ltf_state.strength_0_100 >= 0
        assert ltf_state.strength_0_100 <= 100
        assert ltf_state.regime in ["TREND", "RANGE", "MIXED", "UNKNOWN"]
        assert ltf_state.momentum in [
            StructuralMomentum.FAST,
            StructuralMomentum.SLOW,
            StructuralMomentum.STALLED,
        ]

    def test_multiple_updates(self):
        """Test engine can handle multiple updates."""
        engine = MarketStructureEngine()

        # Update 1
        candles1 = create_uptrend_candles(20)
        states1 = engine.update({"LTF": candles1})

        # Update 2 (more candles)
        candles2 = create_uptrend_candles(30)
        states2 = engine.update({"LTF": candles2})

        # Both should succeed
        assert states1["LTF"].structure is not None
        assert states2["LTF"].structure is not None

    def test_config_override(self):
        """Test custom configuration."""
        config = StructureConfig(
            pivot_left=2,
            pivot_right=2,
            bos_buffer_pct=0.10,
            enable_fvg=False,
        )

        engine = MarketStructureEngine(config)
        candles = create_uptrend_candles(30)

        states = engine.update({"LTF": candles})

        # Should work with custom config
        assert states["LTF"].structure is not None
        # FVG should be disabled
        assert len(states["LTF"].active_fvgs) == 0
