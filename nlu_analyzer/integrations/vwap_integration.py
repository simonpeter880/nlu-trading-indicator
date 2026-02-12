"""
VWAP Engine Integration Example

Demonstrates how to integrate the VWAP Engine into a streaming trading system.
Shows warmup, incremental updates, anchored VWAP creation, and output formatting.
"""

from datetime import datetime, timezone
from typing import Dict, Optional

from nlu_analyzer.indicators.vwap_engine import (
    Candle,
    PriceSource,
    VWAPConfig,
    VWAPEngine,
    format_vwap_output,
)


def example_streaming_integration():
    """
    Example: Integrate VWAP Engine into a continuous streaming loop.

    Workflow:
    1. Initialize engine with config
    2. Warmup with historical data
    3. Process new candles incrementally
    4. Add anchored VWAPs on events
    5. Display compact output
    """

    # ========== CONFIGURATION ==========
    config = VWAPConfig(
        price_source=PriceSource.TYPICAL,
        timeframes=["1m", "5m", "1h"],
        session_reset="UTC_DAY",
        timezone="UTC",
        enable_std_bands=True,
        std_band_multipliers=[1.0, 2.0],
        min_bars_for_std=30,
        hold_bars=3,
        touch_tolerance=0.0001,
        reclaim_tolerance=0.0002,
        max_anchors_per_tf=3,
    )

    # Alternative: Use close price only
    # config = VWAPConfig(price_source=PriceSource.CLOSE)

    # ========== INITIALIZATION ==========
    engine = VWAPEngine(config)

    # ========== WARMUP WITH HISTORICAL DATA ==========
    print("Warming up VWAP Engine...")

    # Simulate historical candles (replace with actual data)
    historical_candles_by_tf = {
        "1m": generate_test_candles(100.0, 100, 0.002),
        "5m": generate_test_candles(100.0, 80, 0.003),
        "1h": generate_test_candles(100.0, 60, 0.004),
    }

    # Optional: ATR% values if available
    atr_percent_by_tf = {"1m": 0.008, "5m": 0.012, "1h": 0.020}

    # Warmup all timeframes
    engine.warmup(historical_candles_by_tf, atr_percent_by_tf)
    print("Warmup complete.\n")

    # ========== STREAMING LOOP ==========
    print("=" * 80)
    print("STREAMING MODE - Processing new candles")
    print("=" * 80)

    # Example: New 1m candle closes
    new_candle_1m = Candle(
        timestamp=datetime(2024, 1, 10, 15, 30, 0, tzinfo=timezone.utc).timestamp(),
        open=110.5,
        high=110.8,
        low=110.3,
        close=110.7,
        volume=50000.0,
    )

    result_1m = engine.on_candle_close(tf="1m", candle=new_candle_1m, atr_percent=0.008)

    print(f"\n[1m CANDLE CLOSED] Price: {new_candle_1m.close:.2f}")
    print(format_vwap_output(result_1m, compact=False))

    # ========== ADDING ANCHORED VWAP ==========
    print("\n" + "=" * 80)
    print("ADDING ANCHORED VWAP")
    print("=" * 80)

    # Example: BOS (Break of Structure) detected at current time
    bos_time = new_candle_1m.timestamp
    anchor = engine.add_anchor(
        tf="1m", anchor_time=bos_time, anchor_id="BOS_110.7", note="BOS@110.7", kind="BOS"
    )

    print(f"Added anchored VWAP: {anchor.anchor_id} at {bos_time}")

    # Process next candle
    next_candle = Candle(
        timestamp=bos_time + 60, open=111.0, high=111.5, low=110.8, close=111.2, volume=45000.0
    )

    result_with_anchor = engine.on_candle_close("1m", next_candle, atr_percent=0.008)

    print(f"\n[1m CANDLE CLOSED] Price: {next_candle.close:.2f}")
    print(f"Anchored VWAP count: {len(result_with_anchor.anchors_by_tf.get('1m', []))}")

    # ========== MULTI-TIMEFRAME UPDATE ==========
    print("\n" + "=" * 80)
    print("MULTI-TIMEFRAME UPDATE")
    print("=" * 80)

    latest_candles = {
        "1m": [next_candle],
        "5m": [
            Candle(
                timestamp=bos_time, open=110.5, high=111.5, low=110.0, close=111.0, volume=250000.0
            )
        ],
        "1h": [
            Candle(
                timestamp=bos_time, open=109.0, high=112.0, low=108.5, close=111.5, volume=3000000.0
            )
        ],
    }

    all_results = engine.update(candles_by_tf=latest_candles, atr_percent_by_tf=atr_percent_by_tf)

    print(format_vwap_output(all_results, compact=True))

    # ========== ACCESSING SPECIFIC METRICS ==========
    print("\n" + "=" * 80)
    print("DETAILED METRICS ACCESS")
    print("=" * 80)

    session_1m = all_results.session_by_tf["1m"]

    print("\nTimeframe: 1m Session VWAP")
    print(f"  VWAP: {session_1m.vwap:.2f}")
    print(f"  Price Position: {session_1m.price_position.value}")
    print(f"  Interaction State: {session_1m.interaction_state.value}")
    print(f"  Distance: {session_1m.distance.get('pct', 0)*100:+.2f}%")

    if session_1m.distance.get("sigma") is not None:
        print(f"  Sigma Distance: {session_1m.distance['sigma']:+.2f}σ")

    print(f"  Bar Count: {session_1m.bar_count}")
    print(f"  Volume Sum: {session_1m.v_sum:.0f}")

    if session_1m.bands:
        print(f"\n  Bands ({session_1m.bands.method.value}):")
        if session_1m.bands.std is not None:
            print(f"    Standard Deviation: {session_1m.bands.std:.4f}")
        for k, (lower, upper) in sorted(session_1m.bands.bands.items()):
            print(f"    {k}σ: [{lower:.2f}, {upper:.2f}]")

    # ========== TRADING LOGIC EXAMPLE ==========
    print("\n" + "=" * 80)
    print("TRADING LOGIC EXAMPLE")
    print("=" * 80)

    def analyze_vwap_context(session_state, weekly_state, anchors):
        """Example: Use VWAP context for trading decisions"""

        signals = []

        # Session VWAP context
        if session_state.interaction_state.value == "RECLAIM":
            signals.append("✓ SESSION RECLAIM - Bullish bias, look for longs")
        elif session_state.interaction_state.value == "LOSS":
            signals.append("✗ SESSION LOSS - Bearish bias, look for shorts")
        elif session_state.interaction_state.value == "ACCEPT":
            if session_state.price_position.value == "ABOVE":
                signals.append("✓ ACCEPTING ABOVE - Bullish continuation")
            else:
                signals.append("✗ ACCEPTING BELOW - Bearish continuation")

        # Weekly VWAP alignment
        if (
            session_state.price_position.value == "ABOVE"
            and weekly_state.price_position.value == "ABOVE"
        ):
            signals.append("✓ ALIGNED ABOVE - Multi-timeframe bullish")
        elif (
            session_state.price_position.value == "BELOW"
            and weekly_state.price_position.value == "BELOW"
        ):
            signals.append("✗ ALIGNED BELOW - Multi-timeframe bearish")

        # Anchored VWAP
        for anchor in anchors:
            if anchor.interaction_state.value == "RECLAIM":
                signals.append(f"✓ {anchor.anchor_note} RECLAIMED - Local bullish reversal")

        # Distance warnings
        if session_state.distance.get("sigma"):
            sigma_dist = abs(session_state.distance["sigma"])
            if sigma_dist > 2.0:
                signals.append(f"⚠ EXTENDED {sigma_dist:.1f}σ - Mean reversion risk")

        return signals

    for tf in ["1m", "5m", "1h"]:
        if tf in all_results.session_by_tf:
            session = all_results.session_by_tf[tf]
            weekly = all_results.weekly_by_tf.get(tf)
            anchors = all_results.anchors_by_tf.get(tf, [])

            signals = analyze_vwap_context(session, weekly, anchors)

            print(f"\n{tf} Signals:")
            for signal in signals:
                print(f"  {signal}")

    # ========== COMPACT DASHBOARD OUTPUT ==========
    print("\n" + "=" * 80)
    print("COMPACT DASHBOARD OUTPUT")
    print("=" * 80 + "\n")

    dashboard = format_vwap_output(all_results, compact=True)
    print(dashboard)


def generate_test_candles(start_price: float, count: int, trend: float):
    """Generate synthetic test candles"""
    candles = []
    price = start_price
    base_time = datetime(2024, 1, 10, 0, 0, 0, tzinfo=timezone.utc).timestamp()

    for i in range(count):
        high = price * 1.002
        low = price * 0.998
        close = price * (1 + trend)

        candles.append(
            Candle(
                timestamp=base_time + i * 60,
                open=price,
                high=high,
                low=low,
                close=close,
                volume=100000.0,
            )
        )

        price = close

    return candles


# ========== PRODUCTION INTEGRATION TEMPLATE ==========


class ProductionVWAPIntegration:
    """
    Template for production integration.

    Replace placeholder methods with your actual data sources.
    """

    def __init__(self):
        # Initialize VWAP engine
        self.vwap_engine = VWAPEngine(
            VWAPConfig(
                price_source=PriceSource.TYPICAL,
                timeframes=["1m", "5m", "1h"],
                max_anchors_per_tf=3,
            )
        )

        # Track warmed up timeframes
        self.warmed_up_tfs = set()

    def on_startup(self):
        """Called once on system startup"""
        # Fetch historical candles
        historical_data = self.fetch_historical_candles()
        atr_data = self.fetch_current_atr_percent()

        # Warmup
        self.vwap_engine.warmup(historical_data, atr_data)

        for tf in historical_data.keys():
            self.warmed_up_tfs.add(tf)

        print(f"VWAP Engine warmed up for: {self.warmed_up_tfs}")

    def on_candle_close(self, tf: str, candle: Candle):
        """
        Called whenever a candle closes on any timeframe.

        Integrate into your existing candle aggregation system.
        """
        if tf not in self.warmed_up_tfs:
            print(f"Warning: {tf} not warmed up yet")
            return

        # Get current ATR% (optional)
        atr_percent = self.get_current_atr_percent(tf)

        # Update VWAP
        vwap_result = self.vwap_engine.on_candle_close(
            tf=tf, candle=candle, atr_percent=atr_percent
        )

        # Process VWAP context
        self.process_vwap_context(tf, vwap_result)

        # Display if needed
        if self.should_display(tf):
            print(format_vwap_output(vwap_result, compact=True))

    def on_structure_event(self, tf: str, event_type: str, timestamp: float, price: float):
        """
        Called when market structure event detected (BOS, CHoCH, etc.).

        Creates anchored VWAP at the event.
        """
        anchor_id = f"{event_type}_{price:.2f}"
        note = f"{event_type}@{price:.2f}"

        self.vwap_engine.add_anchor(
            tf=tf, anchor_time=timestamp, anchor_id=anchor_id, note=note, kind=event_type
        )

        print(f"Added anchor: {note} on {tf}")

    def fetch_historical_candles(self) -> Dict[str, list]:
        """Fetch historical candles from your data source"""
        # TODO: Replace with actual data fetching
        return {"1m": [], "5m": [], "1h": []}

    def fetch_current_atr_percent(self) -> Dict[str, float]:
        """Fetch current ATR% values"""
        # TODO: Replace with actual ATR calculation
        return {"1m": 0.008, "5m": 0.012, "1h": 0.020}

    def get_current_atr_percent(self, tf: str) -> Optional[float]:
        """Get current ATR% for a timeframe"""
        # TODO: Replace with actual ATR lookup
        return None

    def process_vwap_context(self, tf: str, vwap_result):
        """Process VWAP context for trading decisions"""
        # TODO: Add your logic here
        pass

    def should_display(self, tf: str) -> bool:
        """Determine if output should be displayed"""
        # TODO: Add your display logic
        return False


if __name__ == "__main__":
    print("VWAP ENGINE - INTEGRATION EXAMPLE")
    print("=" * 80)
    example_streaming_integration()
