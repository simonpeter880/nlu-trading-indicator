"""
Supertrend Filter Integration Example

Demonstrates how to integrate the Supertrend Engine for regime labeling
and directional bias filtering in a streaming trading system.
"""

from typing import Dict

from nlu_analyzer.indicators.supertrend_filter import (
    Candle,
    Direction,
    Regime,
    SupertrendConfig,
    SupertrendEngine,
    format_supertrend_output,
)


def example_streaming_integration():
    """
    Example: Integrate Supertrend Engine into continuous streaming loop.

    Workflow:
    1. Initialize engine with config
    2. Warmup with historical data
    3. Process new candles incrementally
    4. Use regime label and direction for filtering.
    """

    # ========== CONFIGURATION ==========
    config = SupertrendConfig(
        atr_period=10,
        multiplier=3.0,
        timeframes=["1m", "5m", "1h"],
        flip_window=20,
        flip_rate_chop=0.10,
        flip_rate_trend=0.05,
        min_hold_bars=3,
        st_distance_factor=0.15,
        atrp_min_chop=0.0010,
    )

    # Alternative: Per-timeframe overrides
    # config = SupertrendConfig(
    #     atr_period=10,
    #     per_tf_overrides={
    #         "1m": {"atr_period": 7, "multiplier": 2.5},
    #         "1h": {"atr_period": 14, "multiplier": 3.5}
    #     }
    # )

    # ========== INITIALIZATION ==========
    engine = SupertrendEngine(config)

    # ========== WARMUP WITH HISTORICAL DATA ==========
    print("Warming up Supertrend Engine...")

    # Simulate historical candles (replace with actual data)
    historical_candles_by_tf = {
        "1m": generate_test_candles(100.0, 100, 0.002),
        "5m": generate_test_candles(100.0, 80, 0.003),
        "1h": generate_test_candles(100.0, 60, 0.004),
    }

    # Warmup all timeframes
    warmup_results = engine.warmup(historical_candles_by_tf)
    print(f"Warmed up {len(warmup_results)} timeframes.\n")

    # ========== STREAMING LOOP ==========
    print("=" * 80)
    print("STREAMING MODE - Processing new candles")
    print("=" * 80)

    # Example: New 1m candle closes
    new_candle_1m = Candle(
        timestamp=100.0, open=110.5, high=110.8, low=110.3, close=110.7, volume=50000.0
    )

    result_1m = engine.on_candle_close("1m", new_candle_1m)

    print(f"\n[1m CANDLE CLOSED] Price: {new_candle_1m.close:.2f}")
    print(f"Direction: {result_1m.st_direction.value}")
    print(f"Regime: {result_1m.regime.value}")
    print(f"ST Line: {result_1m.st_line:.2f}")
    print(f"ATR%: {result_1m.atr_percent*100:.2f}%")
    print(f"Flip Event: {result_1m.flip_event}")

    # ========== MULTI-TIMEFRAME UPDATE ==========
    print("\n" + "=" * 80)
    print("MULTI-TIMEFRAME UPDATE")
    print("=" * 80)

    latest_candles = {
        "1m": [new_candle_1m],
        "5m": [
            Candle(timestamp=100.0, open=110.5, high=111.5, low=110.0, close=111.0, volume=250000.0)
        ],
        "1h": [
            Candle(
                timestamp=100.0, open=109.0, high=112.0, low=108.5, close=111.5, volume=3000000.0
            )
        ],
    }

    all_results = engine.update(latest_candles)
    print(format_supertrend_output(all_results, compact=True))

    # ========== DETAILED METRICS ACCESS ==========
    print("\n" + "=" * 80)
    print("DETAILED METRICS ACCESS")
    print("=" * 80)

    state = result_1m

    print("\nTimeframe: 1m")
    print(f"  Direction: {state.st_direction.value}")
    print(f"  Regime: {state.regime.value}")
    print(f"  Strength: {state.regime_strength_0_100:.1f}/100")
    print(f"  ST Line: {state.st_line:.2f}")
    print(f"  Final Upper: {state.final_upper:.2f}")
    print(f"  Final Lower: {state.final_lower:.2f}")
    print(f"  ATR: {state.atr:.2f} ({state.atr_percent*100:.2f}%)")
    print(
        f"  Flips (last {state.debug['flip_window']}): {state.flips_last_n} ({state.flip_rate:.2%})"
    )
    print(f"  Hold Count: {state.direction_hold_count}")
    print(f"  Distance Avg: {state.distance_avg*100:.2f}%")

    # ========== TRADING LOGIC EXAMPLE ==========
    print("\n" + "=" * 80)
    print("TRADING LOGIC EXAMPLE")
    print("=" * 80)

    def analyze_supertrend_context(st_state):
        """Example: Use Supertrend for filtering (NOT entry trigger)"""

        signals = []

        # Regime filter
        if st_state.regime == Regime.TREND:
            if st_state.st_direction == Direction.UP:
                signals.append("✓ TREND UP - Allow long setups from other systems")
                signals.append(f"  Strength: {st_state.regime_strength_0_100:.0f}/100")
            else:
                signals.append("✓ TREND DOWN - Allow short setups from other systems")
                signals.append(f"  Strength: {st_state.regime_strength_0_100:.0f}/100")

        elif st_state.regime == Regime.CHOP:
            signals.append("✗ CHOP - Avoid trend-following strategies")
            signals.append(
                f"  Flip rate: {st_state.flip_rate:.2%} (threshold: {st_state.debug['flip_rate_chop_thr']:.2%})"
            )
            if st_state.distance_avg < 0.001:
                signals.append("  Price hugging ST line - very noisy")

        # Direction filter
        if st_state.regime == Regime.TREND:
            if st_state.flip_event:
                signals.append("⚠ Direction flipped - wait for confirmation")
            elif st_state.direction_hold_count < st_state.debug["min_hold_bars"]:
                signals.append(
                    f"⚠ Hold count low ({st_state.direction_hold_count}) - wait for stability"
                )

        # Strength warnings
        if st_state.regime == Regime.TREND and st_state.regime_strength_0_100 < 60:
            signals.append("⚠ Trend strength weakening - tighten risk management")

        return signals

    for tf in ["1m", "5m", "1h"]:
        if tf in all_results:
            signals = analyze_supertrend_context(all_results[tf])

            print(f"\n{tf} Analysis:")
            for signal in signals:
                print(f"  {signal}")

    # ========== MULTI-TIMEFRAME ALIGNMENT ==========
    print("\n" + "=" * 80)
    print("MULTI-TIMEFRAME ALIGNMENT EXAMPLE")
    print("=" * 80)

    def check_mtf_alignment(results):
        """Check multi-timeframe alignment."""

        # Get HTF regime and direction
        htf_trend = results["1h"].regime == Regime.TREND
        htf_dir = results["1h"].st_direction

        # Get MTF regime
        mtf_trend = results["5m"].regime == Regime.TREND
        mtf_dir = results["5m"].st_direction

        # Get LTF
        ltf_dir = results["1m"].st_direction

        alignment = []

        if htf_trend and mtf_trend:
            if htf_dir == mtf_dir == ltf_dir:
                alignment.append(f"✓✓ FULL ALIGNMENT - All timeframes {htf_dir.value}")
                alignment.append("  → Strongest trend-following conditions")
            elif htf_dir == mtf_dir:
                alignment.append(f"✓ HTF+MTF ALIGNED {htf_dir.value}, LTF {ltf_dir.value}")
                alignment.append("  → Good for counter-trend scalps on LTF")
            else:
                alignment.append("⚠ MIXED SIGNALS - Check higher timeframes")
        else:
            alignment.append("✗ NO ALIGNMENT - Avoid trend strategies")
            if not htf_trend:
                alignment.append("  HTF in CHOP - wait for clarity")
            if not mtf_trend:
                alignment.append("  MTF in CHOP - local noise")

        return alignment

    if len(all_results) >= 3:
        alignment = check_mtf_alignment(all_results)
        for msg in alignment:
            print(msg)

    # ========== COMPACT DASHBOARD OUTPUT ==========
    print("\n" + "=" * 80)
    print("COMPACT DASHBOARD OUTPUT")
    print("=" * 80 + "\n")

    dashboard = format_supertrend_output(all_results, compact=True)
    print(dashboard)


def generate_test_candles(start_price: float, count: int, trend: float):
    """Generate synthetic test candles."""
    candles = []
    price = start_price

    for i in range(count):
        high = price * 1.002
        low = price * 0.998
        close = price * (1 + trend)

        candles.append(
            Candle(timestamp=float(i), open=price, high=high, low=low, close=close, volume=100000.0)
        )

        price = close

    return candles


# ========== PRODUCTION INTEGRATION TEMPLATE ==========


class ProductionSupertrendIntegration:
    """
    Template for production integration.

    Replace placeholder methods with your actual systems.
    """

    def __init__(self):
        # Initialize Supertrend engine
        self.st_engine = SupertrendEngine(
            SupertrendConfig(atr_period=10, multiplier=3.0, flip_window=20)
        )

        # Track warmed up timeframes
        self.warmed_up_tfs = set()

    def on_startup(self):
        """Called once on system startup."""
        # Fetch historical candles
        historical_data = self.fetch_historical_candles()

        # Warmup
        self.st_engine.warmup(historical_data)

        for tf in historical_data.keys():
            self.warmed_up_tfs.add(tf)

        print(f"Supertrend warmed up for: {self.warmed_up_tfs}")

    def on_candle_close(self, tf: str, candle: Candle):
        """
        Called whenever a candle closes on any timeframe.

        Integrate into your existing candle aggregation system.
        """
        if tf not in self.warmed_up_tfs:
            print(f"Warning: {tf} not warmed up yet")
            return

        # Update Supertrend
        st_state = self.st_engine.on_candle_close(tf, candle)

        # Use for filtering (NOT entry trigger)
        self.apply_supertrend_filter(tf, st_state)

        # Display if needed
        if self.should_display(tf):
            print(format_supertrend_output({tf: st_state}, compact=True))

    def apply_supertrend_filter(self, tf: str, st_state):
        """
        Apply Supertrend as filter for other trading systems.

        DO NOT use flip_event directly for entries.
        Use regime and direction as context.
        """

        # Example: Block trades in CHOP regime
        if st_state.regime == Regime.CHOP:
            self.disable_trend_strategies(tf)
            return

        # Example: Only allow longs when ST is UP in TREND
        if st_state.regime == Regime.TREND:
            if st_state.st_direction == Direction.UP:
                self.allow_long_entries(tf)
                self.block_short_entries(tf)
            else:
                self.allow_short_entries(tf)
                self.block_long_entries(tf)

    def fetch_historical_candles(self) -> Dict[str, list]:
        """Fetch historical candles from your data source."""
        # TODO: Replace with actual data fetching
        return {"1m": [], "5m": [], "1h": []}

    def disable_trend_strategies(self, tf: str):
        """Disable trend-following on this timeframe."""
        # TODO: Implement
        pass

    def allow_long_entries(self, tf: str):
        """Allow long entries on this timeframe."""
        # TODO: Implement
        pass

    def block_long_entries(self, tf: str):
        """Block long entries on this timeframe."""
        # TODO: Implement
        pass

    def allow_short_entries(self, tf: str):
        """Allow short entries on this timeframe."""
        # TODO: Implement
        pass

    def block_short_entries(self, tf: str):
        """Block short entries on this timeframe."""
        # TODO: Implement
        pass

    def should_display(self, tf: str) -> bool:
        """Determine if output should be displayed."""
        # TODO: Add your display logic
        return False


if __name__ == "__main__":
    print("SUPERTREND FILTER - INTEGRATION EXAMPLE")
    print("=" * 80)
    example_streaming_integration()
