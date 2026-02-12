"""
EMA Ribbon Integration Example

Demonstrates how to integrate the EMA Ribbon Engine into a streaming trading system.
Shows warmup, incremental updates, and compact output formatting.
"""

from nlu_analyzer.indicators.ema_ribbon import (
    EMARibbonEngine,
    EMARibbonConfig,
    Candle,
    format_ribbon_output
)
from typing import Dict, Optional


def example_streaming_integration():
    """
    Example: Integrate EMA Ribbon into a continuous streaming loop.

    This demonstrates the typical workflow:
    1. Initialize engine with config
    2. Warmup with historical data
    3. Process new candles incrementally on close
    4. Display compact output
    """

    # ========== CONFIGURATION ==========
    config = EMARibbonConfig(
        ribbon_periods=[9, 12, 15, 18, 21, 25, 30, 35, 40, 50],  # Default dense ribbon
        atr_period=14,
        width_smooth_period=5,
        slope_lookback_by_tf={
            "1m": 15,
            "5m": 10,
            "1h": 4
        }
    )

    # Alternative: Use lighter ribbon preset
    # config = EMARibbonConfig(
    #     ribbon_periods=[9, 12, 15, 21, 30, 40, 50]
    # )

    # ========== INITIALIZATION ==========
    engine = EMARibbonEngine(config)

    # ========== WARMUP WITH HISTORICAL DATA ==========
    # In production, fetch historical candles from your data source
    print("Warming up EMA Ribbon Engine...")

    # Simulate historical candles (replace with actual data)
    historical_candles_by_tf = {
        "1m": generate_test_candles(100.0, 100, trend="up"),
        "5m": generate_test_candles(100.0, 80, trend="up"),
        "1h": generate_test_candles(100.0, 60, trend="up")
    }

    # Optional: ATR% values if available (decimal, e.g., 0.01 = 1%)
    atr_percent_by_tf = {
        "1m": 0.008,   # 0.8%
        "5m": 0.012,   # 1.2%
        "1h": 0.020    # 2.0%
    }

    # Warmup all timeframes
    engine.warmup(historical_candles_by_tf, atr_percent_by_tf)
    print("Warmup complete.\n")

    # ========== STREAMING LOOP ==========
    print("=" * 80)
    print("STREAMING MODE - Processing new candles as they close")
    print("=" * 80)

    # Simulate receiving new candles
    # In production, this would be triggered by your candle aggregation system

    # Example: New 1m candle closes
    new_candle_1m = Candle(
        timestamp=100.0,
        open=110.5,
        high=110.8,
        low=110.3,
        close=110.7,
        volume=50000.0
    )

    ribbon_state_1m = engine.on_candle_close(
        tf="1m",
        candle=new_candle_1m,
        atr_percent=0.008,
        ema_system_state=None  # Optional: pass EMA system state if available
    )

    print(f"\n[1m CANDLE CLOSED] Price: {new_candle_1m.close:.2f}")
    print(format_ribbon_output({"1m": ribbon_state_1m}, compact=True))

    # Example: New 5m candle closes
    new_candle_5m = Candle(
        timestamp=500.0,
        open=110.6,
        high=111.2,
        low=110.4,
        close=111.0,
        volume=250000.0
    )

    ribbon_state_5m = engine.on_candle_close(
        tf="5m",
        candle=new_candle_5m,
        atr_percent=0.012
    )

    print(f"\n[5m CANDLE CLOSED] Price: {new_candle_5m.close:.2f}")
    print(format_ribbon_output({"5m": ribbon_state_5m}, compact=True))

    # ========== MULTI-TIMEFRAME UPDATE ==========
    # Alternative: Update multiple timeframes at once (convenience method)
    print("\n" + "=" * 80)
    print("MULTI-TIMEFRAME UPDATE")
    print("=" * 80)

    latest_candles = {
        "1m": [new_candle_1m],
        "5m": [new_candle_5m]
    }

    all_ribbon_states = engine.update(
        candles_by_tf=latest_candles,
        atr_percent_by_tf={"1m": 0.008, "5m": 0.012}
    )

    print(format_ribbon_output(all_ribbon_states, compact=True))

    # ========== ACCESSING SPECIFIC METRICS ==========
    print("\n" + "=" * 80)
    print("DETAILED METRICS ACCESS")
    print("=" * 80)

    state = ribbon_state_1m

    print(f"\nTimeframe: 1m")
    print(f"  Direction: {state.ribbon_direction.value}")
    print(f"  State: {state.ribbon_state.value}")
    print(f"  Strength: {state.ribbon_strength_0_100:.1f}")
    print(f"  Stack Score: {state.stack_score:.2%}")
    print(f"  Ribbon Width: {state.ribbon_width_smooth:.4f}")
    print(f"  Width Rate: {state.width_rate:+.2%}")
    print(f"  Center: {state.ribbon_center:.2f}")
    print(f"  Center Slope: {state.ribbon_center_slope:+.4%}")
    print(f"  Pullback: {state.pullback_into_ribbon}")

    print(f"\n  EMA Values:")
    for period, ema_value in sorted(state.emas.items()):
        print(f"    EMA{period}: {ema_value:.2f}")

    print(f"\n  Debug Info:")
    for key, value in state.debug.items():
        print(f"    {key}: {value}")

    # ========== TRADING LOGIC EXAMPLE ==========
    print("\n" + "=" * 80)
    print("TRADING LOGIC EXAMPLE")
    print("=" * 80)

    def analyze_ribbon_health(ribbon_state):
        """Example: Use ribbon state for trading decisions"""

        # This module is focused on TREND HEALTH, not entry signals
        # Use it to filter/qualify entries from your EMA system

        if ribbon_state.ribbon_state.value == "HEALTHY":
            if ribbon_state.ribbon_direction.value == "BULL":
                return "✓ HEALTHY UPTREND - Favor long entries"
            else:
                return "✓ HEALTHY DOWNTREND - Favor short entries"

        elif ribbon_state.ribbon_state.value == "WEAKENING":
            return "⚠ TREND WEAKENING - Reduce position size, tighten stops"

        elif ribbon_state.ribbon_state.value == "EXHAUSTING":
            return "⚠⚠ TREND EXHAUSTING - Avoid new entries, consider exits"

        else:  # CHOP
            return "✗ CHOPPY MARKET - Avoid trend-following strategies"

    for tf in ["1m", "5m"]:
        if tf in all_ribbon_states:
            state = all_ribbon_states[tf]
            analysis = analyze_ribbon_health(state)
            print(f"\n{tf}: {analysis}")

    # ========== COMPLETE COMPACT OUTPUT ==========
    print("\n" + "=" * 80)
    print("COMPACT DASHBOARD OUTPUT")
    print("=" * 80 + "\n")

    # This is what you'd display in your live trading dashboard
    dashboard = format_ribbon_output(all_ribbon_states, compact=True)
    print(dashboard)


def generate_test_candles(start_price: float, count: int, trend: str = "up"):
    """Generate synthetic test candles"""
    candles = []
    price = start_price
    trend_strength = 0.002

    for i in range(count):
        candles.append(Candle(
            timestamp=float(i),
            open=price,
            high=price * 1.001,
            low=price * 0.999,
            close=price,
            volume=10000.0
        ))

        if trend == "up":
            price *= (1 + trend_strength)
        elif trend == "down":
            price *= (1 - trend_strength)

    return candles


# ========== PRODUCTION INTEGRATION TEMPLATE ==========

class ProductionRibbonIntegration:
    """
    Template for production integration.

    Replace the placeholder methods with your actual data sources.
    """

    def __init__(self):
        # Initialize ribbon engine
        self.ribbon_engine = EMARibbonEngine(EMARibbonConfig(
            ribbon_periods=[9, 12, 15, 21, 30, 40, 50]  # Lighter preset
        ))

        # Track which timeframes are warmed up
        self.warmed_up_tfs = set()

    def on_startup(self):
        """Called once on system startup"""
        # Fetch historical candles for all timeframes
        historical_data = self.fetch_historical_candles()
        atr_data = self.fetch_current_atr_percent()

        # Warmup
        self.ribbon_engine.warmup(historical_data, atr_data)

        # Mark all timeframes as ready
        for tf in historical_data.keys():
            self.warmed_up_tfs.add(tf)

        print(f"EMA Ribbon warmed up for timeframes: {self.warmed_up_tfs}")

    def on_candle_close(self, tf: str, candle: Candle):
        """
        Called whenever a candle closes on any timeframe.

        This is where you integrate into your existing candle aggregation system.
        """
        if tf not in self.warmed_up_tfs:
            print(f"Warning: {tf} not warmed up yet")
            return

        # Get current ATR% (optional)
        atr_percent = self.get_current_atr_percent(tf)

        # Get EMA system state (optional, if you want to reuse existing EMAs)
        ema_system_state = self.get_ema_system_state(tf)

        # Update ribbon
        ribbon_state = self.ribbon_engine.on_candle_close(
            tf=tf,
            candle=candle,
            atr_percent=atr_percent,
            ema_system_state=ema_system_state
        )

        # Store or process the ribbon state
        self.process_ribbon_state(tf, ribbon_state)

        # Display if needed
        if self.should_display(tf):
            print(format_ribbon_output({tf: ribbon_state}, compact=True))

    def fetch_historical_candles(self) -> Dict[str, list]:
        """Fetch historical candles from your data source"""
        # TODO: Replace with actual data fetching
        return {
            "1m": [],
            "5m": [],
            "1h": []
        }

    def fetch_current_atr_percent(self) -> Dict[str, float]:
        """Fetch current ATR% values"""
        # TODO: Replace with actual ATR calculation
        return {
            "1m": 0.008,
            "5m": 0.012,
            "1h": 0.020
        }

    def get_current_atr_percent(self, tf: str) -> Optional[float]:
        """Get current ATR% for a timeframe"""
        # TODO: Replace with actual ATR lookup
        return None

    def get_ema_system_state(self, tf: str) -> Optional[Dict]:
        """Get current EMA system state (optional)"""
        # TODO: Replace with actual EMA system lookup
        return None

    def process_ribbon_state(self, tf: str, ribbon_state):
        """Process ribbon state (store, log, trade decision, etc.)"""
        # TODO: Add your logic here
        pass

    def should_display(self, tf: str) -> bool:
        """Determine if output should be displayed"""
        # TODO: Add your display logic
        return False


if __name__ == "__main__":
    print("EMA RIBBON ENGINE - INTEGRATION EXAMPLE")
    print("=" * 80)
    example_streaming_integration()
