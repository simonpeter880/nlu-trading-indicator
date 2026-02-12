#!/usr/bin/env python3
"""
EMA Filter Example

Demonstrates real-time EMA trend filter with incremental updates.

Usage:
    python example_ema_filter.py [SYMBOL]
"""

import asyncio
import argparse
from indicator.engines.data_fetcher import BinanceIndicatorFetcher
from indicator.engines.ema_filter import (
    EMAFilterEngine,
    EMAConfig,
    Candle,
    EMABias,
    EMARegime,
    MTFAlignment,
    print_ema_block,
)
from indicator.display.colors import Colors


async def demo_ema_filter(symbol: str = "BTCUSDT"):
    """Demonstrate EMA filter."""

    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}EMA FILTER DEMO - {symbol}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.RESET}\n")

    # Initialize
    fetcher = BinanceIndicatorFetcher()

    print(f"{Colors.DIM}Fetching data...{Colors.RESET}")

    # Fetch multi-timeframe data
    data_1m = await fetcher.fetch_ohlcv(symbol, interval="1m", limit=100)
    data_5m = await fetcher.fetch_ohlcv(symbol, interval="5m", limit=100)
    data_1h = await fetcher.fetch_ohlcv(symbol, interval="1h", limit=100)

    await fetcher.close()

    if not data_1m.closes:
        print(f"{Colors.RED}Error: Could not fetch data{Colors.RESET}")
        return

    print(f"{Colors.GREEN}Data fetched!{Colors.RESET}\n")

    # Convert to Candles
    candles_1m = [
        Candle(
            timestamp=int(data_1m.timestamps[i]),
            open=data_1m.opens[i],
            high=data_1m.highs[i],
            low=data_1m.lows[i],
            close=data_1m.closes[i],
            volume=data_1m.volumes[i],
        )
        for i in range(len(data_1m.closes))
    ]

    candles_5m = [
        Candle(
            timestamp=int(data_5m.timestamps[i]),
            open=data_5m.opens[i],
            high=data_5m.highs[i],
            low=data_5m.lows[i],
            close=data_5m.closes[i],
            volume=data_5m.volumes[i],
        )
        for i in range(len(data_5m.closes))
    ]

    candles_1h = [
        Candle(
            timestamp=int(data_1h.timestamps[i]),
            open=data_1h.opens[i],
            high=data_1h.highs[i],
            low=data_1h.lows[i],
            close=data_1h.closes[i],
            volume=data_1h.volumes[i],
        )
        for i in range(len(data_1h.closes))
    ]

    # Create engine with config
    config = EMAConfig(
        ema_periods=[9, 21, 50],
        slope_threshold_factor=0.15,
        width_threshold_factor=0.10,
    )

    engine = EMAFilterEngine(config)

    # Warmup
    print(f"{Colors.DIM}Warming up EMA engine...{Colors.RESET}\n")

    states = engine.warmup({
        "1m": candles_1m,
        "5m": candles_5m,
        "1h": candles_1h,
    })

    # Print results
    print_ema_block(engine, ltf="1m", htf="1h")

    # Detailed analysis
    print(f"\n{Colors.BOLD}{Colors.CYAN}DETAILED ANALYSIS{Colors.RESET}")
    print(f"{Colors.CYAN}{'─' * 70}{Colors.RESET}\n")

    current_price = candles_1m[-1].close

    for tf in ["1m", "5m", "1h"]:
        state = engine.get_state(tf)
        if not state:
            continue

        print(f"{Colors.BOLD}{tf} Timeframe:{Colors.RESET}")
        print(f"  Current Price: ${current_price:,.2f}")
        print(f"  EMA9:  ${state.ema9:,.2f}")
        print(f"  EMA21: ${state.ema21:,.2f}")
        print(f"  EMA50: ${state.ema50:,.2f}")
        print()

        regime_color = Colors.GREEN if state.ema_regime == EMARegime.TREND else Colors.YELLOW
        print(f"  Regime: {regime_color}{state.ema_regime.value}{Colors.RESET}")

        bias_color = Colors.GREEN if state.ema_bias == EMABias.BULL else \
                    Colors.RED if state.ema_bias == EMABias.BEAR else Colors.DIM
        print(f"  Bias: {bias_color}{state.ema_bias.value}{Colors.RESET}")

        align_color = Colors.GREEN if "UP" in state.ema_alignment.value else \
                     Colors.RED if "DOWN" in state.ema_alignment.value else Colors.YELLOW
        print(f"  Alignment: {align_color}{state.ema_alignment.value}{Colors.RESET}")

        strength_color = Colors.GREEN if state.trend_strength_0_100 >= 70 else \
                        Colors.YELLOW if state.trend_strength_0_100 >= 50 else Colors.RED
        print(f"  Strength: {strength_color}{state.trend_strength_0_100:.0f}%{Colors.RESET}")
        print()

        print(f"  Slope21: {state.slope_21:+.5f} ({state.slope_21*100:+.3f}%)")
        print(f"  Slope50: {state.slope_50:+.5f} ({state.slope_50*100:+.3f}%)")
        print(f"  Ribbon Width: {state.ribbon_width:.4f} ({state.ribbon_width*100:.2f}%)")
        print(f"  Extension: {state.ext_21:.4f} ({state.ext_21*100:.2f}%)")
        print()

        extended_str = f"{Colors.YELLOW}YES{Colors.RESET}" if state.extended else "NO"
        pullback_str = f"{Colors.GREEN}YES{Colors.RESET}" if state.pullback_zone_hit else "NO"

        print(f"  Extended: {extended_str}")
        print(f"  Pullback Zone Hit: {pullback_str}")
        print()

    # Trading guidance
    print(f"{Colors.BOLD}{Colors.CYAN}TRADING GUIDANCE{Colors.RESET}")
    print(f"{Colors.CYAN}{'─' * 70}{Colors.RESET}\n")

    mtf = engine.get_mtf_state("1h", "1m")

    if mtf:
        if mtf.alignment_summary == MTFAlignment.ALIGNED:
            if mtf.htf_bias == EMABias.BULL:
                print(f"  {Colors.GREEN}✓ LONG BIAS{Colors.RESET}")
                print(f"    HTF and LTF both bullish")

                ltf_state = engine.get_state("1m")
                if ltf_state.pullback_zone_hit and not ltf_state.extended:
                    print(f"    {Colors.GREEN}Entry zone: Price at EMA21 pullback${Colors.RESET}")
                    print(f"    Target: ${ltf_state.ema21:,.2f}")
                elif ltf_state.extended:
                    print(f"    {Colors.YELLOW}Wait: Price extended from EMA21{Colors.RESET}")

            elif mtf.htf_bias == EMABias.BEAR:
                print(f"  {Colors.RED}✓ SHORT BIAS{Colors.RESET}")
                print(f"    HTF and LTF both bearish")

        elif mtf.alignment_summary == MTFAlignment.RANGE_DOMINANT:
            print(f"  {Colors.YELLOW}⊡ RANGE MODE{Colors.RESET}")
            print(f"    HTF in range - mean reversion trades")

        else:
            print(f"  {Colors.YELLOW}⚠ MIXED{Colors.RESET}")
            print(f"    HTF and LTF disagree - scalp only or wait")

    print()

    # Demo incremental update
    print(f"{Colors.BOLD}{Colors.CYAN}INCREMENTAL UPDATE DEMO{Colors.RESET}")
    print(f"{Colors.CYAN}{'─' * 70}{Colors.RESET}\n")

    print(f"{Colors.DIM}Simulating new candle close on 1m...{Colors.RESET}")

    # Create simulated new candle (slightly up from last)
    last_candle = candles_1m[-1]
    new_candle = Candle(
        timestamp=last_candle.timestamp + 60000,
        open=last_candle.close,
        high=last_candle.close + 5,
        low=last_candle.close - 2,
        close=last_candle.close + 3,
        volume=last_candle.volume * 1.1,
    )

    # Incremental update
    new_state = engine.on_candle_close("1m", new_candle, atr_percent=0.003)

    print(f"New candle: ${new_candle.close:,.2f} "
          f"(was ${last_candle.close:,.2f}, +${new_candle.close - last_candle.close:,.2f})")
    print()

    print(f"Updated EMA state:")
    print(f"  EMA9:  ${new_state.ema9:,.2f}")
    print(f"  EMA21: ${new_state.ema21:,.2f}")
    print(f"  EMA50: ${new_state.ema50:,.2f}")
    print(f"  Bias: {new_state.ema_bias.value}")
    print(f"  Strength: {new_state.trend_strength_0_100:.0f}%")

    print(f"\n{Colors.DIM}Demo complete.{Colors.RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="EMA Filter Demo")
    parser.add_argument(
        "symbol",
        nargs="?",
        default="BTCUSDT",
        help="Trading pair symbol (default: BTCUSDT)"
    )
    args = parser.parse_args()

    symbol = args.symbol.upper().replace("/", "").replace("-", "")

    asyncio.run(demo_ema_filter(symbol))


if __name__ == "__main__":
    main()
