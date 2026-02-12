#!/usr/bin/env python3
"""
Market Structure Analysis Example

Demonstrates the complete market structure detection system:
1. Swing detection (HH/HL/LH/LL)
2. BOS and CHoCH detection
3. Range classification
4. Acceptance/Rejection
5. Fair Value Gaps
6. Multi-timeframe alignment
7. Trade veto system

Usage:
    python example_market_structure.py [SYMBOL]
"""

import asyncio
import argparse
from indicator.engines.data_fetcher import BinanceIndicatorFetcher
from indicator.engines.market_structure import (
    MarketStructureDetector,
    TrendDirection,
    get_allowed_trade_direction,
    structure_veto_signal,
)
from indicator.display import (
    print_structure_deep_dive,
    print_structure_allowed_trades,
    Colors,
)


async def analyze_structure(symbol: str = "BTCUSDT"):
    """Analyze market structure for a symbol."""

    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}MARKET STRUCTURE ANALYSIS - {symbol}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}\n")

    # Initialize
    fetcher = BinanceIndicatorFetcher()
    detector = MarketStructureDetector(
        swing_lookback=5,
        bos_acceptance_bars=3,
        bos_acceptance_volume=1.2,
        followthrough_time_max=300,
        range_threshold_pct=2.0,
        compression_periods=10,
    )

    print(f"{Colors.DIM}Fetching data...{Colors.RESET}")

    # Fetch LTF data (15m candles, last 12 hours)
    ltf_data = await fetcher.fetch_ohlcv(
        symbol=symbol,
        interval="15m",
        limit=48,  # 12 hours of 15m candles
    )

    # Fetch HTF data (1h candles, last 3 days)
    htf_data = await fetcher.fetch_ohlcv(
        symbol=symbol,
        interval="1h",
        limit=72,  # 3 days of 1h candles
    )

    await fetcher.close()

    if not ltf_data.closes or not htf_data.closes:
        print(f"{Colors.RED}Error: Could not fetch data{Colors.RESET}")
        return

    print(f"{Colors.GREEN}Data fetched successfully!{Colors.RESET}\n")

    # Analyze HTF structure first
    print(f"{Colors.BOLD}Analyzing Higher Timeframe (1H)...{Colors.RESET}")
    htf_state = detector.analyze(
        highs=htf_data.highs,
        lows=htf_data.lows,
        closes=htf_data.closes,
        volumes=htf_data.volumes,
        timestamps=htf_data.timestamps,
    )

    print(f"  HTF Trend: {Colors.BOLD}{htf_state.trend_direction.value.upper()}{Colors.RESET}")
    print(f"  HTF Confidence: {htf_state.structure_confidence:.0f}%")

    # Analyze LTF structure with HTF context
    print(f"\n{Colors.BOLD}Analyzing Lower Timeframe (15M) with HTF context...{Colors.RESET}")
    ltf_state = detector.analyze(
        highs=ltf_data.highs,
        lows=ltf_data.lows,
        closes=ltf_data.closes,
        volumes=ltf_data.volumes,
        timestamps=ltf_data.timestamps,
        htf_trend=htf_state.trend_direction,
    )

    # Print detailed structure
    print_structure_deep_dive(ltf_state)

    # Print trade guidance
    print_structure_allowed_trades(ltf_state)

    # Test trade veto system
    print(f"{Colors.BOLD}Trade Veto System Test{Colors.RESET}")
    print(f"  {'─' * 60}")

    test_signals = [
        ("long", 75),
        ("short", 75),
        ("long", 45),
        ("short", 45),
    ]

    for direction, confidence in test_signals:
        vetoed, reason = structure_veto_signal(ltf_state, direction, confidence)

        status_color = Colors.RED if vetoed else Colors.GREEN
        status = "VETOED" if vetoed else "ALLOWED"

        print(f"  {direction.upper():5} @ {confidence}% conf: "
              f"{status_color}{status}{Colors.RESET}")
        if reason:
            print(f"        Reason: {reason}")

    # Summary
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}SUMMARY{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 80}{Colors.RESET}")

    current_price = ltf_data.closes[-1]
    print(f"  Symbol: {symbol}")
    print(f"  Current Price: ${current_price:,.2f}")
    print(f"  LTF Trend: {ltf_state.ltf_trend.value.upper()}")
    print(f"  HTF Trend: {ltf_state.htf_trend.value.upper()}")
    print(f"  MTF Alignment: {ltf_state.tf_alignment.value}")
    print(f"  Structure Confidence: {ltf_state.structure_confidence:.0f}%")

    allowed = get_allowed_trade_direction(ltf_state)
    if allowed:
        allowed_color = Colors.GREEN if allowed == "long" else Colors.RED if allowed == "short" else Colors.YELLOW
        print(f"  Allowed Direction: {allowed_color}{allowed.upper()}{Colors.RESET}")
    else:
        print(f"  Allowed Direction: {Colors.RED}NO TRADE{Colors.RESET}")

    print(f"\n{Colors.DIM}Analysis complete.{Colors.RESET}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Market Structure Analysis Example"
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default="BTCUSDT",
        help="Trading pair symbol (default: BTCUSDT)"
    )
    args = parser.parse_args()

    symbol = args.symbol.upper().replace("/", "").replace("-", "")

    asyncio.run(analyze_structure(symbol))


if __name__ == "__main__":
    main()
