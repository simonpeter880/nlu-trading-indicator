#!/usr/bin/env python3
"""
Trading Pair Indicator Analysis - Entry Point.
Prompts for a trading pair and runs the full analysis.
"""

import asyncio
import sys

from indicator.display.colors import Colors
from indicator.apps.runner import analyze_pair


def get_user_input() -> tuple:
    """Get trading pair, timeframe, and OI history window from user."""
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}╔════════════════════════════════════════════════════════════╗{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}║         TRADING PAIR INDICATOR ANALYSIS                    ║{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}╚════════════════════════════════════════════════════════════╝{Colors.RESET}")
    print()

    # Get symbol
    print(f"{Colors.BOLD}Enter trading pair:{Colors.RESET}")
    print(f"{Colors.DIM}Examples: BTCUSDT, ETHUSDT, SOLUSDT, BTC/USDT{Colors.RESET}")
    symbol = input(f"{Colors.CYAN}> {Colors.RESET}").strip().upper()

    if not symbol:
        print(f"{Colors.RED}No symbol entered. Using BTCUSDT as default.{Colors.RESET}")
        symbol = "BTCUSDT"

    # Normalize symbol
    symbol = symbol.replace("/", "").replace("-", "").replace("_", "")

    # Get timeframe
    print()
    print(f"{Colors.BOLD}Select timeframe:{Colors.RESET}")
    print(f"{Colors.DIM}1=1m  2=5m  3=15m  4=1h  5=4h  6=1d  (default: 4=1h){Colors.RESET}")
    tf_input = input(f"{Colors.CYAN}> {Colors.RESET}").strip()

    timeframe_map = {
        '1': '1m', '1m': '1m',
        '2': '5m', '5m': '5m',
        '3': '15m', '15m': '15m',
        '4': '1h', '1h': '1h',
        '5': '4h', '4h': '4h',
        '6': '1d', '1d': '1d',
        '': '1h'
    }

    timeframe = timeframe_map.get(tf_input.lower(), '1h')

    # OI history window length
    print()
    print(f"{Colors.BOLD}OI history window (bars):{Colors.RESET}")
    print(f"{Colors.DIM}Default: 30  (min: 2){Colors.RESET}")
    oi_input = input(f"{Colors.CYAN}> {Colors.RESET}").strip()

    oi_history_limit = 30
    if oi_input:
        try:
            oi_history_limit = max(2, int(oi_input))
        except ValueError:
            print(f"{Colors.YELLOW}Invalid input. Using default: 30{Colors.RESET}")

    return symbol, timeframe, oi_history_limit


def main():
    """Main entry point."""
    try:
        symbol, timeframe, oi_history_limit = get_user_input()
        print(f"\n{Colors.DIM}Analyzing {symbol} on {timeframe} timeframe...{Colors.RESET}")

        asyncio.run(analyze_pair(symbol, timeframe, oi_history_limit=oi_history_limit))

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Analysis cancelled.{Colors.RESET}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
