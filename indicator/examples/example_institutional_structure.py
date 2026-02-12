#!/usr/bin/env python3
"""
Institutional Market Structure Engine - Example Usage

Demonstrates complete market structure analysis with the engine.

Usage:
    python example_institutional_structure.py [SYMBOL]
"""

import argparse
import asyncio

from indicator.display.colors import Colors
from indicator.engines.data_fetcher import BinanceIndicatorFetcher
from indicator.engines.institutional_structure import (
    Candle,
    EventType,
    MarketStructureEngine,
    StructureConfig,
    StructureState,
    TradingMode,
)


def print_structure_state(tf_name: str, state, current_price: float):
    """Print market structure state in a nice format."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{tf_name} STRUCTURE{Colors.RESET}")
    print(f"{Colors.CYAN}{'═' * 70}{Colors.RESET}")

    # Structure
    structure_color = (
        Colors.GREEN
        if state.structure == StructureState.UP
        else Colors.RED if state.structure == StructureState.DOWN else Colors.YELLOW
    )

    print(
        f"  {Colors.BOLD}Structure:{Colors.RESET} "
        f"{structure_color}{state.structure.value}{Colors.RESET} ({state.structure_label})"
    )

    print(f"  {Colors.BOLD}Regime:{Colors.RESET} {state.regime}")
    print(f"  {Colors.BOLD}Strength:{Colors.RESET} {state.strength_0_100:.0f}%")
    print(f"  {Colors.BOLD}Momentum:{Colors.RESET} {state.momentum.value}")

    # Last events
    if state.last_bos:
        bos = state.last_bos
        side_color = Colors.GREEN if bos.side.value == "BULL" else Colors.RED
        print(
            f"\n  {Colors.BOLD}Last BOS:{Colors.RESET} {side_color}{bos.side.value}{Colors.RESET} "
            f"@ ${bos.level:,.2f}"
        )

    if state.last_choch:
        choch = state.last_choch
        side_color = Colors.GREEN if choch.side.value == "BULL" else Colors.RED
        print(
            f"  {Colors.BOLD}Last CHoCH:{Colors.RESET} {side_color}{choch.side.value}{Colors.RESET} "
            f"@ ${choch.level:,.2f} {Colors.YELLOW}(reversal warning){Colors.RESET}"
        )

    # Active range
    if state.active_range:
        r = state.active_range
        print(f"\n  {Colors.BOLD}Active Range:{Colors.RESET} " f"${r.bottom:,.2f} - ${r.top:,.2f}")

    # FVGs
    if state.active_fvgs:
        print(f"\n  {Colors.BOLD}Active FVGs:{Colors.RESET}")
        for i, fvg in enumerate(state.active_fvgs[:5], 1):
            side_color = Colors.GREEN if fvg.side.value == "BULL" else Colors.RED
            print(
                f"    {i}. {side_color}{fvg.side.value} FVG{Colors.RESET}: "
                f"${fvg.bottom:,.2f} - ${fvg.top:,.2f}"
            )

    # Recent swings
    if state.last_swing_high or state.last_swing_low:
        print(f"\n  {Colors.BOLD}Recent Swings:{Colors.RESET}")
        if state.last_swing_high:
            print(
                f"    High: ${state.last_swing_high.price:,.2f} "
                f"(strength: {state.last_swing_high.strength:.2f})"
            )
        if state.last_swing_low:
            print(
                f"    Low:  ${state.last_swing_low.price:,.2f} "
                f"(strength: {state.last_swing_low.strength:.2f})"
            )

    # Recent events summary
    if state.recent_events:
        sweeps = [e for e in state.recent_events if e.event_type == EventType.SWEEP]
        accepts = [e for e in state.recent_events if e.event_type == EventType.ACCEPT]
        rejects = [e for e in state.recent_events if e.event_type == EventType.REJECT]

        if sweeps or accepts or rejects:
            print(f"\n  {Colors.BOLD}Recent Events:{Colors.RESET}")
            if sweeps:
                print(f"    Sweeps: {len(sweeps)}")
                for sweep in sweeps[-2:]:
                    confirmed = sweep.details.get("confirmed", False)
                    conf_str = (
                        f"{Colors.GREEN}✓{Colors.RESET}"
                        if confirmed
                        else f"{Colors.RED}✗{Colors.RESET}"
                    )
                    print(f"      {sweep.side.value} sweep @ ${sweep.level:,.2f} {conf_str}")
            if accepts:
                print(f"    Acceptances: {len(accepts)}")
            if rejects:
                print(f"    Rejections: {len(rejects)}")


async def analyze_structure(symbol: str = "BTCUSDT"):
    """Run complete market structure analysis."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}INSTITUTIONAL MARKET STRUCTURE - {symbol}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 70}{Colors.RESET}\n")

    # Initialize
    fetcher = BinanceIndicatorFetcher()

    print(f"{Colors.DIM}Fetching data...{Colors.RESET}")

    # Fetch LTF (5m) and HTF (1h)
    ltf_data = await fetcher.fetch_ohlcv(symbol, interval="5m", limit=100)
    htf_data = await fetcher.fetch_ohlcv(symbol, interval="1h", limit=100)

    await fetcher.close()

    if not ltf_data.closes or not htf_data.closes:
        print(f"{Colors.RED}Error: Could not fetch data{Colors.RESET}")
        return

    print(f"{Colors.GREEN}Data fetched!{Colors.RESET}")

    # Convert to Candle objects
    ltf_candles = [
        Candle(
            timestamp=int(ltf_data.timestamps[i]),
            open=ltf_data.opens[i],
            high=ltf_data.highs[i],
            low=ltf_data.lows[i],
            close=ltf_data.closes[i],
            volume=ltf_data.volumes[i],
        )
        for i in range(len(ltf_data.closes))
    ]

    htf_candles = [
        Candle(
            timestamp=int(htf_data.timestamps[i]),
            open=htf_data.opens[i],
            high=htf_data.highs[i],
            low=htf_data.lows[i],
            close=htf_data.closes[i],
            volume=htf_data.volumes[i],
        )
        for i in range(len(htf_data.closes))
    ]

    # Create engine with custom config
    config = StructureConfig(
        pivot_left=3,
        pivot_right=3,
        bos_buffer_pct=0.05,
        enable_fvg=True,
        accept_hold_candles=3,
    )

    engine = MarketStructureEngine(config)

    # Update engine
    print(f"\n{Colors.DIM}Analyzing market structure...{Colors.RESET}")

    states = engine.update(
        {
            "LTF": ltf_candles,
            "HTF": htf_candles,
        }
    )

    # Get current price
    current_price = ltf_candles[-1].close

    # Print results
    print_structure_state("LTF (5m)", states["LTF"], current_price)
    print_structure_state("HTF (1h)", states["HTF"], current_price)

    # Multi-timeframe alignment
    alignment = engine.get_mtf_alignment("HTF", "LTF")

    if alignment:
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'═' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}MULTI-TIMEFRAME ALIGNMENT{Colors.RESET}")
        print(f"{Colors.CYAN}{'═' * 70}{Colors.RESET}")

        align_color = (
            Colors.GREEN
            if alignment.alignment.value == "ALIGNED"
            else Colors.YELLOW if alignment.alignment.value == "MIXED" else Colors.RED
        )

        print(
            f"  {Colors.BOLD}Alignment:{Colors.RESET} "
            f"{align_color}{alignment.alignment.value}{Colors.RESET}"
        )

        mode_color = (
            Colors.GREEN if alignment.recommended_mode == TradingMode.TREND_MODE else Colors.YELLOW
        )

        print(
            f"  {Colors.BOLD}Recommended Mode:{Colors.RESET} "
            f"{mode_color}{alignment.recommended_mode.value}{Colors.RESET}"
        )

        print(f"\n  {Colors.BOLD}HTF:{Colors.RESET} {alignment.htf_structure.value}")
        print(f"  {Colors.BOLD}LTF:{Colors.RESET} {alignment.ltf_structure.value}")

    # Trading guidance
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}TRADING GUIDANCE{Colors.RESET}")
    print(f"{Colors.CYAN}{'═' * 70}{Colors.RESET}")

    ltf = states["LTF"]
    htf = states["HTF"]

    if alignment and alignment.alignment.value == "ALIGNED":
        if alignment.htf_structure == StructureState.UP:
            print(f"  {Colors.GREEN}✓ LONG BIAS{Colors.RESET}")
            print(f"    Both timeframes in uptrend")
            if ltf.active_fvgs:
                bull_fvgs = [f for f in ltf.active_fvgs if f.side.value == "BULL"]
                if bull_fvgs:
                    print(
                        f"    Pullback target: ${bull_fvgs[0].bottom:,.2f} - ${bull_fvgs[0].top:,.2f}"
                    )
        elif alignment.htf_structure == StructureState.DOWN:
            print(f"  {Colors.RED}✓ SHORT BIAS{Colors.RESET}")
            print(f"    Both timeframes in downtrend")
            if ltf.active_fvgs:
                bear_fvgs = [f for f in ltf.active_fvgs if f.side.value == "BEAR"]
                if bear_fvgs:
                    print(
                        f"    Pullback target: ${bear_fvgs[0].bottom:,.2f} - ${bear_fvgs[0].top:,.2f}"
                    )
    elif alignment and alignment.alignment.value == "RANGE_DOMINANT":
        print(f"  {Colors.YELLOW}⊡ RANGE MODE{Colors.RESET}")
        print(f"    HTF in range - mean reversion trades")
        if htf.active_range:
            print(f"    Range: ${htf.active_range.bottom:,.2f} - ${htf.active_range.top:,.2f}")
    else:
        print(f"  {Colors.YELLOW}⚠ SCALP ONLY{Colors.RESET}")
        print(f"    Timeframes not aligned - reduce size or wait")

    # Warnings
    warnings = []

    if ltf.last_choch and (not ltf.last_bos or ltf.last_choch.time > ltf.last_bos.time):
        warnings.append("Recent CHoCH on LTF - trend may be reversing")

    if ltf.strength_0_100 < 50:
        warnings.append(f"Low LTF structure confidence ({ltf.strength_0_100:.0f}%)")

    if ltf.momentum.value == "STALLED":
        warnings.append("Momentum stalled - weak follow-through")

    if warnings:
        print(f"\n  {Colors.YELLOW}⚠ Warnings:{Colors.RESET}")
        for w in warnings:
            print(f"    • {w}")

    # Summary
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'═' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}SUMMARY{Colors.RESET}")
    print(f"{Colors.CYAN}{'═' * 70}{Colors.RESET}")

    print(f"  Symbol: {symbol}")
    print(f"  Current Price: ${current_price:,.2f}")
    print(f"  LTF Structure: {ltf.structure.value} ({ltf.strength_0_100:.0f}% confidence)")
    print(f"  HTF Structure: {htf.structure.value} ({htf.strength_0_100:.0f}% confidence)")
    if alignment:
        print(f"  Alignment: {alignment.alignment.value}")
        print(f"  Mode: {alignment.recommended_mode.value}")

    print(f"\n{Colors.DIM}Analysis complete.{Colors.RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="Institutional Market Structure Analysis")
    parser.add_argument(
        "symbol", nargs="?", default="BTCUSDT", help="Trading pair symbol (default: BTCUSDT)"
    )
    args = parser.parse_args()

    symbol = args.symbol.upper().replace("/", "").replace("-", "")

    asyncio.run(analyze_structure(symbol))


if __name__ == "__main__":
    main()
