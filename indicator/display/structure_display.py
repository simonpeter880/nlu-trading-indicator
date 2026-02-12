"""
Market Structure Display Functions

Beautiful terminal output for market structure analysis.
"""

from typing import Optional, List
from market_structure import (
    MarketStructureState,
    TrendDirection,
    StructureEvent,
    RangeType,
    StructuralMomentum,
    TimeframeAlignment,
    FairValueGap,
    StructureBreak,
    SwingPoint,
)
from continuous.market_structure_adapter import MarketStructureSignal
from .colors import Colors


def print_structure_header() -> None:
    """Print market structure section header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}┌{'─' * 78}┐{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}│  MARKET STRUCTURE - THE FOUNDATION{' ' * 44}│{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}└{'─' * 78}┘{Colors.RESET}")


def _trend_color(trend: TrendDirection) -> str:
    """Get color for trend direction."""
    if trend == TrendDirection.UPTREND:
        return Colors.GREEN
    elif trend == TrendDirection.DOWNTREND:
        return Colors.RED
    elif trend == TrendDirection.RANGE:
        return Colors.YELLOW
    else:
        return Colors.DIM


def _event_color(event_type: StructureEvent) -> str:
    """Get color for structure event."""
    if event_type == StructureEvent.BOS:
        return Colors.CYAN
    elif event_type == StructureEvent.CHOCH:
        return Colors.YELLOW
    elif event_type == StructureEvent.ACCEPTANCE:
        return Colors.GREEN
    elif event_type == StructureEvent.REJECTION:
        return Colors.RED
    else:
        return Colors.RESET


def _alignment_icon(alignment: TimeframeAlignment) -> str:
    """Get icon for MTF alignment."""
    if alignment == TimeframeAlignment.FULL_ALIGN:
        return f"{Colors.GREEN}✓✓{Colors.RESET}"
    elif alignment == TimeframeAlignment.PARTIAL_ALIGN:
        return f"{Colors.YELLOW}✓~{Colors.RESET}"
    elif alignment == TimeframeAlignment.COUNTER_TREND:
        return f"{Colors.RED}✗✗{Colors.RESET}"
    else:
        return f"{Colors.DIM}~~{Colors.RESET}"


def print_structure_summary(state: MarketStructureState) -> None:
    """
    Print compact market structure summary.

    Shows:
    - Trend direction (LTF + HTF)
    - Last structure event (BOS/CHoCH)
    - Range status
    - Confidence
    """
    print(f"\n{Colors.BOLD}Structure Summary{Colors.RESET}")
    print(f"  {'─' * 60}")

    # Trend
    ltf_color = _trend_color(state.ltf_trend)
    htf_color = _trend_color(state.htf_trend)
    align_icon = _alignment_icon(state.tf_alignment)

    print(f"  {Colors.BOLD}Trend:{Colors.RESET} "
          f"LTF: {ltf_color}{state.ltf_trend.value.upper()}{Colors.RESET}  │  "
          f"HTF: {htf_color}{state.htf_trend.value.upper()}{Colors.RESET}  │  "
          f"Alignment: {align_icon}")

    # Last event
    if state.last_event:
        event = state.last_event
        event_color = _event_color(event.event_type)
        direction_icon = "↑" if event.direction == "up" else "↓"

        acceptance_str = ""
        if event.accepted is True:
            acceptance_str = f"{Colors.GREEN}ACCEPTED{Colors.RESET}"
        elif event.accepted is False:
            acceptance_str = f"{Colors.RED}REJECTED{Colors.RESET}"
        else:
            acceptance_str = f"{Colors.YELLOW}PENDING{Colors.RESET}"

        print(f"  {Colors.BOLD}Last Event:{Colors.RESET} "
              f"{event_color}{event.event_type.value.upper()}{Colors.RESET} {direction_icon} "
              f"${event.broken_level:,.2f} → {acceptance_str}")
    else:
        print(f"  {Colors.BOLD}Last Event:{Colors.RESET} {Colors.DIM}None{Colors.RESET}")

    # Range status
    if state.in_range:
        range_color = Colors.YELLOW
        range_icon = "⊡"
        range_desc = f"{state.range_type.value.upper()} (${state.range_low:,.2f} - ${state.range_high:,.2f})"
        if state.range_type == RangeType.COMPRESSION:
            range_color = Colors.MAGENTA
            tightness_pct = state.range_tightness * 100
            range_desc += f" {Colors.BOLD}[{tightness_pct:.0f}% tight]{Colors.RESET}"

        print(f"  {Colors.BOLD}Range:{Colors.RESET} {range_color}{range_icon} {range_desc}{Colors.RESET}")

    # Momentum
    momentum_icons = {
        StructuralMomentum.FAST: f"{Colors.GREEN}⚡ FAST{Colors.RESET}",
        StructuralMomentum.SLOW: f"{Colors.YELLOW}⏱ SLOW{Colors.RESET}",
        StructuralMomentum.STALLED: f"{Colors.RED}⏸ STALLED{Colors.RESET}",
    }
    print(f"  {Colors.BOLD}Momentum:{Colors.RESET} {momentum_icons.get(state.structural_momentum, 'UNKNOWN')}")

    # Confidence
    conf = state.structure_confidence
    conf_color = Colors.GREEN if conf >= 70 else Colors.YELLOW if conf >= 50 else Colors.RED
    conf_bar_width = 20
    filled = int(conf / 100 * conf_bar_width)
    conf_bar = f"{Colors.GREEN}{'█' * filled}{Colors.DIM}{'░' * (conf_bar_width - filled)}{Colors.RESET}"

    print(f"  {Colors.BOLD}Confidence:{Colors.RESET} {conf_color}{conf:.0f}%{Colors.RESET} {conf_bar}")


def print_structure_deep_dive(state: MarketStructureState) -> None:
    """
    Print detailed market structure analysis.

    Shows:
    - Swing points (HH/HL/LH/LL)
    - BOS and CHoCH events
    - FVG zones
    - Warnings
    """
    print_structure_header()

    # Summary section
    print_structure_summary(state)

    # Swings section
    if state.recent_swings:
        print(f"\n  {Colors.BOLD}Recent Swings:{Colors.RESET}")
        highs = [s for s in state.recent_swings if s.swing_type.value == "high"]
        lows = [s for s in state.recent_swings if s.swing_type.value == "low"]

        # Show pattern
        if len(highs) >= 2:
            h1, h2 = highs[-2:]
            if h2.price > h1.price:
                print(f"    Highs: {Colors.GREEN}HH{Colors.RESET} "
                      f"(${h1.price:,.2f} → ${h2.price:,.2f})")
            else:
                print(f"    Highs: {Colors.RED}LH{Colors.RESET} "
                      f"(${h1.price:,.2f} → ${h2.price:,.2f})")

        if len(lows) >= 2:
            l1, l2 = lows[-2:]
            if l2.price > l1.price:
                print(f"    Lows:  {Colors.GREEN}HL{Colors.RESET} "
                      f"(${l1.price:,.2f} → ${l2.price:,.2f})")
            else:
                print(f"    Lows:  {Colors.RED}LL{Colors.RESET} "
                      f"(${l1.price:,.2f} → ${l2.price:,.2f})")

    # BOS/CHoCH section
    events_to_show = []
    if state.last_bos:
        events_to_show.append(("BOS", state.last_bos))
    if state.last_choch:
        events_to_show.append(("CHoCH", state.last_choch))

    if events_to_show:
        print(f"\n  {Colors.BOLD}Structure Events:{Colors.RESET}")
        for label, event in events_to_show:
            event_color = _event_color(event.event_type)
            direction_icon = "↑" if event.direction == "up" else "↓"

            acceptance_str = ""
            if event.accepted is True:
                acceptance_str = f" │ {Colors.GREEN}✓ ACCEPTED{Colors.RESET}"
                if event.acceptance_volume_ratio:
                    acceptance_str += f" (RV: {event.acceptance_volume_ratio:.2f}x)"
            elif event.accepted is False:
                acceptance_str = f" │ {Colors.RED}✗ REJECTED{Colors.RESET}"

            followthrough_str = ""
            if event.time_to_followthrough:
                ft = event.time_to_followthrough
                if ft < 60:
                    followthrough_str = f" │ Follow: {Colors.GREEN}{ft:.0f}s{Colors.RESET}"
                elif ft < 300:
                    followthrough_str = f" │ Follow: {Colors.YELLOW}{ft:.0f}s{Colors.RESET}"
                else:
                    followthrough_str = f" │ Follow: {Colors.RED}{ft:.0f}s{Colors.RESET}"

            print(f"    {event_color}{label:6}{Colors.RESET} {direction_icon} "
                  f"${event.broken_level:>10,.2f} → ${event.break_price:>10,.2f}"
                  f"{acceptance_str}{followthrough_str}")

    # FVG section
    if state.active_fvgs:
        print(f"\n  {Colors.BOLD}Active Fair Value Gaps (FVG):{Colors.RESET}")
        for i, fvg in enumerate(state.active_fvgs[-5:], 1):  # Show last 5
            fvg_color = Colors.GREEN if fvg.is_bullish else Colors.RED
            fvg_type = "Bullish" if fvg.is_bullish else "Bearish"
            print(f"    {i}. {fvg_color}{fvg_type} FVG{Colors.RESET}: "
                  f"${fvg.price_bottom:,.2f} - ${fvg.price_top:,.2f} "
                  f"({fvg.gap_size_pct:.2f}%)")

    # Warnings
    if state.warnings:
        print(f"\n  {Colors.YELLOW}⚠ Warnings:{Colors.RESET}")
        for warning in state.warnings:
            print(f"    • {warning}")

    print()


def print_structure_signal(signal: MarketStructureSignal) -> None:
    """
    Print market structure signal for continuous runner.

    Compact format for live updates.
    """
    # Trend line
    trend_color = _trend_color(signal.trend_direction)
    allowed_str = ""
    if signal.allowed_direction:
        if signal.allowed_direction == "long":
            allowed_str = f"{Colors.GREEN}LONG OK{Colors.RESET}"
        elif signal.allowed_direction == "short":
            allowed_str = f"{Colors.RED}SHORT OK{Colors.RESET}"
        elif signal.allowed_direction == "both":
            allowed_str = f"{Colors.YELLOW}RANGE (BOTH){Colors.RESET}"
    else:
        allowed_str = f"{Colors.DIM}NO TRADE{Colors.RESET}"

    print(f"  Structure: {trend_color}{signal.trend_direction.value.upper()}{Colors.RESET} "
          f"│ Score: {signal.score:+.2f} "
          f"│ Allowed: {allowed_str} "
          f"│ Conf: {signal.structure_confidence:.0f}%")

    # Events line
    event_parts = []
    if signal.has_bos:
        if signal.bos_accepted is True:
            event_parts.append(f"{Colors.CYAN}BOS✓{Colors.RESET}")
        elif signal.bos_accepted is False:
            event_parts.append(f"{Colors.CYAN}BOS✗{Colors.RESET}")
        else:
            event_parts.append(f"{Colors.CYAN}BOS?{Colors.RESET}")

    if signal.has_choch:
        event_parts.append(f"{Colors.YELLOW}CHoCH{Colors.RESET}")

    if signal.in_range:
        event_parts.append(f"{Colors.YELLOW}RANGE{Colors.RESET}")

    if signal.near_fvg:
        event_parts.append(f"{Colors.MAGENTA}Near FVG{Colors.RESET}")

    if event_parts:
        print(f"  Events: {' │ '.join(event_parts)}")

    # MTF line
    align_icon = _alignment_icon(signal.tf_alignment)
    htf_color = _trend_color(signal.htf_trend)
    ltf_color = _trend_color(signal.ltf_trend)
    momentum_icon = {
        StructuralMomentum.FAST: f"{Colors.GREEN}⚡{Colors.RESET}",
        StructuralMomentum.SLOW: f"{Colors.YELLOW}⏱{Colors.RESET}",
        StructuralMomentum.STALLED: f"{Colors.RED}⏸{Colors.RESET}",
    }.get(signal.structural_momentum, "")

    print(f"  MTF: HTF:{htf_color}{signal.htf_trend.value}{Colors.RESET} "
          f"│ LTF:{ltf_color}{signal.ltf_trend.value}{Colors.RESET} "
          f"│ Align:{align_icon} "
          f"│ Mom:{momentum_icon}")

    # Warnings
    if signal.warnings:
        print(f"  {Colors.YELLOW}⚠{Colors.RESET} {', '.join(signal.warnings[:2])}")


def print_structure_allowed_trades(state: MarketStructureState) -> None:
    """
    Print what trades are allowed based on structure.

    Clear guidance for the trader.
    """
    from ..market_structure import get_allowed_trade_direction

    allowed = get_allowed_trade_direction(state)

    print(f"\n{Colors.BOLD}Structure Trade Guidance{Colors.RESET}")
    print(f"  {'─' * 60}")

    if allowed is None:
        print(f"  {Colors.RED}✗ NO TRADE{Colors.RESET}")
        print(f"    Structure unclear or conflicting")
        print(f"    Wait for clearer setup")

    elif allowed == "long":
        print(f"  {Colors.GREEN}✓ LONG ALLOWED{Colors.RESET}")
        print(f"    Uptrend structure confirmed")
        if state.active_fvgs:
            bull_fvgs = [f for f in state.active_fvgs if f.is_bullish]
            if bull_fvgs:
                nearest = bull_fvgs[-1]
                print(f"    Pullback target: ${nearest.price_bottom:,.2f} - ${nearest.price_top:,.2f}")

    elif allowed == "short":
        print(f"  {Colors.RED}✓ SHORT ALLOWED{Colors.RESET}")
        print(f"    Downtrend structure confirmed")
        if state.active_fvgs:
            bear_fvgs = [f for f in state.active_fvgs if not f.is_bullish]
            if bear_fvgs:
                nearest = bear_fvgs[-1]
                print(f"    Pullback target: ${nearest.price_bottom:,.2f} - ${nearest.price_top:,.2f}")

    elif allowed == "both":
        print(f"  {Colors.YELLOW}◆ RANGE - BOTH DIRECTIONS{Colors.RESET}")
        print(f"    Mean reversion trades")
        if state.range_high and state.range_low:
            print(f"    Range: ${state.range_low:,.2f} - ${state.range_high:,.2f}")
            if state.range_type == RangeType.COMPRESSION:
                print(f"    {Colors.MAGENTA}Compression - breakout imminent{Colors.RESET}")

    # Confidence warning
    if state.structure_confidence < 50:
        print(f"\n  {Colors.YELLOW}⚠ Low confidence ({state.structure_confidence:.0f}%) - "
              f"reduce position size{Colors.RESET}")

    # CHoCH warning
    if state.last_choch:
        if not state.last_bos or state.last_choch.timestamp > state.last_bos.timestamp:
            print(f"\n  {Colors.YELLOW}⚠ Recent CHoCH - trend may be reversing{Colors.RESET}")

    # MTF alignment
    if state.tf_alignment == TimeframeAlignment.COUNTER_TREND:
        print(f"\n  {Colors.YELLOW}⚠ Counter-trend on LTF - scalp only or wait{Colors.RESET}")

    print()


def get_structure_status_line(signal: Optional[MarketStructureSignal]) -> str:
    """
    Get compact one-line structure status for continuous runner status bar.

    Returns:
        Colored status string
    """
    if signal is None:
        return f"{Colors.DIM}Structure: N/A{Colors.RESET}"

    trend_color = _trend_color(signal.trend_direction)
    trend_short = signal.trend_direction.value[:4].upper()  # UP/DOWN/RANG

    # Allowed indicator
    allowed_icon = ""
    if signal.allowed_direction == "long":
        allowed_icon = f"{Colors.GREEN}↑{Colors.RESET}"
    elif signal.allowed_direction == "short":
        allowed_icon = f"{Colors.RED}↓{Colors.RESET}"
    elif signal.allowed_direction == "both":
        allowed_icon = f"{Colors.YELLOW}↕{Colors.RESET}"
    else:
        allowed_icon = f"{Colors.DIM}•{Colors.RESET}"

    # Score
    score = signal.score
    score_color = Colors.GREEN if score > 0.3 else Colors.RED if score < -0.3 else Colors.YELLOW
    score_str = f"{score:+.1f}"

    return (f"Struct: {trend_color}{trend_short}{Colors.RESET} "
            f"{allowed_icon} {score_color}{score_str}{Colors.RESET}")
