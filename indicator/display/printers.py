"""Print functions for indicator analysis output."""

from datetime import datetime
from typing import List, Optional

from .colors import Colors
from .formatters import signal_color, strength_bar, score_bar
from signals import Signal, signal_value

# Type imports for annotations
from volume_analysis import (
    VolumeAnalysisSummary,
    VolumeContext,
    VolumeLocation,
    AbsorptionType,
)
from volume_engine import (
    VolumeEngineResult,
    AggressionBias,
    VolumeAcceleration,
    ExhaustionRisk,
    MTFAgreement,
)
from unified_score import UnifiedScore
from breakout_validation import BreakoutValidation
from oi_analysis import (
    OIAnalysisSummary,
    OIRegime,
    OISignal,
)
from funding_analysis import (
    FundingAnalysisSummary,
    FundingZone,
    CrowdPosition,
    FundingWarning,
    FundingOICombo,
)
from orderbook_analysis import (
    OrderbookAnalysisSummary as OBAnalysisSummary,
    AbsorptionSide,
    ImbalanceDirection,
    SpoofType,
)
from indicators import IndicatorResult


def print_volume_deep_dive(summary: VolumeAnalysisSummary):
    """Print detailed volume analysis - 'Was the move REAL?'"""
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}┌{'─' * 78}┐{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}│  WAS THE MOVE REAL? - Deep Volume Analysis{' ' * 34}│{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}└{'─' * 78}┘{Colors.RESET}")
    print()

    # Main verdict
    if summary.move_is_real:
        verdict_color = Colors.GREEN
        verdict = "YES - REAL MOVE"
    else:
        verdict_color = Colors.RED
        verdict = "NO - SUSPICIOUS"

    signal_clr = signal_color(summary.signal)
    print(f"  {Colors.BOLD}Verdict:{Colors.RESET} {verdict_color}{verdict}{Colors.RESET}")
    print(f"  {Colors.BOLD}Signal:{Colors.RESET}  {signal_clr}{signal_value(summary.signal).upper()}{Colors.RESET}")
    print(f"  {Colors.BOLD}Confidence:{Colors.RESET} {strength_bar(summary.confidence)} {summary.confidence:.0f}%")
    print()
    print(f"  {Colors.DIM}{summary.summary}{Colors.RESET}")
    print()
    print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")

    # Relative Volume
    rv = summary.relative_volume
    ctx_colors = {
        VolumeContext.EXTREME: Colors.MAGENTA,
        VolumeContext.HIGH: Colors.GREEN,
        VolumeContext.NORMAL: Colors.YELLOW,
        VolumeContext.LOW: Colors.RED,
        VolumeContext.DEAD: Colors.RED + Colors.DIM,
    }
    ctx_color = ctx_colors.get(rv.context, Colors.YELLOW)

    print()
    print(f"  {Colors.BOLD}1. RELATIVE VOLUME{Colors.RESET}")
    print(f"     Current: {rv.current_volume:,.0f}  |  20-bar Avg: {rv.avg_volume_20:,.0f}")
    print(f"     Ratio: {ctx_color}{rv.relative_ratio:.2f}x{Colors.RESET} ({rv.context.value.upper()})")
    print(f"     {Colors.DIM}{rv.description}{Colors.RESET}")

    meaningful_icon = "✓" if rv.is_meaningful else "✗"
    meaningful_color = Colors.GREEN if rv.is_meaningful else Colors.RED
    print(f"     Meaningful (>1.5x): {meaningful_color}{meaningful_icon}{Colors.RESET}")

    # Volume Location
    vl = summary.volume_location
    loc_colors = {
        VolumeLocation.RANGE_HIGH: Colors.RED,
        VolumeLocation.RANGE_LOW: Colors.GREEN,
        VolumeLocation.MID_RANGE: Colors.YELLOW,
        VolumeLocation.BREAKOUT_HIGH: Colors.CYAN,
        VolumeLocation.BREAKOUT_LOW: Colors.MAGENTA,
    }
    loc_color = loc_colors.get(vl.location, Colors.YELLOW)

    print()
    print(f"  {Colors.BOLD}2. VOLUME LOCATION{Colors.RESET}")
    print(f"     Range: ${vl.range_low:,.2f} - ${vl.range_high:,.2f}")
    print(f"     Current: ${vl.current_price:,.2f} ({vl.percentile_in_range:.0f}% in range)")
    print(f"     Location: {loc_color}{vl.location.value.upper()}{Colors.RESET}")
    print(f"     {Colors.DIM}{vl.interpretation}{Colors.RESET}")

    # Location meaning table
    print()
    print(f"     {Colors.DIM}┌─────────────────┬──────────────────┐{Colors.RESET}")
    print(f"     {Colors.DIM}│ Location        │ Meaning          │{Colors.RESET}")
    print(f"     {Colors.DIM}├─────────────────┼──────────────────┤{Colors.RESET}")
    print(f"     {Colors.DIM}│{Colors.RESET} {Colors.RED}Range High{Colors.RESET}      {Colors.DIM}│{Colors.RESET} Distribution     {Colors.DIM}│{Colors.RESET}")
    print(f"     {Colors.DIM}│{Colors.RESET} {Colors.GREEN}Range Low{Colors.RESET}       {Colors.DIM}│{Colors.RESET} Accumulation     {Colors.DIM}│{Colors.RESET}")
    print(f"     {Colors.DIM}│{Colors.RESET} {Colors.YELLOW}Mid-Range{Colors.RESET}       {Colors.DIM}│{Colors.RESET} Noise            {Colors.DIM}│{Colors.RESET}")
    print(f"     {Colors.DIM}│{Colors.RESET} {Colors.CYAN}After Sweep{Colors.RESET}     {Colors.DIM}│{Colors.RESET} Confirmation     {Colors.DIM}│{Colors.RESET}")
    print(f"     {Colors.DIM}└─────────────────┴──────────────────┘{Colors.RESET}")

    # Absorption
    ab = summary.absorption
    print()
    print(f"  {Colors.BOLD}3. ABSORPTION DETECTION{Colors.RESET}")
    print(f"     Volume Ratio: {ab.volume_ratio:.2f}x  |  Price Move: {ab.price_move_percent:.2f}%")
    print(f"     Efficiency: {ab.efficiency:.3f} (lower = more absorption)")

    if ab.detected:
        ab_color = Colors.GREEN if ab.absorption_type == AbsorptionType.BID_ABSORPTION else Colors.RED
        ab_icon = "⚠"
        print(f"     {ab_color}{ab_icon} ABSORPTION DETECTED: {ab.absorption_type.value.upper()}{Colors.RESET}")
    else:
        print(f"     {Colors.DIM}No absorption detected{Colors.RESET}")
    print(f"     {Colors.DIM}{ab.description}{Colors.RESET}")

    # Liquidity Sweep
    sw = summary.liquidity_sweep
    print()
    print(f"  {Colors.BOLD}4. LIQUIDITY SWEEP{Colors.RESET}")

    if sw.sweep_detected:
        sw_color = Colors.GREEN if sw.sweep_direction == "low" else Colors.RED
        conf_icon = "✓" if sw.volume_confirmation else "✗"
        conf_color = Colors.GREEN if sw.volume_confirmation else Colors.RED
        print(f"     {sw_color}SWEEP DETECTED at {sw.sweep_direction.upper()}{Colors.RESET} (${sw.sweep_level:,.2f})")
        print(f"     Volume Confirmation: {conf_color}{conf_icon}{Colors.RESET}")
    else:
        print(f"     {Colors.DIM}No sweep detected{Colors.RESET}")
    print(f"     {Colors.DIM}{sw.description}{Colors.RESET}")

    print()
    print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")
    print(f"  {Colors.BOLD}Pro Insight:{Colors.RESET} {Colors.CYAN}Volume confirms ACCEPTANCE, not direction.{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")


def print_oi_deep_dive(summary: OIAnalysisSummary):
    """Print detailed OI analysis - 'Is money entering or leaving?'"""
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}┌{'─' * 78}┐{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}│  IS MONEY ENTERING OR LEAVING? - Open Interest Analysis{' ' * 21}│{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}└{'─' * 78}┘{Colors.RESET}")
    print()

    # Main verdict
    if summary.is_money_entering:
        verdict_color = Colors.GREEN
        verdict = "MONEY ENTERING"
    else:
        verdict_color = Colors.RED
        verdict = "MONEY LEAVING"

    signal_colors = {
        Signal.BULLISH.value: Colors.GREEN,
        Signal.BEARISH.value: Colors.RED,
        Signal.CAUTION.value: Colors.MAGENTA,
        Signal.NEUTRAL.value: Colors.YELLOW,
    }
    overall_signal = signal_value(summary.overall_signal)
    signal_clr = signal_colors.get(overall_signal, Colors.YELLOW)

    print(f"  {Colors.BOLD}Verdict:{Colors.RESET} {verdict_color}{verdict}{Colors.RESET}")
    print(f"  {Colors.BOLD}Signal:{Colors.RESET}  {signal_clr}{overall_signal.upper()}{Colors.RESET}")
    print(f"  {Colors.BOLD}Confidence:{Colors.RESET} {strength_bar(summary.confidence)} {summary.confidence:.0f}%")
    print()
    print(f"  {Colors.DIM}{summary.summary}{Colors.RESET}")
    print()
    print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")

    # The 4 OI Regimes Table
    print()
    print(f"  {Colors.BOLD}THE 4 OI REGIMES (Memorize This!){Colors.RESET}")
    print()
    print(f"  {Colors.DIM}┌─────────┬────────┬─────────────────────┬────────────────────┐{Colors.RESET}")
    print(f"  {Colors.DIM}│{Colors.RESET} Price   {Colors.DIM}│{Colors.RESET} OI     {Colors.DIM}│{Colors.RESET} Interpretation      {Colors.DIM}│{Colors.RESET} Trade Meaning      {Colors.DIM}│{Colors.RESET}")
    print(f"  {Colors.DIM}├─────────┼────────┼─────────────────────┼────────────────────┤{Colors.RESET}")

    regime = summary.regime
    regimes_display = [
        ("↑", "↑", "New longs entering", "Trend continuation", OIRegime.NEW_LONGS),
        ("↑", "↓", "Shorts closing", "Fake strength", OIRegime.SHORT_COVERING),
        ("↓", "↑", "New shorts entering", "Squeeze fuel", OIRegime.NEW_SHORTS),
        ("↓", "↓", "Longs closing", "Trend exhaustion", OIRegime.LONG_LIQUIDATION),
    ]

    for p, o, interp, meaning, r_type in regimes_display:
        if regime.regime == r_type:
            row_color = Colors.CYAN + Colors.BOLD
            marker = "► "
        else:
            row_color = Colors.DIM
            marker = "  "

        print(f"  {Colors.DIM}│{Colors.RESET}{marker}{row_color}{p:^7}{Colors.RESET}{Colors.DIM}│{Colors.RESET}"
              f"{row_color}{o:^8}{Colors.RESET}{Colors.DIM}│{Colors.RESET}"
              f"{row_color}{interp:<21}{Colors.RESET}{Colors.DIM}│{Colors.RESET}"
              f"{row_color}{meaning:<20}{Colors.RESET}{Colors.DIM}│{Colors.RESET}")

    print(f"  {Colors.DIM}└─────────┴────────┴─────────────────────┴────────────────────┘{Colors.RESET}")

    # Current Regime Details
    print()
    print(f"  {Colors.BOLD}1. CURRENT REGIME{Colors.RESET}")

    regime_colors = {
        OIRegime.NEW_LONGS: Colors.GREEN,
        OIRegime.SHORT_COVERING: Colors.YELLOW,
        OIRegime.NEW_SHORTS: Colors.RED,
        OIRegime.LONG_LIQUIDATION: Colors.MAGENTA,
        OIRegime.NEUTRAL: Colors.DIM
    }
    r_color = regime_colors.get(regime.regime, Colors.YELLOW)

    print(f"     Regime: {r_color}{regime.regime.value.upper()}{Colors.RESET}")
    print(f"     Price: {regime.price_direction.upper()}  |  OI: {regime.oi_direction.upper()}")
    print(f"     {Colors.DIM}{regime.interpretation}{Colors.RESET}")
    print(f"     {Colors.BOLD}Trade:{Colors.RESET} {regime.trade_meaning}")

    # Rate of Change
    roc = summary.rate_of_change
    print()
    print(f"  {Colors.BOLD}2. RATE OF CHANGE{Colors.RESET}")
    print(f"     Current OI: {roc.current_oi:,.0f}  |  Previous: {roc.previous_oi:,.0f}")

    change_color = Colors.GREEN if roc.oi_change_percent > 0 else Colors.RED if roc.oi_change_percent < 0 else Colors.YELLOW
    print(f"     Change: {change_color}{roc.oi_change_percent:+.2f}%{Colors.RESET} ({roc.oi_change_absolute:+,.0f})")

    rate_color = Colors.MAGENTA if roc.rate_vs_average > 1.5 else Colors.GREEN if roc.rate_vs_average > 1 else Colors.DIM
    print(f"     Rate vs Avg: {rate_color}{roc.rate_vs_average:.2f}x{Colors.RESET}")

    if roc.acceleration > 0.3:
        acc_display = f"{Colors.GREEN}ACCELERATING ↑{Colors.RESET}"
    elif roc.acceleration < -0.3:
        acc_display = f"{Colors.RED}DECELERATING ↓{Colors.RESET}"
    else:
        acc_display = f"{Colors.DIM}Steady{Colors.RESET}"
    print(f"     Momentum: {acc_display}")
    print(f"     {Colors.DIM}{roc.description}{Colors.RESET}")

    # High-Edge Signal
    sig = summary.high_edge_signal
    print()
    print(f"  {Colors.BOLD}3. HIGH-EDGE SIGNALS{Colors.RESET}")

    if sig.detected:
        sig_colors = {
            OISignal.COMPRESSION: Colors.YELLOW,
            OISignal.BREAKOUT_TRAP: Colors.MAGENTA,
            OISignal.EXPANSION: Colors.GREEN,
            OISignal.EXHAUSTION: Colors.RED,
        }
        s_color = sig_colors.get(sig.signal, Colors.YELLOW)
        print(f"     {s_color}⚠ {sig.signal.value.upper()} DETECTED{Colors.RESET}")
        print(f"     Confidence: {sig.confidence:.0f}%")
        print(f"     {Colors.DIM}{sig.description}{Colors.RESET}")
        print(f"     {Colors.BOLD}Action:{Colors.RESET} {sig.action}")
    else:
        print(f"     {Colors.DIM}No high-edge signal detected{Colors.RESET}")
        print(f"     {Colors.DIM}{sig.description}{Colors.RESET}")

    # Key OI signals reference
    print()
    print(f"  {Colors.DIM}┌─────────────────────────────────────────────────────────────────────┐{Colors.RESET}")
    print(f"  {Colors.DIM}│{Colors.RESET} {Colors.BOLD}High-Edge Signals:{Colors.RESET}                                              {Colors.DIM}│{Colors.RESET}")
    print(f"  {Colors.DIM}│{Colors.RESET} • {Colors.YELLOW}COMPRESSION{Colors.RESET}: OI rising, price stalls → Breakout coming     {Colors.DIM}│{Colors.RESET}")
    print(f"  {Colors.DIM}│{Colors.RESET} • {Colors.MAGENTA}BREAKOUT TRAP{Colors.RESET}: OI drops after breakout → Fade the move   {Colors.DIM}│{Colors.RESET}")
    print(f"  {Colors.DIM}│{Colors.RESET} • {Colors.GREEN}EXPANSION{Colors.RESET}: OI expanding at key level → Genuine interest    {Colors.DIM}│{Colors.RESET}")
    print(f"  {Colors.DIM}│{Colors.RESET} • {Colors.RED}EXHAUSTION{Colors.RESET}: OI collapsing → Move is ending                 {Colors.DIM}│{Colors.RESET}")
    print(f"  {Colors.DIM}└─────────────────────────────────────────────────────────────────────┘{Colors.RESET}")

    print()
    print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")
    print(f"  {Colors.BOLD}Pro Insight:{Colors.RESET} {Colors.CYAN}OI measures ACTIVE CONTRACTS - not direction, not volume.{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")


def print_funding_deep_dive(summary: FundingAnalysisSummary, oi_change_pct: Optional[float] = None):
    """Print detailed Funding analysis - 'Where is the crowd leaning?'"""
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}┌{'─' * 78}┐{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}│  WHERE IS THE CROWD LEANING? - Funding Rate Analysis{' ' * 24}│{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}└{'─' * 78}┘{Colors.RESET}")
    print()

    # Main verdict - Can you chase?
    if summary.should_chase:
        chase_color = Colors.GREEN
        chase_verdict = "CAN TRADE WITH TREND"
    else:
        chase_color = Colors.RED
        chase_verdict = "DON'T CHASE"

    # Warning severity colors
    severity_colors = {
        "extreme": Colors.RED + Colors.BOLD,
        "high": Colors.MAGENTA,
        "medium": Colors.YELLOW,
        "low": Colors.DIM
    }
    warn_color = severity_colors.get(summary.warning.severity, Colors.DIM)

    print(f"  {Colors.BOLD}Verdict:{Colors.RESET} {chase_color}{chase_verdict}{Colors.RESET}")
    print(f"  {Colors.BOLD}Warning:{Colors.RESET} {warn_color}{summary.warning.warning.value.upper()}{Colors.RESET}")
    print(f"  {Colors.BOLD}Confidence:{Colors.RESET} {strength_bar(summary.confidence)} {summary.confidence:.0f}%")
    print()
    print(f"  {Colors.DIM}{summary.summary}{Colors.RESET}")
    print()
    print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")

    # Key concept box
    print()
    print(f"  {Colors.DIM}┌─────────────────────────────────────────────────────────────────────┐{Colors.RESET}")
    print(f"  {Colors.DIM}│{Colors.RESET} {Colors.BOLD}KEY CONCEPT:{Colors.RESET} Funding is a WARNING SYSTEM, not an entry trigger {Colors.DIM}│{Colors.RESET}")
    print(f"  {Colors.DIM}│{Colors.RESET} • High funding = DON'T CHASE                                      {Colors.DIM}│{Colors.RESET}")
    print(f"  {Colors.DIM}│{Colors.RESET} • Extreme funding = HUNT REVERSALS                                {Colors.DIM}│{Colors.RESET}")
    print(f"  {Colors.DIM}└─────────────────────────────────────────────────────────────────────┘{Colors.RESET}")

    # Percentile Analysis
    pct = summary.percentile
    print()
    print(f"  {Colors.BOLD}1. FUNDING PERCENTILE{Colors.RESET}")
    print(f"     Current Rate: {pct.current_rate_percent:.4f}% (per 8h)")
    print(f"     Annualized: {pct.annualized_rate:.1f}%")

    zone_colors = {
        FundingZone.EXTREME_POSITIVE: Colors.RED + Colors.BOLD,
        FundingZone.HIGH_POSITIVE: Colors.RED,
        FundingZone.NORMAL_POSITIVE: Colors.YELLOW,
        FundingZone.NEUTRAL: Colors.DIM,
        FundingZone.NORMAL_NEGATIVE: Colors.YELLOW,
        FundingZone.HIGH_NEGATIVE: Colors.GREEN,
        FundingZone.EXTREME_NEGATIVE: Colors.GREEN + Colors.BOLD,
    }
    z_color = zone_colors.get(pct.zone, Colors.YELLOW)

    # Percentile bar
    pct_bar_width = 40
    pct_pos = int(pct.percentile / 100 * pct_bar_width)
    pct_bar = "░" * pct_pos + "█" + "░" * (pct_bar_width - pct_pos - 1)
    print(f"     Percentile: {z_color}{pct.percentile:.0f}th{Colors.RESET}")
    print(f"     [{Colors.GREEN}Short{Colors.RESET}] {pct_bar} [{Colors.RED}Long{Colors.RESET}]")
    print(f"     Zone: {z_color}{pct.zone.value.upper()}{Colors.RESET}")
    print(f"     {Colors.DIM}{pct.description}{Colors.RESET}")

    # Crowd Analysis
    crowd = summary.crowd
    print()
    print(f"  {Colors.BOLD}2. CROWD POSITIONING{Colors.RESET}")

    crowd_colors = {
        CrowdPosition.HEAVILY_LONG: Colors.RED + Colors.BOLD,
        CrowdPosition.MODERATELY_LONG: Colors.RED,
        CrowdPosition.BALANCED: Colors.YELLOW,
        CrowdPosition.MODERATELY_SHORT: Colors.GREEN,
        CrowdPosition.HEAVILY_SHORT: Colors.GREEN + Colors.BOLD,
    }
    c_color = crowd_colors.get(crowd.position, Colors.YELLOW)

    print(f"     Position: {c_color}{crowd.position.value.upper()}{Colors.RESET}")
    print(f"     Lean Strength: {strength_bar(crowd.lean_strength, 15)} {crowd.lean_strength:.0f}%")
    print(f"     {Colors.DIM}{crowd.crowd_description}{Colors.RESET}")

    if crowd.contrarian_bias != "none":
        contra_color = Colors.GREEN if "bullish" in crowd.contrarian_bias else Colors.RED
        print(f"     Contrarian Bias: {contra_color}{crowd.contrarian_bias.upper()}{Colors.RESET}")

    # Warning Signal
    warn = summary.warning
    print()
    print(f"  {Colors.BOLD}3. WARNING SIGNAL{Colors.RESET}")

    if warn.warning != FundingWarning.NONE:
        print(f"     {warn_color}⚠ {warn.warning.value.upper()}{Colors.RESET}")
        print(f"     Severity: {warn_color}{warn.severity.upper()}{Colors.RESET}")
        print(f"     {Colors.DIM}{warn.description}{Colors.RESET}")
        print(f"     {Colors.BOLD}Action:{Colors.RESET} {warn.action}")
    else:
        print(f"     {Colors.GREEN}✓ No warning - Funding at normal levels{Colors.RESET}")
        print(f"     {Colors.DIM}{warn.description}{Colors.RESET}")

    # Funding + OI Combo (if available)
    if summary.funding_oi_combo:
        combo = summary.funding_oi_combo
        print()
        print(f"  {Colors.BOLD}4. FUNDING + OI COMBO{Colors.RESET}")

        combo_colors = {
            FundingOICombo.CROWDED_LONGS_RISING: Colors.RED,
            FundingOICombo.CROWDED_SHORTS_RISING: Colors.GREEN,
            FundingOICombo.EXTREME_EXHAUSTION: Colors.MAGENTA,
            FundingOICombo.CAPITULATION: Colors.RED + Colors.BOLD,
            FundingOICombo.HEALTHY_TREND: Colors.GREEN,
            FundingOICombo.NO_EDGE: Colors.DIM,
        }
        combo_color = combo_colors.get(combo.combo, Colors.YELLOW)

        print(f"     Combo: {combo_color}{combo.combo.value.upper()}{Colors.RESET}")
        print(f"     Funding: {combo.funding_direction.upper()}  |  OI: {combo.oi_direction.upper()}")
        print(f"     Probability: {combo.probability:.0f}%")
        print(f"     {Colors.DIM}{combo.expected_outcome}{Colors.RESET}")
        print(f"     {Colors.BOLD}Trade:{Colors.RESET} {combo.trade_action}")

        # Combo reference table
        print()
        print(f"  {Colors.DIM}┌────────────────┬──────────────┬─────────────────────────────┐{Colors.RESET}")
        print(f"  {Colors.DIM}│{Colors.RESET} Funding        {Colors.DIM}│{Colors.RESET} OI           {Colors.DIM}│{Colors.RESET} Meaning                     {Colors.DIM}│{Colors.RESET}")
        print(f"  {Colors.DIM}├────────────────┼──────────────┼─────────────────────────────┤{Colors.RESET}")
        print(f"  {Colors.DIM}│{Colors.RESET} {Colors.RED}Very positive{Colors.RESET}  {Colors.DIM}│{Colors.RESET} Rising       {Colors.DIM}│{Colors.RESET} Crowded longs → downside    {Colors.DIM}│{Colors.RESET}")
        print(f"  {Colors.DIM}│{Colors.RESET} {Colors.GREEN}Very negative{Colors.RESET}  {Colors.DIM}│{Colors.RESET} Rising       {Colors.DIM}│{Colors.RESET} Crowded shorts → upside     {Colors.DIM}│{Colors.RESET}")
        print(f"  {Colors.DIM}│{Colors.RESET} {Colors.MAGENTA}Extreme{Colors.RESET}        {Colors.DIM}│{Colors.RESET} Flat         {Colors.DIM}│{Colors.RESET} Exhaustion                  {Colors.DIM}│{Colors.RESET}")
        print(f"  {Colors.DIM}│{Colors.RESET} {Colors.MAGENTA}Extreme{Colors.RESET}        {Colors.DIM}│{Colors.RESET} Dropping     {Colors.DIM}│{Colors.RESET} Capitulation / flush        {Colors.DIM}│{Colors.RESET}")
        print(f"  {Colors.DIM}└────────────────┴──────────────┴─────────────────────────────┘{Colors.RESET}")

    print()
    print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")
    print(f"  {Colors.BOLD}Pro Rules:{Colors.RESET} {Colors.CYAN}High funding = don't chase | Extreme funding = hunt reversals{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")


def print_volume_engine_deep_dive(result: VolumeEngineResult):
    """Print institutional-grade volume analysis - 'Who initiated, who absorbed, who is trapped?'"""
    print()
    print(f"{Colors.BOLD}{Colors.MAGENTA}┌{'─' * 78}┐{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}│  WHO INITIATED, WHO ABSORBED, WHO IS TRAPPED? - Institutional Volume{' ' * 7}│{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}└{'─' * 78}┘{Colors.RESET}")
    print()

    # The 3 core questions
    init_color = Colors.GREEN if result.who_initiated == "BUYERS" else Colors.RED if result.who_initiated == "SELLERS" else Colors.YELLOW
    print(f"  {Colors.BOLD}Who Initiated?{Colors.RESET}  {init_color}{result.who_initiated}{Colors.RESET}")
    print(f"  {Colors.BOLD}Who Absorbed?{Colors.RESET}   {Colors.CYAN}{result.who_absorbed}{Colors.RESET}")

    trap_color = Colors.MAGENTA if result.who_is_trapped != "NONE DETECTED" else Colors.DIM
    print(f"  {Colors.BOLD}Who is Trapped?{Colors.RESET} {trap_color}{result.who_is_trapped}{Colors.RESET}")
    print()

    # Quality assessment
    quality_colors = {
        "institutional": Colors.GREEN,
        "retail": Colors.RED,
        "mixed": Colors.YELLOW
    }
    q_color = quality_colors.get(result.volume_quality, Colors.YELLOW)
    signal_clr = signal_color(result.signal)

    print(f"  {Colors.BOLD}Volume Quality:{Colors.RESET} {q_color}{result.volume_quality.upper()}{Colors.RESET}")
    print(f"  {Colors.BOLD}Signal:{Colors.RESET}         {signal_clr}{signal_value(result.signal).upper()}{Colors.RESET}")
    print(f"  {Colors.BOLD}Confidence:{Colors.RESET}     {strength_bar(result.confidence)} {result.confidence:.0f}%")
    print()
    print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")

    # 1. Volume Delta
    delta = result.delta
    print()
    print(f"  {Colors.BOLD}1. VOLUME DELTA (Who is Aggressive?){Colors.RESET}")

    bias_colors = {
        AggressionBias.STRONG_BUY: Colors.GREEN + Colors.BOLD,
        AggressionBias.BUY: Colors.GREEN,
        AggressionBias.NEUTRAL: Colors.YELLOW,
        AggressionBias.SELL: Colors.RED,
        AggressionBias.STRONG_SELL: Colors.RED + Colors.BOLD,
    }
    b_color = bias_colors.get(delta.aggression_bias, Colors.YELLOW)

    print(f"     Bias: {b_color}{delta.aggression_bias.value.upper()}{Colors.RESET}")
    print(f"     Delta: {delta.delta_percent:+.1f}%  |  Cumulative: {delta.cumulative_delta:+,.0f}")
    print(f"     Strength: {strength_bar(delta.strength, 15)} {delta.strength:.0f}%")

    if delta.delta_divergence:
        print(f"     {Colors.MAGENTA}⚠ DELTA DIVERGENCE - Price and delta moving opposite{Colors.RESET}")

    print(f"     {Colors.CYAN}{delta.interpretation}{Colors.RESET}")

    # Delta explanation box
    print()
    print(f"     {Colors.DIM}┌────────────────────────────────────────────────────────────┐{Colors.RESET}")
    print(f"     {Colors.DIM}│{Colors.RESET} {Colors.BOLD}Volume Delta Interpretation:{Colors.RESET}                                {Colors.DIM}│{Colors.RESET}")
    print(f"     {Colors.DIM}│{Colors.RESET} • {Colors.GREEN}Positive delta{Colors.RESET} = Buyers lifting offers (aggressive)     {Colors.DIM}│{Colors.RESET}")
    print(f"     {Colors.DIM}│{Colors.RESET} • {Colors.RED}Negative delta{Colors.RESET} = Sellers hitting bids (aggressive)      {Colors.DIM}│{Colors.RESET}")
    print(f"     {Colors.DIM}│{Colors.RESET} • Divergence = Hidden accumulation/distribution           {Colors.DIM}│{Colors.RESET}")
    print(f"     {Colors.DIM}└────────────────────────────────────────────────────────────┘{Colors.RESET}")

    # 2. Volume Acceleration
    accel = result.acceleration
    print()
    print(f"  {Colors.BOLD}2. VOLUME ACCELERATION/DECELERATION{Colors.RESET}")

    accel_colors = {
        VolumeAcceleration.ACCELERATING: Colors.GREEN,
        VolumeAcceleration.STEADY: Colors.YELLOW,
        VolumeAcceleration.DECELERATING: Colors.RED,
        VolumeAcceleration.CLIMAX: Colors.MAGENTA + Colors.BOLD,
    }
    a_color = accel_colors.get(accel.acceleration, Colors.YELLOW)

    accel_icon = {
        VolumeAcceleration.ACCELERATING: "↑",
        VolumeAcceleration.STEADY: "→",
        VolumeAcceleration.DECELERATING: "↓",
        VolumeAcceleration.CLIMAX: "⚠",
    }
    icon = accel_icon.get(accel.acceleration, "")

    print(f"     State: {a_color}{icon} {accel.acceleration.value.upper()}{Colors.RESET}")
    print(f"     Rate: {accel.rate:.2f}x previous  |  Momentum: {accel.momentum:+.2f}")

    if accel.bars_accelerating > 0:
        print(f"     {Colors.GREEN}Accelerating for {accel.bars_accelerating} bars{Colors.RESET}")
    elif accel.bars_accelerating < 0:
        print(f"     {Colors.RED}Decelerating for {abs(accel.bars_accelerating)} bars{Colors.RESET}")

    if accel.is_climax:
        print(f"     {Colors.MAGENTA}⚠ CLIMAX DETECTED - High reversal probability{Colors.RESET}")

    print(f"     {Colors.DIM}{accel.description}{Colors.RESET}")

    # 3. MTF Agreement (if available)
    if result.mtf_agreement:
        mtf = result.mtf_agreement
        print()
        print(f"  {Colors.BOLD}3. MULTI-TIMEFRAME AGREEMENT{Colors.RESET}")

        mtf_colors = {
            MTFAgreement.CONFIRMED: Colors.GREEN,
            MTFAgreement.STOP_RUN: Colors.MAGENTA,
            MTFAgreement.REACCUMULATION: Colors.CYAN,
            MTFAgreement.UNCLEAR: Colors.YELLOW,
        }
        m_color = mtf_colors.get(mtf.agreement, Colors.YELLOW)

        print(f"     Agreement: {m_color}{mtf.agreement.value.upper()}{Colors.RESET}")
        print(f"     LTF Volume: {mtf.ltf_volume_ratio:.1f}x avg  |  HTF Volume: {mtf.htf_volume_ratio:.1f}x avg")

        accepted_icon = "✓" if mtf.is_accepted else "✗"
        accepted_color = Colors.GREEN if mtf.is_accepted else Colors.RED
        print(f"     Volume Accepted: {accepted_color}{accepted_icon}{Colors.RESET}")

        print(f"     {Colors.DIM}{mtf.description}{Colors.RESET}")

        # MTF table
        print()
        print(f"     {Colors.DIM}┌─────────────┬─────────────┬─────────────────────┐{Colors.RESET}")
        print(f"     {Colors.DIM}│{Colors.RESET} LTF Volume  {Colors.DIM}│{Colors.RESET} HTF Volume  {Colors.DIM}│{Colors.RESET} Verdict             {Colors.DIM}│{Colors.RESET}")
        print(f"     {Colors.DIM}├─────────────┼─────────────┼─────────────────────┤{Colors.RESET}")
        print(f"     {Colors.DIM}│{Colors.RESET} {Colors.GREEN}High{Colors.RESET}        {Colors.DIM}│{Colors.RESET} {Colors.GREEN}High{Colors.RESET}        {Colors.DIM}│{Colors.RESET} {Colors.GREEN}CONFIRMED{Colors.RESET}           {Colors.DIM}│{Colors.RESET}")
        print(f"     {Colors.DIM}│{Colors.RESET} {Colors.RED}High{Colors.RESET}        {Colors.DIM}│{Colors.RESET} {Colors.RED}Low{Colors.RESET}         {Colors.DIM}│{Colors.RESET} {Colors.MAGENTA}STOP RUN{Colors.RESET}            {Colors.DIM}│{Colors.RESET}")
        print(f"     {Colors.DIM}│{Colors.RESET} Low         {Colors.DIM}│{Colors.RESET} High        {Colors.DIM}│{Colors.RESET} {Colors.CYAN}REACCUMULATION{Colors.RESET}      {Colors.DIM}│{Colors.RESET}")
        print(f"     {Colors.DIM}│{Colors.RESET} Low         {Colors.DIM}│{Colors.RESET} Low         {Colors.DIM}│{Colors.RESET} NO INTEREST         {Colors.DIM}│{Colors.RESET}")
        print(f"     {Colors.DIM}└─────────────┴─────────────┴─────────────────────┘{Colors.RESET}")
    else:
        print()
        print(f"  {Colors.BOLD}3. MULTI-TIMEFRAME AGREEMENT{Colors.RESET}")
        print(f"     {Colors.DIM}(HTF data not available - single timeframe analysis){Colors.RESET}")

    # 4. Exhaustion Detection
    exhaust = result.exhaustion
    print()
    print(f"  {Colors.BOLD}4. EXHAUSTION DETECTION (Is Offense Dying?){Colors.RESET}")

    exhaust_colors = {
        ExhaustionRisk.LOW: Colors.GREEN,
        ExhaustionRisk.MEDIUM: Colors.YELLOW,
        ExhaustionRisk.HIGH: Colors.RED,
        ExhaustionRisk.EXTREME: Colors.RED + Colors.BOLD,
    }
    e_color = exhaust_colors.get(exhaust.risk, Colors.YELLOW)

    print(f"     Risk: {e_color}{exhaust.risk.value.upper()}{Colors.RESET}")
    print(f"     Confidence: {strength_bar(exhaust.confidence, 15)} {exhaust.confidence:.0f}%")

    # Exhaustion signals
    signals_present = []
    if exhaust.body_shrinking:
        signals_present.append(("Body Shrinking", True))
    else:
        signals_present.append(("Body Shrinking", False))
    if exhaust.volume_declining:
        signals_present.append(("Volume Declining", True))
    else:
        signals_present.append(("Volume Declining", False))
    if exhaust.failed_continuation:
        signals_present.append(("Failed Continuation", True))
    else:
        signals_present.append(("Failed Continuation", False))
    if exhaust.oi_stagnant:
        signals_present.append(("OI Stagnant", True))
    else:
        signals_present.append(("OI Stagnant", False))

    print()
    print(f"     {Colors.BOLD}Signals:{Colors.RESET}")
    for sig_name, present in signals_present:
        icon = "✓" if present else "✗"
        color = Colors.RED if present else Colors.DIM
        print(f"       {color}{icon} {sig_name}{Colors.RESET}")

    print()
    print(f"     {Colors.DIM}{exhaust.description}{Colors.RESET}")
    print(f"     {Colors.BOLD}Action:{Colors.RESET} {exhaust.action}")

    # Final insight
    print()
    print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")
    print(f"  {Colors.BOLD}Pro Insight:{Colors.RESET} {Colors.MAGENTA}Volume tells WHO moved the market. Delta tells WHO was aggressive.{Colors.RESET}")
    print(f"              {Colors.MAGENTA}Exhaustion tells WHEN the move dies. MTF tells if it's ACCEPTED.{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")


def print_orderbook_deep_dive(summary: OBAnalysisSummary):
    """Print detailed Orderbook analysis - 'Where is price FORCED to go?'"""
    print()
    print(f"{Colors.BOLD}{Colors.CYAN}┌{'─' * 78}┐{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}│  WHERE IS PRICE FORCED TO GO? - Order Book Analysis{' ' * 25}│{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}└{'─' * 78}┘{Colors.RESET}")
    print()

    # Main verdict
    path_colors = {
        "UP": Colors.GREEN,
        "DOWN": Colors.RED,
        "UP (based on flow)": Colors.GREEN,
        "DOWN (based on flow)": Colors.RED,
        "UNCLEAR": Colors.YELLOW
    }
    path_color = path_colors.get(summary.where_price_forced, Colors.YELLOW)

    signal_colors = {
        Signal.BULLISH.value: Colors.GREEN,
        Signal.BEARISH.value: Colors.RED,
        Signal.TRAP.value: Colors.MAGENTA,
        Signal.NEUTRAL.value: Colors.YELLOW,
    }
    overall_signal = signal_value(summary.overall_signal)
    sig_color = signal_colors.get(overall_signal, Colors.YELLOW)

    print(f"  {Colors.BOLD}Path:{Colors.RESET}   {path_color}{summary.where_price_forced}{Colors.RESET}")
    print(f"  {Colors.BOLD}Signal:{Colors.RESET} {sig_color}{overall_signal.upper()}{Colors.RESET}")
    print(f"  {Colors.BOLD}Confidence:{Colors.RESET} {strength_bar(summary.confidence)} {summary.confidence:.0f}%")
    print()
    print(f"  {Colors.DIM}{summary.summary}{Colors.RESET}")
    print()
    print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")

    # Key concept box
    print()
    print(f"  {Colors.DIM}┌─────────────────────────────────────────────────────────────────────┐{Colors.RESET}")
    print(f"  {Colors.DIM}│{Colors.RESET} {Colors.BOLD}WHAT TO IGNORE:{Colors.RESET}                                                  {Colors.DIM}│{Colors.RESET}")
    print(f"  {Colors.DIM}│{Colors.RESET} ❌ Single walls | ❌ Static snapshots | ❌ Pretty heatmaps         {Colors.DIM}│{Colors.RESET}")
    print(f"  {Colors.DIM}│{Colors.RESET} {Colors.BOLD}WHAT MATTERS:{Colors.RESET}                                                    {Colors.DIM}│{Colors.RESET}")
    print(f"  {Colors.DIM}│{Colors.RESET} ✓ Absorption | ✓ Imbalance+Volume | ✓ Spoofs | ✓ Liquidity path   {Colors.DIM}│{Colors.RESET}")
    print(f"  {Colors.DIM}└─────────────────────────────────────────────────────────────────────┘{Colors.RESET}")

    # Snapshot info
    snap = summary.snapshot
    print()
    print(f"  {Colors.BOLD}ORDERBOOK SNAPSHOT{Colors.RESET}")
    print(f"     Mid Price: ${snap.mid_price:,.2f}  |  Spread: ${snap.spread:.2f} ({snap.spread_percent:.4f}%)")
    print(f"     Bid Depth (10): {snap.bid_depth_10:,.2f}  |  Ask Depth (10): {snap.ask_depth_10:,.2f}")
    print(f"     Bid Value: ${snap.bid_value_10:,.0f}  |  Ask Value: ${snap.ask_value_10:,.0f}")

    # 1. Absorption
    ab = summary.absorption
    print()
    print(f"  {Colors.BOLD}1. ABSORPTION DETECTION{Colors.RESET}")
    print(f"     {Colors.DIM}\"Smart money defending a level\"{Colors.RESET}")

    if ab.detected:
        ab_colors = {
            AbsorptionSide.BID_ABSORPTION: Colors.GREEN,
            AbsorptionSide.ASK_ABSORPTION: Colors.RED,
        }
        ab_color = ab_colors.get(ab.side, Colors.YELLOW)
        print(f"     {ab_color}⚠ {ab.side.value.upper()} DETECTED{Colors.RESET}")
        print(f"     Strength: {strength_bar(ab.strength, 15)} {ab.strength:.0f}%")
        print(f"     Volume: {ab.volume_absorbed:,.0f}  |  Price Impact: {ab.price_impact:.2f}%")
        print(f"     Efficiency: {ab.efficiency:.3f} (lower = more absorption)")
        print(f"     {Colors.DIM}{ab.description}{Colors.RESET}")
        print(f"     {Colors.BOLD}Action:{Colors.RESET} {ab.action}")
    else:
        print(f"     {Colors.DIM}No absorption detected{Colors.RESET}")
        print(f"     {Colors.DIM}{ab.description}{Colors.RESET}")

    # 2. Imbalance
    imb = summary.imbalance
    print()
    print(f"  {Colors.BOLD}2. LIQUIDITY IMBALANCE{Colors.RESET}")
    print(f"     {Colors.DIM}\"2-3x imbalance = directional pressure (with volume)\"{Colors.RESET}")

    imb_colors = {
        ImbalanceDirection.BID_HEAVY: Colors.GREEN,
        ImbalanceDirection.ASK_HEAVY: Colors.RED,
        ImbalanceDirection.BALANCED: Colors.YELLOW,
    }
    imb_color = imb_colors.get(imb.direction, Colors.YELLOW)

    print(f"     Direction: {imb_color}{imb.direction.value.upper()}{Colors.RESET}")
    print(f"     Ratio: {imb.ratio:.2f}x (bid/ask)")

    if imb.is_actionable:
        print(f"     {Colors.GREEN}✓ ACTIONABLE - Volume confirms imbalance{Colors.RESET}")
    elif imb.is_bait:
        print(f"     {Colors.MAGENTA}⚠ BAIT WARNING - Imbalance without volume = manipulation{Colors.RESET}")
    else:
        print(f"     {Colors.DIM}Balanced - No actionable signal{Colors.RESET}")

    print(f"     {Colors.BOLD}Action:{Colors.RESET} {imb.action}")

    # 3. Spoof Detection
    sp = summary.spoof
    print()
    print(f"  {Colors.BOLD}3. SPOOF DETECTION{Colors.RESET}")
    print(f"     {Colors.DIM}\"Behavioral patterns: Wall appears → Price approaches → Wall pulls\"{Colors.RESET}")

    if sp.detected:
        sp_colors = {
            SpoofType.BID_SPOOF: Colors.RED,  # Fake support = bearish
            SpoofType.ASK_SPOOF: Colors.GREEN,  # Fake resistance = bullish
        }
        sp_color = sp_colors.get(sp.spoof_type, Colors.MAGENTA)
        print(f"     {sp_color}⚠ {sp.spoof_type.value.upper()} DETECTED{Colors.RESET}")
        print(f"     Confidence: {sp.confidence:.0f}%")
        print(f"     Wall Size: {sp.wall_size:,.0f} at ${sp.wall_price:,.2f}")
        print(f"     Times Pulled: {sp.times_pulled}")
        if sp.trapped_traders:
            print(f"     {Colors.MAGENTA}Traders TRAPPED (OI increased){Colors.RESET}")
        print(f"     {Colors.DIM}{sp.description}{Colors.RESET}")
        print(f"     {Colors.BOLD}Action:{Colors.RESET} {sp.action}")
    else:
        print(f"     {Colors.DIM}No spoof patterns detected{Colors.RESET}")
        print(f"     {Colors.DIM}(Need multiple snapshots to track wall behavior){Colors.RESET}")

    # 4. Liquidity Ladders
    lad = summary.liquidity_ladder
    print()
    print(f"  {Colors.BOLD}4. LIQUIDITY LADDERS{Colors.RESET}")
    print(f"     {Colors.DIM}\"Price moves toward THIN zones, away from THICK zones\"{Colors.RESET}")

    path_display = {
        "up": f"{Colors.GREEN}UP ↑{Colors.RESET}",
        "down": f"{Colors.RED}DOWN ↓{Colors.RESET}",
        "unclear": f"{Colors.YELLOW}UNCLEAR{Colors.RESET}"
    }
    print(f"     Path of Least Resistance: {path_display.get(lad.path_of_least_resistance, 'UNCLEAR')}")

    # Show zones
    if lad.nearest_thick_above:
        print(f"     {Colors.RED}THICK above:{Colors.RESET} ${lad.nearest_thick_above:,.2f} (resistance)")
    if lad.nearest_thick_below:
        print(f"     {Colors.GREEN}THICK below:{Colors.RESET} ${lad.nearest_thick_below:,.2f} (support)")
    if lad.nearest_thin_above:
        print(f"     Thin above: ${lad.nearest_thin_above:,.2f} (price attracted)")
    if lad.nearest_thin_below:
        print(f"     Thin below: ${lad.nearest_thin_below:,.2f} (price attracted)")

    print(f"     {Colors.DIM}{lad.description}{Colors.RESET}")

    # Zones summary
    print()
    print(f"     {Colors.DIM}Thick zones above: {len(lad.thick_zones_above)} | Thin zones above: {len(lad.thin_zones_above)}{Colors.RESET}")
    print(f"     {Colors.DIM}Thick zones below: {len(lad.thick_zones_below)} | Thin zones below: {len(lad.thin_zones_below)}{Colors.RESET}")

    print()
    print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")
    print(f"  {Colors.BOLD}Pro Insight:{Colors.RESET} {Colors.CYAN}Orderbook is your SHARPEST weapon. Track behavior, not snapshots.{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")


def print_unified_score(score: UnifiedScore):
    """Print unified market score - THE decisive signal."""
    print()
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'═' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}  UNIFIED MARKET SCORE - ONE ACTIONABLE NUMBER{' ' * 31}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}{'═' * 80}{Colors.RESET}")
    print()

    # Main score display
    score_color = Colors.GREEN if score.total_score > 0.25 else Colors.RED if score.total_score < -0.25 else Colors.YELLOW

    # Big score display
    score_display = f"{score.total_score:+.2f}"
    print(f"  {Colors.BOLD}TOTAL SCORE:{Colors.RESET} {score_color}{Colors.BOLD}{score_display:>6}{Colors.RESET}  "
          f"(Range: -1.00 to +1.00)")

    # Action
    action_colors = {
        "long": Colors.GREEN,
        "short": Colors.RED,
        "neutral": Colors.YELLOW
    }
    action_color = action_colors.get(score.bias, Colors.YELLOW)

    print(f"  {Colors.BOLD}ACTION:{Colors.RESET}      {action_color}{Colors.BOLD}{score.action}{Colors.RESET}")
    print(f"  {Colors.BOLD}CONFIDENCE:{Colors.RESET}  {strength_bar(score.confidence, 30)} {score.confidence:.0f}%")
    print()
    print(f"{Colors.DIM}{'─' * 80}{Colors.RESET}")

    # Component scores
    print()
    print(f"  {Colors.BOLD}Component Breakdown:{Colors.RESET}")
    print()

    # Volume
    vol_color = Colors.GREEN if score.volume_score > 0.2 else Colors.RED if score.volume_score < -0.2 else Colors.YELLOW
    vol_bar = score_bar(score.volume_score)
    print(f"    Volume/Delta ({score.volume_weight*100:.0f}%):  {vol_bar}  {vol_color}{score.volume_score:+.2f}{Colors.RESET}")

    # Orderbook
    book_color = Colors.GREEN if score.orderbook_score > 0.2 else Colors.RED if score.orderbook_score < -0.2 else Colors.YELLOW
    book_bar = score_bar(score.orderbook_score)
    print(f"    Orderbook    ({score.orderbook_weight*100:.0f}%):  {book_bar}  {book_color}{score.orderbook_score:+.2f}{Colors.RESET}")

    # OI
    oi_color = Colors.GREEN if score.oi_score > 0.2 else Colors.RED if score.oi_score < -0.2 else Colors.YELLOW
    oi_bar = score_bar(score.oi_score)
    print(f"    Open Interest({score.oi_weight*100:.0f}%):  {oi_bar}  {oi_color}{score.oi_score:+.2f}{Colors.RESET}")

    # Funding
    fund_color = Colors.GREEN if score.funding_score > 0.2 else Colors.RED if score.funding_score < -0.2 else Colors.YELLOW
    fund_bar = score_bar(score.funding_score)
    print(f"    Funding      ({score.funding_weight*100:.0f}%):  {fund_bar}  {fund_color}{score.funding_score:+.2f}{Colors.RESET}")

    print()
    print(f"{Colors.DIM}{'─' * 80}{Colors.RESET}")

    # Interpretation
    print()
    print(f"  {Colors.BOLD}Interpretation:{Colors.RESET}")
    print(f"  {Colors.DIM}{score.description}{Colors.RESET}")

    # Warning
    if score.warning:
        print()
        print(f"  {Colors.BOLD}{Colors.MAGENTA}⚠ WARNING:{Colors.RESET} {Colors.YELLOW}{score.warning}{Colors.RESET}")

    # Action guide
    print()
    print(f"{Colors.DIM}{'─' * 80}{Colors.RESET}")
    print(f"  {Colors.BOLD}Action Thresholds:{Colors.RESET}")
    print(f"    {Colors.GREEN}+0.55 or higher{Colors.RESET}  → LONG bias")
    print(f"    {Colors.RED}-0.55 or lower{Colors.RESET}   → SHORT bias")
    print(f"    {Colors.YELLOW}-0.25 to +0.25{Colors.RESET}  → NO TRADE (neutral)")
    print(f"{Colors.DIM}{'─' * 80}{Colors.RESET}")


def print_breakout_validation(validation: Optional[BreakoutValidation]):
    """Print breakout validation result."""
    if not validation:
        return

    print()
    print(f"{Colors.BOLD}{Colors.CYAN}┌{'─' * 78}┐{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}│  BREAKOUT VALIDATION - Real or Fake?{' ' * 41}│{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}└{'─' * 78}┘{Colors.RESET}")
    print()

    event = validation.event
    features = validation.features

    # Breakout type and level
    type_color = Colors.GREEN if event.breakout_type.value == "upward" else Colors.RED
    print(f"  {Colors.BOLD}Breakout Type:{Colors.RESET} {type_color}{event.breakout_type.value.upper()}{Colors.RESET}")
    print(f"  {Colors.BOLD}Breakout Level:{Colors.RESET} ${event.breakout_level:,.2f}")
    print(f"  {Colors.BOLD}Breakout Price:{Colors.RESET} ${event.breakout_price:,.2f} ({event.breakout_margin_pct:+.2f}%)")

    # Hard veto check
    if validation.hard_veto:
        print()
        print(f"  {Colors.BOLD}{Colors.RED}⚠ HARD VETO - FAKE BREAKOUT{Colors.RESET}")
        print(f"  {Colors.RED}{validation.veto_reason}{Colors.RESET}")
        print()
        print(f"  {Colors.BOLD}Action:{Colors.RESET} {Colors.YELLOW}AVOID{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")
        return

    # Quality and confidence
    quality_colors = {
        "institutional": Colors.GREEN,
        "retail": Colors.RED,
        "mixed": Colors.YELLOW
    }
    q_color = quality_colors.get(validation.quality.value, Colors.YELLOW)

    print()
    print(f"  {Colors.BOLD}Quality:{Colors.RESET}     {q_color}{validation.quality.value.upper()}{Colors.RESET}")
    print(f"  {Colors.BOLD}Confidence:{Colors.RESET}  {strength_bar(validation.confidence, 25)} {validation.confidence:.0f}%")
    print()
    print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")

    # Signal alignment
    print()
    print(f"  {Colors.BOLD}Signal Alignment:{Colors.RESET}")

    flow_icon = "✓" if validation.flow_aligned else "✗"
    flow_color = Colors.GREEN if validation.flow_aligned else Colors.RED
    print(f"    {flow_color}{flow_icon} Flow/Delta{Colors.RESET}     (ΔVr: {features.delta_ratio:+.3f}, RV: {features.relative_volume:.1f}x)")

    oi_icon = "✓" if validation.oi_aligned else "✗"
    oi_color = Colors.GREEN if validation.oi_aligned else Colors.RED
    print(f"    {oi_color}{oi_icon} Open Interest{Colors.RESET}  (ΔOI: {features.oi_change_pct:+.1f}%)")

    book_icon = "✓" if validation.book_aligned else "✗"
    book_color = Colors.GREEN if validation.book_aligned else Colors.RED
    print(f"    {book_color}{book_icon} Orderbook{Colors.RESET}      (Imb: {features.depth_imbalance_25bps:+.2f})")

    # Action
    print()
    print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")
    print(f"  {Colors.BOLD}Validation Result:{Colors.RESET}")

    action_colors = {
        "enter_long": Colors.GREEN,
        "enter_short": Colors.RED,
        "avoid": Colors.YELLOW,
        "wait": Colors.YELLOW
    }
    action_color = action_colors.get(validation.action, Colors.YELLOW)

    print(f"  {Colors.BOLD}ACTION:{Colors.RESET} {action_color}{validation.action.upper()}{Colors.RESET}")

    if validation.entry_price:
        print(f"  Entry:  ${validation.entry_price:,.2f}")
        print(f"  Stop:   ${validation.stop_loss:,.2f}")
        print(f"  Target: ${validation.target:,.2f}")

    # Warnings
    if validation.warnings:
        print()
        print(f"  {Colors.BOLD}Warnings:{Colors.RESET}")
        for warning in validation.warnings:
            print(f"    {Colors.YELLOW}• {warning}{Colors.RESET}")

    print()
    print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")
    print(f"  {Colors.BOLD}Insight:{Colors.RESET} {Colors.CYAN}Flow + OI alignment > Pattern recognition{Colors.RESET}")
    print(f"           {Colors.CYAN}Price breaks level ≠ real breakout. Check the FLOW.{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 78}{Colors.RESET}")


def print_header(symbol: str, price: float, change_pct: float, timeframe: str):
    """Print analysis header."""
    print()
    print(f"{Colors.BOLD}{'═' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  INDICATOR ANALYSIS: {symbol}{Colors.RESET}")
    print(f"{Colors.BOLD}{'═' * 80}{Colors.RESET}")

    change_color = Colors.GREEN if change_pct >= 0 else Colors.RED
    print(f"  Price: {Colors.BOLD}${price:,.2f}{Colors.RESET}  |  "
          f"{timeframe} Change: {change_color}{change_pct:+.2f}%{Colors.RESET}  |  "
          f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{Colors.DIM}{'─' * 80}{Colors.RESET}")


def print_section(title: str, emoji: str = ""):
    """Print section header."""
    print()
    print(f"{Colors.BOLD}{Colors.BLUE}{emoji} {title}{Colors.RESET}")
    print(f"{Colors.DIM}{'─' * 40}{Colors.RESET}")


def print_indicator(result: IndicatorResult):
    """Print a single indicator result."""
    color = signal_color(result.signal)
    signal_display = signal_value(result.signal).upper()

    print(f"  {Colors.BOLD}{result.name:<20}{Colors.RESET} "
          f"{strength_bar(result.strength)} "
          f"{color}{signal_display:^8}{Colors.RESET}")
    print(f"  {Colors.DIM}{result.description}{Colors.RESET}")
    print()


def print_summary(results: List[IndicatorResult]):
    """Print overall summary."""
    bullish = sum(1 for r in results if signal_value(r.signal) == Signal.BULLISH.value)
    bearish = sum(1 for r in results if signal_value(r.signal) == Signal.BEARISH.value)
    neutral = sum(1 for r in results if signal_value(r.signal) == Signal.NEUTRAL.value)

    total = len(results)
    avg_strength = sum(r.strength for r in results) / total if total > 0 else 50

    # Calculate weighted bias
    bias_score = sum(
        r.strength * (1 if signal_value(r.signal) == Signal.BULLISH.value else -1 if signal_value(r.signal) == Signal.BEARISH.value else 0)
        for r in results
    ) / total if total > 0 else 0

    print()
    print(f"{Colors.BOLD}{'═' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.MAGENTA}  SUMMARY{Colors.RESET}")
    print(f"{Colors.BOLD}{'═' * 80}{Colors.RESET}")
    print()

    # Signal distribution
    print(f"  {Colors.GREEN}Bullish: {bullish}/{total}{Colors.RESET}  |  "
          f"{Colors.RED}Bearish: {bearish}/{total}{Colors.RESET}  |  "
          f"{Colors.YELLOW}Neutral: {neutral}/{total}{Colors.RESET}")
    print()

    # Overall bias
    if bias_score > 20:
        bias_text = f"{Colors.GREEN}{Colors.BOLD}BULLISH{Colors.RESET}"
        bias_desc = "Multiple indicators favor upside"
    elif bias_score > 5:
        bias_text = f"{Colors.GREEN}Mildly Bullish{Colors.RESET}"
        bias_desc = "Slight bullish lean"
    elif bias_score < -20:
        bias_text = f"{Colors.RED}{Colors.BOLD}BEARISH{Colors.RESET}"
        bias_desc = "Multiple indicators favor downside"
    elif bias_score < -5:
        bias_text = f"{Colors.RED}Mildly Bearish{Colors.RESET}"
        bias_desc = "Slight bearish lean"
    else:
        bias_text = f"{Colors.YELLOW}Neutral{Colors.RESET}"
        bias_desc = "Mixed or indecisive signals"

    print(f"  Overall Bias: {bias_text}")
    print(f"  {Colors.DIM}{bias_desc}{Colors.RESET}")
    print(f"  Bias Score: {bias_score:+.1f} | Avg Strength: {avg_strength:.1f}%")
    print()

    # Key signals
    strong_signals = [r for r in results if r.strength >= 70]
    if strong_signals:
        print(f"  {Colors.BOLD}Key Signals:{Colors.RESET}")
        for r in strong_signals:
            color = signal_color(r.signal)
            print(f"    • {color}{r.name}: {signal_value(r.signal).upper()} ({r.strength:.0f}%){Colors.RESET}")
    print()
    print(f"{Colors.DIM}{'─' * 80}{Colors.RESET}")
