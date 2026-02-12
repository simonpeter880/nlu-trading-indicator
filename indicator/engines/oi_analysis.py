"""
Advanced Open Interest Analysis Module
"Is money entering or leaving?"

Key Concepts:
- OI measures change in ACTIVE CONTRACTS, not direction or volume
- 4 OI Regimes: The price + OI matrix
- Rate of change matters more than absolute value
- High-edge signals: compression and trap detection
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING
from enum import Enum

from .signals import Signal
if TYPE_CHECKING:
    from .indicator_config import IndicatorConfig

from .indicator_config import DEFAULT_CONFIG

class OIRegime(Enum):
    """The 4 fundamental OI regimes."""
    NEW_LONGS = "new_longs"          # Price ↑, OI ↑ - Trend continuation
    SHORT_COVERING = "short_covering" # Price ↑, OI ↓ - Fake strength
    NEW_SHORTS = "new_shorts"         # Price ↓, OI ↑ - Squeeze fuel
    LONG_LIQUIDATION = "long_liquidation"  # Price ↓, OI ↓ - Trend exhaustion
    NEUTRAL = "neutral"               # No clear regime


class OISignal(Enum):
    """High-edge OI signals."""
    COMPRESSION = "compression"       # OI rising while price stalls
    BREAKOUT_TRAP = "breakout_trap"   # OI dropping after breakout
    EXPANSION = "expansion"           # OI expanding at key level
    EXHAUSTION = "exhaustion"         # OI collapsing
    NONE = "none"


@dataclass
class OIRegimeResult:
    """Result of OI regime analysis."""
    regime: OIRegime
    price_direction: str  # 'up', 'down', 'flat'
    oi_direction: str     # 'up', 'down', 'flat'
    interpretation: str
    trade_meaning: str
    strength: float       # 0-100, how clear the regime is


@dataclass
class OIRateOfChange:
    """OI rate of change analysis."""
    current_oi: float
    previous_oi: float
    oi_change_absolute: float
    oi_change_percent: float
    rate_vs_average: float    # How fast compared to normal
    acceleration: float       # Is the rate increasing or decreasing
    description: str


@dataclass
class OIHighEdgeSignal:
    """High-edge OI signal detection."""
    signal: OISignal
    detected: bool
    confidence: float
    description: str
    action: str  # What to do


@dataclass
class OIAnalysisSummary:
    """Complete OI analysis summary."""
    regime: OIRegimeResult
    rate_of_change: OIRateOfChange
    high_edge_signal: OIHighEdgeSignal

    is_money_entering: bool
    overall_signal: Signal
    confidence: float
    summary: str


class AdvancedOIAnalyzer:
    """
    Professional OI analysis focusing on:
    - The 4 OI regimes (memorize these!)
    - Rate of change over absolute values
    - High-edge signals for entries
    """

    def __init__(
        self,
        price_threshold: Optional[float] = None,
        oi_threshold: Optional[float] = None,
        config: Optional['IndicatorConfig'] = None
    ):
        self.config = config or DEFAULT_CONFIG
        cfg = self.config.open_interest
        self.price_threshold = price_threshold if price_threshold is not None else cfg.price_move_threshold_pct
        self.oi_threshold = oi_threshold if oi_threshold is not None else cfg.oi_change_threshold_pct

    def determine_regime(
        self,
        price_change_percent: float,
        oi_change_percent: float
    ) -> OIRegimeResult:
        """
        Determine which of the 4 OI regimes we're in.

        The Matrix:
        ┌─────────┬────────────┬─────────────────────┬──────────────────┐
        │ Price   │ OI         │ Interpretation      │ Trade Meaning    │
        ├─────────┼────────────┼─────────────────────┼──────────────────┤
        │ ↑       │ ↑          │ New longs entering  │ Trend continuation│
        │ ↑       │ ↓          │ Shorts closing      │ Fake strength    │
        │ ↓       │ ↑          │ New shorts entering │ Squeeze fuel     │
        │ ↓       │ ↓          │ Longs closing       │ Trend exhaustion │
        └─────────┴────────────┴─────────────────────┴──────────────────┘
        """
        # Determine directions
        if price_change_percent > self.price_threshold:
            price_dir = "up"
        elif price_change_percent < -self.price_threshold:
            price_dir = "down"
        else:
            price_dir = "flat"

        if oi_change_percent > self.oi_threshold:
            oi_dir = "up"
        elif oi_change_percent < -self.oi_threshold:
            oi_dir = "down"
        else:
            oi_dir = "flat"

        # Determine regime
        if price_dir == "up" and oi_dir == "up":
            regime = OIRegime.NEW_LONGS
            interpretation = "New longs entering the market"
            trade_meaning = "TREND CONTINUATION - Strong bullish, ride the trend"
            strength = min(90, 50 + abs(price_change_percent) * 5 + abs(oi_change_percent) * 3)

        elif price_dir == "up" and oi_dir == "down":
            regime = OIRegime.SHORT_COVERING
            interpretation = "Shorts are closing positions (covering)"
            trade_meaning = "FAKE STRENGTH - Rally may fade, be cautious with longs"
            strength = min(85, 50 + abs(price_change_percent) * 3 + abs(oi_change_percent) * 3)

        elif price_dir == "down" and oi_dir == "up":
            regime = OIRegime.NEW_SHORTS
            interpretation = "New shorts entering the market"
            trade_meaning = "SQUEEZE FUEL - Building energy for potential short squeeze"
            strength = min(90, 50 + abs(price_change_percent) * 3 + abs(oi_change_percent) * 5)

        elif price_dir == "down" and oi_dir == "down":
            regime = OIRegime.LONG_LIQUIDATION
            interpretation = "Longs are closing/liquidating positions"
            trade_meaning = "TREND EXHAUSTION - Selloff may be ending"
            strength = min(85, 50 + abs(price_change_percent) * 3 + abs(oi_change_percent) * 3)

        else:
            regime = OIRegime.NEUTRAL
            interpretation = "No clear regime - market in consolidation"
            trade_meaning = "WAIT - No clear edge, let the market show direction"
            strength = 40

        return OIRegimeResult(
            regime=regime,
            price_direction=price_dir,
            oi_direction=oi_dir,
            interpretation=interpretation,
            trade_meaning=trade_meaning,
            strength=strength
        )

    def analyze_rate_of_change(
        self,
        oi_history: List[float],
        lookback: int = 20
    ) -> OIRateOfChange:
        """
        Analyze the RATE of OI change, not just the absolute value.

        Rate of change is more important than where OI is.
        """
        if len(oi_history) < 2:
            return OIRateOfChange(
                current_oi=oi_history[-1] if oi_history else 0,
                previous_oi=0,
                oi_change_absolute=0,
                oi_change_percent=0,
                rate_vs_average=1.0,
                acceleration=0,
                description="Insufficient OI history"
            )

        current = oi_history[-1]
        previous = oi_history[-2]

        # Absolute and percent change
        change_abs = current - previous
        change_pct = (change_abs / previous * 100) if previous > 0 else 0

        # Calculate average rate of change over lookback
        if len(oi_history) >= lookback:
            changes = []
            for i in range(1, lookback):
                if oi_history[-i-1] > 0:
                    pct = (oi_history[-i] - oi_history[-i-1]) / oi_history[-i-1] * 100
                    changes.append(abs(pct))
            avg_change = sum(changes) / len(changes) if changes else 1.0
        else:
            avg_change = abs(change_pct) if change_pct != 0 else 1.0

        # Rate vs average
        rate_vs_avg = abs(change_pct) / avg_change if avg_change > 0 else 1.0

        # Calculate acceleration (is the rate speeding up or slowing down)
        acceleration = 0
        cfg = self.config.open_interest
        if len(oi_history) >= cfg.rate_acceleration_periods + 1:
            recent_changes = []
            older_changes = []
            mid = cfg.rate_acceleration_periods // 2

            for i in range(1, mid + 1):
                if oi_history[-i-1] > 0:
                    recent_changes.append(abs(oi_history[-i] - oi_history[-i-1]))

            for i in range(mid + 1, cfg.rate_acceleration_periods + 1):
                if oi_history[-i-1] > 0:
                    older_changes.append(abs(oi_history[-i] - oi_history[-i-1]))

            if recent_changes and older_changes:
                recent_avg = sum(recent_changes) / len(recent_changes)
                older_avg = sum(older_changes) / len(older_changes)
                acceleration = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0

        # Generate description
        if rate_vs_avg > cfg.rate_vs_avg_significant:
            desc = f"OI changing {rate_vs_avg:.1f}x faster than normal - SIGNIFICANT"
        elif rate_vs_avg > cfg.rate_vs_avg_elevated:
            desc = f"OI change elevated ({rate_vs_avg:.1f}x normal) - Watch closely"
        elif rate_vs_avg < cfg.rate_vs_avg_low:
            desc = f"OI change slow ({rate_vs_avg:.1f}x normal) - Low activity"
        else:
            desc = f"OI change rate normal ({rate_vs_avg:.1f}x)"

        if acceleration > cfg.acceleration_positive_threshold:
            desc += " | Accelerating!"
        elif acceleration < cfg.acceleration_negative_threshold:
            desc += " | Decelerating"

        return OIRateOfChange(
            current_oi=current,
            previous_oi=previous,
            oi_change_absolute=change_abs,
            oi_change_percent=change_pct,
            rate_vs_average=rate_vs_avg,
            acceleration=acceleration,
            description=desc
        )

    def detect_high_edge_signal(
        self,
        prices: List[float],
        oi_history: List[float],
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None,
        lookback: int = 10
    ) -> OIHighEdgeSignal:
        """
        Detect high-edge OI signals:

        1. COMPRESSION: OI rising while price stalls
           - Building energy for explosive move
           - Look for breakout direction

        2. BREAKOUT TRAP: OI dropping after breakout
           - Breakout is failing
           - Fade the move

        3. EXPANSION: OI expanding at key level
           - Genuine interest, likely continuation

        4. EXHAUSTION: OI collapsing
           - Move is ending
        """
        if len(prices) < lookback or len(oi_history) < lookback:
            return OIHighEdgeSignal(
                signal=OISignal.NONE,
                detected=False,
                confidence=0,
                description="Insufficient data for signal detection",
                action="Wait for more data"
            )

        cfg = self.config.open_interest

        # Calculate price range over lookback
        price_range = max(prices[-lookback:]) - min(prices[-lookback:])
        price_volatility = price_range / prices[-lookback] * 100 if prices[-lookback] > 0 else 0

        # Calculate OI change over lookback
        oi_start = oi_history[-lookback]
        oi_end = oi_history[-1]
        oi_change_pct = (oi_end - oi_start) / oi_start * 100 if oi_start > 0 else 0

        # Recent price move (last few bars)
        recent_price_move = (prices[-1] - prices[-3]) / prices[-3] * 100 if len(prices) >= 3 and prices[-3] > 0 else 0

        # Check for COMPRESSION: OI up, price flat
        if oi_change_pct > cfg.compression_oi_change_pct and price_volatility < cfg.compression_price_volatility_pct:
            return OIHighEdgeSignal(
                signal=OISignal.COMPRESSION,
                detected=True,
                confidence=min(85, 50 + oi_change_pct * 5),
                description=f"COMPRESSION: OI +{oi_change_pct:.1f}% while price ranged {price_volatility:.1f}%",
                action="Prepare for breakout! Set alerts at range extremes"
            )

        # Check for BREAKOUT TRAP: Recent breakout but OI dropping
        if highs and lows and len(highs) >= lookback:
            recent_window = min(cfg.breakout_trap_recent_window_bars, lookback)
            recent_high = max(highs[-recent_window:])
            range_high = max(highs[-lookback:-recent_window]) if len(highs) > recent_window else recent_high

            recent_low = min(lows[-recent_window:])
            range_low = min(lows[-lookback:-recent_window]) if len(lows) > recent_window else recent_low

            # Breakout high with OI drop
            breakout_ratio = cfg.breakout_trap_breakout_pct / 100
            broke_high = recent_high > range_high * (1 + breakout_ratio)
            broke_low = recent_low < range_low * (1 - breakout_ratio)

            recent_oi_start_idx = -recent_window if len(oi_history) >= recent_window else -1
            recent_oi_start = oi_history[recent_oi_start_idx] if oi_history else 0
            recent_oi_change = (
                (oi_history[-1] - recent_oi_start) / recent_oi_start * 100
                if recent_oi_start > 0 else 0
            )

            if broke_high and recent_oi_change < cfg.breakout_trap_oi_drop_pct:
                return OIHighEdgeSignal(
                    signal=OISignal.BREAKOUT_TRAP,
                    detected=True,
                    confidence=min(80, 50 + abs(recent_oi_change) * 3),
                    description=f"TRAP: Broke highs but OI dropped {recent_oi_change:.1f}%",
                    action="Fade the breakout! Short entries with tight stops"
                )

            if broke_low and recent_oi_change < cfg.breakout_trap_oi_drop_pct:
                return OIHighEdgeSignal(
                    signal=OISignal.BREAKOUT_TRAP,
                    detected=True,
                    confidence=min(80, 50 + abs(recent_oi_change) * 3),
                    description=f"TRAP: Broke lows but OI dropped {recent_oi_change:.1f}%",
                    action="Fade the breakdown! Long entries with tight stops"
                )

        # Check for EXPANSION: OI expanding significantly
        if oi_change_pct > cfg.expansion_oi_change_pct and abs(recent_price_move) > cfg.expansion_price_move_pct:
            direction = "bullish" if recent_price_move > 0 else "bearish"
            return OIHighEdgeSignal(
                signal=OISignal.EXPANSION,
                detected=True,
                confidence=min(80, 50 + oi_change_pct * 3),
                description=f"EXPANSION: OI +{oi_change_pct:.1f}% with {direction} price action",
                action=f"Genuine interest - trade with the trend ({direction})"
            )

        # Check for EXHAUSTION: OI collapsing
        if oi_change_pct < cfg.exhaustion_oi_change_pct:
            return OIHighEdgeSignal(
                signal=OISignal.EXHAUSTION,
                detected=True,
                confidence=min(75, 50 + abs(oi_change_pct) * 2),
                description=f"EXHAUSTION: OI collapsed {oi_change_pct:.1f}%",
                action="Move is ending - take profits or prepare for reversal"
            )

        # No clear signal
        return OIHighEdgeSignal(
            signal=OISignal.NONE,
            detected=False,
            confidence=40,
            description="No high-edge signal detected",
            action="No clear edge - wait for setup"
        )

    def full_analysis(
        self,
        prices: List[float],
        oi_history: List[float],
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None,
        price_change_percent: Optional[float] = None
    ) -> OIAnalysisSummary:
        """
        Complete OI analysis answering: "Is money entering or leaving?"

        Combines:
        - Regime identification
        - Rate of change analysis
        - High-edge signal detection
        """
        if not oi_history or not prices:
            return OIAnalysisSummary(
                regime=OIRegimeResult(
                    regime=OIRegime.NEUTRAL,
                    price_direction="flat",
                    oi_direction="flat",
                    interpretation="No data",
                    trade_meaning="Cannot analyze",
                    strength=0
                ),
                rate_of_change=OIRateOfChange(
                    current_oi=0, previous_oi=0,
                    oi_change_absolute=0, oi_change_percent=0,
                    rate_vs_average=0, acceleration=0,
                    description="No data"
                ),
                high_edge_signal=OIHighEdgeSignal(
                    signal=OISignal.NONE, detected=False,
                    confidence=0, description="No data", action="Wait"
                ),
                is_money_entering=False,
                overall_signal=Signal.NEUTRAL,
                confidence=0,
                summary="Insufficient OI data"
            )

        # Calculate price change if not provided
        if price_change_percent is None and len(prices) >= 2:
            price_change_percent = (prices[-1] - prices[0]) / prices[0] * 100

        # OI change
        oi_change_pct = 0
        if len(oi_history) >= 2:
            oi_change_pct = (oi_history[-1] - oi_history[0]) / oi_history[0] * 100 if oi_history[0] > 0 else 0

        # Run analyses
        regime = self.determine_regime(price_change_percent or 0, oi_change_pct)
        rate = self.analyze_rate_of_change(oi_history)
        signal = self.detect_high_edge_signal(prices, oi_history, highs, lows)

        # Determine if money is entering (with epsilon to filter noise)
        money_flow_eps = self.config.open_interest.money_flow_epsilon_pct
        is_money_entering = oi_change_pct > money_flow_eps
        is_money_leaving = oi_change_pct < -money_flow_eps

        # Determine overall signal
        if signal.detected:
            if signal.signal == OISignal.COMPRESSION:
                overall_signal = Signal.NEUTRAL  # Wait for direction
            elif signal.signal == OISignal.BREAKOUT_TRAP:
                overall_signal = Signal.CAUTION
            elif signal.signal == OISignal.EXPANSION:
                overall_signal = Signal.BULLISH if regime.price_direction == "up" else Signal.BEARISH
            elif signal.signal == OISignal.EXHAUSTION:
                overall_signal = Signal.CAUTION
            else:
                overall_signal = Signal.NEUTRAL
        else:
            if regime.regime == OIRegime.NEW_LONGS:
                overall_signal = Signal.BULLISH
            elif regime.regime == OIRegime.NEW_SHORTS:
                overall_signal = Signal.BEARISH
            elif regime.regime == OIRegime.SHORT_COVERING:
                overall_signal = Signal.CAUTION  # Fake strength
            elif regime.regime == OIRegime.LONG_LIQUIDATION:
                overall_signal = Signal.CAUTION  # Exhaustion
            else:
                overall_signal = Signal.NEUTRAL

        # Calculate confidence
        confidence = (regime.strength + signal.confidence) / 2 if signal.detected else regime.strength

        # Generate summary
        if is_money_entering:
            money_flow = "ENTERING"
        elif is_money_leaving:
            money_flow = "LEAVING"
        else:
            money_flow = "FLAT"
        summary = f"Money is {money_flow} | Regime: {regime.regime.value.upper()} | {regime.trade_meaning}"

        return OIAnalysisSummary(
            regime=regime,
            rate_of_change=rate,
            high_edge_signal=signal,
            is_money_entering=is_money_entering,
            overall_signal=overall_signal,
            confidence=confidence,
            summary=summary
        )


# Convenience functions
def get_oi_regime(
    price_change_percent: float,
    oi_change_percent: float,
    config: Optional['IndicatorConfig'] = None
) -> Tuple[OIRegime, str]:
    """
    Quick regime check.

    Returns:
        (regime, trade_meaning)
    """
    analyzer = AdvancedOIAnalyzer(config=config)
    result = analyzer.determine_regime(price_change_percent, oi_change_percent)
    return result.regime, result.trade_meaning


def is_money_entering(oi_history: List[float], epsilon_pct: float = 0.1) -> Tuple[bool, float]:
    """
    Quick check: Is money entering?

    Args:
        oi_history: List of OI values
        epsilon_pct: Minimum change to count as entering (filters noise)

    Returns:
        (is_entering, oi_change_percent)
    """
    if len(oi_history) < 2:
        return False, 0

    change_pct = (oi_history[-1] - oi_history[0]) / oi_history[0] * 100 if oi_history[0] > 0 else 0
    return change_pct > epsilon_pct, change_pct
