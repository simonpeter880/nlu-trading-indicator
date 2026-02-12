"""
Advanced Funding Rate Analysis Module
"Where is the crowd leaning?"

Key Concepts:
- Funding is a WARNING SYSTEM, not an entry trigger
- Payment between longs and shorts to keep perp aligned with spot
- Focus on EXTREMES (percentiles, not raw values)
- Funding + OI combo for high-probability setups

Pro Rules:
- High funding = don't chase
- Extreme funding = hunt reversals
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Tuple

from .signals import Signal

if TYPE_CHECKING:
    from .indicator_config import IndicatorConfig

from .indicator_config import DEFAULT_CONFIG


class FundingZone(Enum):
    """Funding rate zones based on percentiles."""

    EXTREME_POSITIVE = "extreme_positive"  # > 95th percentile
    HIGH_POSITIVE = "high_positive"  # > 75th percentile
    NORMAL_POSITIVE = "normal_positive"  # 25-75th percentile, positive
    NEUTRAL = "neutral"  # Near zero
    NORMAL_NEGATIVE = "normal_negative"  # 25-75th percentile, negative
    HIGH_NEGATIVE = "high_negative"  # < 25th percentile
    EXTREME_NEGATIVE = "extreme_negative"  # < 5th percentile


class CrowdPosition(Enum):
    """Where the crowd is positioned."""

    HEAVILY_LONG = "heavily_long"
    MODERATELY_LONG = "moderately_long"
    BALANCED = "balanced"
    MODERATELY_SHORT = "moderately_short"
    HEAVILY_SHORT = "heavily_short"


class FundingWarning(Enum):
    """Warning signals from funding."""

    DONT_CHASE_LONGS = "dont_chase_longs"  # High positive funding
    DONT_CHASE_SHORTS = "dont_chase_shorts"  # High negative funding
    SQUEEZE_RISK_UP = "squeeze_risk_up"  # Extreme negative = short squeeze
    SQUEEZE_RISK_DOWN = "squeeze_risk_down"  # Extreme positive = long squeeze
    EXHAUSTION = "exhaustion"  # Extreme funding + flat OI
    NONE = "none"


class FundingOICombo(Enum):
    """High-probability Funding + OI combinations."""

    CROWDED_LONGS_RISING = "crowded_longs_rising"  # Very positive + OI rising = downside risk
    CROWDED_SHORTS_RISING = "crowded_shorts_rising"  # Very negative + OI rising = upside risk
    EXTREME_EXHAUSTION = "extreme_exhaustion"  # Extreme + flat OI = exhaustion
    CAPITULATION = "capitulation"  # Extreme + OI dropping = flush
    HEALTHY_TREND = "healthy_trend"  # Normal funding + OI rising
    NO_EDGE = "no_edge"


@dataclass
class FundingPercentileResult:
    """Result of funding percentile analysis."""

    current_rate: float
    current_rate_percent: float
    annualized_rate: float
    zone: FundingZone
    percentile: float  # 0-100, where current rate sits historically
    description: str


@dataclass
class CrowdAnalysisResult:
    """Analysis of crowd positioning."""

    position: CrowdPosition
    lean_strength: float  # 0-100, how strong the lean is
    crowd_description: str
    contrarian_bias: str  # What direction contrarian would favor


@dataclass
class FundingWarningResult:
    """Warning signals from funding analysis."""

    warning: FundingWarning
    severity: str  # 'low', 'medium', 'high', 'extreme'
    action: str
    description: str


@dataclass
class FundingOIComboResult:
    """Combined Funding + OI analysis."""

    combo: FundingOICombo
    funding_direction: str  # 'positive', 'negative', 'neutral'
    oi_direction: str  # 'rising', 'falling', 'flat'
    probability: float  # Probability of expected outcome
    expected_outcome: str
    trade_action: str


@dataclass
class FundingAnalysisSummary:
    """Complete funding analysis summary."""

    percentile: FundingPercentileResult
    crowd: CrowdAnalysisResult
    warning: FundingWarningResult
    funding_oi_combo: Optional[FundingOIComboResult]

    is_extreme: bool
    should_chase: bool  # Can you chase the trend?
    overall_signal: Signal
    confidence: float
    summary: str


class AdvancedFundingAnalyzer:
    """
    Professional Funding Rate analysis.

    Key principles:
    - Funding is a WARNING SYSTEM, not an entry trigger
    - Use percentiles, not raw values
    - Combine with OI for high-probability setups
    """

    def __init__(
        self,
        extreme_percentile: Optional[float] = None,
        high_percentile: Optional[float] = None,
        config: Optional["IndicatorConfig"] = None,
    ):
        self.config = config or DEFAULT_CONFIG
        cfg = self.config.funding
        self.extreme_percentile = (
            extreme_percentile if extreme_percentile is not None else cfg.extreme_percentile
        )
        self.high_percentile = (
            high_percentile if high_percentile is not None else cfg.high_percentile
        )

    def _calculate_percentile(self, current_value: float, historical_values: List[float]) -> float:
        """Calculate what percentile the current value is at."""
        if not historical_values:
            return 50.0

        sorted_values = sorted(historical_values)
        count_below = sum(1 for v in sorted_values if v < current_value)
        return (count_below / len(sorted_values)) * 100

    def analyze_percentile(
        self, current_rate: float, historical_rates: Optional[List[float]] = None
    ) -> FundingPercentileResult:
        """
        Analyze funding rate using percentiles, not raw values.

        Focus on WHERE the current rate sits historically.
        """
        rate_percent = current_rate * 100
        annualized = current_rate * 1095 * 100  # 3x per day * 365
        cfg = self.config.funding

        # Calculate percentile if history available
        if historical_rates and len(historical_rates) >= 10:
            percentile = self._calculate_percentile(current_rate, historical_rates)
        else:
            # Estimate based on typical ranges
            extreme_span = max(1.0, self.extreme_percentile - self.high_percentile)
            mid_span = max(1.0, self.high_percentile - 50.0)
            if rate_percent > cfg.extreme_positive_pct:
                percentile = self.extreme_percentile + min(
                    4, (rate_percent - cfg.extreme_positive_pct) * 20
                )
            elif rate_percent > cfg.high_positive_pct:
                percentile = self.high_percentile + (
                    (rate_percent - cfg.high_positive_pct)
                    / (cfg.extreme_positive_pct - cfg.high_positive_pct)
                    * extreme_span
                )
            elif rate_percent > cfg.neutral_pct:
                percentile = 50 + (
                    (rate_percent - cfg.neutral_pct)
                    / (cfg.high_positive_pct - cfg.neutral_pct)
                    * mid_span
                )
            elif rate_percent > -cfg.neutral_pct:
                percentile = 50
            elif rate_percent > cfg.high_negative_pct:
                percentile = (100 - self.high_percentile) + (
                    (rate_percent - cfg.high_negative_pct)
                    / (cfg.neutral_pct - cfg.high_negative_pct)
                    * mid_span
                )
            elif rate_percent > cfg.extreme_negative_pct:
                percentile = (100 - self.extreme_percentile) + (
                    (rate_percent - cfg.extreme_negative_pct)
                    / (cfg.high_negative_pct - cfg.extreme_negative_pct)
                    * extreme_span
                )
            else:
                percentile = max(
                    1,
                    (100 - self.extreme_percentile)
                    - (cfg.extreme_negative_pct - rate_percent) * extreme_span,
                )

        # Determine zone
        low_extreme = 100 - self.extreme_percentile
        low_high = 100 - self.high_percentile

        if percentile >= self.extreme_percentile:
            zone = FundingZone.EXTREME_POSITIVE
            desc = (
                f"EXTREME positive funding ({percentile:.0f}th percentile) - Longs paying heavily"
            )
        elif percentile >= self.high_percentile:
            zone = FundingZone.HIGH_POSITIVE
            desc = f"High positive funding ({percentile:.0f}th percentile) - Longs paying shorts"
        elif percentile >= 50 and rate_percent > cfg.neutral_pct:
            zone = FundingZone.NORMAL_POSITIVE
            desc = f"Normal positive funding ({percentile:.0f}th percentile)"
        elif percentile <= low_extreme:
            zone = FundingZone.EXTREME_NEGATIVE
            desc = (
                f"EXTREME negative funding ({percentile:.0f}th percentile) - Shorts paying heavily"
            )
        elif percentile <= low_high:
            zone = FundingZone.HIGH_NEGATIVE
            desc = f"High negative funding ({percentile:.0f}th percentile) - Shorts paying longs"
        elif percentile < 50 and rate_percent < -cfg.neutral_pct:
            zone = FundingZone.NORMAL_NEGATIVE
            desc = f"Normal negative funding ({percentile:.0f}th percentile)"
        else:
            zone = FundingZone.NEUTRAL
            desc = f"Neutral funding ({percentile:.0f}th percentile) - Balanced"

        return FundingPercentileResult(
            current_rate=current_rate,
            current_rate_percent=rate_percent,
            annualized_rate=annualized,
            zone=zone,
            percentile=percentile,
            description=desc,
        )

    def analyze_crowd(self, funding_zone: FundingZone, percentile: float) -> CrowdAnalysisResult:
        """
        Determine where the crowd is leaning.

        High positive = crowd is long
        High negative = crowd is short
        """
        if funding_zone in [FundingZone.EXTREME_POSITIVE, FundingZone.HIGH_POSITIVE]:
            if funding_zone == FundingZone.EXTREME_POSITIVE:
                position = CrowdPosition.HEAVILY_LONG
                lean_strength = min(100, 70 + (percentile - 95) * 6)
                crowd_desc = "Crowd is HEAVILY LONG - Everyone is bullish"
                contrarian = "bearish"
            else:
                position = CrowdPosition.MODERATELY_LONG
                lean_strength = 50 + (percentile - 75) * 1
                crowd_desc = "Crowd is moderately long - Bullish sentiment"
                contrarian = "cautiously bearish"

        elif funding_zone in [FundingZone.EXTREME_NEGATIVE, FundingZone.HIGH_NEGATIVE]:
            if funding_zone == FundingZone.EXTREME_NEGATIVE:
                position = CrowdPosition.HEAVILY_SHORT
                lean_strength = min(100, 70 + (5 - percentile) * 6)
                crowd_desc = "Crowd is HEAVILY SHORT - Everyone is bearish"
                contrarian = "bullish"
            else:
                position = CrowdPosition.MODERATELY_SHORT
                lean_strength = 50 + (25 - percentile) * 1
                crowd_desc = "Crowd is moderately short - Bearish sentiment"
                contrarian = "cautiously bullish"

        else:
            position = CrowdPosition.BALANCED
            lean_strength = 30
            crowd_desc = "Crowd is balanced - No strong consensus"
            contrarian = "none"

        return CrowdAnalysisResult(
            position=position,
            lean_strength=lean_strength,
            crowd_description=crowd_desc,
            contrarian_bias=contrarian,
        )

    def get_warning(self, funding_zone: FundingZone, percentile: float) -> FundingWarningResult:
        """
        Generate warning signals.

        Funding is a WARNING SYSTEM:
        - High funding = don't chase
        - Extreme funding = hunt reversals
        """
        if funding_zone == FundingZone.EXTREME_POSITIVE:
            return FundingWarningResult(
                warning=FundingWarning.SQUEEZE_RISK_DOWN,
                severity="extreme",
                action="HUNT REVERSAL - Look for short entries",
                description="Extreme crowding on long side. High probability of long squeeze.",
            )

        elif funding_zone == FundingZone.HIGH_POSITIVE:
            return FundingWarningResult(
                warning=FundingWarning.DONT_CHASE_LONGS,
                severity="high",
                action="DON'T CHASE - Wait for pullback to enter longs",
                description="High funding means longs are expensive. Don't add here.",
            )

        elif funding_zone == FundingZone.EXTREME_NEGATIVE:
            return FundingWarningResult(
                warning=FundingWarning.SQUEEZE_RISK_UP,
                severity="extreme",
                action="HUNT REVERSAL - Look for long entries",
                description="Extreme crowding on short side. High probability of short squeeze.",
            )

        elif funding_zone == FundingZone.HIGH_NEGATIVE:
            return FundingWarningResult(
                warning=FundingWarning.DONT_CHASE_SHORTS,
                severity="high",
                action="DON'T CHASE - Wait for bounce to enter shorts",
                description="High negative funding means shorts are crowded. Don't pile on.",
            )

        else:
            return FundingWarningResult(
                warning=FundingWarning.NONE,
                severity="low",
                action="No warning - Normal funding levels",
                description="Funding is neutral. No positioning extremes detected.",
            )

    def analyze_funding_oi_combo(
        self,
        funding_zone: FundingZone,
        oi_change_percent: float,
        oi_direction: Optional[str] = None,
    ) -> FundingOIComboResult:
        """
        Combine Funding + OI for high-probability scenarios.

        The Matrix:
        ┌────────────────┬──────────────┬─────────────────────────────┐
        │ Funding        │ OI           │ Meaning                     │
        ├────────────────┼──────────────┼─────────────────────────────┤
        │ Very positive  │ Rising       │ Crowded longs → downside    │
        │ Very negative  │ Rising       │ Crowded shorts → upside     │
        │ Extreme        │ Flat         │ Exhaustion                  │
        │ Extreme        │ Dropping     │ Capitulation / flush        │
        │ Normal         │ Rising       │ Healthy trend               │
        └────────────────┴──────────────┴─────────────────────────────┘
        """
        # Determine OI direction
        if oi_direction is None:
            direction_threshold = self.config.open_interest.direction_change_pct
            if oi_change_percent > direction_threshold:
                oi_dir = "rising"
            elif oi_change_percent < -direction_threshold:
                oi_dir = "falling"
            else:
                oi_dir = "flat"
        else:
            oi_dir = oi_direction

        # Determine funding direction
        if funding_zone in [FundingZone.EXTREME_POSITIVE, FundingZone.HIGH_POSITIVE]:
            funding_dir = "positive"
        elif funding_zone in [FundingZone.EXTREME_NEGATIVE, FundingZone.HIGH_NEGATIVE]:
            funding_dir = "negative"
        else:
            funding_dir = "neutral"

        is_extreme = funding_zone in [FundingZone.EXTREME_POSITIVE, FundingZone.EXTREME_NEGATIVE]

        # Determine combo
        if (
            funding_zone in [FundingZone.EXTREME_POSITIVE, FundingZone.HIGH_POSITIVE]
            and oi_dir == "rising"
        ):
            return FundingOIComboResult(
                combo=FundingOICombo.CROWDED_LONGS_RISING,
                funding_direction=funding_dir,
                oi_direction=oi_dir,
                probability=75 if is_extreme else 65,
                expected_outcome="DOWNSIDE RISK - New longs entering crowded market",
                trade_action="Avoid longs. Look for short setups on failed breakouts.",
            )

        elif (
            funding_zone in [FundingZone.EXTREME_NEGATIVE, FundingZone.HIGH_NEGATIVE]
            and oi_dir == "rising"
        ):
            return FundingOIComboResult(
                combo=FundingOICombo.CROWDED_SHORTS_RISING,
                funding_direction=funding_dir,
                oi_direction=oi_dir,
                probability=75 if is_extreme else 65,
                expected_outcome="UPSIDE RISK - New shorts entering crowded market",
                trade_action="Avoid shorts. Look for long setups on failed breakdowns.",
            )

        elif is_extreme and oi_dir == "flat":
            return FundingOIComboResult(
                combo=FundingOICombo.EXTREME_EXHAUSTION,
                funding_direction=funding_dir,
                oi_direction=oi_dir,
                probability=70,
                expected_outcome="EXHAUSTION - Extreme positioning but no new money",
                trade_action="Reversal imminent. Position for mean reversion.",
            )

        elif is_extreme and oi_dir == "falling":
            direction = "down" if funding_dir == "positive" else "up"
            return FundingOIComboResult(
                combo=FundingOICombo.CAPITULATION,
                funding_direction=funding_dir,
                oi_direction=oi_dir,
                probability=80,
                expected_outcome=f"CAPITULATION - Positions being flushed {direction}",
                trade_action=f"Wait for flush to complete, then fade the move.",
            )

        elif funding_dir == "neutral" and oi_dir == "rising":
            return FundingOIComboResult(
                combo=FundingOICombo.HEALTHY_TREND,
                funding_direction=funding_dir,
                oi_direction=oi_dir,
                probability=60,
                expected_outcome="HEALTHY TREND - New positions without crowding",
                trade_action="Trade with the trend. Funding allows for continuation.",
            )

        else:
            return FundingOIComboResult(
                combo=FundingOICombo.NO_EDGE,
                funding_direction=funding_dir,
                oi_direction=oi_dir,
                probability=50,
                expected_outcome="No clear edge from Funding + OI combo",
                trade_action="Use other indicators for direction.",
            )

    def full_analysis(
        self,
        current_rate: float,
        historical_rates: Optional[List[float]] = None,
        oi_change_percent: Optional[float] = None,
    ) -> FundingAnalysisSummary:
        """
        Complete funding analysis answering: "Where is the crowd leaning?"

        Key output: Warning signals and Funding + OI combo.
        """
        # Percentile analysis
        percentile = self.analyze_percentile(current_rate, historical_rates)

        # Crowd analysis
        crowd = self.analyze_crowd(percentile.zone, percentile.percentile)

        # Warning signals
        warning = self.get_warning(percentile.zone, percentile.percentile)

        # Funding + OI combo (if OI data available)
        funding_oi = None
        if oi_change_percent is not None:
            funding_oi = self.analyze_funding_oi_combo(percentile.zone, oi_change_percent)

        # Determine if extreme
        is_extreme = percentile.zone in [FundingZone.EXTREME_POSITIVE, FundingZone.EXTREME_NEGATIVE]

        # Should you chase?
        should_chase = percentile.zone in [
            FundingZone.NEUTRAL,
            FundingZone.NORMAL_POSITIVE,
            FundingZone.NORMAL_NEGATIVE,
        ]

        # Overall signal
        if is_extreme:
            if percentile.zone == FundingZone.EXTREME_POSITIVE:
                overall_signal = Signal.WARNING  # Contrarian bearish
            else:
                overall_signal = Signal.WARNING  # Contrarian bullish
        elif percentile.zone in [FundingZone.HIGH_POSITIVE]:
            overall_signal = Signal.NEUTRAL  # Cautious
        elif percentile.zone in [FundingZone.HIGH_NEGATIVE]:
            overall_signal = Signal.NEUTRAL
        else:
            overall_signal = Signal.NEUTRAL

        # Override with Funding+OI combo if strong signal
        if funding_oi:
            if funding_oi.combo == FundingOICombo.CROWDED_LONGS_RISING:
                overall_signal = Signal.BEARISH
            elif funding_oi.combo == FundingOICombo.CROWDED_SHORTS_RISING:
                overall_signal = Signal.BULLISH
            elif funding_oi.combo == FundingOICombo.CAPITULATION:
                overall_signal = Signal.WARNING

        # Confidence
        confidence = 50.0
        if is_extreme:
            confidence += 20
        if warning.severity == "high":
            confidence += 10
        if funding_oi and funding_oi.probability > 60:
            confidence += 10
        confidence = min(90, confidence)

        # Summary
        chase_text = "CAN trade with trend" if should_chase else "DON'T CHASE"
        summary = f"Crowd: {crowd.position.value.upper()} | {chase_text} | {warning.action}"

        return FundingAnalysisSummary(
            percentile=percentile,
            crowd=crowd,
            warning=warning,
            funding_oi_combo=funding_oi,
            is_extreme=is_extreme,
            should_chase=should_chase,
            overall_signal=overall_signal,
            confidence=confidence,
            summary=summary,
        )


# Convenience functions
def should_i_chase(
    current_rate: float, config: Optional["IndicatorConfig"] = None
) -> Tuple[bool, str]:
    """
    Quick check: Should I chase this trade?

    Returns:
        (can_chase, reason)
    """
    analyzer = AdvancedFundingAnalyzer(config=config)
    result = analyzer.full_analysis(current_rate)
    return result.should_chase, result.warning.action


def get_crowd_lean(
    current_rate: float, config: Optional["IndicatorConfig"] = None
) -> Tuple[CrowdPosition, str]:
    """
    Quick check: Where is the crowd?

    Returns:
        (position, contrarian_bias)
    """
    analyzer = AdvancedFundingAnalyzer(config=config)
    result = analyzer.full_analysis(current_rate)
    return result.crowd.position, result.crowd.contrarian_bias
