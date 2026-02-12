"""
Advanced Volume Analysis Module
"Was the move REAL?" - Volume confirms acceptance, not direction.

Key Concepts:
- Relative Volume: Compare to 20-bar avg and session context
- Volume Location: WHERE volume appears matters more than how much
- Absorption: Volume spike + no price move = hidden accumulation/distribution
- Liquidity Sweeps: Volume after sweep = confirmation
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import math


class VolumeContext(Enum):
    """Volume context relative to average."""
    EXTREME = "extreme"      # > 3x average
    HIGH = "high"            # > 1.5x average
    NORMAL = "normal"        # 0.7x - 1.5x average
    LOW = "low"              # < 0.7x average
    DEAD = "dead"            # < 0.3x average


class VolumeLocation(Enum):
    """Where volume occurred in the price range."""
    RANGE_HIGH = "range_high"      # Distribution zone
    RANGE_LOW = "range_low"        # Accumulation zone
    MID_RANGE = "mid_range"        # Noise zone
    BREAKOUT_HIGH = "breakout_high"  # Above range
    BREAKOUT_LOW = "breakout_low"    # Below range


class AbsorptionType(Enum):
    """Type of absorption detected."""
    BID_ABSORPTION = "bid_absorption"    # Buyers absorbing sells
    ASK_ABSORPTION = "ask_absorption"    # Sellers absorbing buys
    NONE = "none"


@dataclass
class RelativeVolumeResult:
    """Result of relative volume analysis."""
    current_volume: float
    avg_volume_20: float
    relative_ratio: float
    context: VolumeContext
    is_meaningful: bool  # > 1.5x average
    description: str


@dataclass
class VolumeLocationResult:
    """Result of volume location analysis."""
    location: VolumeLocation
    range_high: float
    range_low: float
    current_price: float
    percentile_in_range: float  # 0-100, where in range
    interpretation: str


@dataclass
class AbsorptionResult:
    """Result of absorption detection."""
    detected: bool
    absorption_type: AbsorptionType
    volume_ratio: float        # How much volume relative to avg
    price_move_percent: float  # How much price actually moved
    efficiency: float          # price_move / volume_ratio (low = absorption)
    description: str


@dataclass
class LiquiditySweepResult:
    """Result of liquidity sweep detection."""
    sweep_detected: bool
    sweep_direction: str  # 'high', 'low', or 'none'
    sweep_level: float
    volume_confirmation: bool  # Did volume confirm after sweep
    description: str


@dataclass
class VolumeAnalysisSummary:
    """Complete volume analysis summary."""
    relative_volume: RelativeVolumeResult
    volume_location: VolumeLocationResult
    absorption: AbsorptionResult
    liquidity_sweep: LiquiditySweepResult

    move_is_real: bool
    confidence: float  # 0-100
    signal: str  # 'bullish', 'bearish', 'neutral', 'suspicious'
    summary: str


class AdvancedVolumeAnalyzer:
    """
    Professional volume analysis focusing on:
    - Was the move REAL? (participation)
    - WHERE did volume occur? (context)
    - What's being hidden? (absorption)
    """

    def __init__(
        self,
        lookback_period: int = 20,
        range_period: int = 20,
        relative_threshold: float = 1.5,
        absorption_threshold: float = 0.3
    ):
        """
        Args:
            lookback_period: Bars to look back for average volume
            range_period: Bars to define high/low range
            relative_threshold: Volume ratio to be considered meaningful (default 1.5x)
            absorption_threshold: Max price efficiency to detect absorption
        """
        self.lookback_period = lookback_period
        self.range_period = range_period
        self.relative_threshold = relative_threshold
        self.absorption_threshold = absorption_threshold

    def analyze_relative_volume(
        self,
        volumes: List[float],
        session_volumes: Optional[List[float]] = None
    ) -> RelativeVolumeResult:
        """
        Analyze volume relative to recent history.

        Rule: Volume > 1.5× average → meaningful

        Args:
            volumes: List of volume values (most recent last)
            session_volumes: Optional - same session from previous days

        Returns:
            RelativeVolumeResult with context and interpretation
        """
        if len(volumes) < 2:
            return RelativeVolumeResult(
                current_volume=volumes[-1] if volumes else 0,
                avg_volume_20=0,
                relative_ratio=1.0,
                context=VolumeContext.NORMAL,
                is_meaningful=False,
                description="Insufficient data"
            )

        current = volumes[-1]

        # Calculate 20-bar average (or available)
        lookback = min(self.lookback_period, len(volumes) - 1)
        avg_20 = sum(volumes[-lookback-1:-1]) / lookback if lookback > 0 else current

        # Calculate relative ratio
        ratio = current / avg_20 if avg_20 > 0 else 1.0

        # Determine context
        if ratio > 3.0:
            context = VolumeContext.EXTREME
            desc = f"EXTREME volume ({ratio:.1f}x avg) - Major event"
        elif ratio > self.relative_threshold:
            context = VolumeContext.HIGH
            desc = f"HIGH volume ({ratio:.1f}x avg) - Meaningful participation"
        elif ratio > 0.7:
            context = VolumeContext.NORMAL
            desc = f"Normal volume ({ratio:.1f}x avg)"
        elif ratio > 0.3:
            context = VolumeContext.LOW
            desc = f"LOW volume ({ratio:.1f}x avg) - Weak conviction"
        else:
            context = VolumeContext.DEAD
            desc = f"DEAD volume ({ratio:.1f}x avg) - No participation"

        # Session comparison if available
        if session_volumes and len(session_volumes) > 0:
            session_avg = sum(session_volumes) / len(session_volumes)
            session_ratio = current / session_avg if session_avg > 0 else 1.0
            desc += f" | vs session: {session_ratio:.1f}x"

        return RelativeVolumeResult(
            current_volume=current,
            avg_volume_20=avg_20,
            relative_ratio=ratio,
            context=context,
            is_meaningful=ratio > self.relative_threshold,
            description=desc
        )

    def analyze_volume_location(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[float]
    ) -> VolumeLocationResult:
        """
        Determine WHERE in the range volume is occurring.

        Location meanings:
        - Range high: Distribution
        - Range low: Accumulation
        - Mid-range: Noise
        - After sweep: Confirmation

        Args:
            closes: Close prices
            highs: High prices
            lows: Low prices
            volumes: Volume values
        """
        if len(closes) < self.range_period:
            return VolumeLocationResult(
                location=VolumeLocation.MID_RANGE,
                range_high=closes[-1] if closes else 0,
                range_low=closes[-1] if closes else 0,
                current_price=closes[-1] if closes else 0,
                percentile_in_range=50,
                interpretation="Insufficient data for range analysis"
            )

        current_price = closes[-1]

        # Calculate range from lookback period
        range_highs = highs[-self.range_period:]
        range_lows = lows[-self.range_period:]

        range_high = max(range_highs)
        range_low = min(range_lows)
        range_size = range_high - range_low

        if range_size == 0:
            return VolumeLocationResult(
                location=VolumeLocation.MID_RANGE,
                range_high=range_high,
                range_low=range_low,
                current_price=current_price,
                percentile_in_range=50,
                interpretation="No range (flat price)"
            )

        # Calculate position in range (0-100)
        percentile = ((current_price - range_low) / range_size) * 100

        # Determine location
        if current_price > range_high:
            location = VolumeLocation.BREAKOUT_HIGH
            interpretation = "BREAKOUT above range - Watch for continuation or rejection"
        elif current_price < range_low:
            location = VolumeLocation.BREAKOUT_LOW
            interpretation = "BREAKDOWN below range - Watch for continuation or rejection"
        elif percentile > 80:
            location = VolumeLocation.RANGE_HIGH
            interpretation = "Volume at RANGE HIGH - Potential distribution zone"
        elif percentile < 20:
            location = VolumeLocation.RANGE_LOW
            interpretation = "Volume at RANGE LOW - Potential accumulation zone"
        else:
            location = VolumeLocation.MID_RANGE
            interpretation = "Volume in MID-RANGE - Likely noise, wait for extremes"

        return VolumeLocationResult(
            location=location,
            range_high=range_high,
            range_low=range_low,
            current_price=current_price,
            percentile_in_range=percentile,
            interpretation=interpretation
        )

    def detect_absorption(
        self,
        closes: List[float],
        volumes: List[float],
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None
    ) -> AbsorptionResult:
        """
        Detect absorption: Volume spike + minimal price movement.

        When large volume doesn't move price, someone is absorbing.
        - High volume + price up small = bid absorption (bullish)
        - High volume + price down small = ask absorption (bearish)

        Args:
            closes: Close prices
            volumes: Volume values
            highs: Optional high prices for wick analysis
            lows: Optional low prices for wick analysis
        """
        if len(closes) < 3 or len(volumes) < 3:
            return AbsorptionResult(
                detected=False,
                absorption_type=AbsorptionType.NONE,
                volume_ratio=1.0,
                price_move_percent=0,
                efficiency=1.0,
                description="Insufficient data"
            )

        # Get relative volume
        rel_vol = self.analyze_relative_volume(volumes)
        volume_ratio = rel_vol.relative_ratio

        # Calculate price movement
        price_change = closes[-1] - closes[-2]
        price_change_percent = abs(price_change / closes[-2] * 100) if closes[-2] > 0 else 0

        # Calculate expected move based on volume
        # High volume should produce proportional price movement
        # Efficiency = price_move / volume_ratio
        efficiency = price_change_percent / volume_ratio if volume_ratio > 0 else 0

        # Absorption detection: High volume + low efficiency
        is_absorption = (
            volume_ratio > self.relative_threshold and
            efficiency < self.absorption_threshold
        )

        if is_absorption:
            if price_change >= 0:
                absorption_type = AbsorptionType.BID_ABSORPTION
                desc = f"BID ABSORPTION detected - {volume_ratio:.1f}x volume moved price only {price_change_percent:.2f}%"
                desc += " | Buyers absorbing sell pressure (BULLISH)"
            else:
                absorption_type = AbsorptionType.ASK_ABSORPTION
                desc = f"ASK ABSORPTION detected - {volume_ratio:.1f}x volume moved price only {price_change_percent:.2f}%"
                desc += " | Sellers absorbing buy pressure (BEARISH)"
        else:
            absorption_type = AbsorptionType.NONE
            if volume_ratio > self.relative_threshold:
                desc = f"High volume ({volume_ratio:.1f}x) with {price_change_percent:.2f}% move - Normal"
            else:
                desc = f"Normal volume ({volume_ratio:.1f}x) - No absorption signal"

        # Additional wick analysis if available
        if highs and lows and len(highs) >= 1 and len(lows) >= 1:
            body = abs(closes[-1] - closes[-2]) if len(closes) >= 2 else 0
            total_range = highs[-1] - lows[-1]

            if total_range > 0:
                wick_ratio = (total_range - body) / total_range
                if wick_ratio > 0.7 and volume_ratio > 1.0:
                    desc += f" | Large wicks ({wick_ratio:.0%}) suggest rejection"

        return AbsorptionResult(
            detected=is_absorption,
            absorption_type=absorption_type,
            volume_ratio=volume_ratio,
            price_move_percent=price_change_percent,
            efficiency=efficiency,
            description=desc
        )

    def detect_liquidity_sweep(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
        sweep_lookback: int = 10
    ) -> LiquiditySweepResult:
        """
        Detect liquidity sweeps with volume confirmation.

        A liquidity sweep occurs when price briefly takes out a level
        then reverses. Volume after sweep = confirmation.

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            volumes: Volume values
            sweep_lookback: Bars to look for sweep levels
        """
        if len(highs) < sweep_lookback + 2:
            return LiquiditySweepResult(
                sweep_detected=False,
                sweep_direction="none",
                sweep_level=0,
                volume_confirmation=False,
                description="Insufficient data for sweep detection"
            )

        current_close = closes[-1]
        current_high = highs[-1]
        current_low = lows[-1]

        # Find recent swing highs and lows (excluding last 2 bars)
        lookback_highs = highs[-sweep_lookback-2:-2]
        lookback_lows = lows[-sweep_lookback-2:-2]

        swing_high = max(lookback_highs)
        swing_low = min(lookback_lows)

        # Check for high sweep (price went above swing high then closed below)
        high_sweep = current_high > swing_high and current_close < swing_high

        # Check for low sweep (price went below swing low then closed above)
        low_sweep = current_low < swing_low and current_close > swing_low

        # Volume confirmation
        rel_vol = self.analyze_relative_volume(volumes)
        vol_confirms = rel_vol.is_meaningful

        if high_sweep:
            sweep_detected = True
            direction = "high"
            level = swing_high
            if vol_confirms:
                desc = f"LIQUIDITY SWEEP at high ${swing_high:.2f} with VOLUME CONFIRMATION"
                desc += " | Trapped longs, potential reversal (BEARISH)"
            else:
                desc = f"Liquidity sweep at high ${swing_high:.2f} but NO volume confirmation"
                desc += " | Weak signal, needs follow-through"
        elif low_sweep:
            sweep_detected = True
            direction = "low"
            level = swing_low
            if vol_confirms:
                desc = f"LIQUIDITY SWEEP at low ${swing_low:.2f} with VOLUME CONFIRMATION"
                desc += " | Trapped shorts, potential reversal (BULLISH)"
            else:
                desc = f"Liquidity sweep at low ${swing_low:.2f} but NO volume confirmation"
                desc += " | Weak signal, needs follow-through"
        else:
            sweep_detected = False
            direction = "none"
            level = 0
            desc = "No liquidity sweep detected"

        return LiquiditySweepResult(
            sweep_detected=sweep_detected,
            sweep_direction=direction,
            sweep_level=level,
            volume_confirmation=vol_confirms,
            description=desc
        )

    def full_analysis(
        self,
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float]
    ) -> VolumeAnalysisSummary:
        """
        Perform complete volume analysis answering: "Was the move REAL?"

        Combines:
        - Relative volume analysis
        - Volume location context
        - Absorption detection
        - Liquidity sweep detection

        Returns comprehensive summary with actionable signal.
        """
        # Run all analyses
        rel_vol = self.analyze_relative_volume(volumes)
        vol_location = self.analyze_volume_location(closes, highs, lows, volumes)
        absorption = self.detect_absorption(closes, volumes, highs, lows)
        sweep = self.detect_liquidity_sweep(highs, lows, closes, volumes)

        # Determine if move is real
        # A "real" move has: meaningful volume + appropriate location + no absorption
        move_is_real = (
            rel_vol.is_meaningful and
            not absorption.detected and
            vol_location.location != VolumeLocation.MID_RANGE
        )

        # Calculate confidence
        confidence = 50.0

        # Volume factor
        if rel_vol.context == VolumeContext.EXTREME:
            confidence += 20
        elif rel_vol.context == VolumeContext.HIGH:
            confidence += 10
        elif rel_vol.context == VolumeContext.LOW:
            confidence -= 15
        elif rel_vol.context == VolumeContext.DEAD:
            confidence -= 25

        # Location factor
        if vol_location.location in [VolumeLocation.RANGE_HIGH, VolumeLocation.RANGE_LOW]:
            confidence += 10
        elif vol_location.location in [VolumeLocation.BREAKOUT_HIGH, VolumeLocation.BREAKOUT_LOW]:
            confidence += 15 if rel_vol.is_meaningful else -10
        elif vol_location.location == VolumeLocation.MID_RANGE:
            confidence -= 10

        # Absorption factor (suspicious)
        if absorption.detected:
            confidence -= 5  # Not necessarily bad, but needs attention

        # Sweep factor
        if sweep.sweep_detected and sweep.volume_confirmation:
            confidence += 15
        elif sweep.sweep_detected and not sweep.volume_confirmation:
            confidence -= 5

        confidence = max(0, min(100, confidence))

        # Determine signal
        price_direction = closes[-1] - closes[-2] if len(closes) >= 2 else 0

        if absorption.detected:
            if absorption.absorption_type == AbsorptionType.BID_ABSORPTION:
                signal = "bullish"
                summary = "ABSORPTION: Hidden buying detected"
            else:
                signal = "bearish"
                summary = "ABSORPTION: Hidden selling detected"
        elif sweep.sweep_detected and sweep.volume_confirmation:
            if sweep.sweep_direction == "low":
                signal = "bullish"
                summary = "SWEEP & CONFIRM: Trapped shorts, reversal likely"
            else:
                signal = "bearish"
                summary = "SWEEP & CONFIRM: Trapped longs, reversal likely"
        elif move_is_real:
            if price_direction > 0:
                signal = "bullish"
                summary = "REAL MOVE: Meaningful volume confirms upward acceptance"
            else:
                signal = "bearish"
                summary = "REAL MOVE: Meaningful volume confirms downward acceptance"
        elif not rel_vol.is_meaningful:
            signal = "suspicious"
            summary = "WEAK: Low volume move - likely to reverse or fade"
        elif vol_location.location == VolumeLocation.MID_RANGE:
            signal = "neutral"
            summary = "NOISE: Mid-range volume - wait for extremes"
        else:
            signal = "neutral"
            summary = "MIXED: Conflicting signals - no clear edge"

        # Add context to summary
        summary += f" | Vol: {rel_vol.relative_ratio:.1f}x | Location: {vol_location.location.value}"

        return VolumeAnalysisSummary(
            relative_volume=rel_vol,
            volume_location=vol_location,
            absorption=absorption,
            liquidity_sweep=sweep,
            move_is_real=move_is_real,
            confidence=confidence,
            signal=signal,
            summary=summary
        )


# Convenience functions
def was_move_real(
    closes: List[float],
    volumes: List[float],
    highs: Optional[List[float]] = None,
    lows: Optional[List[float]] = None
) -> Tuple[bool, str]:
    """
    Quick check: Was the move REAL?

    Returns:
        (is_real, explanation)
    """
    analyzer = AdvancedVolumeAnalyzer()

    if highs is None:
        highs = closes
    if lows is None:
        lows = closes

    opens = closes  # Approximation if not available

    result = analyzer.full_analysis(opens, highs, lows, closes, volumes)
    return result.move_is_real, result.summary


def get_volume_context(volumes: List[float]) -> Tuple[VolumeContext, float]:
    """
    Get quick volume context.

    Returns:
        (context_enum, relative_ratio)
    """
    analyzer = AdvancedVolumeAnalyzer()
    result = analyzer.analyze_relative_volume(volumes)
    return result.context, result.relative_ratio
