"""
Market Structure Detection - The Foundation of Trend Analysis

RULE: Structure defines allowed trades. Indicators validate structure.
      If structure ≠ clear → NO indicator can save the trade.

Components:
1. Swing Highs/Lows (HH/HL/LH/LL)
2. Break of Structure (BOS) - Continuation
3. Change of Character (CHoCH) - Trend weakening/reversal
4. Range Structure - Compression/Distribution/Accumulation
5. Structural Acceptance vs Rejection
6. Fair Value Gaps (FVG) - Imbalance zones
7. Time-to-Followthrough - Structural momentum
8. Multi-Timeframe Alignment

Architecture:
    PRICE DATA
        ↓
    SWING DETECTION (HH/HL/LH/LL)
        ↓
    STRUCTURE CLASSIFICATION
    ├─ Trend (BOS/CHoCH)
    ├─ Range (Compression/Distribution/Accumulation)
    └─ Acceptance/Rejection
        ↓
    FVG DETECTION
        ↓
    TIME MOMENTUM
        ↓
    MULTI-TF ALIGNMENT
        ↓
    FINAL STRUCTURE STATE
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class SwingType(Enum):
    """Swing point type."""

    HIGH = "high"
    LOW = "low"


class TrendDirection(Enum):
    """Market trend direction."""

    UPTREND = "uptrend"  # HH + HL
    DOWNTREND = "downtrend"  # LH + LL
    RANGE = "range"  # No clear HH/HL or LH/LL
    UNKNOWN = "unknown"  # Insufficient data


class StructureEvent(Enum):
    """Market structure events."""

    BOS = "bos"  # Break of Structure (continuation)
    CHOCH = "choch"  # Change of Character (reversal warning)
    SWEEP = "sweep"  # Liquidity sweep
    ACCEPTANCE = "acceptance"  # Break + acceptance
    REJECTION = "rejection"  # Break + rejection


class RangeType(Enum):
    """Range classification."""

    COMPRESSION = "compression"  # Narrowing range, breakout imminent
    DISTRIBUTION = "distribution"  # Top formation, bearish
    ACCUMULATION = "accumulation"  # Bottom formation, bullish
    NEUTRAL = "neutral"  # No clear bias


class StructuralMomentum(Enum):
    """Time-based structural momentum."""

    FAST = "fast"  # Quick followthrough
    SLOW = "slow"  # Delayed followthrough
    STALLED = "stalled"  # No followthrough


class TimeframeAlignment(Enum):
    """Multi-timeframe structure alignment."""

    FULL_ALIGN = "full_align"  # HTF and LTF agree
    PARTIAL_ALIGN = "partial_align"  # HTF trend, LTF range
    COUNTER_TREND = "counter_trend"  # HTF and LTF disagree
    HTF_RANGE = "htf_range"  # HTF ranging


@dataclass
class SwingPoint:
    """A swing high or swing low."""

    idx: int  # Index in price array
    price: float  # Price at swing
    timestamp: int  # Timestamp (ms)
    swing_type: SwingType
    strength: int = 1  # How many bars on each side
    tested: int = 0  # Times price returned to this level
    broken: bool = False  # Has this swing been broken?


@dataclass
class FairValueGap:
    """Fair Value Gap (imbalance zone)."""

    idx_start: int  # Start candle index
    idx_end: int  # End candle index (gap is between start and end)
    price_top: float  # Top of gap
    price_bottom: float  # Bottom of gap
    gap_size_pct: float  # Size as % of price
    is_bullish: bool  # True if bullish FVG (gap up)
    filled: bool = False  # Has gap been filled?
    timestamp: int = 0  # Creation timestamp


@dataclass
class StructureBreak:
    """Record of a structure break (BOS or CHoCH)."""

    timestamp: int
    event_type: StructureEvent  # BOS or CHOCH
    direction: str  # "up" or "down"
    broken_level: float  # The swing level that was broken
    break_price: float  # Price at break
    bar_idx: int  # Index where break occurred

    # Acceptance tracking
    acceptance_bars: int = 0  # Bars since break
    accepted: Optional[bool] = None
    acceptance_volume_ratio: float = 0.0

    # Time tracking
    time_to_followthrough: Optional[float] = None  # Seconds
    followthrough_achieved: bool = False


@dataclass
class MarketStructureState:
    """Complete market structure state."""

    # Core structure
    trend_direction: TrendDirection
    recent_swings: List[SwingPoint] = field(default_factory=list)

    # Last structure event
    last_event: Optional[StructureBreak] = None
    last_bos: Optional[StructureBreak] = None
    last_choch: Optional[StructureBreak] = None

    # Range classification
    in_range: bool = False
    range_type: RangeType = RangeType.NEUTRAL
    range_high: Optional[float] = None
    range_low: Optional[float] = None
    range_tightness: float = 0.0  # 0-1, higher = tighter

    # Fair Value Gaps
    active_fvgs: List[FairValueGap] = field(default_factory=list)

    # Structural momentum
    structural_momentum: StructuralMomentum = StructuralMomentum.SLOW

    # Multi-timeframe
    htf_trend: TrendDirection = TrendDirection.UNKNOWN
    ltf_trend: TrendDirection = TrendDirection.UNKNOWN
    tf_alignment: TimeframeAlignment = TimeframeAlignment.PARTIAL_ALIGN

    # Confidence
    structure_confidence: float = 0.0  # 0-100

    # Warnings
    warnings: List[str] = field(default_factory=list)


class MarketStructureDetector:
    """
    Detects market structure from price data.

    This is the foundation - structure defines what trades are allowed.
    """

    def __init__(
        self,
        swing_lookback: int = 5,  # Bars on each side for swing
        bos_acceptance_bars: int = 3,  # Bars to confirm acceptance
        bos_acceptance_volume: float = 1.2,  # Volume ratio for acceptance
        followthrough_time_max: float = 300,  # Max seconds for followthrough (5min)
        followthrough_distance_pct: float = 0.5,  # Required move % for followthrough
        range_threshold_pct: float = 2.0,  # Max range size for "range" classification
        compression_periods: int = 10,  # Periods to detect compression
        fvg_min_gap_pct: float = 0.1,  # Minimum gap size %
    ):
        self.swing_lookback = swing_lookback
        self.bos_acceptance_bars = bos_acceptance_bars
        self.bos_acceptance_volume = bos_acceptance_volume
        self.followthrough_time_max = followthrough_time_max
        self.followthrough_distance_pct = followthrough_distance_pct
        self.range_threshold_pct = range_threshold_pct
        self.compression_periods = compression_periods
        self.fvg_min_gap_pct = fvg_min_gap_pct

        # State
        self._swing_highs: List[SwingPoint] = []
        self._swing_lows: List[SwingPoint] = []
        self._structure_breaks: List[StructureBreak] = []
        self._fvgs: List[FairValueGap] = []

    def detect_swings(
        self, highs: List[float], lows: List[float], timestamps: Optional[List[int]] = None
    ) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """
        Detect swing highs and swing lows.

        Swing High: high[i] > high[i-n:i] and high[i] > high[i+1:i+n+1]
        Swing Low: low[i] < low[i-n:i] and low[i] < low[i+1:i+n+1]

        Returns:
            (swing_highs, swing_lows)
        """
        if len(highs) < self.swing_lookback * 2 + 1:
            return [], []

        swing_highs = []
        swing_lows = []

        # Use default timestamps if not provided
        if timestamps is None:
            timestamps = [int(time.time() * 1000) + i * 60000 for i in range(len(highs))]

        for i in range(self.swing_lookback, len(highs) - self.swing_lookback):
            # Check swing high
            is_swing_high = True
            for j in range(i - self.swing_lookback, i + self.swing_lookback + 1):
                if j == i:
                    continue
                if highs[j] >= highs[i]:
                    is_swing_high = False
                    break

            if is_swing_high:
                swing_highs.append(
                    SwingPoint(
                        idx=i,
                        price=highs[i],
                        timestamp=timestamps[i],
                        swing_type=SwingType.HIGH,
                        strength=self.swing_lookback,
                    )
                )

            # Check swing low
            is_swing_low = True
            for j in range(i - self.swing_lookback, i + self.swing_lookback + 1):
                if j == i:
                    continue
                if lows[j] <= lows[i]:
                    is_swing_low = False
                    break

            if is_swing_low:
                swing_lows.append(
                    SwingPoint(
                        idx=i,
                        price=lows[i],
                        timestamp=timestamps[i],
                        swing_type=SwingType.LOW,
                        strength=self.swing_lookback,
                    )
                )

        return swing_highs, swing_lows

    def classify_trend(
        self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]
    ) -> TrendDirection:
        """
        Classify trend based on swing structure.

        Uptrend: HH (higher highs) + HL (higher lows)
        Downtrend: LH (lower highs) + LL (lower lows)
        Range: No clear HH/HL or LH/LL pattern
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return TrendDirection.UNKNOWN

        # Check last 3 swings for pattern
        recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
        recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows

        # Count higher/lower movements
        hh_count = 0  # Higher highs
        lh_count = 0  # Lower highs
        hl_count = 0  # Higher lows
        ll_count = 0  # Lower lows

        # Check highs
        for i in range(1, len(recent_highs)):
            if recent_highs[i].price > recent_highs[i - 1].price:
                hh_count += 1
            else:
                lh_count += 1

        # Check lows
        for i in range(1, len(recent_lows)):
            if recent_lows[i].price > recent_lows[i - 1].price:
                hl_count += 1
            else:
                ll_count += 1

        # Classify
        if hh_count >= 1 and hl_count >= 1 and lh_count == 0:
            return TrendDirection.UPTREND
        elif lh_count >= 1 and ll_count >= 1 and hh_count == 0:
            return TrendDirection.DOWNTREND
        else:
            return TrendDirection.RANGE

    def detect_bos_and_choch(
        self,
        current_idx: int,
        current_high: float,
        current_low: float,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        trend: TrendDirection,
    ) -> Optional[StructureBreak]:
        """
        Detect Break of Structure (BOS) or Change of Character (CHoCH).

        BOS (Continuation):
        - Uptrend: Break above last swing high
        - Downtrend: Break below last swing low

        CHoCH (Reversal Warning):
        - Uptrend: Break BELOW last swing low (HL)
        - Downtrend: Break ABOVE last swing high (LH)
        """
        if trend == TrendDirection.UPTREND:
            # Check for BOS (break above last high)
            if swing_highs:
                last_high = swing_highs[-1]
                if current_high > last_high.price and not last_high.broken:
                    last_high.broken = True
                    return StructureBreak(
                        timestamp=int(time.time() * 1000),
                        event_type=StructureEvent.BOS,
                        direction="up",
                        broken_level=last_high.price,
                        break_price=current_high,
                        bar_idx=current_idx,
                    )

            # Check for CHoCH (break below last low)
            if swing_lows:
                last_low = swing_lows[-1]
                if current_low < last_low.price and not last_low.broken:
                    last_low.broken = True
                    return StructureBreak(
                        timestamp=int(time.time() * 1000),
                        event_type=StructureEvent.CHOCH,
                        direction="down",
                        broken_level=last_low.price,
                        break_price=current_low,
                        bar_idx=current_idx,
                    )

        elif trend == TrendDirection.DOWNTREND:
            # Check for BOS (break below last low)
            if swing_lows:
                last_low = swing_lows[-1]
                if current_low < last_low.price and not last_low.broken:
                    last_low.broken = True
                    return StructureBreak(
                        timestamp=int(time.time() * 1000),
                        event_type=StructureEvent.BOS,
                        direction="down",
                        broken_level=last_low.price,
                        break_price=current_low,
                        bar_idx=current_idx,
                    )

            # Check for CHoCH (break above last high)
            if swing_highs:
                last_high = swing_highs[-1]
                if current_high > last_high.price and not last_high.broken:
                    last_high.broken = True
                    return StructureBreak(
                        timestamp=int(time.time() * 1000),
                        event_type=StructureEvent.CHOCH,
                        direction="up",
                        broken_level=last_high.price,
                        break_price=current_high,
                        bar_idx=current_idx,
                    )

        return None

    def check_acceptance(
        self,
        structure_break: StructureBreak,
        closes: List[float],
        volumes: List[float],
        current_idx: int,
    ) -> bool:
        """
        Check if structure break is ACCEPTED or REJECTED.

        Acceptance requires:
        1. Close beyond broken level
        2. Volume confirmation (RV > threshold)
        3. Price holds for N bars

        Returns:
            True if accepted, False if rejected, None if pending
        """
        bars_since_break = current_idx - structure_break.bar_idx
        structure_break.acceptance_bars = bars_since_break

        if bars_since_break < self.bos_acceptance_bars:
            return None  # Too early

        # Get bars since break
        start_idx = structure_break.bar_idx
        end_idx = min(current_idx + 1, len(closes))
        recent_closes = closes[start_idx:end_idx]
        recent_volumes = volumes[start_idx:end_idx] if volumes else []

        if len(recent_closes) < self.bos_acceptance_bars:
            return None

        # Check 1: Closes beyond level
        level = structure_break.broken_level
        if structure_break.direction == "up":
            closes_beyond = sum(1 for c in recent_closes if c > level)
        else:
            closes_beyond = sum(1 for c in recent_closes if c < level)

        close_rate = closes_beyond / len(recent_closes)

        # Check 2: Volume confirmation
        volume_confirmed = True
        if recent_volumes:
            avg_volume = sum(recent_volumes) / len(recent_volumes)
            prev_volume_window = volumes[max(0, start_idx - 10) : start_idx] if volumes else []
            if prev_volume_window:
                prev_avg = sum(prev_volume_window) / len(prev_volume_window)
                volume_ratio = avg_volume / prev_avg if prev_avg > 0 else 1.0
                structure_break.acceptance_volume_ratio = volume_ratio
                volume_confirmed = volume_ratio >= self.bos_acceptance_volume

        # Acceptance: Most closes beyond + volume confirmed
        accepted = close_rate >= 0.67 and volume_confirmed
        structure_break.accepted = accepted

        return accepted

    def detect_range_structure(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: Optional[List[float]] = None,
    ) -> Tuple[bool, RangeType, Optional[float], Optional[float], float]:
        """
        Detect if market is in a range and classify range type.

        Range indicators:
        - Multiple rejections at same levels
        - Low volatility (range < threshold%)
        - Failed breakout attempts

        Range types:
        - COMPRESSION: Tightening range (Bollinger squeeze-like)
        - DISTRIBUTION: Higher volume at highs (top formation)
        - ACCUMULATION: Higher volume at lows (bottom formation)

        Returns:
            (in_range, range_type, range_high, range_low, tightness)
        """
        if len(highs) < self.compression_periods:
            return False, RangeType.NEUTRAL, None, None, 0.0

        recent_highs = highs[-self.compression_periods :]
        recent_lows = lows[-self.compression_periods :]
        recent_closes = closes[-self.compression_periods :]

        # Calculate range
        range_high = max(recent_highs)
        range_low = min(recent_lows)
        range_size_pct = (range_high - range_low) / range_low * 100

        # Check if in range (small volatility)
        in_range = range_size_pct < self.range_threshold_pct

        if not in_range:
            return False, RangeType.NEUTRAL, None, None, 0.0

        # Calculate range tightness (0-1, higher = tighter)
        # Compare recent range to historical range
        if len(highs) >= self.compression_periods * 2:
            historical_highs = highs[-self.compression_periods * 2 : -self.compression_periods]
            historical_lows = lows[-self.compression_periods * 2 : -self.compression_periods]
            historical_range = max(historical_highs) - min(historical_lows)
            current_range = range_high - range_low
            tightness = 1.0 - (current_range / historical_range) if historical_range > 0 else 0.0
            tightness = max(0.0, min(1.0, tightness))
        else:
            tightness = 0.5

        # Classify range type
        range_type = RangeType.NEUTRAL

        if tightness > 0.7:
            # Very tight range = compression
            range_type = RangeType.COMPRESSION
        elif volumes and len(volumes) >= len(recent_closes):
            # Check volume distribution
            recent_volumes = volumes[-self.compression_periods :]
            mid_price = (range_high + range_low) / 2

            # Volume at highs vs lows
            volume_at_highs = sum(
                v for i, v in enumerate(recent_volumes) if recent_closes[i] > mid_price
            )
            volume_at_lows = sum(
                v for i, v in enumerate(recent_volumes) if recent_closes[i] < mid_price
            )
            total_vol = sum(recent_volumes)

            if total_vol > 0:
                high_vol_pct = volume_at_highs / total_vol
                low_vol_pct = volume_at_lows / total_vol

                if high_vol_pct > 0.6:
                    range_type = RangeType.DISTRIBUTION  # Selling at highs
                elif low_vol_pct > 0.6:
                    range_type = RangeType.ACCUMULATION  # Buying at lows

        return in_range, range_type, range_high, range_low, tightness

    def detect_fvg(
        self, idx: int, highs: List[float], lows: List[float], closes: List[float]
    ) -> Optional[FairValueGap]:
        """
        Detect Fair Value Gap (FVG) / Imbalance.

        Bullish FVG:
        - Low[i] > High[i-2]
        - Gap between candles i-2 and i

        Bearish FVG:
        - High[i] < Low[i-2]
        - Gap between candles i-2 and i
        """
        if idx < 2:
            return None

        # Bullish FVG check
        if lows[idx] > highs[idx - 2]:
            gap_top = lows[idx]
            gap_bottom = highs[idx - 2]
            gap_size_pct = (gap_top - gap_bottom) / gap_bottom * 100

            if gap_size_pct >= self.fvg_min_gap_pct:
                return FairValueGap(
                    idx_start=idx - 2,
                    idx_end=idx,
                    price_top=gap_top,
                    price_bottom=gap_bottom,
                    gap_size_pct=gap_size_pct,
                    is_bullish=True,
                    timestamp=int(time.time() * 1000),
                )

        # Bearish FVG check
        if highs[idx] < lows[idx - 2]:
            gap_top = lows[idx - 2]
            gap_bottom = highs[idx]
            gap_size_pct = (gap_top - gap_bottom) / gap_bottom * 100

            if gap_size_pct >= self.fvg_min_gap_pct:
                return FairValueGap(
                    idx_start=idx - 2,
                    idx_end=idx,
                    price_top=gap_top,
                    price_bottom=gap_bottom,
                    gap_size_pct=gap_size_pct,
                    is_bullish=False,
                    timestamp=int(time.time() * 1000),
                )

        return None

    def check_fvg_filled(self, fvg: FairValueGap, current_high: float, current_low: float) -> bool:
        """Check if FVG has been filled (price returned to gap)."""
        if fvg.is_bullish:
            # Bullish gap filled if price drops back into gap
            return current_low <= fvg.price_top and current_low >= fvg.price_bottom
        else:
            # Bearish gap filled if price rises back into gap
            return current_high >= fvg.price_bottom and current_high <= fvg.price_top

    def calculate_structural_momentum(
        self, structure_break: Optional[StructureBreak], current_price: float, time_elapsed: float
    ) -> StructuralMomentum:
        """
        Calculate structural momentum based on time-to-followthrough.

        Fast: Quick continuation after break
        Slow: Delayed continuation
        Stalled: No continuation within expected time
        """
        if structure_break is None:
            return StructuralMomentum.SLOW

        if structure_break.accepted is False:
            return StructuralMomentum.STALLED

        # Check if followthrough achieved
        if not structure_break.followthrough_achieved:
            distance_from_break = (
                abs(current_price - structure_break.break_price) / structure_break.break_price * 100
            )

            if distance_from_break >= self.followthrough_distance_pct:
                structure_break.followthrough_achieved = True
                structure_break.time_to_followthrough = time_elapsed

        # Classify based on time
        if structure_break.time_to_followthrough is not None:
            if structure_break.time_to_followthrough < self.followthrough_time_max * 0.33:
                return StructuralMomentum.FAST
            elif structure_break.time_to_followthrough < self.followthrough_time_max:
                return StructuralMomentum.SLOW
            else:
                return StructuralMomentum.STALLED
        elif time_elapsed > self.followthrough_time_max:
            return StructuralMomentum.STALLED

        return StructuralMomentum.SLOW

    def analyze_mtf_alignment(
        self, htf_trend: TrendDirection, ltf_trend: TrendDirection
    ) -> TimeframeAlignment:
        """
        Analyze multi-timeframe structure alignment.

        Best: HTF trend + LTF trend aligned
        OK: HTF trend + LTF range
        Risky: HTF trend + LTF counter-trend
        Neutral: HTF range
        """
        if htf_trend == TrendDirection.RANGE:
            return TimeframeAlignment.HTF_RANGE

        if htf_trend == ltf_trend:
            return TimeframeAlignment.FULL_ALIGN

        if ltf_trend == TrendDirection.RANGE:
            return TimeframeAlignment.PARTIAL_ALIGN

        # HTF and LTF disagree
        return TimeframeAlignment.COUNTER_TREND

    def get_structure_confidence(
        self,
        trend: TrendDirection,
        in_range: bool,
        last_event: Optional[StructureBreak],
        tf_alignment: TimeframeAlignment,
        structural_momentum: StructuralMomentum,
    ) -> float:
        """
        Calculate structure confidence (0-100).

        High confidence:
        - Clear trend + BOS accepted + MTF aligned + fast momentum

        Low confidence:
        - Range + no clear structure + counter-trend + stalled
        """
        confidence = 50.0  # Base

        # Trend clarity
        if trend == TrendDirection.UPTREND or trend == TrendDirection.DOWNTREND:
            confidence += 15
        elif trend == TrendDirection.RANGE and in_range:
            confidence += 10  # Range is also a clear structure
        else:
            confidence -= 20

        # Last event
        if last_event and last_event.event_type == StructureEvent.BOS:
            if last_event.accepted is True:
                confidence += 20
            elif last_event.accepted is False:
                confidence -= 15
        elif last_event and last_event.event_type == StructureEvent.CHOCH:
            confidence -= 10  # Trend changing, less confident

        # MTF alignment
        if tf_alignment == TimeframeAlignment.FULL_ALIGN:
            confidence += 15
        elif tf_alignment == TimeframeAlignment.COUNTER_TREND:
            confidence -= 20
        elif tf_alignment == TimeframeAlignment.PARTIAL_ALIGN:
            confidence += 5

        # Structural momentum
        if structural_momentum == StructuralMomentum.FAST:
            confidence += 10
        elif structural_momentum == StructuralMomentum.STALLED:
            confidence -= 15

        return max(0.0, min(100.0, confidence))

    def analyze(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: Optional[List[float]] = None,
        timestamps: Optional[List[int]] = None,
        htf_trend: TrendDirection = TrendDirection.UNKNOWN,
    ) -> MarketStructureState:
        """
        Complete market structure analysis.

        Returns:
            MarketStructureState with all components analyzed
        """
        warnings = []

        # 1. Detect swings
        swing_highs, swing_lows = self.detect_swings(highs, lows, timestamps)

        # 2. Classify trend
        ltf_trend = self.classify_trend(swing_highs, swing_lows)

        # 3. Detect BOS/CHoCH
        last_event = None
        if len(highs) > 0:
            current_idx = len(highs) - 1
            last_event = self.detect_bos_and_choch(
                current_idx, highs[-1], lows[-1], swing_highs, swing_lows, ltf_trend
            )

            if last_event:
                self._structure_breaks.append(last_event)

        # Get last BOS and CHoCH
        last_bos = None
        last_choch = None
        for sb in reversed(self._structure_breaks[-10:]):  # Check recent 10
            if sb.event_type == StructureEvent.BOS and last_bos is None:
                last_bos = sb
            if sb.event_type == StructureEvent.CHOCH and last_choch is None:
                last_choch = sb

        # 4. Check acceptance for last event
        if last_event and volumes:
            accepted = self.check_acceptance(last_event, closes, volumes, len(closes) - 1)
            if accepted is False:
                warnings.append(
                    f"{last_event.event_type.value.upper()} rejected - likely fake break"
                )

        # 5. Detect range structure
        in_range, range_type, range_high, range_low, tightness = self.detect_range_structure(
            highs, lows, closes, volumes
        )

        if in_range and range_type == RangeType.COMPRESSION:
            warnings.append("Compression range detected - breakout imminent")

        # 6. Detect FVGs
        active_fvgs = []
        if len(highs) >= 3:
            for i in range(2, len(highs)):
                fvg = self.detect_fvg(i, highs, lows, closes)
                if fvg:
                    self._fvgs.append(fvg)

        # Check which FVGs are still active (not filled)
        if len(highs) > 0:
            for fvg in self._fvgs[-20:]:  # Keep recent 20
                if not fvg.filled:
                    filled = self.check_fvg_filled(fvg, highs[-1], lows[-1])
                    if filled:
                        fvg.filled = True
                    else:
                        active_fvgs.append(fvg)

        # 7. Calculate structural momentum
        time_elapsed = 0
        if last_event:
            time_elapsed = (int(time.time() * 1000) - last_event.timestamp) / 1000.0

        structural_momentum = self.calculate_structural_momentum(
            last_event, closes[-1] if closes else 0, time_elapsed
        )

        # 8. Multi-timeframe alignment
        tf_alignment = self.analyze_mtf_alignment(htf_trend, ltf_trend)

        if tf_alignment == TimeframeAlignment.COUNTER_TREND:
            warnings.append("Counter-trend on LTF - scalp only or wait for HTF alignment")

        # 9. Calculate structure confidence
        structure_confidence = self.get_structure_confidence(
            ltf_trend, in_range, last_event, tf_alignment, structural_momentum
        )

        # 10. Add warnings based on CHoCH
        if last_choch and not last_bos:
            warnings.append("CHoCH detected - trend may be reversing")
        elif last_choch and last_bos and last_choch.timestamp > last_bos.timestamp:
            warnings.append("Recent CHoCH after BOS - trend weakening")

        return MarketStructureState(
            trend_direction=ltf_trend,
            recent_swings=swing_highs[-5:] + swing_lows[-5:],
            last_event=last_event,
            last_bos=last_bos,
            last_choch=last_choch,
            in_range=in_range,
            range_type=range_type,
            range_high=range_high,
            range_low=range_low,
            range_tightness=tightness,
            active_fvgs=active_fvgs,
            structural_momentum=structural_momentum,
            htf_trend=htf_trend,
            ltf_trend=ltf_trend,
            tf_alignment=tf_alignment,
            structure_confidence=structure_confidence,
            warnings=warnings,
        )


# Convenience functions


def get_allowed_trade_direction(structure: MarketStructureState) -> Optional[str]:
    """
    Determine allowed trade direction based on structure.

    Returns:
        "long", "short", "both" (range), or None (no trade)
    """
    # If structure not confident, no trade
    if structure.structure_confidence < 40:
        return None

    # Range structure: mean reversion (both directions)
    if structure.in_range:
        return "both"

    # Trend structure with MTF alignment
    if structure.trend_direction == TrendDirection.UPTREND:
        # Only allow long if no recent CHoCH or CHoCH is old
        if structure.last_choch:
            if structure.last_bos and structure.last_bos.timestamp > structure.last_choch.timestamp:
                return "long"  # BOS more recent than CHoCH
            else:
                return None  # CHoCH warning active
        return "long"

    elif structure.trend_direction == TrendDirection.DOWNTREND:
        # Only allow short if no recent CHoCH
        if structure.last_choch:
            if structure.last_bos and structure.last_bos.timestamp > structure.last_choch.timestamp:
                return "short"  # BOS more recent than CHoCH
            else:
                return None  # CHoCH warning active
        return "short"

    return None


def structure_veto_signal(
    structure: MarketStructureState, signal_direction: str, signal_confidence: float
) -> Tuple[bool, Optional[str]]:
    """
    HARD VETO: Structure can veto any signal.

    Args:
        structure: Current market structure state
        signal_direction: "long" or "short"
        signal_confidence: Signal confidence (0-100)

    Returns:
        (vetoed, reason)
        If vetoed=True, DO NOT take the trade regardless of indicators.
    """
    allowed = get_allowed_trade_direction(structure)

    # No trade allowed by structure
    if allowed is None:
        return True, "Structure unclear - no trade allowed"

    # Check direction
    if allowed == "both":
        # Range: allow both but warn
        if signal_confidence < 60:
            return True, "Range market - need high confidence for mean reversion"
        return False, None

    if allowed == "long" and signal_direction == "short":
        return True, f"Structure is {structure.trend_direction.value} - SHORT vetoed"

    if allowed == "short" and signal_direction == "long":
        return True, f"Structure is {structure.trend_direction.value} - LONG vetoed"

    # Counter-trend trades need very high confidence
    if structure.tf_alignment == TimeframeAlignment.COUNTER_TREND:
        if signal_confidence < 75:
            return True, "Counter-trend setup - need very high confidence (75+)"

    # CHoCH warning: reduce position or avoid
    if structure.last_choch and structure.last_choch.event_type == StructureEvent.CHOCH:
        if (
            structure.last_bos is None
            or structure.last_choch.timestamp > structure.last_bos.timestamp
        ):
            if signal_confidence < 70:
                return True, "Recent CHoCH (trend reversal warning) - need high confidence"

    # Rejected structure break
    if structure.last_event and structure.last_event.accepted is False:
        return True, "Last structure break rejected - avoid trades until new structure"

    return False, None
