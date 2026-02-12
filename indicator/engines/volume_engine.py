"""
Institutional-Grade Volume Engine
"Who initiated, who absorbed, and who is trapped?"

UPGRADES from basic volume analysis:
1. Volume Delta - Who was aggressive (buyers lifting vs sellers hitting)
2. Volume Acceleration - Is volume accelerating or decaying?
3. Multi-Timeframe Agreement - Is volume accepted across timeframes?
4. Exhaustion Detection - Is offense dying?

This transforms:
BEFORE: "Was the move real?"
AFTER:  "Who initiated, who absorbed, and who is trapped?"
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, TYPE_CHECKING
from enum import Enum
import math

from .signals import Signal
if TYPE_CHECKING:
    from .data_fetcher import AggTradeData
    from .precise_volume_delta import (
        PreciseVolumeDeltaResult,
        BarVolumeDelta,
        AbsorptionDetectionResult
    )
    from .indicator_config import IndicatorConfig

from .indicator_config import DEFAULT_CONFIG, safe_divide
from .calculations import average_last


class AggressionBias(Enum):
    """Who is aggressive - buyers or sellers."""
    STRONG_BUY = "strong_buy"      # Buyers lifting aggressively
    BUY = "buy"                     # Moderate buy aggression
    NEUTRAL = "neutral"             # Balanced
    SELL = "sell"                   # Moderate sell aggression
    STRONG_SELL = "strong_sell"    # Sellers hitting aggressively


class VolumeAcceleration(Enum):
    """Volume momentum state."""
    ACCELERATING = "accelerating"   # Volume increasing
    STEADY = "steady"               # Volume stable
    DECELERATING = "decelerating"   # Volume decreasing
    CLIMAX = "climax"               # Extreme volume spike (potential reversal)


class ExhaustionRisk(Enum):
    """Risk of move exhaustion."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class MTFAgreement(Enum):
    """Multi-timeframe volume agreement."""
    CONFIRMED = "confirmed"         # LTF + HTF agree
    STOP_RUN = "stop_run"          # LTF high, HTF low
    REACCUMULATION = "reaccumulation"  # LTF low, HTF high
    UNCLEAR = "unclear"


@dataclass
class VolumeDeltaResult:
    """Volume Delta analysis - Who is aggressive."""
    delta: float                    # buy_volume - sell_volume
    delta_percent: float            # As percentage of total
    aggression_bias: AggressionBias
    strength: float                 # 0-100
    cumulative_delta: float         # Running total
    delta_divergence: bool          # Delta diverging from price?
    description: str
    interpretation: str


@dataclass
class AccelerationResult:
    """Volume acceleration/deceleration analysis."""
    acceleration: VolumeAcceleration
    rate: float                     # Current / Previous ratio
    momentum: float                 # First derivative
    is_climax: bool                 # Potential reversal signal
    bars_accelerating: int          # Consecutive bars
    description: str


@dataclass
class MTFAgreementResult:
    """Multi-timeframe volume agreement."""
    agreement: MTFAgreement
    ltf_volume_ratio: float         # LTF vs its average
    htf_volume_ratio: float         # HTF vs its average
    is_accepted: bool               # Volume accepted across TFs
    confidence: float
    description: str


@dataclass
class ExhaustionResult:
    """Volume exhaustion detection."""
    risk: ExhaustionRisk
    signals_present: List[str]      # Which exhaustion signals triggered
    body_shrinking: bool            # Candle bodies getting smaller
    volume_declining: bool          # Volume falling
    failed_continuation: bool       # Price failed to continue
    oi_stagnant: bool              # OI not growing
    confidence: float
    description: str
    action: str


@dataclass
class VolumeEngineResult:
    """Complete institutional-grade volume analysis."""
    # Core question answers
    who_initiated: str              # Buyers, Sellers, or Unclear
    who_absorbed: str               # Bids, Asks, or None
    who_is_trapped: str             # Longs, Shorts, or None

    # Components
    delta: VolumeDeltaResult
    acceleration: AccelerationResult
    mtf_agreement: Optional[MTFAgreementResult]
    exhaustion: ExhaustionResult

    # Overall assessment
    volume_quality: str             # 'institutional', 'retail', 'mixed'
    confidence: float
    signal: Signal
    summary: str


class InstitutionalVolumeEngine:
    """
    Institutional-grade volume analysis engine.

    Answers the critical questions:
    1. Who initiated the move?
    2. Who absorbed the pressure?
    3. Who is trapped?
    """

    def __init__(self, config: Optional['IndicatorConfig'] = None):
        """
        Initialize with optional config.

        Args:
            config: IndicatorConfig instance. If None, uses DEFAULT_CONFIG.
        """
        self.config = config or DEFAULT_CONFIG

    # Properties for backward compatibility (read from config)
    @property
    def DELTA_STRONG_THRESHOLD(self) -> float:
        """Strong delta threshold percentage."""
        return self.config.delta.strong_threshold_pct

    @property
    def DELTA_WEAK_THRESHOLD(self) -> float:
        """Weak delta threshold percentage."""
        return self.config.delta.weak_threshold_pct

    @property
    def ACCELERATION_THRESHOLD(self) -> float:
        """Volume acceleration ratio threshold."""
        return self.config.acceleration.accelerating

    @property
    def DECELERATION_THRESHOLD(self) -> float:
        """Volume deceleration ratio threshold."""
        return self.config.acceleration.decelerating

    @property
    def CLIMAX_THRESHOLD(self) -> float:
        """Climax volume ratio threshold."""
        return self.config.acceleration.climax

    @property
    def MTF_HIGH_THRESHOLD(self) -> float:
        """High multi-timeframe volume ratio threshold."""
        return self.config.mtf.high

    @property
    def MTF_LOW_THRESHOLD(self) -> float:
        """Low multi-timeframe volume ratio threshold."""
        return self.config.mtf.low

    @property
    def BODY_SHRINK_THRESHOLD(self) -> float:
        """Body shrink ratio threshold for exhaustion."""
        return self.config.exhaustion.body_shrink

    def _validate_ohlcv(
        self,
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
        min_length: int = 2
    ) -> None:
        """
        Validate OHLCV data arrays.

        Args:
            opens: Open prices
            highs: High prices
            lows: Low prices
            closes: Close prices
            volumes: Volume data
            min_length: Minimum required data points

        Raises:
            ValueError: If validation fails
        """
        arrays = [opens, highs, lows, closes, volumes]
        names = ['opens', 'highs', 'lows', 'closes', 'volumes']

        # Check all arrays have same length
        lengths = [len(arr) for arr in arrays]
        if len(set(lengths)) > 1:
            length_info = ", ".join(f"{n}={l}" for n, l in zip(names, lengths))
            raise ValueError(f"All OHLCV arrays must have equal length: {length_info}")

        # Check minimum length
        if lengths[0] < min_length:
            raise ValueError(f"Need at least {min_length} data points, got {lengths[0]}")

        # Check for negative volumes
        if any(v < 0 for v in volumes):
            raise ValueError("Volumes cannot be negative")

        # Check OHLC consistency (high >= low)
        for i, (h, l) in enumerate(zip(highs, lows)):
            if h < l:
                raise ValueError(f"Invalid OHLC at index {i}: high ({h}) < low ({l})")

    def calculate_volume_delta(
        self,
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float]
    ) -> VolumeDeltaResult:
        """
        Calculate Volume Delta - approximation without order flow data.

        Without tape data, we approximate using candle structure:
        - Up candles with close near high = buy aggression
        - Down candles with close near low = sell aggression

        More sophisticated: Use (close - open) / (high - low) * volume

        Raises:
            ValueError: If OHLCV data is invalid
        """
        # Validate input data
        self._validate_ohlcv(opens, highs, lows, closes, volumes, min_length=2)

        if len(closes) < 2:
            return VolumeDeltaResult(
                delta=0, delta_percent=0,
                aggression_bias=AggressionBias.NEUTRAL,
                strength=50, cumulative_delta=0,
                delta_divergence=False,
                description="Insufficient data",
                interpretation="Need more bars for delta calculation"
            )

        deltas = []
        cumulative = 0

        for i in range(len(closes)):
            o, h, l, c, v = opens[i], highs[i], lows[i], closes[i], volumes[i]

            # Calculate bar range
            bar_range = h - l
            if bar_range == 0:
                bar_delta = 0
            else:
                # Position of close within range: 1 = at high, 0 = at low
                close_position = (c - l) / bar_range

                # Delta approximation:
                # close_position > 0.5 = more buy aggression
                # close_position < 0.5 = more sell aggression
                # Scale: -1 (full sell) to +1 (full buy)
                aggression_factor = (close_position - 0.5) * 2

                # Weight by volume
                bar_delta = aggression_factor * v

            deltas.append(bar_delta)
            cumulative += bar_delta

        # Recent delta (last 5 bars)
        recent_deltas = deltas[-5:] if len(deltas) >= 5 else deltas
        recent_delta = sum(recent_deltas)
        recent_volume = sum(volumes[-5:]) if len(volumes) >= 5 else sum(volumes)

        # Delta as percentage of volume
        delta_percent = safe_divide(recent_delta, recent_volume, default=0.0) * 100

        # Determine aggression bias
        cfg = self.config.delta
        if delta_percent > self.DELTA_STRONG_THRESHOLD:
            bias = AggressionBias.STRONG_BUY
            strength = min(cfg.strong_max_strength, cfg.strong_base_strength + (delta_percent - cfg.strong_threshold_pct) * cfg.strong_strength_multiplier)
        elif delta_percent > self.DELTA_WEAK_THRESHOLD:
            bias = AggressionBias.BUY
            strength = cfg.weak_base_strength + (delta_percent - cfg.weak_threshold_pct) * cfg.weak_strength_multiplier
        elif delta_percent < -self.DELTA_STRONG_THRESHOLD:
            bias = AggressionBias.STRONG_SELL
            strength = min(cfg.strong_max_strength, cfg.strong_base_strength + (abs(delta_percent) - cfg.strong_threshold_pct) * cfg.strong_strength_multiplier)
        elif delta_percent < -self.DELTA_WEAK_THRESHOLD:
            bias = AggressionBias.SELL
            strength = cfg.weak_base_strength + (abs(delta_percent) - cfg.weak_threshold_pct) * cfg.weak_strength_multiplier
        else:
            bias = AggressionBias.NEUTRAL
            strength = cfg.neutral_base_strength + abs(delta_percent) * cfg.neutral_strength_multiplier

        # Check for delta divergence (price going one way, delta the other)
        price_change = closes[-1] - closes[-5] if len(closes) >= 5 else closes[-1] - closes[0]
        delta_divergence = (price_change > 0 and recent_delta < 0) or (price_change < 0 and recent_delta > 0)

        # Interpretation based on delta + price
        if delta_divergence:
            if price_change > 0 and recent_delta < 0:
                interp = "HIDDEN DISTRIBUTION - Price up but sellers aggressive"
            else:
                interp = "HIDDEN ACCUMULATION - Price down but buyers aggressive"
        else:
            if bias in [AggressionBias.STRONG_BUY, AggressionBias.BUY]:
                interp = "GENUINE BUYING - Buyers lifting offers aggressively"
            elif bias in [AggressionBias.STRONG_SELL, AggressionBias.SELL]:
                interp = "GENUINE SELLING - Sellers hitting bids aggressively"
            else:
                interp = "BALANCED - No clear aggressor"

        desc = f"Delta: {delta_percent:+.1f}% | Cumulative: {cumulative:+,.0f}"

        return VolumeDeltaResult(
            delta=recent_delta,
            delta_percent=delta_percent,
            aggression_bias=bias,
            strength=strength,
            cumulative_delta=cumulative,
            delta_divergence=delta_divergence,
            description=desc,
            interpretation=interp
        )

    def analyze_acceleration(
        self,
        volumes: List[float],
        lookback: int = 10
    ) -> AccelerationResult:
        """
        Analyze volume acceleration/deceleration.

        Markets move on CHANGE, not level.
        - Rising acceleration = initiative (trend likely to continue)
        - Falling acceleration = weak continuation
        - Climax = potential reversal
        """
        if len(volumes) < 3:
            return AccelerationResult(
                acceleration=VolumeAcceleration.STEADY,
                rate=1.0, momentum=0, is_climax=False,
                bars_accelerating=0,
                description="Insufficient data"
            )

        # Calculate volume ratios (current / previous)
        ratios = []
        for i in range(1, min(lookback, len(volumes))):
            if volumes[-(i+1)] > 0:
                ratios.append(volumes[-i] / volumes[-(i+1)])
            else:
                ratios.append(1.0)

        current_ratio = ratios[0] if ratios else 1.0

        # Calculate average volume for climax detection
        avg_vol = average_last(volumes, lookback, default=0.0)
        current_vol = volumes[-1]
        vol_vs_avg = safe_divide(current_vol, avg_vol, default=1.0)

        # Count consecutive accelerating/decelerating bars
        accel_cfg = self.config.acceleration
        bars_accel = 0
        bars_decel = 0
        for r in ratios:
            if r > accel_cfg.accel_bar_threshold:
                bars_accel += 1
                bars_decel = 0
            elif r < accel_cfg.decel_bar_threshold:
                bars_decel += 1
                bars_accel = 0
            else:
                break

        # Calculate momentum (second derivative approximation)
        if len(ratios) >= 2:
            momentum = ratios[0] - ratios[1]
        else:
            momentum = 0

        # Determine state
        is_climax = vol_vs_avg >= self.CLIMAX_THRESHOLD

        if is_climax:
            state = VolumeAcceleration.CLIMAX
            desc = f"CLIMAX! Volume {vol_vs_avg:.1f}x average - Potential reversal"
        elif current_ratio >= self.ACCELERATION_THRESHOLD:
            state = VolumeAcceleration.ACCELERATING
            desc = f"ACCELERATING - Volume {current_ratio:.2f}x previous ({bars_accel} bars)"
        elif current_ratio <= self.DECELERATION_THRESHOLD:
            state = VolumeAcceleration.DECELERATING
            desc = f"DECELERATING - Volume {current_ratio:.2f}x previous ({bars_decel} bars down)"
        else:
            state = VolumeAcceleration.STEADY
            desc = f"Steady volume - {current_ratio:.2f}x previous"

        return AccelerationResult(
            acceleration=state,
            rate=current_ratio,
            momentum=momentum,
            is_climax=is_climax,
            bars_accelerating=bars_accel if state == VolumeAcceleration.ACCELERATING else -bars_decel,
            description=desc
        )

    def analyze_mtf_agreement(
        self,
        ltf_volumes: List[float],
        htf_volumes: List[float],
        ltf_lookback: int = 20,
        htf_lookback: int = 20
    ) -> MTFAgreementResult:
        """
        Multi-timeframe volume agreement.

        Rule: LTF spike must align with HTF acceptance.

        ┌─────────────┬─────────────┬─────────────────────┐
        │ LTF Volume  │ HTF Volume  │ Verdict             │
        ├─────────────┼─────────────┼─────────────────────┤
        │ High        │ Low         │ STOP RUN            │
        │ High        │ High        │ ACCEPTED MOVE       │
        │ Low         │ High        │ REACCUMULATION      │
        │ Low         │ Low         │ NO INTEREST         │
        └─────────────┴─────────────┴─────────────────────┘
        """
        if not ltf_volumes or not htf_volumes:
            return MTFAgreementResult(
                agreement=MTFAgreement.UNCLEAR,
                ltf_volume_ratio=1.0,
                htf_volume_ratio=1.0,
                is_accepted=False,
                confidence=30,
                description="Insufficient MTF data"
            )

        # Calculate LTF ratio vs average
        ltf_avg = average_last(ltf_volumes, ltf_lookback, default=0.0)
        ltf_current = ltf_volumes[-1]
        ltf_ratio = safe_divide(ltf_current, ltf_avg, default=1.0)

        # Calculate HTF ratio vs average
        htf_avg = average_last(htf_volumes, htf_lookback, default=0.0)
        htf_current = htf_volumes[-1]
        htf_ratio = safe_divide(htf_current, htf_avg, default=1.0)

        # Determine agreement
        ltf_high = ltf_ratio >= self.MTF_HIGH_THRESHOLD
        ltf_low = ltf_ratio <= self.MTF_LOW_THRESHOLD
        htf_high = htf_ratio >= self.MTF_HIGH_THRESHOLD
        htf_low = htf_ratio <= self.MTF_LOW_THRESHOLD

        if ltf_high and htf_high:
            agreement = MTFAgreement.CONFIRMED
            is_accepted = True
            confidence = min(90, 60 + (ltf_ratio + htf_ratio - 2) * 10)
            desc = f"CONFIRMED - LTF {ltf_ratio:.1f}x + HTF {htf_ratio:.1f}x = Accepted move"
        elif ltf_high and htf_low:
            agreement = MTFAgreement.STOP_RUN
            is_accepted = False
            confidence = min(85, 55 + abs(ltf_ratio - htf_ratio) * 10)
            desc = f"STOP RUN - LTF spike ({ltf_ratio:.1f}x) but HTF quiet ({htf_ratio:.1f}x)"
        elif ltf_low and htf_high:
            agreement = MTFAgreement.REACCUMULATION
            is_accepted = True
            confidence = 65
            desc = f"REACCUMULATION - LTF quiet ({ltf_ratio:.1f}x) but HTF absorbing ({htf_ratio:.1f}x)"
        else:
            agreement = MTFAgreement.UNCLEAR
            is_accepted = False
            confidence = 40
            desc = f"UNCLEAR - LTF {ltf_ratio:.1f}x, HTF {htf_ratio:.1f}x - No strong signal"

        return MTFAgreementResult(
            agreement=agreement,
            ltf_volume_ratio=ltf_ratio,
            htf_volume_ratio=htf_ratio,
            is_accepted=is_accepted,
            confidence=confidence,
            description=desc
        )

    def detect_exhaustion(
        self,
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
        oi_change_percent: Optional[float] = None,
        lookback: int = 5
    ) -> ExhaustionResult:
        """
        Detect volume exhaustion - offense dying.

        Different from absorption:
        - Absorption = DEFENSE (someone defending a level)
        - Exhaustion = OFFENSE DYING (move running out of steam)

        Detect via:
        1. High volume
        2. Shrinking candle bodies
        3. Failed continuation
        4. OI stagnation or drop
        """
        if len(closes) < lookback + 1:
            return ExhaustionResult(
                risk=ExhaustionRisk.LOW,
                signals_present=[],
                body_shrinking=False,
                volume_declining=False,
                failed_continuation=False,
                oi_stagnant=False,
                confidence=30,
                description="Insufficient data for exhaustion detection",
                action="Need more bars"
            )

        signals = []
        confidence = 30

        # 1. Check for high volume (was there a significant move?)
        exh_cfg = self.config.exhaustion
        if len(volumes) >= lookback * 2:
            avg_vol = average_last(volumes[:-lookback], lookback, default=0.0)
        else:
            avg_vol = average_last(volumes, len(volumes), default=0.0)
        recent_max_vol = max(volumes[-lookback:])
        had_volume_spike = recent_max_vol > avg_vol * exh_cfg.volume_spike

        # 2. Check for shrinking candle bodies
        bodies = [abs(closes[i] - opens[i]) for i in range(-lookback, 0)]
        if len(bodies) >= 3:
            early_body = sum(bodies[:2]) / 2
            late_body = sum(bodies[-2:]) / 2
            body_shrinking = late_body < early_body * exh_cfg.body_shrink
        else:
            body_shrinking = False

        if body_shrinking and had_volume_spike:
            signals.append("Shrinking bodies despite volume")
            confidence += 20

        # 3. Check for volume declining
        vol_early = sum(volumes[-lookback:-lookback//2]) / (lookback//2) if lookback > 1 else volumes[-lookback]
        vol_late = sum(volumes[-lookback//2:]) / (lookback//2 + lookback%2)
        volume_declining = vol_late < vol_early * exh_cfg.volume_decline

        if volume_declining:
            signals.append("Volume declining")
            confidence += 15

        # 4. Check for failed continuation
        price_direction = closes[-lookback] - closes[-lookback-1] if len(closes) > lookback else 0
        recent_progress = closes[-1] - closes[-lookback]

        # Failed continuation: initial move but couldn't continue
        failed_continuation = False
        if abs(price_direction) > 0:
            if price_direction > 0:
                # Was going up, check if it continued
                failed_continuation = recent_progress < price_direction * exh_cfg.continuation_failure
            else:
                # Was going down, check if it continued
                failed_continuation = recent_progress > price_direction * exh_cfg.continuation_failure

        if failed_continuation and had_volume_spike:
            signals.append("Failed continuation")
            confidence += 20

        # 5. Check OI stagnation
        oi_stagnant = False
        if oi_change_percent is not None:
            oi_stagnant = abs(oi_change_percent) < exh_cfg.oi_stagnant_pct  # Less than threshold change
            if oi_stagnant and had_volume_spike:
                signals.append("OI stagnant despite volume")
                confidence += 15
            elif oi_change_percent < exh_cfg.oi_dropping_pct:
                signals.append("OI dropping - positions closing")
                confidence += 10

        # Determine risk level
        num_signals = len(signals)
        if num_signals >= 4 or (num_signals >= 3 and had_volume_spike):
            risk = ExhaustionRisk.EXTREME
            action = "HIGH PROBABILITY REVERSAL - Take profits or prepare to fade"
        elif num_signals >= 3 or (num_signals >= 2 and had_volume_spike):
            risk = ExhaustionRisk.HIGH
            action = "Move likely exhausted - Reduce position or tighten stops"
        elif num_signals >= 2:
            risk = ExhaustionRisk.MEDIUM
            action = "Watch for reversal signs - Don't add to position"
        else:
            risk = ExhaustionRisk.LOW
            action = "No exhaustion detected - Trend may continue"

        desc = f"Exhaustion signals: {num_signals}/5"
        if signals:
            desc += f" | {', '.join(signals)}"

        return ExhaustionResult(
            risk=risk,
            signals_present=signals,
            body_shrinking=body_shrinking,
            volume_declining=volume_declining,
            failed_continuation=failed_continuation,
            oi_stagnant=oi_stagnant,
            confidence=min(95, confidence),
            description=desc,
            action=action
        )

    def full_analysis_with_precise_delta(
        self,
        agg_trades: List['AggTradeData'],
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
        bar_size_ms: int = 60000,
        htf_volumes: Optional[List[float]] = None,
        oi_change_percent: Optional[float] = None,
        window_start_ms: Optional[int] = None,
        window_end_ms: Optional[int] = None
    ) -> VolumeEngineResult:
        """
        Complete institutional-grade volume analysis using PRECISE delta from aggTrades.

        This is the premium version that uses real order flow data.

        Args:
            agg_trades: List of aggregated trades for precise delta calculation
            opens, highs, lows, closes, volumes: OHLCV data
            bar_size_ms: Bar size in milliseconds (should match OHLCV timeframe)
            htf_volumes: Higher timeframe volumes for MTF agreement
            oi_change_percent: OI change percentage for exhaustion detection
            window_start_ms: Optional analysis window start (ms)
            window_end_ms: Optional analysis window end (ms)

        Returns:
            VolumeEngineResult with precise delta calculations

        Raises:
            ValueError: If OHLCV data is invalid
        """
        # Validate OHLCV data
        self._validate_ohlcv(opens, highs, lows, closes, volumes, min_length=2)

        try:
            from precise_volume_delta import PreciseVolumeDeltaEngine
        except ImportError:
            # Fallback to approximation if precise module not available
            return self.full_analysis(opens, highs, lows, closes, volumes, htf_volumes, oi_change_percent)

        if bar_size_ms <= 0:
            raise ValueError("bar_size_ms must be positive")

        # Calculate precise delta
        precise_engine = PreciseVolumeDeltaEngine(config=self.config)
        bars = precise_engine.bucket_trades_to_bars(
            agg_trades,
            bar_size_ms,
            apply_filters=True,
            start_time_ms=window_start_ms,
            end_time_ms=window_end_ms
        )
        if not bars:
            return self.full_analysis(opens, highs, lows, closes, volumes, htf_volumes, oi_change_percent)
        precise_result = precise_engine.analyze_volume_delta(bars)

        # Map precise delta to our VolumeDeltaResult format
        # Convert PreciseAggressionBias to AggressionBias
        bias_mapping = {
            "strong_buy": AggressionBias.STRONG_BUY,
            "buy": AggressionBias.BUY,
            "neutral": AggressionBias.NEUTRAL,
            "sell": AggressionBias.SELL,
            "strong_sell": AggressionBias.STRONG_SELL,
        }
        aggression_bias = bias_mapping.get(precise_result.aggression_bias.value, AggressionBias.NEUTRAL)

        delta_result = VolumeDeltaResult(
            delta=precise_result.delta,
            delta_percent=precise_result.delta_ratio * 100,  # Convert to percentage
            aggression_bias=aggression_bias,
            strength=precise_result.confidence,
            cumulative_delta=precise_result.cvd,
            delta_divergence=precise_result.delta_divergence,
            description=precise_result.description,
            interpretation=precise_result.interpretation
        )

        # Use standard acceleration analysis
        acceleration = self.analyze_acceleration(volumes)

        # MTF analysis if HTF data available
        mtf = None
        if htf_volumes and len(htf_volumes) >= 5:
            mtf = self.analyze_mtf_agreement(volumes, htf_volumes)

        # Exhaustion detection
        exhaustion = self.detect_exhaustion(
            opens, highs, lows, closes, volumes, oi_change_percent
        )

        # Determine WHO INITIATED
        if aggression_bias in [AggressionBias.STRONG_BUY, AggressionBias.BUY]:
            who_initiated = "BUYERS"
        elif aggression_bias in [AggressionBias.STRONG_SELL, AggressionBias.SELL]:
            who_initiated = "SELLERS"
        else:
            who_initiated = "UNCLEAR"

        # Determine WHO ABSORBED (from delta divergence)
        if delta_result.delta_divergence:
            price_change = closes[-1] - closes[0] if len(closes) > 1 else 0
            if price_change > 0 and delta_result.delta < 0:
                who_absorbed = "ASKS (sellers absorbing)"
            elif price_change < 0 and delta_result.delta > 0:
                who_absorbed = "BIDS (buyers absorbing)"
            else:
                who_absorbed = "NONE DETECTED"
        else:
            who_absorbed = "NONE DETECTED"

        # Determine WHO IS TRAPPED
        who_trapped = "NONE DETECTED"
        if mtf and mtf.agreement == MTFAgreement.STOP_RUN:
            price_change = closes[-1] - closes[0] if len(closes) > 1 else 0
            if price_change > 0:
                who_trapped = "SHORTS (stop run up)"
            else:
                who_trapped = "LONGS (stop run down)"
        elif exhaustion.risk in [ExhaustionRisk.HIGH, ExhaustionRisk.EXTREME]:
            if who_initiated == "BUYERS":
                who_trapped = "LATE LONGS (exhaustion)"
            elif who_initiated == "SELLERS":
                who_trapped = "LATE SHORTS (exhaustion)"

        # Determine volume quality (use precise metrics)
        if mtf and mtf.is_accepted and delta_result.strength > 70:
            volume_quality = "institutional"
        elif mtf and mtf.agreement == MTFAgreement.STOP_RUN:
            volume_quality = "retail"
        elif abs(precise_result.delta_ratio) > 0.25:  # Strong precise delta
            volume_quality = "institutional"
        else:
            volume_quality = "mixed"

        # Calculate overall confidence (use precise confidence)
        confidence = (delta_result.strength + exhaustion.confidence) / 2
        if mtf:
            confidence = (confidence + mtf.confidence) / 2

        # Determine signal
        if exhaustion.risk == ExhaustionRisk.EXTREME:
            signal = Signal.CAUTION
        elif delta_result.delta_divergence:
            if delta_result.delta > 0:
                signal = Signal.BULLISH  # Hidden accumulation
            else:
                signal = Signal.BEARISH  # Hidden distribution
        elif aggression_bias in [AggressionBias.STRONG_BUY, AggressionBias.BUY]:
            signal = Signal.BULLISH if exhaustion.risk != ExhaustionRisk.HIGH else Signal.NEUTRAL
        elif aggression_bias in [AggressionBias.STRONG_SELL, AggressionBias.SELL]:
            signal = Signal.BEARISH if exhaustion.risk != ExhaustionRisk.HIGH else Signal.NEUTRAL
        else:
            signal = Signal.NEUTRAL

        # Generate summary
        parts = []
        parts.append(f"Initiated: {who_initiated}")
        parts.append(f"Absorbed: {who_absorbed}")
        if who_trapped != "NONE DETECTED":
            parts.append(f"Trapped: {who_trapped}")
        parts.append(f"Quality: {volume_quality}")

        summary = " | ".join(parts)

        return VolumeEngineResult(
            who_initiated=who_initiated,
            who_absorbed=who_absorbed,
            who_is_trapped=who_trapped,
            delta=delta_result,
            acceleration=acceleration,
            mtf_agreement=mtf,
            exhaustion=exhaustion,
            volume_quality=volume_quality,
            confidence=confidence,
            signal=signal,
            summary=summary
        )

    def full_analysis(
        self,
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
        htf_volumes: Optional[List[float]] = None,
        oi_change_percent: Optional[float] = None
    ) -> VolumeEngineResult:
        """
        Complete institutional-grade volume analysis.

        Answers:
        1. Who initiated the move?
        2. Who absorbed the pressure?
        3. Who is trapped?
        """
        # Run all analyses
        delta = self.calculate_volume_delta(opens, highs, lows, closes, volumes)
        acceleration = self.analyze_acceleration(volumes)

        # MTF analysis if HTF data available
        mtf = None
        if htf_volumes and len(htf_volumes) >= 5:
            mtf = self.analyze_mtf_agreement(volumes, htf_volumes)

        exhaustion = self.detect_exhaustion(
            opens, highs, lows, closes, volumes, oi_change_percent
        )

        # Determine WHO INITIATED
        if delta.aggression_bias in [AggressionBias.STRONG_BUY, AggressionBias.BUY]:
            who_initiated = "BUYERS"
        elif delta.aggression_bias in [AggressionBias.STRONG_SELL, AggressionBias.SELL]:
            who_initiated = "SELLERS"
        else:
            who_initiated = "UNCLEAR"

        # Determine WHO ABSORBED (from delta divergence)
        if delta.delta_divergence:
            price_change = closes[-1] - closes[0] if len(closes) > 1 else 0
            if price_change > 0 and delta.delta < 0:
                who_absorbed = "ASKS (sellers absorbing)"
            elif price_change < 0 and delta.delta > 0:
                who_absorbed = "BIDS (buyers absorbing)"
            else:
                who_absorbed = "NONE DETECTED"
        else:
            who_absorbed = "NONE DETECTED"

        # Determine WHO IS TRAPPED
        who_trapped = "NONE DETECTED"
        if mtf and mtf.agreement == MTFAgreement.STOP_RUN:
            # Stop run = someone got trapped
            price_change = closes[-1] - closes[0] if len(closes) > 1 else 0
            if price_change > 0:
                who_trapped = "SHORTS (stop run up)"
            else:
                who_trapped = "LONGS (stop run down)"
        elif exhaustion.risk in [ExhaustionRisk.HIGH, ExhaustionRisk.EXTREME]:
            # Exhaustion = late chasers trapped
            if who_initiated == "BUYERS":
                who_trapped = "LATE LONGS (exhaustion)"
            elif who_initiated == "SELLERS":
                who_trapped = "LATE SHORTS (exhaustion)"

        # Determine volume quality
        if mtf and mtf.is_accepted and delta.strength > 70:
            volume_quality = "institutional"
        elif mtf and mtf.agreement == MTFAgreement.STOP_RUN:
            volume_quality = "retail"
        else:
            volume_quality = "mixed"

        # Calculate overall confidence
        confidence = (delta.strength + exhaustion.confidence) / 2
        if mtf:
            confidence = (confidence + mtf.confidence) / 2

        # Determine signal
        if exhaustion.risk == ExhaustionRisk.EXTREME:
            signal = Signal.CAUTION
        elif delta.delta_divergence:
            # Divergence is contrarian
            if delta.delta > 0:
                signal = Signal.BULLISH  # Hidden accumulation
            else:
                signal = Signal.BEARISH  # Hidden distribution
        elif delta.aggression_bias in [AggressionBias.STRONG_BUY, AggressionBias.BUY]:
            signal = Signal.BULLISH if exhaustion.risk != ExhaustionRisk.HIGH else Signal.NEUTRAL
        elif delta.aggression_bias in [AggressionBias.STRONG_SELL, AggressionBias.SELL]:
            signal = Signal.BEARISH if exhaustion.risk != ExhaustionRisk.HIGH else Signal.NEUTRAL
        else:
            signal = Signal.NEUTRAL

        # Generate summary
        parts = []
        parts.append(f"Initiated: {who_initiated}")
        parts.append(f"Absorbed: {who_absorbed}")
        if who_trapped != "NONE DETECTED":
            parts.append(f"Trapped: {who_trapped}")
        parts.append(f"Quality: {volume_quality}")

        summary = " | ".join(parts)

        return VolumeEngineResult(
            who_initiated=who_initiated,
            who_absorbed=who_absorbed,
            who_is_trapped=who_trapped,
            delta=delta,
            acceleration=acceleration,
            mtf_agreement=mtf,
            exhaustion=exhaustion,
            volume_quality=volume_quality,
            confidence=confidence,
            signal=signal,
            summary=summary
        )


# Convenience functions
def get_aggression_bias(
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
    config: Optional['IndicatorConfig'] = None
) -> Tuple[AggressionBias, float]:
    """Quick check: Who is aggressive?"""
    engine = InstitutionalVolumeEngine(config)
    delta = engine.calculate_volume_delta(opens, highs, lows, closes, volumes)
    return delta.aggression_bias, delta.strength


def is_volume_exhausted(
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
    oi_change: Optional[float] = None,
    config: Optional['IndicatorConfig'] = None
) -> Tuple[bool, ExhaustionRisk]:
    """Quick check: Is the move exhausted?"""
    engine = InstitutionalVolumeEngine(config)
    result = engine.detect_exhaustion(opens, highs, lows, closes, volumes, oi_change)
    is_exhausted = result.risk in [ExhaustionRisk.HIGH, ExhaustionRisk.EXTREME]
    return is_exhausted, result.risk
