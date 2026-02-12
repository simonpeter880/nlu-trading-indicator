"""
Market Structure Adapter for Continuous Analysis

Integrates market structure detection into the continuous streaming system.
Provides rolling window structure analysis with multiple timeframes.
"""

import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from market_structure import (
    MarketStructureDetector,
    MarketStructureState,
    TrendDirection,
    StructureEvent,
    TimeframeAlignment,
    StructuralMomentum,
    get_allowed_trade_direction,
    structure_veto_signal,
)
from .rolling_window import TradeWindow, WindowStats


@dataclass
class MarketStructureSignal:
    """Market structure signal for state machine."""

    # Structure state
    trend_direction: TrendDirection
    structure_confidence: float  # 0-100
    allowed_direction: Optional[str]  # "long", "short", "both", None

    # Events
    has_bos: bool
    has_choch: bool
    bos_accepted: Optional[bool]
    in_range: bool

    # Multi-timeframe
    htf_trend: TrendDirection
    ltf_trend: TrendDirection
    tf_alignment: TimeframeAlignment

    # Momentum
    structural_momentum: StructuralMomentum

    # FVG
    active_fvg_count: int
    near_fvg: bool  # Price near an FVG

    # Warnings
    warnings: List[str]

    # Raw state (for detailed analysis)
    full_state: MarketStructureState

    @property
    def score(self) -> float:
        """
        Structure score (-1 to +1).

        Positive: Bullish structure
        Negative: Bearish structure
        Near zero: Range or unclear
        """
        if self.trend_direction == TrendDirection.RANGE:
            return 0.0

        base_score = 0.0

        # Direction
        if self.trend_direction == TrendDirection.UPTREND:
            base_score = 0.5
        elif self.trend_direction == TrendDirection.DOWNTREND:
            base_score = -0.5

        # Confidence modifier
        confidence_mult = self.structure_confidence / 100.0
        base_score *= confidence_mult

        # BOS boost
        if self.has_bos and self.bos_accepted is True:
            base_score *= 1.3
        elif self.has_bos and self.bos_accepted is False:
            base_score *= 0.5

        # CHoCH penalty
        if self.has_choch:
            base_score *= 0.6

        # MTF alignment boost
        if self.tf_alignment == TimeframeAlignment.FULL_ALIGN:
            base_score *= 1.2
        elif self.tf_alignment == TimeframeAlignment.COUNTER_TREND:
            base_score *= 0.4

        # Structural momentum
        if self.structural_momentum == StructuralMomentum.FAST:
            base_score *= 1.15
        elif self.structural_momentum == StructuralMomentum.STALLED:
            base_score *= 0.7

        return max(-1.0, min(1.0, base_score))


class MarketStructureAdapter:
    """
    Adapter to integrate market structure into continuous analysis.

    Manages multiple timeframe structure detection using rolling windows.
    """

    def __init__(
        self,
        # LTF (lower timeframe) settings
        ltf_window_seconds: int = 180,  # 3 minutes for LTF
        ltf_candle_seconds: int = 15,   # 15s candles

        # HTF (higher timeframe) settings
        htf_window_seconds: int = 3600,  # 1 hour for HTF
        htf_candle_seconds: int = 60,    # 1m candles

        # Structure detector settings
        swing_lookback: int = 5,
        bos_acceptance_bars: int = 3,
        bos_acceptance_volume: float = 1.2,
        followthrough_time_max: float = 300,  # 5 minutes
        followthrough_distance_pct: float = 0.5,
        range_threshold_pct: float = 2.0,
        compression_periods: int = 10,
        fvg_min_gap_pct: float = 0.1,
    ):
        self.ltf_window_seconds = ltf_window_seconds
        self.ltf_candle_seconds = ltf_candle_seconds
        self.htf_window_seconds = htf_window_seconds
        self.htf_candle_seconds = htf_candle_seconds

        # Structure detectors (one per timeframe)
        self.ltf_detector = MarketStructureDetector(
            swing_lookback=swing_lookback,
            bos_acceptance_bars=bos_acceptance_bars,
            bos_acceptance_volume=bos_acceptance_volume,
            followthrough_time_max=followthrough_time_max,
            followthrough_distance_pct=followthrough_distance_pct,
            range_threshold_pct=range_threshold_pct,
            compression_periods=compression_periods,
            fvg_min_gap_pct=fvg_min_gap_pct,
        )

        self.htf_detector = MarketStructureDetector(
            swing_lookback=swing_lookback,
            bos_acceptance_bars=bos_acceptance_bars,
            bos_acceptance_volume=bos_acceptance_volume,
            followthrough_time_max=followthrough_time_max * 2,  # HTF needs more time
            followthrough_distance_pct=followthrough_distance_pct,
            range_threshold_pct=range_threshold_pct * 1.5,  # HTF ranges are larger
            compression_periods=compression_periods,
            fvg_min_gap_pct=fvg_min_gap_pct,
        )

        # State
        self._last_ltf_state: Optional[MarketStructureState] = None
        self._last_htf_state: Optional[MarketStructureState] = None
        self._last_signal: Optional[MarketStructureSignal] = None

    def _build_candles(
        self,
        window: TradeWindow,
        candle_seconds: int
    ) -> Dict[str, List[float]]:
        """
        Build OHLCV candles from trade window.

        Returns:
            Dict with: opens, highs, lows, closes, volumes, timestamps
        """
        if not window.prices:
            return {
                'opens': [],
                'highs': [],
                'lows': [],
                'closes': [],
                'volumes': [],
                'timestamps': []
            }

        # Get all trades
        trades_count = len(window.prices)

        # Build candles by grouping trades
        candles = []
        candle_ms = candle_seconds * 1000

        # Find start time
        start_time = window.start_time
        current_time = window.start_time

        i = 0
        while i < trades_count:
            candle_start = current_time
            candle_end = candle_start + candle_ms

            # Collect trades in this candle
            candle_prices = []
            candle_volumes = []

            while i < trades_count:
                # Estimate timestamp for this trade
                # (approximate: trades are evenly distributed in window)
                trade_time = start_time + (i / trades_count) * (window.end_time - start_time)

                if trade_time >= candle_end:
                    break

                candle_prices.append(window.prices[i])
                candle_volumes.append(window.volumes[i])
                i += 1

            if candle_prices:
                candles.append({
                    'timestamp': int(candle_start),
                    'open': candle_prices[0],
                    'high': max(candle_prices),
                    'low': min(candle_prices),
                    'close': candle_prices[-1],
                    'volume': sum(candle_volumes)
                })

            current_time = candle_end

        # Extract OHLCV
        if not candles:
            return {
                'opens': [],
                'highs': [],
                'lows': [],
                'closes': [],
                'volumes': [],
                'timestamps': []
            }

        return {
            'opens': [c['open'] for c in candles],
            'highs': [c['high'] for c in candles],
            'lows': [c['low'] for c in candles],
            'closes': [c['close'] for c in candles],
            'volumes': [c['volume'] for c in candles],
            'timestamps': [c['timestamp'] for c in candles],
        }

    def compute(
        self,
        ltf_window: TradeWindow,
        htf_window: TradeWindow
    ) -> Optional[MarketStructureSignal]:
        """
        Compute market structure from LTF and HTF windows.

        Args:
            ltf_window: Lower timeframe window (e.g., 3 minutes)
            htf_window: Higher timeframe window (e.g., 1 hour)

        Returns:
            MarketStructureSignal with complete structure state
        """
        # Build candles for each timeframe
        ltf_candles = self._build_candles(ltf_window, self.ltf_candle_seconds)
        htf_candles = self._build_candles(htf_window, self.htf_candle_seconds)

        # Need minimum candles
        if len(ltf_candles['closes']) < 10 or len(htf_candles['closes']) < 10:
            return None

        # Analyze HTF structure first (context)
        htf_state = self.htf_detector.analyze(
            highs=htf_candles['highs'],
            lows=htf_candles['lows'],
            closes=htf_candles['closes'],
            volumes=htf_candles['volumes'],
            timestamps=htf_candles['timestamps']
        )
        self._last_htf_state = htf_state

        # Analyze LTF structure with HTF context
        ltf_state = self.ltf_detector.analyze(
            highs=ltf_candles['highs'],
            lows=ltf_candles['lows'],
            closes=ltf_candles['closes'],
            volumes=ltf_candles['volumes'],
            timestamps=ltf_candles['timestamps'],
            htf_trend=htf_state.trend_direction
        )
        self._last_ltf_state = ltf_state

        # Build signal
        has_bos = ltf_state.last_bos is not None
        has_choch = ltf_state.last_choch is not None
        bos_accepted = None
        if has_bos:
            bos_accepted = ltf_state.last_bos.accepted

        # Check if near FVG
        near_fvg = False
        if ltf_state.active_fvgs and ltf_candles['closes']:
            current_price = ltf_candles['closes'][-1]
            for fvg in ltf_state.active_fvgs:
                # Check if price within 0.5% of FVG
                distance_to_top = abs(current_price - fvg.price_top) / current_price * 100
                distance_to_bottom = abs(current_price - fvg.price_bottom) / current_price * 100
                if distance_to_top < 0.5 or distance_to_bottom < 0.5:
                    near_fvg = True
                    break

        # Get allowed direction
        allowed_direction = get_allowed_trade_direction(ltf_state)

        signal = MarketStructureSignal(
            trend_direction=ltf_state.ltf_trend,
            structure_confidence=ltf_state.structure_confidence,
            allowed_direction=allowed_direction,
            has_bos=has_bos,
            has_choch=has_choch,
            bos_accepted=bos_accepted,
            in_range=ltf_state.in_range,
            htf_trend=ltf_state.htf_trend,
            ltf_trend=ltf_state.ltf_trend,
            tf_alignment=ltf_state.tf_alignment,
            structural_momentum=ltf_state.structural_momentum,
            active_fvg_count=len(ltf_state.active_fvgs),
            near_fvg=near_fvg,
            warnings=ltf_state.warnings,
            full_state=ltf_state
        )

        self._last_signal = signal
        return signal

    def get_latest_signal(self) -> Optional[MarketStructureSignal]:
        """Get the most recent structure signal."""
        return self._last_signal

    def veto_trade_signal(
        self,
        signal_direction: str,
        signal_confidence: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if structure vetos a trade signal.

        Args:
            signal_direction: "long" or "short"
            signal_confidence: Signal confidence (0-100)

        Returns:
            (vetoed, reason)
        """
        if self._last_ltf_state is None:
            return True, "No structure data available"

        return structure_veto_signal(
            self._last_ltf_state,
            signal_direction,
            signal_confidence
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current structure state."""
        if self._last_signal is None:
            return {
                'available': False,
                'trend': 'unknown',
                'confidence': 0,
                'score': 0.0
            }

        return {
            'available': True,
            'trend': self._last_signal.trend_direction.value,
            'ltf_trend': self._last_signal.ltf_trend.value,
            'htf_trend': self._last_signal.htf_trend.value,
            'confidence': self._last_signal.structure_confidence,
            'score': self._last_signal.score,
            'allowed_direction': self._last_signal.allowed_direction,
            'in_range': self._last_signal.in_range,
            'has_bos': self._last_signal.has_bos,
            'has_choch': self._last_signal.has_choch,
            'bos_accepted': self._last_signal.bos_accepted,
            'tf_alignment': self._last_signal.tf_alignment.value,
            'structural_momentum': self._last_signal.structural_momentum.value,
            'active_fvgs': self._last_signal.active_fvg_count,
            'near_fvg': self._last_signal.near_fvg,
            'warnings': self._last_signal.warnings,
        }

    def get_full_result(self) -> Optional[MarketStructureState]:
        """Get full LTF structure state for detailed analysis."""
        return self._last_ltf_state
