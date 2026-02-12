"""
Engine Adapters - Bridge between rolling windows and existing signal engines.

Wraps your existing sophisticated engines to consume continuous data
from rolling windows instead of discrete REST fetches.
"""

# Import existing engines
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .data_types import (
    BookSignal,
    DeltaSignal,
    FundingUpdate,
    OIFundingSignal,
    OIUpdate,
    OrderbookSnapshot,
    SignalDirection,
    TradeEvent,
    VolumeSignal,
)
from .ring_buffer import TimestampedRingBuffer
from .rolling_window import MultiTimeframeWindows, OrderbookHistory, TradeWindow

sys.path.insert(0, str(Path(__file__).parent.parent))

from funding_analysis import (
    AdvancedFundingAnalyzer,
    CrowdPosition,
    FundingAnalysisSummary,
    FundingZone,
)
from oi_analysis import AdvancedOIAnalyzer, OIAnalysisSummary, OIRegime
from orderbook_analysis import (
    AbsorptionSide,
    AdvancedOrderbookAnalyzer,
    ImbalanceDirection,
    OrderbookAnalysisSummary,
)
from orderbook_analysis import OrderbookSnapshot as AnalyzerOrderbookSnapshot
from signals import Signal
from unified_score import UnifiedScore, calculate_unified_score
from volume_analysis import AdvancedVolumeAnalyzer, VolumeAnalysisSummary
from volume_engine import (
    AggressionBias,
    ExhaustionRisk,
    InstitutionalVolumeEngine,
    VolumeAcceleration,
    VolumeEngineResult,
)


def _bias_to_direction(bias: AggressionBias) -> SignalDirection:
    """Convert AggressionBias to SignalDirection."""
    if bias in [AggressionBias.STRONG_BUY, AggressionBias.BUY]:
        return SignalDirection.BULLISH
    elif bias in [AggressionBias.STRONG_SELL, AggressionBias.SELL]:
        return SignalDirection.BEARISH
    return SignalDirection.NEUTRAL


def _signal_to_direction(signal: Signal) -> SignalDirection:
    """Convert Signal enum to SignalDirection."""
    if signal == Signal.BULLISH:
        return SignalDirection.BULLISH
    elif signal == Signal.BEARISH:
        return SignalDirection.BEARISH
    return SignalDirection.NEUTRAL


class VolumeEngineAdapter:
    """
    Adapter for InstitutionalVolumeEngine.

    Converts rolling window trade data into the format expected
    by the existing volume engine.
    """

    def __init__(self):
        self._engine = InstitutionalVolumeEngine()
        self._baseline_volume: Optional[float] = None
        self._baseline_window_seconds: int = 300  # 5 min baseline
        self._precise_min_trades: int = 20
        self._precise_min_coverage: float = 0.7
        self._precise_bar_target: int = 15
        # Store full analysis result for deep dive display
        self._latest_full_result: Optional[VolumeEngineResult] = None

    def set_baseline_volume(self, volume: float) -> None:
        """Set baseline volume for relative volume calculations."""
        self._baseline_volume = volume

    def compute_signal(
        self,
        trade_window: TradeWindow,
        oi_change_percent: Optional[float] = None,
        htf_volumes: Optional[List[float]] = None,
    ) -> Optional[VolumeSignal]:
        """
        Compute volume signal from trade window.

        Args:
            trade_window: Rolling window of trades
            oi_change_percent: OI change for exhaustion detection

        Returns:
            VolumeSignal or None if insufficient data
        """
        trades = trade_window.items()
        if len(trades) < 10:
            return None

        # Extract OHLCV-like data from trades
        # Group trades into synthetic bars (we'll use the raw trades directly)
        opens, highs, lows, closes, volumes = self._trades_to_ohlcv(trades)

        if len(closes) < 2:
            return None

        result, used_precise = self._run_engine(
            trade_window=trade_window,
            trades=trades,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            volumes=volumes,
            htf_volumes=htf_volumes,
            oi_change_percent=oi_change_percent,
        )

        # Store full result for deep dive display
        self._latest_full_result = result

        # Calculate relative volume
        total_volume = trade_window.total_volume
        if self._baseline_volume and self._baseline_volume > 0:
            relative_volume = total_volume / self._baseline_volume
        else:
            relative_volume = 1.0

        # Check for climax — but suppress in dead volume.
        # A "climax" at RV < 1.0 is noise from trade-count bar artifacts,
        # not a real institutional event. Allowing it fires exhaustion_long/short
        # and generates false reversal signals.
        raw_climax = (
            result.acceleration.acceleration == VolumeAcceleration.CLIMAX
            or result.exhaustion.risk in [ExhaustionRisk.HIGH, ExhaustionRisk.EXTREME]
        )
        is_climax = raw_climax and relative_volume >= 1.0

        # Reduce confidence when volume is dead — the engine's analysis
        # is based on bar patterns that aren't meaningful without participation.
        strength = result.confidence
        if relative_volume < 1.0:
            strength *= relative_volume

        return VolumeSignal(
            timestamp_ms=int(time.time() * 1000),
            total_volume=total_volume,
            buy_volume=trade_window.buy_volume,
            sell_volume=trade_window.sell_volume,
            delta=result.delta.delta if used_precise else trade_window.delta,
            delta_ratio=(
                (result.delta.delta_percent / 100.0) if used_precise else trade_window.delta_ratio
            ),
            relative_volume=relative_volume,
            direction=_bias_to_direction(result.delta.aggression_bias),
            strength=strength,
            is_climax=is_climax,
            window_seconds=trade_window.window_seconds,
        )

    def get_latest_full_result(self) -> Optional[VolumeEngineResult]:
        """Get the latest full volume engine analysis result."""
        return self._latest_full_result

    def _trades_to_ohlcv(
        self,
        trades: List[TradeEvent],
        bar_count: int = 20,
    ) -> tuple[List[float], List[float], List[float], List[float], List[float]]:
        """
        Convert trades to synthetic OHLCV bars.

        Groups trades into N bars for the engine's candle-based calculations.
        """
        if not trades:
            return [], [], [], [], []

        # Determine bar boundaries
        trades_per_bar = max(1, len(trades) // bar_count)

        opens, highs, lows, closes, volumes = [], [], [], [], []

        for i in range(0, len(trades), trades_per_bar):
            bar_trades = trades[i : i + trades_per_bar]
            if not bar_trades:
                continue

            prices = [t.price for t in bar_trades]
            vol = sum(t.quantity for t in bar_trades)

            opens.append(prices[0])
            highs.append(max(prices))
            lows.append(min(prices))
            closes.append(prices[-1])
            volumes.append(vol)

        return opens, highs, lows, closes, volumes

    def _trades_to_agg_trades(self, trades: List[TradeEvent]) -> List[Any]:
        """
        Convert TradeEvent list to AggTradeData-like objects.

        Uses lazy import to avoid heavy dependency at module load.
        """
        try:
            from data_fetcher import AggTradeData
        except Exception:
            return []

        agg_trades: List[AggTradeData] = []
        for trade in trades:
            agg_trades.append(
                AggTradeData(
                    agg_trade_id=trade.trade_id,
                    price=trade.price,
                    quantity=trade.quantity,
                    first_trade_id=trade.trade_id,
                    last_trade_id=trade.trade_id,
                    timestamp=trade.timestamp_ms,
                    is_buyer_maker=trade.is_buyer_maker,
                )
            )
        return agg_trades

    def _run_engine(
        self,
        trade_window: TradeWindow,
        trades: List[TradeEvent],
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
        htf_volumes: Optional[List[float]] = None,
        oi_change_percent: Optional[float] = None,
    ) -> tuple[VolumeEngineResult, bool]:
        """Run volume engine, preferring precise delta when data coverage is sufficient."""
        has_precise = (
            len(trades) >= self._precise_min_trades
            and trade_window.coverage >= self._precise_min_coverage
            and len(closes) >= 2
        )

        if has_precise:
            agg_trades = self._trades_to_agg_trades(trades)
            if agg_trades:
                bar_count = max(5, min(self._precise_bar_target, len(closes)))
                bar_size_ms = max(1000, int((trade_window.window_seconds * 1000) / bar_count))
                window_start = min(t.timestamp_ms for t in trades)
                window_end = max(t.timestamp_ms for t in trades)
                try:
                    return (
                        self._engine.full_analysis_with_precise_delta(
                            agg_trades=agg_trades,
                            opens=opens,
                            highs=highs,
                            lows=lows,
                            closes=closes,
                            volumes=volumes,
                            bar_size_ms=bar_size_ms,
                            htf_volumes=htf_volumes,
                            oi_change_percent=oi_change_percent,
                            window_start_ms=window_start,
                            window_end_ms=window_end,
                        ),
                        True,
                    )
                except Exception as e:
                    logger.debug(
                        f"Precise delta computation failed, falling back to candle approximation: {e}"
                    )
                    pass

        return (
            self._engine.full_analysis(
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                volumes=volumes,
                htf_volumes=htf_volumes,
                oi_change_percent=oi_change_percent,
            ),
            False,
        )

    def compute_from_multi_windows(
        self,
        windows: MultiTimeframeWindows,
        primary_window_seconds: int = 60,
        oi_change_percent: Optional[float] = None,
    ) -> Optional[VolumeSignal]:
        """
        Compute signal using primary window with MTF context.

        Args:
            windows: Multi-timeframe trade windows
            primary_window_seconds: Primary window to use for signal
            oi_change_percent: OI change percentage

        Returns:
            VolumeSignal from the primary window
        """
        primary = windows.get_window(primary_window_seconds)
        if primary is None:
            return None

        # Use longer window for baseline
        context_window = windows.get_window(300)  # 5 min
        if context_window and context_window.total_volume > 0:
            # Normalize to per-minute volume
            baseline = context_window.total_volume / 5
            self.set_baseline_volume(baseline * (primary_window_seconds / 60))

        htf_volumes = None
        htf_window_seconds = self._select_htf_window_seconds(
            windows,
            primary_window_seconds,
        )
        if htf_window_seconds:
            htf_window = windows.get_window(htf_window_seconds)
            if htf_window and len(htf_window) >= 10:
                _, _, _, _, htf_volumes = self._trades_to_ohlcv(
                    htf_window.items(),
                    bar_count=20,
                )

        return self.compute_signal(primary, oi_change_percent, htf_volumes)

    def _select_htf_window_seconds(
        self,
        windows: MultiTimeframeWindows,
        primary_window_seconds: int,
    ) -> Optional[int]:
        """Pick a higher timeframe window for MTF confirmation."""
        candidates = sorted([sec for sec in windows.windows.keys() if sec > primary_window_seconds])
        if not candidates:
            return None
        for sec in candidates:
            if sec >= primary_window_seconds * 3:
                return sec
        return candidates[-1]


class DeltaEngineAdapter:
    """
    Adapter for cumulative volume delta analysis.

    Tracks true CVD across the entire session (not just within windows)
    and detects divergences with price.
    """

    def __init__(self):
        # True cumulative CVD - persists across the session
        self._cumulative_cvd: float = 0.0
        self._last_processed_trade_id: int = 0

        # History for divergence detection
        self._cvd_history: List[float] = []
        self._price_history: List[float] = []
        self._max_history = 100

    def add_trade(self, trade: TradeEvent) -> None:
        """
        Add a trade to the cumulative CVD.

        Call this for every trade received to maintain true cumulative tracking.
        Each trade should only be added once.

        Args:
            trade: Trade event to add
        """
        # Skip if we've already processed this trade
        if trade.trade_id <= self._last_processed_trade_id:
            return

        self._cumulative_cvd += trade.signed_quantity
        self._last_processed_trade_id = trade.trade_id

    def add_trades(self, trades: List[TradeEvent]) -> None:
        """
        Add multiple trades to the cumulative CVD.

        Args:
            trades: List of trade events to add
        """
        for trade in trades:
            self.add_trade(trade)

    @property
    def cvd(self) -> float:
        """Current cumulative volume delta."""
        return self._cumulative_cvd

    def reset_cvd(self) -> None:
        """Reset the cumulative CVD to zero (e.g., at start of new session)."""
        self._cumulative_cvd = 0.0
        self._last_processed_trade_id = 0
        self._cvd_history.clear()
        self._price_history.clear()

    def snapshot(self) -> Dict[str, Any]:
        """Serialize delta engine state for persistence."""
        return {
            "cumulative_cvd": self._cumulative_cvd,
            "last_processed_trade_id": self._last_processed_trade_id,
            "cvd_history": self._cvd_history[-self._max_history :],
            "price_history": self._price_history[-self._max_history :],
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore delta engine state from persisted snapshot."""
        try:
            self._cumulative_cvd = float(snapshot.get("cumulative_cvd", 0.0))
            self._last_processed_trade_id = int(snapshot.get("last_processed_trade_id", 0))
            self._cvd_history = list(snapshot.get("cvd_history", []))[-self._max_history :]
            self._price_history = list(snapshot.get("price_history", []))[-self._max_history :]
        except Exception:
            self.reset_cvd()

    def compute_signal(
        self,
        trade_window: TradeWindow,
        current_price: float,
    ) -> Optional[DeltaSignal]:
        """
        Compute delta/CVD signal.

        Uses true cumulative CVD tracked across the session, not just
        the delta within the rolling window.

        Args:
            trade_window: Rolling window of trades (used for delta_ratio)
            current_price: Current market price

        Returns:
            DeltaSignal or None
        """
        if len(trade_window) < 10:
            return None

        # Use true cumulative CVD, not window-only delta
        cvd = self._cumulative_cvd

        # Track history for divergence detection
        self._cvd_history.append(cvd)
        self._price_history.append(current_price)

        # Trim history
        if len(self._cvd_history) > self._max_history:
            self._cvd_history = self._cvd_history[-self._max_history :]
            self._price_history = self._price_history[-self._max_history :]

        # Calculate changes
        lookback = min(10, len(self._cvd_history))
        if lookback < 2:
            cvd_change = 0.0
            price_change = 0.0
        else:
            cvd_change = cvd - self._cvd_history[-lookback]
            price_change = current_price - self._price_history[-lookback]

        # CVD slope (rate of change)
        cvd_slope = cvd_change / lookback if lookback > 0 else 0.0

        # Determine directions
        price_direction = (
            SignalDirection.BULLISH
            if price_change > 0
            else SignalDirection.BEARISH if price_change < 0 else SignalDirection.NEUTRAL
        )

        cvd_direction = (
            SignalDirection.BULLISH
            if cvd_change > 0
            else SignalDirection.BEARISH if cvd_change < 0 else SignalDirection.NEUTRAL
        )

        # Detect divergence
        is_divergent = (
            price_direction != SignalDirection.NEUTRAL
            and cvd_direction != SignalDirection.NEUTRAL
            and price_direction != cvd_direction
        )

        # Determine who's aggressing (use window delta_ratio for recent activity)
        delta_ratio = trade_window.delta_ratio
        if delta_ratio > 0.15:
            who = "buyers"
        elif delta_ratio < -0.15:
            who = "sellers"
        else:
            who = "balanced"

        # Overall direction (CVD-weighted)
        if is_divergent:
            # Divergence: favor CVD direction (smart money)
            direction = cvd_direction
            strength = min(90, 50 + abs(delta_ratio) * 100)
        else:
            direction = cvd_direction
            strength = min(80, 40 + abs(delta_ratio) * 100)

        return DeltaSignal(
            timestamp_ms=int(time.time() * 1000),
            cvd=cvd,
            cvd_change=cvd_change,
            cvd_slope=cvd_slope,
            price_direction=price_direction,
            cvd_direction=cvd_direction,
            is_divergent=is_divergent,
            who_aggressing=who,
            direction=direction,
            strength=strength,
        )


class VolumeAnalysisAdapter:
    """
    Adapter for AdvancedVolumeAnalyzer.

    Provides "Was the move REAL?" analysis from trade windows.
    """

    def __init__(self):
        self._analyzer = AdvancedVolumeAnalyzer()
        # Store full analysis result for deep dive display
        self._latest_full_result: Optional[VolumeAnalysisSummary] = None

    def compute_analysis(
        self,
        trade_window: TradeWindow,
    ) -> Optional[VolumeAnalysisSummary]:
        """
        Compute volume analysis from trade window.

        Uses time-based bars so each bar covers the same duration,
        making relative volume (current bar / avg) meaningful.

        Args:
            trade_window: Rolling window of trades

        Returns:
            VolumeAnalysisSummary or None if insufficient data
        """
        timestamped_items = trade_window.items_with_timestamps()
        if len(timestamped_items) < 10:
            return None

        # Use time-based bars so the "current" bar is comparable to prior bars.
        # With trade-count chunks, the last bar is often a small remainder,
        # collapsing relative volume to near-zero ("DEAD").
        bar_count = 20
        window_ms = trade_window._window_ms
        bar_duration_ms = max(1, window_ms // bar_count)

        # Align bars to the window end (newest timestamp)
        window_end_ms = timestamped_items[-1].timestamp_ms
        window_start_ms = window_end_ms - window_ms

        opens, highs, lows, closes, volumes = [], [], [], [], []
        trade_idx = 0
        n_trades = len(timestamped_items)

        for bar_i in range(bar_count):
            bar_start = window_start_ms + bar_i * bar_duration_ms
            bar_end = bar_start + bar_duration_ms

            # Collect trades in [bar_start, bar_end)
            bar_prices = []
            bar_vol = 0.0
            while trade_idx < n_trades and timestamped_items[trade_idx].timestamp_ms < bar_end:
                t = timestamped_items[trade_idx].value
                if timestamped_items[trade_idx].timestamp_ms >= bar_start:
                    bar_prices.append(t.price)
                    bar_vol += t.quantity
                trade_idx += 1

            if bar_prices:
                opens.append(bar_prices[0])
                highs.append(max(bar_prices))
                lows.append(min(bar_prices))
                closes.append(bar_prices[-1])
                volumes.append(bar_vol)
            elif closes:
                # Empty bar: carry forward last close, zero volume
                opens.append(closes[-1])
                highs.append(closes[-1])
                lows.append(closes[-1])
                closes.append(closes[-1])
                volumes.append(0.0)

        if len(closes) < 10:
            return None

        # Run full analysis
        result = self._analyzer.full_analysis(opens, highs, lows, closes, volumes)

        # Store full result for deep dive display
        self._latest_full_result = result

        return result

    def get_latest_full_result(self) -> Optional[VolumeAnalysisSummary]:
        """Get the latest full volume analysis result."""
        return self._latest_full_result


class BookEngineAdapter:
    """
    Adapter for AdvancedOrderbookAnalyzer.

    Feeds orderbook snapshots from history into the existing analyzer.
    """

    def __init__(self):
        self._analyzer = AdvancedOrderbookAnalyzer()
        # Store full analysis result for deep dive display
        self._latest_full_result: Optional[OrderbookAnalysisSummary] = None

    def compute_signal(
        self,
        orderbook_history: OrderbookHistory,
        recent_volume: float = 0,
        price_change_percent: float = 0,
        oi_change_percent: Optional[float] = None,
        avg_volume: Optional[float] = None,
    ) -> Optional[BookSignal]:
        """
        Compute orderbook signal from snapshot history.

        Args:
            orderbook_history: History of orderbook snapshots
            recent_volume: Recent trading volume
            price_change_percent: Recent price change
            oi_change_percent: OI change
            avg_volume: Average volume for absorption detection

        Returns:
            BookSignal or None
        """
        current = orderbook_history.latest()
        if current is None:
            return None

        # Convert to format expected by analyzer
        bids = [[level.price, level.quantity] for level in current.bids]
        asks = [[level.price, level.quantity] for level in current.asks]

        if not bids or not asks:
            return None

        # Get previous snapshots and convert to analyzer format for spoof detection
        prev_snapshots_raw = orderbook_history.last_n(5)
        prev_snapshots_converted = self._convert_snapshots_for_analyzer(prev_snapshots_raw)

        # Run analysis
        result = self._analyzer.full_analysis(
            bids=bids,
            asks=asks,
            recent_volume=recent_volume,
            price_change_percent=price_change_percent,
            oi_change_percent=oi_change_percent,
            previous_snapshots=prev_snapshots_converted,
            timestamp=current.timestamp_ms,
            avg_volume=avg_volume,
        )

        # Store full result for deep dive display
        self._latest_full_result = result

        # Extract absorption info
        absorption_detected = result.absorption.detected
        absorption_side = None
        if absorption_detected:
            if result.absorption.side == AbsorptionSide.BID_ABSORPTION:
                absorption_side = "bid"
            elif result.absorption.side == AbsorptionSide.ASK_ABSORPTION:
                absorption_side = "ask"

        # Extract spoof info
        spoof_detected = result.spoof.detected
        spoof_side = None
        if spoof_detected:
            spoof_side = "bid" if "BID" in result.spoof.spoof_type.value else "ask"

        # Get nearest levels from liquidity ladder
        nearest_support = result.liquidity_ladder.nearest_thick_below
        nearest_resistance = result.liquidity_ladder.nearest_thick_above

        return BookSignal(
            timestamp_ms=int(time.time() * 1000),
            bid_depth=current.bid_depth(10),
            ask_depth=current.ask_depth(10),
            imbalance=current.imbalance(10),
            absorption_detected=absorption_detected,
            absorption_side=absorption_side,
            spoof_detected=spoof_detected,
            spoof_side=spoof_side,
            path_of_least_resistance=result.liquidity_ladder.path_of_least_resistance,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            direction=_signal_to_direction(result.overall_signal),
            strength=result.confidence,
        )

    def get_latest_full_result(self) -> Optional[OrderbookAnalysisSummary]:
        """Get the latest full orderbook analysis result."""
        return self._latest_full_result

    def _convert_snapshots_for_analyzer(
        self,
        snapshots: List[OrderbookSnapshot],
    ) -> List[AnalyzerOrderbookSnapshot]:
        """
        Convert OrderbookSnapshot from data_types format to orderbook_analysis format.

        data_types.OrderbookSnapshot uses List[OrderbookLevel] for bids/asks,
        while orderbook_analysis.OrderbookSnapshot expects List[List[float]].

        Args:
            snapshots: List of snapshots from OrderbookHistory (data_types format)

        Returns:
            List of snapshots in orderbook_analysis format for spoof detection
        """
        converted = []
        for snap in snapshots:
            # Convert OrderbookLevel objects to [price, qty] lists
            bids = [[level.price, level.quantity] for level in snap.bids]
            asks = [[level.price, level.quantity] for level in snap.asks]

            # Create analyzer-format snapshot using the analyzer's create_snapshot method
            analyzer_snap = self._analyzer.create_snapshot(
                bids=bids,
                asks=asks,
                timestamp=snap.timestamp_ms,
            )
            converted.append(analyzer_snap)

        return converted


class OIFundingEngineAdapter:
    """
    Adapter for OI and Funding analyzers.

    Combines both into a context-layer signal.
    """

    def __init__(self):
        self._oi_analyzer = AdvancedOIAnalyzer()
        self._funding_analyzer = AdvancedFundingAnalyzer()
        # Store full analysis results for deep dive display
        self._latest_oi_result: Optional[OIAnalysisSummary] = None
        self._latest_funding_result: Optional[FundingAnalysisSummary] = None

    def compute_signal(
        self,
        oi_history: TimestampedRingBuffer[OIUpdate],
        funding_history: TimestampedRingBuffer[FundingUpdate],
        prices: List[float],
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None,
        price_window_seconds: int = 100,
    ) -> Optional[OIFundingSignal]:
        """
        Compute combined OI + Funding context signal.

        Args:
            oi_history: Buffer of OI updates
            funding_history: Buffer of funding updates
            prices: Recent price history (floats) or (timestamp_ms, price) tuples
            highs: Recent highs (optional)
            lows: Recent lows (optional)
            price_window_seconds: How many seconds the prices list spans

        Returns:
            OIFundingSignal or None
        """

        def _extract_price_points(
            price_series: List[float],
        ) -> List[tuple[int, float]]:
            """Extract timestamped price points if provided."""
            if not price_series:
                return []
            first = price_series[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                points: List[tuple[int, float]] = []
                for item in price_series:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        try:
                            points.append((int(item[0]), float(item[1])))
                        except (TypeError, ValueError):
                            continue
                points.sort(key=lambda x: x[0])
                return points
            return []

        def _align_prices_to_oi_exact(
            price_points: List[tuple[int, float]],
            window_seconds: int,
            oi_points: List[OIUpdate],
        ) -> List[float]:
            """Align prices to OI timestamps using exact timestamps."""
            if not price_points or not oi_points:
                return []

            now_ms = int(time.time() * 1000)
            cutoff_ms = now_ms - (window_seconds * 1000)
            filtered = [p for p in price_points if p[0] >= cutoff_ms]
            if not filtered:
                filtered = price_points

            aligned: List[float] = []
            idx = 0
            for oi in oi_points:
                oi_ts = getattr(oi, "timestamp_ms", now_ms)
                while idx + 1 < len(filtered) and filtered[idx + 1][0] <= oi_ts:
                    idx += 1
                # Choose nearest of idx and idx+1
                if idx + 1 < len(filtered):
                    left_ts = filtered[idx][0]
                    right_ts = filtered[idx + 1][0]
                    if abs(right_ts - oi_ts) < abs(oi_ts - left_ts):
                        aligned.append(filtered[idx + 1][1])
                    else:
                        aligned.append(filtered[idx][1])
                else:
                    aligned.append(filtered[idx][1])
            return aligned

        def _align_prices_to_oi_approx(
            price_series: List[float],
            window_seconds: int,
            oi_points: List[OIUpdate],
        ) -> List[float]:
            """Align price series to OI timestamps (approximate, evenly spaced)."""
            if not price_series or not oi_points:
                return []

            now_ms = int(time.time() * 1000)
            if len(price_series) == 1 or window_seconds <= 0:
                return [price_series[-1]] * len(oi_points)

            window_ms = window_seconds * 1000
            start_ms = now_ms - window_ms
            step_ms = window_ms / (len(price_series) - 1)
            if step_ms <= 0:
                return [price_series[-1]] * len(oi_points)

            aligned = []
            for oi in oi_points:
                oi_ts = getattr(oi, "timestamp_ms", now_ms)
                idx = int(round((oi_ts - start_ms) / step_ms))
                idx = max(0, min(idx, len(price_series) - 1))
                aligned.append(price_series[idx])
            return aligned

        def _compute_highs_lows(
            series: List[float],
            window: int = 3,
        ) -> tuple[List[float], List[float]]:
            """Compute simple rolling highs/lows for breakout-trap checks."""
            if not series:
                return [], []
            half = max(1, window // 2)
            highs: List[float] = []
            lows: List[float] = []
            for i in range(len(series)):
                start = max(0, i - half)
                end = min(len(series), i + half + 1)
                seg = series[start:end]
                highs.append(max(seg))
                lows.append(min(seg))
            return highs, lows

        # Get OI values aligned to the price window timeframe.
        # The full OI buffer may span ~16 minutes while prices span ~100s,
        # so comparing them directly creates a timeframe mismatch.
        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - (price_window_seconds * 1000)
        oi_values = oi_history.since(cutoff_ms)

        # Fall back to full buffer if aligned window has too few points
        if len(oi_values) < 2:
            all_oi = oi_history.values()
            if len(all_oi) >= 2:
                oi_values = all_oi[-2:]
        if len(oi_values) < 2:
            return None

        oi_list = [u.open_interest for u in oi_values]
        current_oi = oi_list[-1]

        # Align prices to OI timestamps (so lookbacks are comparable)
        price_points = _extract_price_points(prices)
        if price_points:
            aligned_prices = _align_prices_to_oi_exact(
                price_points, price_window_seconds, oi_values
            )
        else:
            aligned_prices = _align_prices_to_oi_approx(prices, price_window_seconds, oi_values)

        if len(aligned_prices) < 2:
            # Fall back to raw prices if alignment fails
            if price_points:
                aligned_prices = [p for _, p in price_points]
            else:
                aligned_prices = prices

        # Calculate OI change over the aligned window
        oi_change_pct = ((oi_list[-1] - oi_list[0]) / oi_list[0] * 100) if oi_list[0] > 0 else 0

        # OI direction - use config threshold instead of hard-coded 2%
        direction_threshold = self._oi_analyzer.config.open_interest.direction_change_pct
        if oi_change_pct > direction_threshold:
            oi_direction = "rising"
        elif oi_change_pct < -direction_threshold:
            oi_direction = "falling"
        else:
            oi_direction = "flat"

        # Price change (aligned to OI window)
        price_change_pct = 0
        if len(aligned_prices) >= 2:
            price_change_pct = (
                (aligned_prices[-1] - aligned_prices[0]) / aligned_prices[0] * 100
                if aligned_prices[0] > 0
                else 0
            )

        # Run OI analysis with aligned OI data
        highs_aligned, lows_aligned = _compute_highs_lows(aligned_prices)
        oi_result = self._oi_analyzer.full_analysis(
            prices=aligned_prices,
            oi_history=oi_list,
            highs=highs_aligned,
            lows=lows_aligned,
            price_change_percent=price_change_pct,
        )

        # Store OI result for deep dive display
        self._latest_oi_result = oi_result

        # Get funding data
        funding_values = funding_history.values()
        if not funding_values:
            # Return OI-only signal
            return self._build_signal_oi_only(current_oi, oi_change_pct, oi_direction, oi_result)

        current_funding = funding_values[-1].funding_rate
        funding_rates = [f.funding_rate for f in funding_values]

        # Run funding analysis
        funding_result = self._funding_analyzer.full_analysis(
            current_rate=current_funding,
            historical_rates=funding_rates if len(funding_rates) >= 10 else None,
            oi_change_percent=oi_change_pct,
        )

        # Store funding result for deep dive display
        self._latest_funding_result = funding_result

        # Determine regime
        regime = oi_result.regime.regime.value

        # Crowd position
        crowd_position = funding_result.crowd.position.value

        # Squeeze risk
        squeeze_risk = None
        if funding_result.warning.warning.value == "squeeze_risk_up":
            squeeze_risk = "up"
        elif funding_result.warning.warning.value == "squeeze_risk_down":
            squeeze_risk = "down"

        # Is extreme?
        is_extreme = funding_result.is_extreme

        # Determine allows_long/allows_short based on context
        # Context layer doesn't trigger - it allows or disallows
        allows_long = True
        allows_short = True

        # Don't allow longs if crowded long with falling OI
        if crowd_position in ["heavily_long", "moderately_long"] and oi_direction == "falling":
            allows_long = False

        # Don't allow shorts if crowded short with falling OI
        if crowd_position in ["heavily_short", "moderately_short"] and oi_direction == "falling":
            allows_short = False

        # Extreme crowding = contrarian
        if is_extreme:
            if funding_result.percentile.zone == FundingZone.EXTREME_POSITIVE:
                allows_long = False  # Don't chase crowded longs
            elif funding_result.percentile.zone == FundingZone.EXTREME_NEGATIVE:
                allows_short = False  # Don't chase crowded shorts

        # Strength based on confidence
        strength = (oi_result.confidence + funding_result.confidence) / 2

        return OIFundingSignal(
            timestamp_ms=int(time.time() * 1000),
            current_oi=current_oi,
            oi_change_percent=oi_change_pct,
            oi_direction=oi_direction,
            funding_rate=current_funding,
            funding_percentile=funding_result.percentile.percentile,
            crowd_position=crowd_position,
            regime=regime,
            squeeze_risk=squeeze_risk,
            is_extreme=is_extreme,
            allows_long=allows_long,
            allows_short=allows_short,
            strength=strength,
        )

    def _build_signal_oi_only(
        self,
        current_oi: float,
        oi_change_pct: float,
        oi_direction: str,
        oi_result: OIAnalysisSummary,
    ) -> OIFundingSignal:
        """Build signal with OI data only (no funding)."""
        # Store OI result even when no funding
        self._latest_oi_result = oi_result
        return OIFundingSignal(
            timestamp_ms=int(time.time() * 1000),
            current_oi=current_oi,
            oi_change_percent=oi_change_pct,
            oi_direction=oi_direction,
            funding_rate=0.0,
            funding_percentile=50.0,
            crowd_position="balanced",
            regime=oi_result.regime.regime.value,
            squeeze_risk=None,
            is_extreme=False,
            allows_long=True,
            allows_short=True,
            strength=oi_result.confidence,
        )

    def get_latest_oi_result(self) -> Optional[OIAnalysisSummary]:
        """Get the latest full OI analysis result."""
        return self._latest_oi_result

    def get_latest_funding_result(self) -> Optional[FundingAnalysisSummary]:
        """Get the latest full funding analysis result."""
        return self._latest_funding_result


class UnifiedScoreAdapter:
    """
    Adapter for unified score calculation.

    Aggregates all signals into a single actionable score.
    Tracks funding rate history to calculate actual mean/std.
    """

    def __init__(self):
        # Track funding rate history for proper mean/std calculation
        self._funding_history: List[float] = []
        self._max_funding_history = 500  # ~40+ hours at 5min intervals
        # Store latest unified score for deep dive display
        self._latest_unified_score: Optional[UnifiedScore] = None

    def add_funding_rate(self, rate: float) -> None:
        """
        Add a funding rate to history.

        Args:
            rate: Funding rate value
        """
        self._funding_history.append(rate)
        if len(self._funding_history) > self._max_funding_history:
            self._funding_history = self._funding_history[-self._max_funding_history :]

    def _calculate_funding_stats(self) -> tuple[Optional[float], Optional[float]]:
        """
        Calculate mean and standard deviation from funding history.

        Returns:
            (mean, std) or (None, None) if insufficient data
        """
        if len(self._funding_history) < 3:
            return None, None

        n = len(self._funding_history)
        mean = sum(self._funding_history) / n

        # Calculate standard deviation
        variance = sum((x - mean) ** 2 for x in self._funding_history) / n
        std = variance**0.5

        # Ensure std is not too small (avoid division issues)
        if std < 0.00001:
            std = 0.00001

        return mean, std

    def compute_score(
        self,
        volume_signal: Optional[VolumeSignal],
        delta_signal: Optional[DeltaSignal],
        book_signal: Optional[BookSignal],
        oi_funding_signal: Optional[OIFundingSignal],
        price_change_pct: float = 0,
    ) -> UnifiedScore:
        """
        Compute unified score from all signals.

        Args:
            volume_signal: Signal from volume engine
            delta_signal: Signal from delta engine
            book_signal: Signal from orderbook engine
            oi_funding_signal: Signal from OI/funding engine
            price_change_pct: Recent price change

        Returns:
            UnifiedScore
        """
        # Extract values for unified score calculation
        delta_ratio = 0.0
        relative_volume = 1.0
        if volume_signal:
            delta_ratio = volume_signal.delta_ratio
            relative_volume = volume_signal.relative_volume

        oi_change_pct = None
        if oi_funding_signal:
            oi_change_pct = oi_funding_signal.oi_change_percent

        depth_imbalance = None
        absorption_bullish = False
        absorption_bearish = False
        is_bait = False
        spoof_detected = False
        if book_signal:
            depth_imbalance = book_signal.imbalance
            absorption_bullish = (
                book_signal.absorption_detected and book_signal.absorption_side == "bid"
            )
            absorption_bearish = (
                book_signal.absorption_detected and book_signal.absorption_side == "ask"
            )
            spoof_detected = book_signal.spoof_detected
            # Bait = imbalance without volume confirmation.
            # The BookEngineAdapter already carries spoof_detected; for bait we check
            # if the analyzer flagged the overall signal as TRAP or if direction is
            # NEUTRAL despite a strong imbalance (no volume backing it).
            is_bait = (
                depth_imbalance is not None
                and abs(depth_imbalance) > 0.2
                and book_signal.direction == SignalDirection.NEUTRAL
                and not absorption_bullish
                and not absorption_bearish
            )

        # Funding data - pass historical funding for advanced analysis
        current_funding = None
        historical_funding = None
        if oi_funding_signal and oi_funding_signal.funding_rate != 0:
            current_funding = oi_funding_signal.funding_rate

            # Track this funding rate
            self.add_funding_rate(current_funding)

            # Get funding history for percentile-based analysis
            historical_funding = (
                self._funding_history.copy() if len(self._funding_history) >= 10 else None
            )

        unified = calculate_unified_score(
            delta_ratio=delta_ratio,
            relative_volume=relative_volume,
            price_change_pct=price_change_pct,
            oi_change_pct=oi_change_pct,
            current_funding=current_funding,
            historical_funding=historical_funding,
            depth_imbalance=depth_imbalance,
            absorption_bullish=absorption_bullish,
            absorption_bearish=absorption_bearish,
            is_bait=is_bait,
            spoof_detected=spoof_detected,
        )

        # Store for deep dive display
        self._latest_unified_score = unified

        return unified

    def get_latest_unified_score(self) -> Optional[UnifiedScore]:
        """Get the latest unified score."""
        return self._latest_unified_score
