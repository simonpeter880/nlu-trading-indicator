"""
Continuous Analyzer Orchestrator

Wires together:
- Data Ingestion (WebSocket streams)
- Signal Engines (via adapters)
- State Machine (discrete decisions)

This is the main entry point for continuous analysis.
"""

import asyncio
import json
import logging
import os
import time
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass

from .data_types import (
    TradeEvent,
    OrderbookSnapshot,
    OIUpdate,
    FundingUpdate,
    MarketState,
    VolumeSignal,
    DeltaSignal,
    BookSignal,
    OIFundingSignal,
    IngestionConfig,
    WindowConfig,
)
from .ingestion import DataIngestionManager
from .rolling_window import MultiTimeframeWindows
from .engine_adapters import (
    VolumeEngineAdapter,
    VolumeAnalysisAdapter,
    DeltaEngineAdapter,
    BookEngineAdapter,
    OIFundingEngineAdapter,
    UnifiedScoreAdapter,
)
from .atr_expansion_adapter import ATRExpansionAdapter, ATRSignal
from .state_machine import (
    TradingStateMachine,
    MarketRegime,
    StateTransition,
    TradeSignal,
)
from .metrics import MetricsCollector, get_metrics

from indicators import VolatilityIndicators

logger = logging.getLogger(__name__)


@dataclass
class AnalyzerConfig:
    """Configuration for the continuous analyzer."""
    # Ingestion settings
    ingestion: IngestionConfig = None

    # Window settings
    windows: WindowConfig = None

    # Signal computation interval (ms)
    signal_interval_ms: int = 1000  # Compute signals every second

    # State machine settings
    min_confidence: float = 50.0
    min_volume_ratio: float = 0.7
    state_cooldown_ms: int = 5000
    min_state_duration_ms: int = 15000  # Min time in risk-on state before switching
    state_confirmation_count: int = 3   # Consecutive ticks required for risk-on transition

    # Primary decision window (seconds)
    primary_window_seconds: int = 60

    # Warmup settings
    warmup_seconds: int = 60  # Minimum data collection time before signals
    warmup_min_trades: int = 100  # Minimum trades before signals
    warmup_min_orderbook_snapshots: int = 10  # Minimum orderbook snapshots

    # ATR settings (for dynamic stops)
    atr_period: int = 14
    atr_window_seconds: int = 900  # 15 minutes
    atr_bar_count: int = 20

    # State persistence (for restarts)
    state_persistence_enabled: bool = True
    state_persistence_dir: str = "~/.indicator_state"
    state_save_interval_ms: int = 10_000
    state_persistence_path: Optional[str] = None

    def __post_init__(self):
        if self.ingestion is None:
            self.ingestion = IngestionConfig()
        if self.windows is None:
            self.windows = WindowConfig()


class ContinuousAnalyzer:
    """
    Main orchestrator for continuous market analysis.

    Architecture:
    ```
    DATA INGESTION
    ├─ aggTrades (continuous) ──┐
    ├─ orderbook (250ms) ───────┤
    ├─ OI (10s) ────────────────┼──> ROLLING WINDOWS
    └─ Funding (event) ─────────┘    ├─ 15s
                                     ├─ 60s
                                     ├─ 180s
                                     ↓
                               SIGNAL ENGINES
                               ├─ Volume Engine
                               ├─ Delta Engine
                               ├─ Book Engine
                               ├─ OI/Funding Engine
                               ↓
                               UNIFIED SCORE
                               ↓
                               STATE MACHINE
                               ├─ NO_TRADE
                               ├─ COMPRESSION
                               ├─ SQUEEZE_SETUP
                               ├─ TREND_CONTINUATION
                               └─ etc.
    ```

    Usage:
        analyzer = ContinuousAnalyzer("BTCUSDT")

        @analyzer.on_trade_signal
        def handle_signal(signal: TradeSignal):
            print(f"Trade signal: {signal.direction} {signal.signal_type}")

        await analyzer.start()
    """

    def __init__(
        self,
        symbol: str,
        config: Optional[AnalyzerConfig] = None,
    ):
        self.symbol = symbol.upper().replace("/", "").replace("-", "")
        self.config = config or AnalyzerConfig()

        # Data ingestion
        self._ingestion = DataIngestionManager(
            symbol,
            config=self.config.ingestion,
        )

        # Engine adapters
        self._volume_adapter = VolumeEngineAdapter()
        self._volume_analysis_adapter = VolumeAnalysisAdapter()
        self._delta_adapter = DeltaEngineAdapter()
        self._book_adapter = BookEngineAdapter()
        self._oi_funding_adapter = OIFundingEngineAdapter()
        self._unified_adapter = UnifiedScoreAdapter()
        self._atr_adapter = ATRExpansionAdapter()  # ATR expansion timing gate

        # State machine
        self._state_machine = TradingStateMachine(
            min_confidence=self.config.min_confidence,
            min_volume_ratio=self.config.min_volume_ratio,
            transition_cooldown_ms=self.config.state_cooldown_ms,
            min_state_duration_ms=self.config.min_state_duration_ms,
            confirmation_count=self.config.state_confirmation_count,
        )

        # Latest signals (cached)
        self._latest_volume_signal: Optional[VolumeSignal] = None
        self._latest_delta_signal: Optional[DeltaSignal] = None
        self._latest_book_signal: Optional[BookSignal] = None
        self._latest_oi_funding_signal: Optional[OIFundingSignal] = None
        self._latest_market_state: Optional[MarketState] = None

        # Price tracking for price change calculation
        # Stored as (timestamp_ms, price)
        self._price_history: List[tuple[int, float]] = []
        self._max_price_history = 1000

        # Signal computation task
        self._signal_task: Optional[asyncio.Task] = None
        self._running = False

        # Warmup tracking
        self._warmup_start_time: Optional[int] = None
        self._warmup_complete = False

        # External callbacks
        self._on_state_change_callbacks: List[Callable] = []
        self._on_trade_signal_callbacks: List[Callable] = []
        self._on_market_state_callbacks: List[Callable] = []

        # Wire up internal callbacks
        self._state_machine.on_transition(self._handle_state_transition)
        self._state_machine.on_trade_signal(self._handle_trade_signal)

        # Wire delta adapter to receive trades for true cumulative CVD tracking
        self._ingestion.add_update_callback(self._handle_data_update)

        # Metrics collector
        self._metrics = MetricsCollector()

        # State persistence
        self._state_path = self._resolve_state_path()
        self._last_state_save_ms = 0
        self._state_loaded = False

    # === Properties ===

    @property
    def current_state(self) -> MarketRegime:
        return self._state_machine.current_state

    @property
    def latest_market_state(self) -> Optional[MarketState]:
        return self._latest_market_state

    @property
    def latest_price(self) -> float:
        return self._ingestion.latest_price

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_warmed_up(self) -> bool:
        """True if warmup period is complete and signals are being generated."""
        return self._warmup_complete

    @property
    def metrics(self) -> MetricsCollector:
        """Access the metrics collector for telemetry data."""
        return self._metrics

    def get_recent_price_history(self, count: int = 100) -> List[float]:
        """Get recent price history for calculations."""
        if not self._price_history:
            return []
        return [p for _, p in self._price_history[-count:]]

    @property
    def warmup_progress(self) -> dict:
        """Get current warmup progress."""
        if self._warmup_start_time is None:
            return {
                "complete": False,
                "elapsed_seconds": 0,
                "required_seconds": self.config.warmup_seconds,
                "trade_count": 0,
                "required_trades": self.config.warmup_min_trades,
                "orderbook_count": 0,
                "required_orderbooks": self.config.warmup_min_orderbook_snapshots,
            }

        elapsed_ms = int(time.time() * 1000) - self._warmup_start_time
        elapsed_s = elapsed_ms / 1000

        # Get primary window for trade count
        primary_window = self._ingestion.trade_windows.get_window(
            self.config.primary_window_seconds
        )
        trade_count = len(primary_window) if primary_window else 0
        orderbook_count = len(self._ingestion.orderbook_history)

        return {
            "complete": self._warmup_complete,
            "elapsed_seconds": round(elapsed_s, 1),
            "required_seconds": self.config.warmup_seconds,
            "trade_count": trade_count,
            "required_trades": self.config.warmup_min_trades,
            "orderbook_count": orderbook_count,
            "required_orderbooks": self.config.warmup_min_orderbook_snapshots,
        }

    # === Callback Registration ===

    def on_state_change(self, callback: Callable[[StateTransition], Any]) -> None:
        """Register callback for state changes."""
        self._on_state_change_callbacks.append(callback)

    def on_trade_signal(self, callback: Callable[[TradeSignal], Any]) -> None:
        """Register callback for trade signals."""
        self._on_trade_signal_callbacks.append(callback)

    def on_market_state(self, callback: Callable[[MarketState], Any]) -> None:
        """Register callback for market state updates."""
        self._on_market_state_callbacks.append(callback)

    # === Internal Callbacks ===

    def _handle_state_transition(self, transition: StateTransition) -> None:
        """Handle state machine transition."""
        logger.info(
            f"State transition: {transition.from_state.value} -> "
            f"{transition.to_state.value} ({transition.trigger})"
        )
        # Track state transition in metrics
        self._metrics.increment("state_transitions")

        for callback in self._on_state_change_callbacks:
            try:
                callback(transition)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

    def _handle_trade_signal(self, signal: TradeSignal) -> None:
        """Handle trade signal from state machine."""
        logger.info(
            f"Trade signal: {signal.direction.upper()} {signal.signal_type} "
            f"@ {signal.entry_price:.2f} (conf: {signal.confidence:.1f}%)"
        )
        for callback in self._on_trade_signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Trade signal callback error: {e}")

    def _handle_data_update(self, update_type: str, data: Any) -> None:
        """Handle data updates from ingestion for cumulative tracking."""
        if update_type == "trade":
            # Forward trade to delta adapter for true cumulative CVD tracking
            self._delta_adapter.add_trade(data)
            # Track trade rate
            self._metrics.record_event("trades")
        elif update_type == "orderbook":
            # Track orderbook update rate
            self._metrics.record_event("orderbook_updates")
        elif update_type == "funding":
            # Forward funding rate to unified adapter for mean/std calculation
            self._unified_adapter.add_funding_rate(data.funding_rate)

    # === Warmup ===

    def _check_warmup_complete(self) -> bool:
        """
        Check if warmup period is complete.

        Warmup requires:
        1. Minimum elapsed time (default 60s)
        2. Minimum number of trades collected
        3. Minimum number of orderbook snapshots
        """
        if self._warmup_complete:
            return True

        if self._warmup_start_time is None:
            return False

        now = int(time.time() * 1000)
        elapsed_ms = now - self._warmup_start_time
        elapsed_s = elapsed_ms / 1000

        # Check time requirement
        if elapsed_s < self.config.warmup_seconds:
            return False

        # Check trade count
        primary_window = self._ingestion.trade_windows.get_window(
            self.config.primary_window_seconds
        )
        trade_count = len(primary_window) if primary_window else 0
        if trade_count < self.config.warmup_min_trades:
            return False

        # Check orderbook snapshots
        orderbook_count = len(self._ingestion.orderbook_history)
        if orderbook_count < self.config.warmup_min_orderbook_snapshots:
            return False

        # All requirements met
        self._warmup_complete = True
        logger.info(
            f"Warmup complete for {self.symbol}: "
            f"{elapsed_s:.1f}s elapsed, {trade_count} trades, "
            f"{orderbook_count} orderbook snapshots"
        )
        return True

    # === Signal Computation ===

    def _compute_signals(self) -> MarketState:
        """Compute all signals from current data with latency tracking."""
        now = int(time.time() * 1000)

        # Get data sources
        windows = self._ingestion.trade_windows
        book_history = self._ingestion.orderbook_history
        oi_history = self._ingestion.oi_history
        funding_history = self._ingestion.funding_history
        current_price = self._ingestion.latest_price

        # Track price history
        if current_price > 0:
            self._price_history.append((now, current_price))
            if len(self._price_history) > self._max_price_history:
                self._price_history = self._price_history[-self._max_price_history:]

        # Calculate price change
        price_change_pct = 0.0
        if len(self._price_history) >= 2:
            cutoff_ms = now - 60_000  # ~60 seconds
            old_price = None
            for ts, price in reversed(self._price_history):
                if ts <= cutoff_ms:
                    old_price = price
                    break
            if old_price is None:
                old_price = self._price_history[0][1]
            if old_price > 0:
                price_change_pct = (current_price - old_price) / old_price * 100

        # Get OI change for engine inputs (aligned to primary window)
        oi_change_pct = None
        if len(oi_history) >= 2:
            oi_cutoff_ms = now - (self.config.primary_window_seconds * 1000)
            oi_values = oi_history.since(oi_cutoff_ms)
            if len(oi_values) < 2:
                all_oi = oi_history.values()
                if len(all_oi) >= 2:
                    oi_values = all_oi[-2:]
            if len(oi_values) >= 2 and oi_values[0].open_interest > 0:
                oi_change_pct = (oi_values[-1].open_interest - oi_values[0].open_interest) / oi_values[0].open_interest * 100

        # Compute Volume Signal (with latency tracking)
        primary_window = windows.get_window(self.config.primary_window_seconds)
        if primary_window and len(primary_window) >= 10:
            with self._metrics.time("volume_engine"):
                self._latest_volume_signal = self._volume_adapter.compute_from_multi_windows(
                    windows,
                    primary_window_seconds=self.config.primary_window_seconds,
                    oi_change_percent=oi_change_pct,
                )

                # Also compute full volume analysis for deep dive display
            with self._metrics.time("volume_analysis"):
                self._volume_analysis_adapter.compute_analysis(primary_window)

            # Compute ATR expansion signal (timing gate)
            with self._metrics.time("atr_expansion"):
                self._atr_adapter.compute_from_window(
                    f"{self.config.primary_window_seconds}s",
                    primary_window
                )

        # Compute Delta Signal (with latency tracking)
        if primary_window and len(primary_window) >= 10:
            with self._metrics.time("delta_engine"):
                self._latest_delta_signal = self._delta_adapter.compute_signal(
                    primary_window,
                    current_price,
                )

        # Compute Book Signal (with latency tracking)
        recent_volume = primary_window.total_volume if primary_window else 0

        # Calculate average volume from a longer window for absorption detection
        # Use 5-minute window to establish baseline (requires ~4.5min warmup for is_complete)
        avg_volume = None
        baseline_window = windows.get_window(300)  # 5 minutes
        if baseline_window and baseline_window.is_complete:
            # Average per primary window period (e.g., 60s)
            # Scale 5-min total volume down to primary window size
            avg_volume = baseline_window.total_volume * (self.config.primary_window_seconds / 300)

        with self._metrics.time("book_engine"):
            self._latest_book_signal = self._book_adapter.compute_signal(
                book_history,
                recent_volume=recent_volume,
                price_change_percent=price_change_pct,
                oi_change_percent=oi_change_pct,
                avg_volume=avg_volume,
            )

        # Compute OI/Funding Signal (with latency tracking)
        if len(self._price_history) >= 10:
            with self._metrics.time("oi_funding_engine"):
                self._latest_oi_funding_signal = self._oi_funding_adapter.compute_signal(
                    oi_history,
                    funding_history,
                    prices=self._price_history[-100:],  # Last 100 price points
                )

        # Compute Unified Score (with latency tracking)
        with self._metrics.time("unified_score"):
            unified = self._unified_adapter.compute_score(
                self._latest_volume_signal,
                self._latest_delta_signal,
                self._latest_book_signal,
                self._latest_oi_funding_signal,
                price_change_pct,
            )

        # Calculate data quality (how complete is our data?)
        data_quality = self._calculate_data_quality()

        # ATR tracking for dynamic stops
        atr, atr_percent = self._compute_atr(windows)

        # Build market state
        market_state = MarketState(
            timestamp_ms=now,
            symbol=self.symbol,
            current_price=current_price,
            atr=atr,
            atr_percent=atr_percent,
            volume_signal=self._latest_volume_signal,
            delta_signal=self._latest_delta_signal,
            book_signal=self._latest_book_signal,
            oi_funding_signal=self._latest_oi_funding_signal,
            unified_score=unified.total_score,
            confidence=unified.confidence,
            data_quality=data_quality,
        )

        self._latest_market_state = market_state
        return market_state

    def _calculate_data_quality(self) -> float:
        """Calculate data quality score (0-1)."""
        scores = []

        # Trade data quality
        windows = self._ingestion.trade_windows
        primary = windows.get_window(self.config.primary_window_seconds)
        if primary:
            scores.append(min(1.0, primary.coverage))
        else:
            scores.append(0.0)

        # Orderbook data
        if self._ingestion.latest_book:
            scores.append(1.0)
        else:
            scores.append(0.0)

        # OI data
        if len(self._ingestion.oi_history) >= 2:
            scores.append(1.0)
        else:
            scores.append(0.5)

        # Funding data
        if len(self._ingestion.funding_history) >= 1:
            scores.append(1.0)
        else:
            scores.append(0.5)

        return sum(scores) / len(scores) if scores else 0.0

    def _compute_atr(self, windows: MultiTimeframeWindows) -> tuple[Optional[float], Optional[float]]:
        """Compute ATR from recent trade data."""
        atr_window = windows.get_window(self.config.atr_window_seconds)
        if atr_window is None or len(atr_window) < self.config.atr_period + 1:
            atr_window = windows.get_window(self.config.primary_window_seconds)
        if atr_window is None:
            return None, None

        trades = atr_window.items()
        if len(trades) < self.config.atr_period + 1:
            return None, None

        bar_count = max(self.config.atr_bar_count, self.config.atr_period + 1)
        opens, highs, lows, closes, _ = self._trades_to_ohlcv(trades, bar_count=bar_count)
        atr_values = VolatilityIndicators.calculate_atr(highs, lows, closes, period=self.config.atr_period)
        if not atr_values:
            return None, None
        atr = atr_values[-1]
        atr_percent = (atr / closes[-1]) * 100 if closes and closes[-1] > 0 else None
        return atr, atr_percent

    def _trades_to_ohlcv(
        self,
        trades: List[TradeEvent],
        bar_count: int = 20,
    ) -> tuple[List[float], List[float], List[float], List[float], List[float]]:
        """Convert trades to synthetic OHLCV bars."""
        if not trades:
            return [], [], [], [], []

        trades_per_bar = max(1, len(trades) // bar_count)
        opens, highs, lows, closes, volumes = [], [], [], [], []

        for i in range(0, len(trades), trades_per_bar):
            bar_trades = trades[i:i + trades_per_bar]
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

    async def _signal_loop(self) -> None:
        """Main signal computation loop."""
        interval_s = self.config.signal_interval_ms / 1000

        while self._running:
            try:
                start = time.time()

                # Check warmup status
                if not self._check_warmup_complete():
                    # Still warming up - log progress periodically
                    progress = self.warmup_progress
                    if int(progress["elapsed_seconds"]) % 10 == 0:  # Log every 10s
                        logger.debug(
                            f"Warmup in progress: {progress['elapsed_seconds']:.0f}s/"
                            f"{progress['required_seconds']}s, "
                            f"{progress['trade_count']}/{progress['required_trades']} trades, "
                            f"{progress['orderbook_count']}/{progress['required_orderbooks']} orderbooks"
                        )
                    await asyncio.sleep(interval_s)
                    continue

                # Compute signals (with overall latency tracking)
                with self._metrics.time("signal_compute"):
                    market_state = self._compute_signals()

                # Track signal computation event
                self._metrics.record_event("signals_computed")

                # Notify market state callbacks
                for callback in self._on_market_state_callbacks:
                    try:
                        result = callback(market_state)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as e:
                        logger.error(f"Market state callback error: {e}")

                # Update state machine (with latency tracking)
                with self._metrics.time("state_machine"):
                    trade_signal = self._state_machine.update(market_state)
                    if trade_signal:
                        self._metrics.increment("trade_signals")

                # Periodically persist state for restarts
                self._maybe_persist_state()

                # Sleep remaining interval
                elapsed = time.time() - start
                await asyncio.sleep(max(0, interval_s - elapsed))

            except asyncio.CancelledError:
                break
            except Exception as e:
                import traceback
                logger.error(f"Signal loop error: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                self._metrics.increment("errors")
                await asyncio.sleep(1)

    # === State Persistence ===

    def _resolve_state_path(self) -> Optional[str]:
        """Resolve persistence path for this symbol."""
        if not self.config.state_persistence_enabled:
            return None
        if self.config.state_persistence_path:
            return os.path.expanduser(self.config.state_persistence_path)
        state_dir = os.path.expanduser(self.config.state_persistence_dir)
        return os.path.join(state_dir, f"{self.symbol.lower()}.json")

    def _load_state(self) -> bool:
        """Load persisted state if available."""
        if not self.config.state_persistence_enabled or not self._state_path:
            return False
        try:
            if not os.path.exists(self._state_path):
                return False
            with open(self._state_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("symbol") != self.symbol:
                return False

            state_machine = payload.get("state_machine")
            if state_machine:
                self._state_machine.restore(state_machine)

            delta_state = payload.get("delta_engine")
            if delta_state:
                self._delta_adapter.restore(delta_state)

            raw_history = payload.get("price_history", [])
            parsed_history: List[tuple[int, float]] = []
            for item in raw_history:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    try:
                        parsed_history.append((int(item[0]), float(item[1])))
                    except (TypeError, ValueError):
                        continue
            self._price_history = parsed_history[-self._max_price_history:]
            warmup = payload.get("warmup", {})
            self._warmup_start_time = warmup.get("start_time")
            self._warmup_complete = warmup.get("complete", False)

            self._state_loaded = True
            return True
        except Exception as exc:
            logger.debug(f"Failed to load state: {exc}")
            return False

    def _save_state(self) -> None:
        """Persist current state for restart."""
        if not self.config.state_persistence_enabled or not self._state_path:
            return
        try:
            state_dir = os.path.dirname(self._state_path)
            if state_dir:
                os.makedirs(state_dir, exist_ok=True)
            payload = {
                "version": 1,
                "symbol": self.symbol,
                "timestamp_ms": int(time.time() * 1000),
                "state_machine": self._state_machine.snapshot(),
                "delta_engine": self._delta_adapter.snapshot(),
                "price_history": self._price_history[-self._max_price_history:],
                "warmup": {
                    "start_time": self._warmup_start_time,
                    "complete": self._warmup_complete,
                },
            }
            with open(self._state_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        except Exception as exc:
            logger.debug(f"Failed to persist state: {exc}")

    def _maybe_persist_state(self) -> None:
        """Persist state on an interval."""
        if not self.config.state_persistence_enabled:
            return
        now = int(time.time() * 1000)
        if now - self._last_state_save_ms < self.config.state_save_interval_ms:
            return
        self._last_state_save_ms = now
        self._save_state()

    # === Lifecycle ===

    async def start(self) -> None:
        """Start continuous analysis."""
        if self._running:
            return

        self._running = True
        restored = self._load_state()
        if not restored:
            self._warmup_complete = False
            self._warmup_start_time = int(time.time() * 1000)
        elif self._warmup_start_time is None:
            self._warmup_start_time = int(time.time() * 1000)

        logger.info(
            f"Starting continuous analyzer for {self.symbol} "
            f"(warmup: {self.config.warmup_seconds}s)"
        )

        # Start data ingestion
        await self._ingestion.start()

        # Brief pause for WebSocket connections to establish
        await asyncio.sleep(1)

        # Start signal computation loop
        self._signal_task = asyncio.create_task(self._signal_loop())

    async def stop(self) -> None:
        """Stop continuous analysis."""
        self._running = False
        logger.info(f"Stopping continuous analyzer for {self.symbol}")

        # Stop signal loop
        if self._signal_task:
            self._signal_task.cancel()
            try:
                await self._signal_task
            except asyncio.CancelledError:
                pass
            self._signal_task = None

        # Stop data ingestion
        await self._ingestion.stop()

        # Persist state on shutdown
        self._save_state()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    # === Utility Methods ===

    def get_status(self) -> Dict[str, Any]:
        """Get current analyzer status."""
        return {
            "symbol": self.symbol,
            "running": self._running,
            "warmed_up": self._warmup_complete,
            "warmup_progress": self.warmup_progress,
            "current_state": self._state_machine.current_state.value,
            "state_duration_ms": self._state_machine.state_duration_ms,
            "latest_price": self._ingestion.latest_price,
            "unified_score": self._latest_market_state.unified_score if self._latest_market_state else None,
            "confidence": self._latest_market_state.confidence if self._latest_market_state else None,
            "data_quality": self._latest_market_state.data_quality if self._latest_market_state else None,
            "stream_states": {k: v.value for k, v in self._ingestion.get_state().items()},
        }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for monitoring and debugging."""
        return self._metrics.get_summary()

    def get_latency_stats(self, operation: str = "signal_compute") -> Dict[str, float]:
        """
        Get latency statistics for a specific operation.

        Args:
            operation: One of "signal_compute", "volume_engine", "delta_engine",
                      "book_engine", "oi_funding_engine", "unified_score", "state_machine"

        Returns:
            Dict with mean, p50, p95, p99, max latencies in milliseconds
        """
        stats = self._metrics.get_latency_stats(operation)
        if stats is None:
            return {}
        return {
            "count": stats.count,
            "mean_ms": stats.mean_ms,
            "min_ms": stats.min_ms,
            "max_ms": stats.max_ms,
            "p50_ms": stats.p50_ms,
            "p95_ms": stats.p95_ms,
            "p99_ms": stats.p99_ms,
        }

    def get_signals_summary(self) -> Dict[str, Any]:
        """Get summary of current signals."""
        return {
            "volume": {
                "delta_ratio": self._latest_volume_signal.delta_ratio if self._latest_volume_signal else None,
                "relative_volume": self._latest_volume_signal.relative_volume if self._latest_volume_signal else None,
                "direction": self._latest_volume_signal.direction.value if self._latest_volume_signal else None,
                "is_climax": self._latest_volume_signal.is_climax if self._latest_volume_signal else None,
            },
            "delta": {
                "cvd": self._latest_delta_signal.cvd if self._latest_delta_signal else None,
                "is_divergent": self._latest_delta_signal.is_divergent if self._latest_delta_signal else None,
                "who_aggressing": self._latest_delta_signal.who_aggressing if self._latest_delta_signal else None,
            },
            "book": {
                "imbalance": self._latest_book_signal.imbalance if self._latest_book_signal else None,
                "path": self._latest_book_signal.path_of_least_resistance if self._latest_book_signal else None,
                "absorption": self._latest_book_signal.absorption_side if self._latest_book_signal and self._latest_book_signal.absorption_detected else None,
            },
            "oi_funding": {
                "oi_direction": self._latest_oi_funding_signal.oi_direction if self._latest_oi_funding_signal else None,
                "regime": self._latest_oi_funding_signal.regime if self._latest_oi_funding_signal else None,
                "crowd_position": self._latest_oi_funding_signal.crowd_position if self._latest_oi_funding_signal else None,
                "squeeze_risk": self._latest_oi_funding_signal.squeeze_risk if self._latest_oi_funding_signal else None,
            },
        }

    # === Full Analysis Results (for deep-dive display) ===

    def get_volume_analysis_full(self):
        """Get full volume analysis result (VolumeAnalysisSummary)."""
        return self._volume_analysis_adapter.get_latest_full_result()

    def get_volume_engine_full(self):
        """Get full volume engine result (VolumeEngineResult)."""
        return self._volume_adapter.get_latest_full_result()

    def get_orderbook_analysis_full(self):
        """Get full orderbook analysis result (OrderbookAnalysisSummary)."""
        return self._book_adapter.get_latest_full_result()

    def get_oi_analysis_full(self):
        """Get full OI analysis result (OIAnalysisSummary)."""
        return self._oi_funding_adapter.get_latest_oi_result()

    def get_funding_analysis_full(self):
        """Get full funding analysis result (FundingAnalysisSummary)."""
        return self._oi_funding_adapter.get_latest_funding_result()

    def get_unified_score_full(self):
        """Get the most recent unified score computation."""
        return self._unified_adapter.get_latest_unified_score()

    def get_all_full_results(self):
        """Get all full analysis results in a dict for convenience."""
        return {
            'volume_analysis': self.get_volume_analysis_full(),
            'volume_engine': self.get_volume_engine_full(),
            'orderbook': self.get_orderbook_analysis_full(),
            'oi': self.get_oi_analysis_full(),
            'funding': self.get_funding_analysis_full(),
            'unified_score': self.get_unified_score_full(),
            'atr_signals': self.get_atr_signals(),
        }

    def get_atr_signal(self, timeframe: str) -> Optional[ATRSignal]:
        """Get ATR expansion signal for a specific timeframe."""
        return self._atr_adapter.get_signal(timeframe)

    def get_atr_signals(self) -> Dict:
        """Get all ATR expansion signals."""
        return self._atr_adapter.get_all_signals()
