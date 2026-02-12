"""
Continuous Market Data Architecture

3-Layer Model:
1. DATA INGESTION - Raw market data streams (WebSocket)
2. SIGNAL ENGINES - Rolling window computations (adapters to existing engines)
3. STATE MACHINE - Trade decision logic

"Ingest continuously, decide discretely."

Architecture:
```
DATA INGESTION
├─ aggTrades (continuous)
├─ orderbook snapshots (250ms)
├─ OI (10s)
└─ Funding (event-based)
        ↓
ROLLING WINDOWS
├─ 15s  (micro/execution)
├─ 60s  (decision frame)
├─ 180s (decision frame)
        ↓
SIGNAL ENGINES (adapters to existing)
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
├─ EXHAUSTION
├─ SQUEEZE_SETUP
├─ TREND_CONTINUATION
└─ etc.
```

Usage:
    from indicator.continuous import ContinuousAnalyzer

    async def main():
        analyzer = ContinuousAnalyzer("BTCUSDT")

        @analyzer.on_trade_signal
        def handle_signal(signal):
            print(f"Signal: {signal.direction} {signal.signal_type}")

        async with analyzer:
            await asyncio.sleep(3600)  # Run for 1 hour

    asyncio.run(main())
"""

from .atr_expansion_adapter import ATRExpansionAdapter, ATRSignal, format_atr_signals
from .data_types import (
    BookSignal,
    DeltaSignal,
    FundingUpdate,
    IngestionConfig,
    MarketState,
    OIFundingSignal,
    OIUpdate,
    OrderbookLevel,
    OrderbookSnapshot,
    SignalDirection,
    TradeEvent,
    VolumeSignal,
    WindowConfig,
)
from .engine_adapters import (
    BookEngineAdapter,
    DeltaEngineAdapter,
    OIFundingEngineAdapter,
    UnifiedScoreAdapter,
    VolumeAnalysisAdapter,
    VolumeEngineAdapter,
)
from .ingestion import (
    AggTradeStream,
    DataIngestionManager,
    FundingStream,
    OIStream,
    OrderbookStream,
    StreamState,
    StreamStats,
)
from .market_structure_adapter import MarketStructureAdapter, MarketStructureSignal
from .metrics import (
    LatencyStats,
    LatencyTracker,
    MetricsCollector,
    RateStats,
    RateTracker,
    SystemMetrics,
    get_metrics,
    reset_metrics,
)
from .orchestrator import AnalyzerConfig, ContinuousAnalyzer
from .ring_buffer import DeltaBuffer, RingBuffer, SumBuffer, TimestampedRingBuffer
from .rolling_window import (
    MultiTimeframeWindows,
    OrderbookHistory,
    RollingWindow,
    TradeWindow,
    WindowStats,
)
from .state_machine import MarketRegime, StateTransition, TradeSignal, TradingStateMachine

__all__ = [
    # Ring buffers
    "RingBuffer",
    "TimestampedRingBuffer",
    "SumBuffer",
    "DeltaBuffer",
    # Rolling windows
    "RollingWindow",
    "TradeWindow",
    "MultiTimeframeWindows",
    "OrderbookHistory",
    "WindowStats",
    # Data types
    "TradeEvent",
    "OrderbookSnapshot",
    "OrderbookLevel",
    "OIUpdate",
    "FundingUpdate",
    "VolumeSignal",
    "DeltaSignal",
    "BookSignal",
    "OIFundingSignal",
    "MarketState",
    "SignalDirection",
    "IngestionConfig",
    "WindowConfig",
    # Ingestion
    "DataIngestionManager",
    "AggTradeStream",
    "OrderbookStream",
    "OIStream",
    "FundingStream",
    "StreamState",
    "StreamStats",
    # Engine adapters
    "VolumeEngineAdapter",
    "VolumeAnalysisAdapter",
    "DeltaEngineAdapter",
    "BookEngineAdapter",
    "OIFundingEngineAdapter",
    "UnifiedScoreAdapter",
    # State machine
    "MarketRegime",
    "TradingStateMachine",
    "StateTransition",
    "TradeSignal",
    # Orchestrator
    "ContinuousAnalyzer",
    "AnalyzerConfig",
    # Metrics
    "MetricsCollector",
    "LatencyTracker",
    "RateTracker",
    "LatencyStats",
    "RateStats",
    "SystemMetrics",
    "get_metrics",
    "reset_metrics",
    # Market Structure
    "MarketStructureAdapter",
    "MarketStructureSignal",
    # ATR Expansion
    "ATRExpansionAdapter",
    "ATRSignal",
    "format_atr_signals",
]
