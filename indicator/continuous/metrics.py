"""
Metrics and Telemetry for Continuous Analysis.

Tracks latencies, throughput, and system health for monitoring and debugging.
"""

import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import threading


class MetricType(Enum):
    """Types of metrics we track."""
    LATENCY = "latency"          # Time to complete operation (ms)
    COUNTER = "counter"          # Cumulative count
    GAUGE = "gauge"              # Current value (can go up/down)
    RATE = "rate"                # Events per second


@dataclass
class LatencyStats:
    """Statistics for a latency metric."""
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    last_ms: float = 0.0

    # Rolling percentiles (approximate)
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0

    @property
    def mean_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0


@dataclass
class RateStats:
    """Statistics for a rate metric."""
    total_count: int = 0
    window_count: int = 0       # Count in current window
    window_start_ms: int = 0    # Start of current window
    window_duration_ms: int = 1000  # Window size (1 second default)
    rate_per_second: float = 0.0


class LatencyTracker:
    """
    Tracks latency statistics for an operation.

    Uses a sliding window for percentile approximation.
    """

    def __init__(self, window_size: int = 1000):
        """
        Args:
            window_size: Number of samples to keep for percentile calculation
        """
        self._samples: deque[float] = deque(maxlen=window_size)
        self._stats = LatencyStats()
        self._lock = threading.Lock()

    def record(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        with self._lock:
            self._samples.append(latency_ms)
            self._stats.count += 1
            self._stats.total_ms += latency_ms
            self._stats.last_ms = latency_ms
            self._stats.min_ms = min(self._stats.min_ms, latency_ms)
            self._stats.max_ms = max(self._stats.max_ms, latency_ms)

    def get_stats(self) -> LatencyStats:
        """Get current statistics with percentiles."""
        with self._lock:
            stats = LatencyStats(
                count=self._stats.count,
                total_ms=self._stats.total_ms,
                min_ms=self._stats.min_ms if self._stats.count > 0 else 0.0,
                max_ms=self._stats.max_ms,
                last_ms=self._stats.last_ms,
            )

            if self._samples:
                sorted_samples = sorted(self._samples)
                n = len(sorted_samples)
                stats.p50_ms = sorted_samples[int(n * 0.5)]
                stats.p95_ms = sorted_samples[min(int(n * 0.95), n - 1)]
                stats.p99_ms = sorted_samples[min(int(n * 0.99), n - 1)]

            return stats

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._samples.clear()
            self._stats = LatencyStats()


class RateTracker:
    """Tracks event rate (events per second)."""

    def __init__(self, window_ms: int = 1000):
        """
        Args:
            window_ms: Window duration for rate calculation
        """
        self._window_ms = window_ms
        self._stats = RateStats(window_duration_ms=window_ms)
        self._lock = threading.Lock()

    def record(self, count: int = 1) -> None:
        """Record event(s)."""
        now = int(time.time() * 1000)

        with self._lock:
            # Check if we need to start new window
            if now - self._stats.window_start_ms >= self._window_ms:
                # Calculate rate from completed window
                if self._stats.window_start_ms > 0:
                    elapsed_s = (now - self._stats.window_start_ms) / 1000.0
                    self._stats.rate_per_second = self._stats.window_count / elapsed_s if elapsed_s > 0 else 0

                # Start new window
                self._stats.window_start_ms = now
                self._stats.window_count = 0

            self._stats.window_count += count
            self._stats.total_count += count

    def get_stats(self) -> RateStats:
        """Get current rate statistics."""
        now = int(time.time() * 1000)

        with self._lock:
            # Calculate current rate
            elapsed_ms = now - self._stats.window_start_ms
            if elapsed_ms > 0:
                elapsed_s = elapsed_ms / 1000.0
                current_rate = self._stats.window_count / elapsed_s
            else:
                current_rate = 0

            return RateStats(
                total_count=self._stats.total_count,
                window_count=self._stats.window_count,
                window_start_ms=self._stats.window_start_ms,
                window_duration_ms=self._window_ms,
                rate_per_second=current_rate,
            )

    def reset(self) -> None:
        """Reset statistics."""
        with self._lock:
            self._stats = RateStats(window_duration_ms=self._window_ms)


class Timer:
    """Context manager for timing operations."""

    def __init__(self, tracker: LatencyTracker):
        self._tracker = tracker
        self._start: float = 0

    def __enter__(self) -> 'Timer':
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self._tracker.record(elapsed_ms)


@dataclass
class SystemMetrics:
    """Aggregated system metrics snapshot."""
    timestamp_ms: int

    # Data ingestion
    trades_per_second: float = 0.0
    orderbook_updates_per_second: float = 0.0
    total_trades_processed: int = 0
    total_orderbook_updates: int = 0

    # Signal computation latencies (ms)
    signal_compute_latency: LatencyStats = field(default_factory=LatencyStats)
    volume_engine_latency: LatencyStats = field(default_factory=LatencyStats)
    delta_engine_latency: LatencyStats = field(default_factory=LatencyStats)
    book_engine_latency: LatencyStats = field(default_factory=LatencyStats)
    unified_score_latency: LatencyStats = field(default_factory=LatencyStats)

    # State machine
    state_transitions: int = 0
    trade_signals_generated: int = 0

    # Data quality
    data_gaps_detected: int = 0
    reconnections: int = 0

    # Memory (if available)
    window_item_counts: Dict[int, int] = field(default_factory=dict)


class MetricsCollector:
    """
    Central metrics collector for continuous analysis system.

    Usage:
        metrics = MetricsCollector()

        # Time an operation
        with metrics.time("signal_compute"):
            compute_signals()

        # Record event
        metrics.record_event("trades")

        # Get snapshot
        snapshot = metrics.get_snapshot()
    """

    def __init__(self):
        # Latency trackers
        self._latencies: Dict[str, LatencyTracker] = {
            "signal_compute": LatencyTracker(),
            "volume_engine": LatencyTracker(),
            "delta_engine": LatencyTracker(),
            "book_engine": LatencyTracker(),
            "oi_funding_engine": LatencyTracker(),
            "unified_score": LatencyTracker(),
            "state_machine": LatencyTracker(),
            "trade_ingestion": LatencyTracker(),
            "orderbook_ingestion": LatencyTracker(),
        }

        # Rate trackers
        self._rates: Dict[str, RateTracker] = {
            "trades": RateTracker(),
            "orderbook_updates": RateTracker(),
            "signals_computed": RateTracker(),
        }

        # Counters
        self._counters: Dict[str, int] = {
            "state_transitions": 0,
            "trade_signals": 0,
            "data_gaps": 0,
            "reconnections": 0,
            "errors": 0,
        }

        self._lock = threading.Lock()
        self._start_time = time.time()

        # Callbacks for real-time monitoring
        self._on_slow_operation: List[Callable[[str, float], None]] = []
        self._slow_threshold_ms = 100.0  # Alert if operation takes > 100ms

    def time(self, operation: str) -> Timer:
        """
        Create a timer context manager for an operation.

        Usage:
            with metrics.time("signal_compute"):
                compute_signals()
        """
        if operation not in self._latencies:
            self._latencies[operation] = LatencyTracker()
        return Timer(self._latencies[operation])

    def record_latency(self, operation: str, latency_ms: float) -> None:
        """Record a latency measurement directly."""
        if operation not in self._latencies:
            self._latencies[operation] = LatencyTracker()
        self._latencies[operation].record(latency_ms)

        # Check for slow operations
        if latency_ms > self._slow_threshold_ms:
            for callback in self._on_slow_operation:
                try:
                    callback(operation, latency_ms)
                except Exception:
                    pass

    def record_event(self, event_type: str, count: int = 1) -> None:
        """Record event(s) for rate tracking."""
        if event_type not in self._rates:
            self._rates[event_type] = RateTracker()
        self._rates[event_type].record(count)

    def increment(self, counter: str, amount: int = 1) -> None:
        """Increment a counter."""
        with self._lock:
            if counter not in self._counters:
                self._counters[counter] = 0
            self._counters[counter] += amount

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value (stored in counters for simplicity)."""
        with self._lock:
            self._counters[name] = int(value)

    def get_latency_stats(self, operation: str) -> Optional[LatencyStats]:
        """Get latency statistics for an operation."""
        tracker = self._latencies.get(operation)
        return tracker.get_stats() if tracker else None

    def get_rate_stats(self, event_type: str) -> Optional[RateStats]:
        """Get rate statistics for an event type."""
        tracker = self._rates.get(event_type)
        return tracker.get_stats() if tracker else None

    def get_counter(self, name: str) -> int:
        """Get counter value."""
        with self._lock:
            return self._counters.get(name, 0)

    def get_snapshot(self) -> SystemMetrics:
        """Get complete metrics snapshot."""
        now = int(time.time() * 1000)

        trades_rate = self._rates.get("trades")
        orderbook_rate = self._rates.get("orderbook_updates")

        return SystemMetrics(
            timestamp_ms=now,
            trades_per_second=trades_rate.get_stats().rate_per_second if trades_rate else 0,
            orderbook_updates_per_second=orderbook_rate.get_stats().rate_per_second if orderbook_rate else 0,
            total_trades_processed=trades_rate.get_stats().total_count if trades_rate else 0,
            total_orderbook_updates=orderbook_rate.get_stats().total_count if orderbook_rate else 0,
            signal_compute_latency=self._latencies["signal_compute"].get_stats(),
            volume_engine_latency=self._latencies["volume_engine"].get_stats(),
            delta_engine_latency=self._latencies["delta_engine"].get_stats(),
            book_engine_latency=self._latencies["book_engine"].get_stats(),
            unified_score_latency=self._latencies["unified_score"].get_stats(),
            state_transitions=self.get_counter("state_transitions"),
            trade_signals_generated=self.get_counter("trade_signals"),
            data_gaps_detected=self.get_counter("data_gaps"),
            reconnections=self.get_counter("reconnections"),
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get human-readable metrics summary."""
        snapshot = self.get_snapshot()
        uptime = time.time() - self._start_time

        return {
            "uptime_seconds": uptime,
            "ingestion": {
                "trades_per_second": f"{snapshot.trades_per_second:.1f}",
                "orderbook_updates_per_second": f"{snapshot.orderbook_updates_per_second:.1f}",
                "total_trades": snapshot.total_trades_processed,
            },
            "latencies_ms": {
                "signal_compute": {
                    "mean": f"{snapshot.signal_compute_latency.mean_ms:.2f}",
                    "p50": f"{snapshot.signal_compute_latency.p50_ms:.2f}",
                    "p95": f"{snapshot.signal_compute_latency.p95_ms:.2f}",
                    "p99": f"{snapshot.signal_compute_latency.p99_ms:.2f}",
                    "max": f"{snapshot.signal_compute_latency.max_ms:.2f}",
                },
                "volume_engine": {
                    "mean": f"{snapshot.volume_engine_latency.mean_ms:.2f}",
                    "p95": f"{snapshot.volume_engine_latency.p95_ms:.2f}",
                },
                "book_engine": {
                    "mean": f"{snapshot.book_engine_latency.mean_ms:.2f}",
                    "p95": f"{snapshot.book_engine_latency.p95_ms:.2f}",
                },
                "unified_score": {
                    "mean": f"{snapshot.unified_score_latency.mean_ms:.2f}",
                    "p95": f"{snapshot.unified_score_latency.p95_ms:.2f}",
                },
            },
            "state_machine": {
                "transitions": snapshot.state_transitions,
                "trade_signals": snapshot.trade_signals_generated,
            },
            "health": {
                "data_gaps": snapshot.data_gaps_detected,
                "reconnections": snapshot.reconnections,
            },
        }

    def on_slow_operation(self, callback: Callable[[str, float], None]) -> None:
        """Register callback for slow operation alerts."""
        self._on_slow_operation.append(callback)

    def set_slow_threshold(self, threshold_ms: float) -> None:
        """Set threshold for slow operation alerts."""
        self._slow_threshold_ms = threshold_ms

    def reset(self) -> None:
        """Reset all metrics."""
        for tracker in self._latencies.values():
            tracker.reset()
        for tracker in self._rates.values():
            tracker.reset()
        with self._lock:
            self._counters = {k: 0 for k in self._counters}
        self._start_time = time.time()


# Global metrics instance for convenience
_global_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


def reset_metrics() -> None:
    """Reset global metrics."""
    global _global_metrics
    _global_metrics = None
