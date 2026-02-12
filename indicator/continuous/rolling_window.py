"""
Rolling Window computations for continuous data.

Provides efficient time-based aggregations over streaming data.
"""

from typing import Generic, TypeVar, List, Optional, Callable, Dict, Any
from dataclasses import dataclass, field
import time
from collections import deque

from .ring_buffer import RingBuffer, TimestampedRingBuffer, TimestampedItem
from .data_types import TradeEvent, OrderbookSnapshot


T = TypeVar('T')


@dataclass
class WindowStats:
    """Statistics computed over a rolling window."""
    window_seconds: int
    item_count: int
    span_ms: int  # Actual time span of data

    # For numeric data
    sum: float = 0.0
    mean: float = 0.0
    min: float = 0.0
    max: float = 0.0

    # Metadata
    is_complete: bool = False  # True if window has full data


class RollingWindow(Generic[T]):
    """
    Rolling window over timestamped data.

    Efficiently maintains aggregates over a time window,
    updating as new data arrives and old data falls out.

    Example:
        window = RollingWindow[TradeEvent](window_seconds=60)
        window.add(trade)

        # Get all trades in window
        trades = window.items()

        # Get computed stats
        stats = window.compute_stats(lambda t: t.quantity)
    """

    def __init__(
        self,
        window_seconds: int,
        max_items: int = 10000,
    ):
        """
        Initialize rolling window.

        Args:
            window_seconds: Window duration in seconds
            max_items: Maximum items to store (safety limit)
        """
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")

        self._window_ms = window_seconds * 1000
        self._window_seconds = window_seconds
        self._max_items = max_items

        # Use deque for efficient removal from both ends
        self._items: deque[TimestampedItem[T]] = deque(maxlen=max_items)

        # Track window boundaries
        self._oldest_ts: Optional[int] = None
        self._newest_ts: Optional[int] = None

    @property
    def window_seconds(self) -> int:
        return self._window_seconds

    def _current_time_ms(self) -> int:
        """Current time in milliseconds."""
        return int(time.time() * 1000)

    def _evict_old(self, current_ms: Optional[int] = None) -> int:
        """Remove items older than window. Returns count removed."""
        if not self._items:
            return 0

        cutoff = (current_ms or self._current_time_ms()) - self._window_ms
        removed = 0

        while self._items and self._items[0].timestamp_ms < cutoff:
            self._items.popleft()
            removed += 1

        # Update oldest timestamp
        if self._items:
            self._oldest_ts = self._items[0].timestamp_ms
        else:
            self._oldest_ts = None
            self._newest_ts = None

        return removed

    def add(self, item: T, timestamp_ms: Optional[int] = None) -> None:
        """
        Add item to window.

        Args:
            item: Item to add
            timestamp_ms: Optional timestamp (uses current time if not provided)
        """
        ts = timestamp_ms or self._current_time_ms()

        # Evict old items first
        self._evict_old(ts)

        # Add new item
        self._items.append(TimestampedItem(ts, item))

        # Update boundaries
        if self._oldest_ts is None:
            self._oldest_ts = ts
        self._newest_ts = ts

    def add_batch(self, items: List[T], timestamps_ms: Optional[List[int]] = None) -> None:
        """Add multiple items efficiently."""
        if not items:
            return

        if timestamps_ms is None:
            ts = self._current_time_ms()
            timestamps_ms = [ts] * len(items)

        for item, ts in zip(items, timestamps_ms):
            self.add(item, ts)

    def items(self) -> List[T]:
        """Get all items in window (evicts old first)."""
        self._evict_old()
        return [item.value for item in self._items]

    def items_with_timestamps(self) -> List[TimestampedItem[T]]:
        """Get all items with timestamps."""
        self._evict_old()
        return list(self._items)

    def __len__(self) -> int:
        """Current item count."""
        self._evict_old()
        return len(self._items)

    def __bool__(self) -> bool:
        return len(self) > 0

    @property
    def is_complete(self) -> bool:
        """True if window has data spanning full duration."""
        if not self._items:
            return False
        span = (self._newest_ts or 0) - (self._oldest_ts or 0)
        return span >= self._window_ms * 0.9  # 90% coverage

    @property
    def span_ms(self) -> int:
        """Actual time span of data in window."""
        if not self._items:
            return 0
        return (self._newest_ts or 0) - (self._oldest_ts or 0)

    @property
    def coverage(self) -> float:
        """Fraction of window with data (0-1)."""
        return min(1.0, self.span_ms / self._window_ms)

    def compute_stats(
        self,
        value_fn: Callable[[T], float],
    ) -> WindowStats:
        """
        Compute statistics over window.

        Args:
            value_fn: Function to extract numeric value from item

        Returns:
            WindowStats with computed metrics
        """
        items = self.items()

        if not items:
            return WindowStats(
                window_seconds=self._window_seconds,
                item_count=0,
                span_ms=0,
            )

        values = [value_fn(item) for item in items]

        return WindowStats(
            window_seconds=self._window_seconds,
            item_count=len(items),
            span_ms=self.span_ms,
            sum=sum(values),
            mean=sum(values) / len(values),
            min=min(values),
            max=max(values),
            is_complete=self.is_complete,
        )

    def aggregate(
        self,
        agg_fn: Callable[[List[T]], Any],
    ) -> Any:
        """Apply custom aggregation function."""
        return agg_fn(self.items())

    def clear(self) -> None:
        """Clear all items."""
        self._items.clear()
        self._oldest_ts = None
        self._newest_ts = None


class TradeWindow(RollingWindow[TradeEvent]):
    """
    Specialized rolling window for trade events.

    Provides trade-specific aggregations like volume delta, VWAP, etc.

    Optimized: Uses O(1) sum updates on eviction by tracking removed values.
    """

    def __init__(self, window_seconds: int, max_items: int = 50000):
        super().__init__(window_seconds, max_items)

        # Maintain running sums for efficiency
        self._buy_volume = 0.0
        self._sell_volume = 0.0
        self._buy_notional = 0.0
        self._sell_notional = 0.0
        self._total_notional = 0.0

        # Drift correction: periodically recalculate to fix floating point errors
        self._ops_since_recalc = 0
        self._recalc_interval = 10000  # Recalculate every 10k operations

    def _evict_old_with_sums(self, current_ms: Optional[int] = None) -> int:
        """
        Remove items older than window, subtracting their values from sums.

        Returns count removed. O(k) where k = items removed (typically small).
        """
        if not self._items:
            return 0

        cutoff = (current_ms or self._current_time_ms()) - self._window_ms
        removed = 0

        while self._items and self._items[0].timestamp_ms < cutoff:
            evicted = self._items.popleft()
            trade = evicted.value
            notional = trade.notional

            # Subtract evicted trade from running sums
            if trade.is_buy:
                self._buy_volume -= trade.quantity
                self._buy_notional -= notional
            else:
                self._sell_volume -= trade.quantity
                self._sell_notional -= notional
            self._total_notional -= notional

            removed += 1

        # Update oldest timestamp
        if self._items:
            self._oldest_ts = self._items[0].timestamp_ms
        else:
            self._oldest_ts = None
            self._newest_ts = None
            # Reset sums to zero when empty (avoids drift)
            self._buy_volume = 0.0
            self._sell_volume = 0.0
            self._buy_notional = 0.0
            self._sell_notional = 0.0
            self._total_notional = 0.0

        return removed

    def add(self, trade: TradeEvent, timestamp_ms: Optional[int] = None) -> None:
        """Add trade and update running sums in O(1) amortized."""
        # Evict old items, subtracting their values from sums
        self._evict_old_with_sums(timestamp_ms)

        # Add new trade
        ts = timestamp_ms or trade.timestamp_ms
        self._items.append(TimestampedItem(ts, trade))

        # Update boundaries
        if self._oldest_ts is None:
            self._oldest_ts = ts
        self._newest_ts = ts

        # Update running sums
        notional = trade.notional
        if trade.is_buy:
            self._buy_volume += trade.quantity
            self._buy_notional += notional
        else:
            self._sell_volume += trade.quantity
            self._sell_notional += notional
        self._total_notional += notional

        # Periodic drift correction
        self._ops_since_recalc += 1
        if self._ops_since_recalc >= self._recalc_interval:
            self._recalculate_sums()
            self._ops_since_recalc = 0

    def _recalculate_sums(self) -> None:
        """Recalculate running sums from scratch (for drift correction)."""
        self._buy_volume = 0.0
        self._sell_volume = 0.0
        self._buy_notional = 0.0
        self._sell_notional = 0.0
        self._total_notional = 0.0

        for item in self._items:
            trade = item.value
            notional = trade.notional
            if trade.is_buy:
                self._buy_volume += trade.quantity
                self._buy_notional += notional
            else:
                self._sell_volume += trade.quantity
                self._sell_notional += notional
            self._total_notional += notional

    @property
    def buy_volume(self) -> float:
        """Total buy volume in window."""
        self._evict_old_with_sums()
        return self._buy_volume

    @property
    def sell_volume(self) -> float:
        """Total sell volume in window."""
        self._evict_old_with_sums()
        return self._sell_volume

    @property
    def total_volume(self) -> float:
        """Total volume in window."""
        return self.buy_volume + self.sell_volume

    @property
    def delta(self) -> float:
        """Volume delta (buy - sell)."""
        return self.buy_volume - self.sell_volume

    @property
    def delta_ratio(self) -> float:
        """Delta as ratio of total: (buy - sell) / total."""
        total = self.total_volume
        return self.delta / total if total > 0 else 0.0

    @property
    def vwap(self) -> float:
        """Volume-weighted average price."""
        vol = self.total_volume
        if vol <= 0:
            return 0.0
        return self._total_notional / vol

    @property
    def trade_count(self) -> int:
        """Number of trades in window."""
        return len(self)

    def get_cvd(self) -> float:
        """Get cumulative volume delta (same as delta for this window)."""
        return self.delta


class MultiTimeframeWindows:
    """
    Manager for multiple rolling windows at different timeframes.

    Implements the 3-layer timeframe model:
    - Layer 1 (Micro): 5s, 15s, 30s
    - Layer 2 (Decision): 60s, 120s, 180s
    - Layer 3 (Context): 900s (15m), 3600s (1h)
    """

    def __init__(
        self,
        window_seconds_list: Optional[List[int]] = None,
        max_items_per_window: int = 50000,
    ):
        """
        Initialize multi-timeframe windows.

        Args:
            window_seconds_list: List of window durations in seconds
            max_items_per_window: Max items per window
        """
        if window_seconds_list is None:
            # Default 3-layer model
            # Added 300s (5min) for absorption baseline calculation
            window_seconds_list = [5, 15, 30, 60, 120, 180, 300, 900, 3600]

        self._windows: Dict[int, TradeWindow] = {}
        for seconds in window_seconds_list:
            self._windows[seconds] = TradeWindow(seconds, max_items_per_window)

    def add_trade(self, trade: TradeEvent) -> None:
        """Add trade to all windows."""
        for window in self._windows.values():
            window.add(trade, trade.timestamp_ms)

    def add_trades(self, trades: List[TradeEvent]) -> None:
        """Add multiple trades to all windows."""
        for trade in trades:
            self.add_trade(trade)

    def get_window(self, seconds: int) -> Optional[TradeWindow]:
        """Get window by duration."""
        return self._windows.get(seconds)

    def get_all_deltas(self) -> Dict[int, float]:
        """Get delta for all windows."""
        return {sec: w.delta for sec, w in self._windows.items()}

    def get_all_delta_ratios(self) -> Dict[int, float]:
        """Get delta ratio for all windows."""
        return {sec: w.delta_ratio for sec, w in self._windows.items()}

    def get_all_volumes(self) -> Dict[int, float]:
        """Get total volume for all windows."""
        return {sec: w.total_volume for sec, w in self._windows.items()}

    @property
    def windows(self) -> Dict[int, TradeWindow]:
        """Access to all windows."""
        return self._windows

    # Layer accessors
    @property
    def micro_windows(self) -> Dict[int, TradeWindow]:
        """Layer 1: 5s, 15s, 30s windows."""
        return {k: v for k, v in self._windows.items() if k <= 30}

    @property
    def decision_windows(self) -> Dict[int, TradeWindow]:
        """Layer 2: 60s, 120s, 180s windows."""
        return {k: v for k, v in self._windows.items() if 30 < k <= 180}

    @property
    def context_windows(self) -> Dict[int, TradeWindow]:
        """Layer 3: 900s+, 3600s windows."""
        return {k: v for k, v in self._windows.items() if k > 180}

    def clear(self) -> None:
        """Clear all windows."""
        for window in self._windows.values():
            window.clear()


class OrderbookHistory:
    """
    History of orderbook snapshots for change tracking.

    Used for spoof detection and absorption analysis.
    """

    def __init__(self, max_snapshots: int = 100):
        """
        Args:
            max_snapshots: Maximum snapshots to retain
        """
        self._snapshots = RingBuffer[OrderbookSnapshot](max_snapshots)

    def add(self, snapshot: OrderbookSnapshot) -> None:
        """Add new snapshot."""
        self._snapshots.append(snapshot)

    def latest(self) -> Optional[OrderbookSnapshot]:
        """Get most recent snapshot."""
        return self._snapshots.newest()

    def previous(self) -> Optional[OrderbookSnapshot]:
        """Get second most recent snapshot."""
        if len(self._snapshots) < 2:
            return None
        return self._snapshots[-2]

    def last_n(self, n: int) -> List[OrderbookSnapshot]:
        """Get last n snapshots."""
        return self._snapshots.last(n)

    def imbalance_change(self, levels: int = 10) -> Optional[float]:
        """
        Get imbalance change from previous to current snapshot.

        Returns:
            Change in imbalance or None if insufficient data
        """
        current = self.latest()
        prev = self.previous()

        if current is None or prev is None:
            return None

        return current.imbalance(levels) - prev.imbalance(levels)

    def depth_change(self, side: str, levels: int = 10) -> Optional[tuple[float, float]]:
        """
        Get depth change from previous to current.

        Args:
            side: "bid" or "ask"
            levels: Number of levels to consider

        Returns:
            (absolute_change, percent_change) or None
        """
        current = self.latest()
        prev = self.previous()

        if current is None or prev is None:
            return None

        if side == "bid":
            curr_depth = current.bid_depth(levels)
            prev_depth = prev.bid_depth(levels)
        else:
            curr_depth = current.ask_depth(levels)
            prev_depth = prev.ask_depth(levels)

        abs_change = curr_depth - prev_depth
        pct_change = abs_change / prev_depth if prev_depth > 0 else 0.0

        return abs_change, pct_change

    def __len__(self) -> int:
        return len(self._snapshots)

    def clear(self) -> None:
        self._snapshots.clear()
