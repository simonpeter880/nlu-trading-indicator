"""
Ring Buffer - Fixed-size circular buffer for streaming data.

Memory-efficient storage for time-series data with O(1) append and O(1) access.
"""

from typing import Generic, TypeVar, List, Optional, Iterator, Callable
from dataclasses import dataclass, field
import time

T = TypeVar('T')


class RingBuffer(Generic[T]):
    """
    Fixed-size circular buffer with O(1) operations.

    When full, oldest items are overwritten.

    Example:
        buf = RingBuffer[float](maxlen=100)
        buf.append(1.0)
        buf.append(2.0)
        latest = buf[-1]  # 2.0
        all_data = buf.to_list()  # [1.0, 2.0]
    """

    __slots__ = ('_buffer', '_maxlen', '_head', '_size')

    def __init__(self, maxlen: int):
        """
        Initialize ring buffer.

        Args:
            maxlen: Maximum number of elements (must be > 0)
        """
        if maxlen <= 0:
            raise ValueError("maxlen must be positive")
        self._buffer: List[Optional[T]] = [None] * maxlen
        self._maxlen = maxlen
        self._head = 0  # Next write position
        self._size = 0

    def append(self, item: T) -> None:
        """Append item to buffer. O(1)."""
        self._buffer[self._head] = item
        self._head = (self._head + 1) % self._maxlen
        if self._size < self._maxlen:
            self._size += 1

    def extend(self, items: List[T]) -> None:
        """Append multiple items. O(n)."""
        for item in items:
            self.append(item)

    def __len__(self) -> int:
        """Return current number of items."""
        return self._size

    def __bool__(self) -> bool:
        """Return True if buffer has items."""
        return self._size > 0

    def __getitem__(self, index: int) -> T:
        """
        Get item by index. Supports negative indexing.

        buf[0] = oldest item
        buf[-1] = newest item
        """
        if self._size == 0:
            raise IndexError("buffer is empty")

        if index < 0:
            index = self._size + index

        if index < 0 or index >= self._size:
            raise IndexError(f"index {index} out of range for size {self._size}")

        # Calculate actual position
        start = (self._head - self._size) % self._maxlen
        actual_idx = (start + index) % self._maxlen

        item = self._buffer[actual_idx]
        if item is None:
            raise IndexError("unexpected None in buffer")
        return item

    def __iter__(self) -> Iterator[T]:
        """Iterate from oldest to newest."""
        for i in range(self._size):
            yield self[i]

    @property
    def maxlen(self) -> int:
        """Maximum buffer capacity."""
        return self._maxlen

    @property
    def is_full(self) -> bool:
        """True if buffer is at capacity."""
        return self._size == self._maxlen

    def clear(self) -> None:
        """Clear all items."""
        self._buffer = [None] * self._maxlen
        self._head = 0
        self._size = 0

    def to_list(self) -> List[T]:
        """Convert to list (oldest first)."""
        return list(self)

    def last(self, n: int) -> List[T]:
        """Get last n items (oldest first of the n)."""
        if n <= 0:
            return []
        n = min(n, self._size)
        start = self._size - n
        return [self[i] for i in range(start, self._size)]

    def newest(self) -> Optional[T]:
        """Get newest item or None if empty."""
        return self[-1] if self._size > 0 else None

    def oldest(self) -> Optional[T]:
        """Get oldest item or None if empty."""
        return self[0] if self._size > 0 else None


@dataclass
class TimestampedItem(Generic[T]):
    """Item with timestamp."""
    timestamp_ms: int
    value: T


class TimestampedRingBuffer(Generic[T]):
    """
    Ring buffer with timestamp support for time-based queries.

    Optimized for:
    - Appending with timestamps
    - Querying items within time windows
    - Evicting items older than a threshold

    Example:
        buf = TimestampedRingBuffer[float](maxlen=1000, max_age_ms=60000)
        buf.append(1.0)  # Auto-timestamps with current time
        buf.append_at(2.0, timestamp_ms)  # Explicit timestamp

        # Get items from last 5 seconds
        recent = buf.since(time.time() * 1000 - 5000)
    """

    def __init__(self, maxlen: int, max_age_ms: Optional[int] = None):
        """
        Initialize timestamped buffer.

        Args:
            maxlen: Maximum number of elements
            max_age_ms: Optional max age in ms (items older are not returned)
        """
        self._buffer = RingBuffer[TimestampedItem[T]](maxlen)
        self._max_age_ms = max_age_ms

    def append(self, value: T) -> None:
        """Append with current timestamp."""
        ts = int(time.time() * 1000)
        self._buffer.append(TimestampedItem(ts, value))

    def append_at(self, value: T, timestamp_ms: int) -> None:
        """Append with explicit timestamp."""
        self._buffer.append(TimestampedItem(timestamp_ms, value))

    def __len__(self) -> int:
        return len(self._buffer)

    def __bool__(self) -> bool:
        return bool(self._buffer)

    @property
    def is_full(self) -> bool:
        return self._buffer.is_full

    def newest(self) -> Optional[TimestampedItem[T]]:
        """Get newest item."""
        return self._buffer.newest()

    def oldest(self) -> Optional[TimestampedItem[T]]:
        """Get oldest item."""
        return self._buffer.oldest()

    def since(self, cutoff_ms: int) -> List[T]:
        """Get all values since cutoff timestamp."""
        result = []
        for item in self._buffer:
            if item.timestamp_ms >= cutoff_ms:
                result.append(item.value)
        return result

    def since_with_timestamps(self, cutoff_ms: int) -> List[TimestampedItem[T]]:
        """Get all items (with timestamps) since cutoff."""
        result = []
        for item in self._buffer:
            if item.timestamp_ms >= cutoff_ms:
                result.append(item)
        return result

    def window(self, start_ms: int, end_ms: int) -> List[T]:
        """Get values within time window [start, end]."""
        result = []
        for item in self._buffer:
            if start_ms <= item.timestamp_ms <= end_ms:
                result.append(item.value)
        return result

    def last_n_seconds(self, seconds: float) -> List[T]:
        """Get values from last n seconds."""
        cutoff = int(time.time() * 1000) - int(seconds * 1000)
        return self.since(cutoff)

    def values(self) -> List[T]:
        """Get all values (oldest first), respecting max_age."""
        if self._max_age_ms is not None:
            cutoff = int(time.time() * 1000) - self._max_age_ms
            return self.since(cutoff)
        return [item.value for item in self._buffer]

    def clear(self) -> None:
        """Clear all items."""
        self._buffer.clear()

    def span_ms(self) -> int:
        """Get time span of data in buffer (newest - oldest)."""
        if len(self._buffer) < 2:
            return 0
        oldest = self._buffer.oldest()
        newest = self._buffer.newest()
        if oldest is None or newest is None:
            return 0
        return newest.timestamp_ms - oldest.timestamp_ms


class SumBuffer:
    """
    Specialized buffer that maintains a running sum.

    O(1) sum updates on append/eviction.
    Useful for volume totals, trade counts, etc.
    """

    __slots__ = ('_buffer', '_sum')

    def __init__(self, maxlen: int):
        self._buffer = RingBuffer[float](maxlen)
        self._sum = 0.0

    def append(self, value: float) -> None:
        """Append value, updating running sum."""
        if self._buffer.is_full:
            # Subtract the value that will be overwritten
            oldest = self._buffer.oldest()
            if oldest is not None:
                self._sum -= oldest
        self._buffer.append(value)
        self._sum += value

    @property
    def sum(self) -> float:
        """Current running sum."""
        return self._sum

    @property
    def mean(self) -> float:
        """Current mean (sum / count)."""
        n = len(self._buffer)
        return self._sum / n if n > 0 else 0.0

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()
        self._sum = 0.0


class DeltaBuffer:
    """
    Buffer that tracks buy/sell delta.

    Maintains separate sums for buy and sell volume.
    """

    __slots__ = ('_buy_buffer', '_sell_buffer', '_buy_sum', '_sell_sum')

    def __init__(self, maxlen: int):
        self._buy_buffer = RingBuffer[float](maxlen)
        self._sell_buffer = RingBuffer[float](maxlen)
        self._buy_sum = 0.0
        self._sell_sum = 0.0

    def append(self, buy_vol: float, sell_vol: float) -> None:
        """Append buy and sell volumes."""
        # Handle eviction for buy
        if self._buy_buffer.is_full:
            oldest = self._buy_buffer.oldest()
            if oldest is not None:
                self._buy_sum -= oldest
        self._buy_buffer.append(buy_vol)
        self._buy_sum += buy_vol

        # Handle eviction for sell
        if self._sell_buffer.is_full:
            oldest = self._sell_buffer.oldest()
            if oldest is not None:
                self._sell_sum -= oldest
        self._sell_buffer.append(sell_vol)
        self._sell_sum += sell_vol

    @property
    def delta(self) -> float:
        """Buy volume - Sell volume."""
        return self._buy_sum - self._sell_sum

    @property
    def total_volume(self) -> float:
        """Total volume (buy + sell)."""
        return self._buy_sum + self._sell_sum

    @property
    def delta_ratio(self) -> float:
        """Delta as ratio of total: (buy - sell) / (buy + sell)."""
        total = self.total_volume
        return self.delta / total if total > 0 else 0.0

    @property
    def buy_ratio(self) -> float:
        """Buy volume as ratio of total."""
        total = self.total_volume
        return self._buy_sum / total if total > 0 else 0.5

    def __len__(self) -> int:
        return len(self._buy_buffer)

    def clear(self) -> None:
        self._buy_buffer.clear()
        self._sell_buffer.clear()
        self._buy_sum = 0.0
        self._sell_sum = 0.0
