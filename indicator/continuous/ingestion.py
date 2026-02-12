"""
Data Ingestion Layer - WebSocket streams for continuous market data.

Handles:
- aggTrades stream (continuous)
- Orderbook snapshots (sparse, 250ms)
- OI updates (10s polling)
- Funding updates (event-based)
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

import aiohttp

from .data_types import (
    FundingUpdate,
    IngestionConfig,
    OIUpdate,
    OrderbookLevel,
    OrderbookSnapshot,
    TradeEvent,
)
from .ring_buffer import TimestampedRingBuffer
from .rolling_window import MultiTimeframeWindows, OrderbookHistory

logger = logging.getLogger(__name__)


class StreamState(Enum):
    """State of a data stream."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class StreamStats:
    """Statistics for a data stream."""

    messages_received: int = 0
    bytes_received: int = 0
    last_message_time: Optional[int] = None
    reconnect_count: int = 0
    error_count: int = 0


class BaseStream(ABC):
    """Base class for data streams."""

    def __init__(self, symbol: str):
        self.symbol = symbol.upper().replace("/", "").replace("-", "")
        self._state = StreamState.DISCONNECTED
        self._stats = StreamStats()
        self._callbacks: List[Callable] = []
        self._running = False

    @property
    def state(self) -> StreamState:
        return self._state

    @property
    def stats(self) -> StreamStats:
        return self._stats

    def add_callback(self, callback: Callable) -> None:
        """Add callback to be called when new data arrives."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def _notify_callbacks(self, data: Any) -> None:
        """Notify all callbacks with new data."""
        for callback in self._callbacks:
            try:
                result = callback(data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Callback error: {e}")

    @abstractmethod
    async def start(self) -> None:
        """Start the stream."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the stream."""
        pass


class AggTradeStream(BaseStream):
    """
    WebSocket stream for aggregated trades.

    Binance aggTrade stream provides real-time trade data.
    Each message contains:
    - Trade ID, price, quantity
    - Buyer/seller maker flag (for delta calculation)
    """

    WS_BASE = "wss://fstream.binance.com/ws"

    def __init__(
        self,
        symbol: str,
        on_trade: Optional[Callable[[TradeEvent], Awaitable[None]]] = None,
    ):
        super().__init__(symbol)
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._task: Optional[asyncio.Task] = None

        if on_trade:
            self.add_callback(on_trade)

    @property
    def stream_name(self) -> str:
        return f"{self.symbol.lower()}@aggTrade"

    async def start(self) -> None:
        """Connect and start receiving trades."""
        if self._running:
            return

        self._running = True
        self._state = StreamState.CONNECTING

        self._session = aiohttp.ClientSession()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the stream."""
        self._running = False

        if self._ws:
            await self._ws.close()
            self._ws = None

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._session:
            await self._session.close()
            self._session = None

        self._state = StreamState.DISCONNECTED

    async def _run(self) -> None:
        """Main stream loop with reconnection."""
        url = f"{self.WS_BASE}/{self.stream_name}"

        while self._running:
            try:
                logger.info(f"Connecting to {url}")

                async with self._session.ws_connect(
                    url,
                    heartbeat=30,
                    receive_timeout=60,
                ) as ws:
                    self._ws = ws
                    self._state = StreamState.CONNECTED
                    logger.info(f"Connected to {self.stream_name}")

                    async for msg in ws:
                        if not self._running:
                            break

                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_message(msg.data)
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            break

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._stats.error_count += 1
                logger.error(f"Stream error: {e}")

                if self._running:
                    self._state = StreamState.RECONNECTING
                    self._stats.reconnect_count += 1
                    await asyncio.sleep(min(5 * self._stats.reconnect_count, 30))

        self._state = StreamState.DISCONNECTED

    async def _handle_message(self, raw: str) -> None:
        """Parse and process trade message."""
        try:
            data = json.loads(raw)
            self._stats.messages_received += 1
            self._stats.bytes_received += len(raw)
            self._stats.last_message_time = int(time.time() * 1000)

            trade = TradeEvent(
                timestamp_ms=data["T"],
                price=float(data["p"]),
                quantity=float(data["q"]),
                is_buyer_maker=data["m"],
                trade_id=data["a"],
            )

            await self._notify_callbacks(trade)

        except Exception as e:
            logger.error(f"Error parsing trade: {e}")


class OrderbookStream(BaseStream):
    """
    Orderbook snapshot stream with sparse updates.

    Uses depth endpoint polling at configured interval (default 250ms).
    More efficient than full book WebSocket for our use case.
    """

    REST_BASE = "https://fapi.binance.com"

    def __init__(
        self,
        symbol: str,
        interval_ms: int = 250,
        depth: int = 50,  # Increased from 20 to 50 for better liquidity ladder analysis
        on_snapshot: Optional[Callable[[OrderbookSnapshot], Awaitable[None]]] = None,
    ):
        super().__init__(symbol)
        self._interval_ms = interval_ms
        self._depth = depth
        self._session: Optional[aiohttp.ClientSession] = None
        self._task: Optional[asyncio.Task] = None

        if on_snapshot:
            self.add_callback(on_snapshot)

    async def start(self) -> None:
        """Start polling for orderbook snapshots."""
        if self._running:
            return

        self._running = True
        self._state = StreamState.CONNECTING

        self._session = aiohttp.ClientSession()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the stream."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._session:
            await self._session.close()
            self._session = None

        self._state = StreamState.DISCONNECTED

    async def _run(self) -> None:
        """Main polling loop."""
        interval_s = self._interval_ms / 1000
        url = f"{self.REST_BASE}/fapi/v1/depth"

        self._state = StreamState.CONNECTED

        while self._running:
            try:
                start = time.time()

                async with self._session.get(
                    url, params={"symbol": self.symbol, "limit": self._depth}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        await self._handle_snapshot(data)
                    else:
                        logger.warning(f"Orderbook fetch failed: {resp.status}")
                        self._stats.error_count += 1

                # Sleep remaining interval
                elapsed = time.time() - start
                sleep_time = max(0, interval_s - elapsed)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Orderbook error: {e}")
                self._stats.error_count += 1
                await asyncio.sleep(1)

    async def _handle_snapshot(self, data: Dict) -> None:
        """Parse and process orderbook snapshot."""
        try:
            self._stats.messages_received += 1
            self._stats.last_message_time = int(time.time() * 1000)

            snapshot = OrderbookSnapshot(
                timestamp_ms=data.get("T", int(time.time() * 1000)),
                bids=[OrderbookLevel(float(b[0]), float(b[1])) for b in data["bids"]],
                asks=[OrderbookLevel(float(a[0]), float(a[1])) for a in data["asks"]],
                last_update_id=data.get("lastUpdateId", 0),
            )

            await self._notify_callbacks(snapshot)

        except Exception as e:
            logger.error(f"Error parsing orderbook: {e}")


class OIStream(BaseStream):
    """
    Open Interest polling stream.

    OI doesn't need high-frequency updates - every 10 seconds is sufficient.
    """

    REST_BASE = "https://fapi.binance.com"

    def __init__(
        self,
        symbol: str,
        interval_ms: int = 10000,
        on_update: Optional[Callable[[OIUpdate], Awaitable[None]]] = None,
    ):
        super().__init__(symbol)
        self._interval_ms = interval_ms
        self._session: Optional[aiohttp.ClientSession] = None
        self._task: Optional[asyncio.Task] = None

        if on_update:
            self.add_callback(on_update)

    async def start(self) -> None:
        """Start polling OI."""
        if self._running:
            return

        self._running = True
        self._session = aiohttp.ClientSession()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop polling."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._session:
            await self._session.close()
            self._session = None

    async def _run(self) -> None:
        """Main polling loop."""
        interval_s = self._interval_ms / 1000

        while self._running:
            try:
                start = time.time()

                # Fetch OI
                async with self._session.get(
                    f"{self.REST_BASE}/fapi/v1/openInterest", params={"symbol": self.symbol}
                ) as resp:
                    if resp.status == 200:
                        oi_data = await resp.json()

                        # Fetch mark price for OI value
                        async with self._session.get(
                            f"{self.REST_BASE}/fapi/v1/premiumIndex", params={"symbol": self.symbol}
                        ) as mark_resp:
                            if mark_resp.status == 200:
                                mark_data = await mark_resp.json()
                                await self._handle_update(oi_data, mark_data)

                elapsed = time.time() - start
                await asyncio.sleep(max(0, interval_s - elapsed))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"OI error: {e}")
                self._stats.error_count += 1
                await asyncio.sleep(5)

    async def _handle_update(self, oi_data: Dict, mark_data: Dict) -> None:
        """Parse and process OI update."""
        try:
            self._stats.messages_received += 1
            self._stats.last_message_time = int(time.time() * 1000)

            oi = float(oi_data["openInterest"])
            mark_price = float(mark_data["markPrice"])

            update = OIUpdate(
                timestamp_ms=int(time.time() * 1000),
                open_interest=oi,
                open_interest_value=oi * mark_price,
            )

            await self._notify_callbacks(update)

        except Exception as e:
            logger.error(f"Error parsing OI: {e}")


class FundingStream(BaseStream):
    """
    Funding rate stream.

    Funding updates come every 8 hours. We poll less frequently
    but also check for mark price updates.
    """

    REST_BASE = "https://fapi.binance.com"

    def __init__(
        self,
        symbol: str,
        interval_ms: int = 60000,  # Check every minute
        on_update: Optional[Callable[[FundingUpdate], Awaitable[None]]] = None,
    ):
        super().__init__(symbol)
        self._interval_ms = interval_ms
        self._session: Optional[aiohttp.ClientSession] = None
        self._task: Optional[asyncio.Task] = None
        self._last_funding_time: Optional[int] = None

        if on_update:
            self.add_callback(on_update)

    async def start(self) -> None:
        """Start monitoring funding."""
        if self._running:
            return

        self._running = True
        self._session = aiohttp.ClientSession()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop monitoring."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if self._session:
            await self._session.close()
            self._session = None

    async def _run(self) -> None:
        """Main polling loop."""
        interval_s = self._interval_ms / 1000

        while self._running:
            try:
                start = time.time()

                async with self._session.get(
                    f"{self.REST_BASE}/fapi/v1/premiumIndex", params={"symbol": self.symbol}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        await self._handle_update(data)

                elapsed = time.time() - start
                await asyncio.sleep(max(0, interval_s - elapsed))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Funding error: {e}")
                self._stats.error_count += 1
                await asyncio.sleep(5)

    async def _handle_update(self, data: Dict) -> None:
        """Parse and process funding update."""
        try:
            self._stats.messages_received += 1
            self._stats.last_message_time = int(time.time() * 1000)

            next_funding_time = int(data["nextFundingTime"])

            # Only notify if funding time changed (new period)
            is_new_period = (
                self._last_funding_time is None or next_funding_time != self._last_funding_time
            )

            update = FundingUpdate(
                timestamp_ms=int(time.time() * 1000),
                funding_rate=float(data["lastFundingRate"]),
                mark_price=float(data["markPrice"]),
                next_funding_time=next_funding_time,
            )

            self._last_funding_time = next_funding_time

            # Always notify (mark price is useful), but could filter for new periods
            await self._notify_callbacks(update)

        except Exception as e:
            logger.error(f"Error parsing funding: {e}")


class DataIngestionManager:
    """
    Manages all data streams for a symbol.

    Coordinates:
    - aggTrade stream (continuous)
    - Orderbook snapshots (250ms)
    - OI updates (10s)
    - Funding updates (60s)

    Routes data to appropriate storage (rolling windows, buffers).
    """

    def __init__(
        self,
        symbol: str,
        config: Optional[IngestionConfig] = None,
    ):
        self.symbol = symbol.upper().replace("/", "").replace("-", "")
        self.config = config or IngestionConfig()

        # Create streams
        self._trade_stream = AggTradeStream(
            symbol,
            on_trade=self._on_trade,
        )
        self._book_stream = OrderbookStream(
            symbol,
            interval_ms=self.config.orderbook_interval_ms,
            depth=self.config.orderbook_depth,
            on_snapshot=self._on_orderbook,
        )
        self._oi_stream = OIStream(
            symbol,
            interval_ms=self.config.oi_interval_ms,
            on_update=self._on_oi,
        )
        self._funding_stream = FundingStream(
            symbol,
            on_update=self._on_funding,
        )

        # Storage
        self.trade_windows = MultiTimeframeWindows()
        self.orderbook_history = OrderbookHistory(self.config.orderbook_buffer_size)
        self.oi_history = TimestampedRingBuffer[OIUpdate](self.config.oi_buffer_size)
        self.funding_history = TimestampedRingBuffer[FundingUpdate](self.config.funding_buffer_size)

        # Latest values
        self._latest_price: float = 0.0
        self._latest_book: Optional[OrderbookSnapshot] = None
        self._latest_oi: Optional[OIUpdate] = None
        self._latest_funding: Optional[FundingUpdate] = None

        # Callbacks for external consumers
        self._on_update_callbacks: List[Callable] = []

        # State
        self._running = False

    @property
    def latest_price(self) -> float:
        return self._latest_price

    @property
    def latest_book(self) -> Optional[OrderbookSnapshot]:
        return self._latest_book

    @property
    def latest_oi(self) -> Optional[OIUpdate]:
        return self._latest_oi

    @property
    def latest_funding(self) -> Optional[FundingUpdate]:
        return self._latest_funding

    def add_update_callback(self, callback: Callable) -> None:
        """Add callback for any data update."""
        self._on_update_callbacks.append(callback)

    async def _notify_update(self, update_type: str, data: Any) -> None:
        """Notify callbacks of data update."""
        for callback in self._on_update_callbacks:
            try:
                result = callback(update_type, data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Update callback error: {e}")

    async def _on_trade(self, trade: TradeEvent) -> None:
        """Handle incoming trade."""
        self._latest_price = trade.price
        self.trade_windows.add_trade(trade)
        await self._notify_update("trade", trade)

    async def _on_orderbook(self, snapshot: OrderbookSnapshot) -> None:
        """Handle orderbook snapshot."""
        self._latest_book = snapshot
        self.orderbook_history.add(snapshot)
        await self._notify_update("orderbook", snapshot)

    async def _on_oi(self, update: OIUpdate) -> None:
        """Handle OI update."""
        self._latest_oi = update
        self.oi_history.append_at(update, update.timestamp_ms)
        await self._notify_update("oi", update)

    async def _on_funding(self, update: FundingUpdate) -> None:
        """Handle funding update."""
        self._latest_funding = update
        self.funding_history.append_at(update, update.timestamp_ms)
        await self._notify_update("funding", update)

    async def start(self) -> None:
        """Start all data streams."""
        if self._running:
            return

        self._running = True
        logger.info(f"Starting data ingestion for {self.symbol}")

        await asyncio.gather(
            self._trade_stream.start(),
            self._book_stream.start(),
            self._oi_stream.start(),
            self._funding_stream.start(),
        )

    async def stop(self) -> None:
        """Stop all data streams."""
        self._running = False
        logger.info(f"Stopping data ingestion for {self.symbol}")

        await asyncio.gather(
            self._trade_stream.stop(),
            self._book_stream.stop(),
            self._oi_stream.stop(),
            self._funding_stream.stop(),
        )

    def get_stats(self) -> Dict[str, StreamStats]:
        """Get statistics for all streams."""
        return {
            "trades": self._trade_stream.stats,
            "orderbook": self._book_stream.stats,
            "oi": self._oi_stream.stats,
            "funding": self._funding_stream.stats,
        }

    def get_state(self) -> Dict[str, StreamState]:
        """Get state of all streams."""
        return {
            "trades": self._trade_stream.state,
            "orderbook": self._book_stream.state,
            "oi": self._oi_stream.state,
            "funding": self._funding_stream.state,
        }
