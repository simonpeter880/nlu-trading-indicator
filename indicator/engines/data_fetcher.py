"""
Binance Data Fetcher for Indicator Analysis
Fetches OHLCV, volume, open interest, funding rates, and orderbook data.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================


class BinanceAPIError(Exception):
    """Base exception for Binance API errors."""

    def __init__(self, status_code: int, message: str, response_text: str = ""):
        self.status_code = status_code
        self.message = message
        self.response_text = response_text
        super().__init__(f"Binance API error {status_code}: {message}")


class BinanceRateLimitError(BinanceAPIError):
    """Raised when rate limit (HTTP 429) is hit."""

    def __init__(self, retry_after: Optional[int] = None):
        self.retry_after = retry_after
        super().__init__(429, "Rate limit exceeded", "")


class BinanceTimeoutError(BinanceAPIError):
    """Raised when request times out."""

    def __init__(self, timeout: float):
        self.timeout = timeout
        super().__init__(0, f"Request timed out after {timeout}s", "")


class BinanceConnectionError(BinanceAPIError):
    """Raised when connection fails."""

    def __init__(self, original_error: Exception):
        self.original_error = original_error
        super().__init__(0, f"Connection error: {original_error}", "")


# =============================================================================
# REQUEST CONFIGURATION
# =============================================================================


@dataclass
class RequestConfig:
    """Configuration for HTTP requests."""

    timeout_total: float = 30.0  # Total request timeout in seconds
    timeout_connect: float = 10.0  # Connection timeout in seconds
    max_retries: int = 3  # Maximum number of retry attempts
    retry_base_delay: float = 1.0  # Base delay for exponential backoff
    retry_max_delay: float = 30.0  # Maximum delay between retries
    retry_on_status: tuple = (429, 500, 502, 503, 504)  # HTTP status codes to retry


DEFAULT_REQUEST_CONFIG = RequestConfig()


@dataclass
class OHLCVData:
    """OHLCV candle data."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int

    @property
    def datetime(self) -> datetime:
        """UTC datetime for the candle timestamp."""
        return datetime.fromtimestamp(self.timestamp / 1000, tz=timezone.utc)

    @property
    def typical_price(self) -> float:
        """Typical price (H+L+C)/3."""
        return (self.high + self.low + self.close) / 3


@dataclass
class FundingRateData:
    """Funding rate data for perpetual futures."""

    symbol: str
    funding_rate: float
    funding_time: int
    mark_price: float

    @property
    def funding_rate_percent(self) -> float:
        """Funding rate as a percentage."""
        return self.funding_rate * 100

    @property
    def annualized_rate(self) -> float:
        """Annualized funding rate percentage (3 times per day)."""
        # Funding every 8 hours = 3 times per day = 1095 times per year
        return self.funding_rate * 1095 * 100


@dataclass
class OpenInterestData:
    """Open interest data."""

    symbol: str
    open_interest: float
    open_interest_value: float
    timestamp: int


@dataclass
class AggTradeData:
    """Aggregated trade data for volume delta calculation."""

    agg_trade_id: int
    price: float
    quantity: float
    first_trade_id: int
    last_trade_id: int
    timestamp: int
    is_buyer_maker: (
        bool  # m field: True = seller aggressed (hit bid), False = buyer aggressed (lifted ask)
    )

    @property
    def is_buy_aggression(self) -> bool:
        """True if buyer was the market taker (aggressed into ask)."""
        return not self.is_buyer_maker

    @property
    def is_sell_aggression(self) -> bool:
        """True if seller was the market taker (aggressed into bid)."""
        return self.is_buyer_maker

    @property
    def notional(self) -> float:
        """Trade value in quote currency (p * q)."""
        return self.price * self.quantity


@dataclass
class OrderbookData:
    """Orderbook snapshot data."""

    symbol: str
    bids: List[List[float]]  # [[price, qty], ...]
    asks: List[List[float]]
    timestamp: int

    @property
    def best_bid(self) -> float:
        """Best bid price or 0 if empty."""
        return self.bids[0][0] if self.bids else 0

    @property
    def best_ask(self) -> float:
        """Best ask price or 0 if empty."""
        return self.asks[0][0] if self.asks else 0

    @property
    def mid_price(self) -> float:
        """Mid price between best bid and ask."""
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.best_ask - self.best_bid

    @property
    def spread_percent(self) -> float:
        """Spread as a percentage of mid price."""
        return (self.spread / self.mid_price) * 100 if self.mid_price else 0

    def bid_depth(self, levels: int = 10) -> float:
        """Total bid volume within top N levels."""
        return sum(bid[1] for bid in self.bids[:levels])

    def ask_depth(self, levels: int = 10) -> float:
        """Total ask volume within top N levels."""
        return sum(ask[1] for ask in self.asks[:levels])

    def bid_depth_value(self, levels: int = 10) -> float:
        """Total bid value (price * qty) within top N levels."""
        return sum(bid[0] * bid[1] for bid in self.bids[:levels])

    def ask_depth_value(self, levels: int = 10) -> float:
        """Total ask value (price * qty) within top N levels."""
        return sum(ask[0] * ask[1] for ask in self.asks[:levels])

    def imbalance(self, levels: int = 10) -> float:
        """Bid/ask imbalance ratio. > 0 means more bids, < 0 means more asks."""
        bid_vol = self.bid_depth(levels)
        ask_vol = self.ask_depth(levels)
        total = bid_vol + ask_vol
        if total == 0:
            return 0
        return (bid_vol - ask_vol) / total


class BinanceIndicatorFetcher:
    """
    Fetches market data from Binance for indicator calculations.
    Supports both spot and futures markets.
    """

    SPOT_BASE = "https://api.binance.com"
    FUTURES_BASE = "https://fapi.binance.com"

    VALID_INTERVALS = [
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
        "3d",
        "1w",
        "1M",
    ]

    def __init__(
        self,
        session: Optional[aiohttp.ClientSession] = None,
        request_config: Optional[RequestConfig] = None,
    ):
        self._session = session
        self._owns_session = session is None
        self._config = request_config or DEFAULT_REQUEST_CONFIG
        self._exchange_info_cache: Dict[bool, Optional[Dict[str, Any]]] = {}

    async def __aenter__(self):
        if self._session is None:
            timeout = aiohttp.ClientTimeout(
                total=self._config.timeout_total, connect=self._config.timeout_connect
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._owns_session and self._session:
            await self._session.close()

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format (BTC/USDT -> BTCUSDT)."""
        return symbol.upper().replace("/", "").replace("-", "").replace("_", "")

    def _calculate_backoff_delay(self, attempt: int, retry_after: Optional[int] = None) -> float:
        """Calculate exponential backoff delay with jitter."""
        if retry_after is not None:
            return min(retry_after, self._config.retry_max_delay)

        # Exponential backoff: base_delay * 2^attempt
        delay = self._config.retry_base_delay * (2**attempt)
        # Add small jitter to prevent thundering herd
        import random

        jitter = random.uniform(0, 0.1 * delay)
        return min(delay + jitter, self._config.retry_max_delay)

    async def _get_premium_index(self, symbol: str) -> Dict[str, Any]:
        """Fetch premium index (includes funding rate and mark price)."""
        return await self._get(f"{self.FUTURES_BASE}/fapi/v1/premiumIndex", {"symbol": symbol})

    def _build_funding_rate(self, data: Dict[str, Any]) -> FundingRateData:
        """Build FundingRateData from premium index payload."""
        return FundingRateData(
            symbol=data["symbol"],
            funding_rate=float(data["lastFundingRate"]),
            funding_time=int(data["nextFundingTime"]),
            mark_price=float(data["markPrice"]),
        )

    def _build_open_interest(self, data: Dict[str, Any], mark_price: float) -> OpenInterestData:
        """Build OpenInterestData from open interest payload and mark price."""
        oi = float(data["openInterest"])
        return OpenInterestData(
            symbol=data["symbol"],
            open_interest=oi,
            open_interest_value=oi * mark_price,
            timestamp=int(data["time"]) if "time" in data else int(time.time() * 1000),
        )

    async def _get(self, url: str, params: Optional[Dict] = None) -> Any:
        """
        Make GET request with timeout and retry logic.

        Args:
            url: The URL to request
            params: Optional query parameters

        Returns:
            Parsed JSON response

        Raises:
            BinanceAPIError: For non-retryable API errors
            BinanceRateLimitError: When rate limit is exceeded after retries
            BinanceTimeoutError: When request times out after retries
            BinanceConnectionError: When connection fails after retries
        """
        if self._session is None:
            raise RuntimeError(
                "Session not initialized. Use 'async with BinanceIndicatorFetcher()' "
                "or pass a session to __init__."
            )

        last_error: Optional[Exception] = None

        for attempt in range(self._config.max_retries):
            try:
                logger.debug(
                    "GET %s params=%s (attempt %d/%d)",
                    url,
                    params,
                    attempt + 1,
                    self._config.max_retries,
                )
                async with self._session.get(url, params=params) as response:
                    # Handle rate limiting (HTTP 429)
                    if response.status == 429:
                        retry_after = response.headers.get("Retry-After")
                        retry_after_sec = int(retry_after) if retry_after else None
                        delay = self._calculate_backoff_delay(attempt, retry_after_sec)

                        if attempt < self._config.max_retries - 1:
                            logger.warning(
                                f"Rate limited on {url}, attempt {attempt + 1}/{self._config.max_retries}. "
                                f"Retrying in {delay:.1f}s"
                            )
                            await asyncio.sleep(delay)
                            continue
                        else:
                            raise BinanceRateLimitError(retry_after_sec)

                    # Handle other retryable status codes
                    if response.status in self._config.retry_on_status:
                        text = await response.text()
                        if attempt < self._config.max_retries - 1:
                            delay = self._calculate_backoff_delay(attempt)
                            logger.warning(
                                f"Retryable error {response.status} on {url}, "
                                f"attempt {attempt + 1}/{self._config.max_retries}. "
                                f"Retrying in {delay:.1f}s"
                            )
                            await asyncio.sleep(delay)
                            continue
                        else:
                            raise BinanceAPIError(response.status, text, text)

                    # Handle non-retryable errors
                    if response.status != 200:
                        text = await response.text()
                        raise BinanceAPIError(response.status, text, text)

                    return await response.json()

            except asyncio.TimeoutError:
                last_error = BinanceTimeoutError(self._config.timeout_total)
                if attempt < self._config.max_retries - 1:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(
                        f"Timeout on {url}, attempt {attempt + 1}/{self._config.max_retries}. "
                        f"Retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise last_error

            except aiohttp.ClientError as e:
                last_error = BinanceConnectionError(e)
                if attempt < self._config.max_retries - 1:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(
                        f"Connection error on {url}: {e}, "
                        f"attempt {attempt + 1}/{self._config.max_retries}. "
                        f"Retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise last_error

        # Should not reach here, but just in case
        if last_error:
            raise last_error
        raise BinanceAPIError(0, "Unknown error after retries", "")

    async def get_klines(
        self, symbol: str, interval: str = "1h", limit: Optional[int] = 100, futures: bool = True
    ) -> List[OHLCVData]:
        """
        Fetch OHLCV kline/candlestick data.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT' or 'BTC/USDT')
            interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            limit: Number of candles (max 1500). Use None for max.
            futures: If True, fetch from futures market

        Returns:
            List of OHLCVData objects
        """
        if interval not in self.VALID_INTERVALS:
            raise ValueError(f"Invalid interval. Must be one of: {self.VALID_INTERVALS}")

        symbol = self._normalize_symbol(symbol)
        base = self.FUTURES_BASE if futures else self.SPOT_BASE
        endpoint = f"{base}/fapi/v1/klines" if futures else f"{base}/api/v3/klines"

        kline_limit = 1500 if limit is None else min(limit, 1500)
        data = await self._get(
            endpoint, {"symbol": symbol, "interval": interval, "limit": kline_limit}
        )

        return [
            OHLCVData(
                timestamp=int(k[0]),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                quote_volume=float(k[7]),
                trades=int(k[8]),
            )
            for k in data
        ]

    async def get_funding_rate(self, symbol: str) -> FundingRateData:
        """
        Get current funding rate for perpetual futures.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')

        Returns:
            FundingRateData object
        """
        symbol = self._normalize_symbol(symbol)
        data = await self._get_premium_index(symbol)
        return self._build_funding_rate(data)

    async def get_funding_rate_history(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Get historical funding rates.

        Args:
            symbol: Trading pair
            limit: Number of records (max 1000)

        Returns:
            List of funding rate records
        """
        symbol = self._normalize_symbol(symbol)

        data = await self._get(
            f"{self.FUTURES_BASE}/fapi/v1/fundingRate",
            {"symbol": symbol, "limit": min(limit, 1000)},
        )

        return [
            {
                "symbol": d["symbol"],
                "funding_rate": float(d["fundingRate"]),
                "funding_time": int(d["fundingTime"]),
                "funding_rate_percent": float(d["fundingRate"]) * 100,
            }
            for d in data
        ]

    async def get_open_interest(
        self,
        symbol: str,
        mark_price: Optional[float] = None,
        premium_index: Optional[Dict[str, Any]] = None,
    ) -> OpenInterestData:
        """
        Get current open interest for futures.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')

        Returns:
            OpenInterestData object
        """
        symbol = self._normalize_symbol(symbol)

        data = await self._get(f"{self.FUTURES_BASE}/fapi/v1/openInterest", {"symbol": symbol})

        # Get mark price for value calculation
        if mark_price is None:
            if premium_index is None:
                premium_index = await self._get_premium_index(symbol)
            mark_price = float(premium_index["markPrice"])

        return self._build_open_interest(data, mark_price)

    async def get_open_interest_history(
        self, symbol: str, period: str = "5m", limit: int = 100
    ) -> List[Dict]:
        """
        Get historical open interest.

        Args:
            symbol: Trading pair
            period: Data period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
            limit: Number of records (max 500)

        Returns:
            List of open interest records
        """
        symbol = self._normalize_symbol(symbol)

        data = await self._get(
            f"{self.FUTURES_BASE}/futures/data/openInterestHist",
            {"symbol": symbol, "period": period, "limit": min(limit, 500)},
        )

        return [
            {
                "symbol": d["symbol"],
                "sum_open_interest": float(d["sumOpenInterest"]),
                "sum_open_interest_value": float(d["sumOpenInterestValue"]),
                "timestamp": int(d["timestamp"]),
            }
            for d in data
        ]

    async def get_orderbook(
        self, symbol: str, limit: int = 20, futures: bool = True
    ) -> OrderbookData:
        """
        Get current orderbook snapshot.

        Args:
            symbol: Trading pair
            limit: Depth limit (5, 10, 20, 50, 100, 500, 1000)
            futures: If True, fetch from futures market

        Returns:
            OrderbookData object
        """
        symbol = self._normalize_symbol(symbol)
        base = self.FUTURES_BASE if futures else self.SPOT_BASE
        endpoint = f"{base}/fapi/v1/depth" if futures else f"{base}/api/v3/depth"

        data = await self._get(endpoint, {"symbol": symbol, "limit": limit})

        return OrderbookData(
            symbol=symbol,
            bids=[[float(b[0]), float(b[1])] for b in data["bids"]],
            asks=[[float(a[0]), float(a[1])] for a in data["asks"]],
            timestamp=data.get("T", int(time.time() * 1000)),
        )

    async def get_agg_trades(
        self,
        symbol: str,
        limit: int = 1000,
        from_id: Optional[int] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        futures: bool = True,
    ) -> List[AggTradeData]:
        """
        Get aggregated trades data for precise volume delta calculation.

        Each aggTrade represents one or more individual trades aggregated at the same price.
        Critical field: 'm' (is_buyer_maker)
        - m=true → seller was market taker (hit bid) → SELL AGGRESSION
        - m=false → buyer was market taker (lifted ask) → BUY AGGRESSION

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            limit: Number of trades (max 1000)
            from_id: Starting aggregate trade ID (optional)
            start_time: Start time in milliseconds (optional)
            end_time: End time in milliseconds (optional)
            futures: If True, fetch from futures market

        Returns:
            List of AggTradeData objects
        """
        symbol = self._normalize_symbol(symbol)
        base = self.FUTURES_BASE if futures else self.SPOT_BASE
        endpoint = f"{base}/fapi/v1/aggTrades" if futures else f"{base}/api/v3/aggTrades"

        params = {"symbol": symbol, "limit": min(limit, 1000)}

        if from_id is not None:
            params["fromId"] = from_id
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        data = await self._get(endpoint, params)

        return [
            AggTradeData(
                agg_trade_id=int(t["a"]),
                price=float(t["p"]),
                quantity=float(t["q"]),
                first_trade_id=int(t["f"]),
                last_trade_id=int(t["l"]),
                timestamp=int(t["T"]),
                is_buyer_maker=bool(t["m"]),
            )
            for t in data
        ]

    async def get_24h_ticker(self, symbol: str, futures: bool = True) -> Dict:
        """
        Get 24h ticker statistics.

        Args:
            symbol: Trading pair
            futures: If True, fetch from futures market

        Returns:
            Dict with ticker data
        """
        symbol = self._normalize_symbol(symbol)
        base = self.FUTURES_BASE if futures else self.SPOT_BASE
        endpoint = f"{base}/fapi/v1/ticker/24hr" if futures else f"{base}/api/v3/ticker/24hr"

        data = await self._get(endpoint, {"symbol": symbol})

        return {
            "symbol": data["symbol"],
            "price_change": float(data["priceChange"]),
            "price_change_percent": float(data["priceChangePercent"]),
            "weighted_avg_price": float(data["weightedAvgPrice"]),
            "last_price": float(data["lastPrice"]),
            "volume": float(data["volume"]),
            "quote_volume": float(data["quoteVolume"]),
            "high_price": float(data["highPrice"]),
            "low_price": float(data["lowPrice"]),
            "open_price": float(data["openPrice"]),
            "count": int(data["count"]),
        }

    async def get_long_short_ratio(
        self, symbol: str, period: str = "5m", limit: int = 30
    ) -> List[Dict]:
        """
        Get top trader long/short ratio (accounts).

        Args:
            symbol: Trading pair
            period: Data period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
            limit: Number of records (max 500)

        Returns:
            List of long/short ratio records
        """
        symbol = self._normalize_symbol(symbol)

        data = await self._get(
            f"{self.FUTURES_BASE}/futures/data/topLongShortAccountRatio",
            {"symbol": symbol, "period": period, "limit": min(limit, 500)},
        )

        return [
            {
                "symbol": d["symbol"],
                "long_short_ratio": float(d["longShortRatio"]),
                "long_account": float(d["longAccount"]),
                "short_account": float(d["shortAccount"]),
                "timestamp": int(d["timestamp"]),
            }
            for d in data
        ]

    async def get_taker_volume(
        self, symbol: str, period: str = "5m", limit: int = 30
    ) -> List[Dict]:
        """
        Get taker buy/sell volume ratio.

        Args:
            symbol: Trading pair
            period: Data period (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
            limit: Number of records (max 500)

        Returns:
            List of taker volume records
        """
        symbol = self._normalize_symbol(symbol)

        data = await self._get(
            f"{self.FUTURES_BASE}/futures/data/takerlongshortRatio",
            {"symbol": symbol, "period": period, "limit": min(limit, 500)},
        )

        return [
            {
                "symbol": d["symbol"],
                "buy_sell_ratio": float(d["buySellRatio"]),
                "buy_vol": float(d["buyVol"]),
                "sell_vol": float(d["sellVol"]),
                "timestamp": int(d["timestamp"]),
            }
            for d in data
        ]

    async def validate_symbol(self, symbol: str, futures: bool = True) -> bool:
        """
        Check if a symbol is valid/tradable.

        Args:
            symbol: Trading pair
            futures: Check futures or spot market

        Returns:
            True if valid, False otherwise
        """
        symbol = self._normalize_symbol(symbol)

        try:
            cached = self._exchange_info_cache.get(futures)
            if cached is None:
                base = self.FUTURES_BASE if futures else self.SPOT_BASE
                endpoint = (
                    f"{base}/fapi/v1/exchangeInfo" if futures else f"{base}/api/v3/exchangeInfo"
                )
                cached = await self._get(endpoint)
                self._exchange_info_cache[futures] = cached

            symbols = [s["symbol"] for s in cached["symbols"]]
            return symbol in symbols
        except (BinanceAPIError, BinanceTimeoutError, BinanceConnectionError) as e:
            logger.warning(f"Failed to validate symbol {symbol}: {e}")
            return False
        except (KeyError, TypeError) as e:
            logger.warning(f"Unexpected response format when validating {symbol}: {e}")
            return False
        except aiohttp.ClientError as e:
            logger.warning(f"Network error validating symbol {symbol}: {e}")
            return False

    async def get_all_data(
        self,
        symbol: str,
        kline_interval: str = "1h",
        kline_limit: Optional[int] = 100,
        ls_ratio_period: str = "5m",
        taker_volume_period: str = "5m",
    ) -> Dict:
        """
        Fetch all required data for indicator analysis in parallel.

        Args:
            symbol: Trading pair
            kline_interval: Timeframe for klines
            kline_limit: Number of klines (max 1500). Use None for max.
            ls_ratio_period: Period for long/short ratio history
            taker_volume_period: Period for taker volume history

        Returns:
            Dict containing all market data
        """
        symbol = self._normalize_symbol(symbol)

        # Fetch all data concurrently (premium index reused for funding + OI)
        results = await asyncio.gather(
            self.get_klines(symbol, kline_interval, kline_limit, futures=True),
            self._get_premium_index(symbol),
            self._get(f"{self.FUTURES_BASE}/fapi/v1/openInterest", {"symbol": symbol}),
            self.get_orderbook(symbol, limit=50, futures=True),
            self.get_24h_ticker(symbol, futures=True),
            self.get_long_short_ratio(symbol, period=ls_ratio_period, limit=30),
            self.get_taker_volume(symbol, period=taker_volume_period, limit=30),
            return_exceptions=True,
        )

        klines, premium, oi_raw, orderbook, ticker, ls_ratio, taker_vol = results

        if isinstance(premium, Exception):
            funding = premium
            oi = premium if not isinstance(oi_raw, Exception) else oi_raw
        else:
            funding = self._build_funding_rate(premium)
            if isinstance(oi_raw, Exception):
                oi = oi_raw
            else:
                oi = self._build_open_interest(oi_raw, float(premium["markPrice"]))

        return {
            "symbol": symbol,
            "timestamp": int(time.time() * 1000),
            "klines": klines if not isinstance(klines, Exception) else None,
            "funding_rate": funding if not isinstance(funding, Exception) else None,
            "open_interest": oi if not isinstance(oi, Exception) else None,
            "orderbook": orderbook if not isinstance(orderbook, Exception) else None,
            "ticker": ticker if not isinstance(ticker, Exception) else None,
            "long_short_ratio": ls_ratio if not isinstance(ls_ratio, Exception) else None,
            "taker_volume": taker_vol if not isinstance(taker_vol, Exception) else None,
            "errors": {
                "klines": str(klines) if isinstance(klines, Exception) else None,
                "funding_rate": str(funding) if isinstance(funding, Exception) else None,
                "open_interest": str(oi) if isinstance(oi, Exception) else None,
                "orderbook": str(orderbook) if isinstance(orderbook, Exception) else None,
                "ticker": str(ticker) if isinstance(ticker, Exception) else None,
                "long_short_ratio": str(ls_ratio) if isinstance(ls_ratio, Exception) else None,
                "taker_volume": str(taker_vol) if isinstance(taker_vol, Exception) else None,
            },
        }
