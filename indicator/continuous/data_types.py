"""
Core data types for continuous market data.

These are the atomic units flowing through the system.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# =============================================================================
# RAW DATA EVENTS (from exchanges)
# =============================================================================


@dataclass(slots=True)
class TradeEvent:
    """
    Single aggregated trade event.

    From Binance aggTrade stream:
    - is_buyer_maker=True  -> Seller aggressed (hit bid) -> SELL
    - is_buyer_maker=False -> Buyer aggressed (lifted ask) -> BUY
    """

    timestamp_ms: int
    price: float
    quantity: float
    is_buyer_maker: bool  # True = sell aggression, False = buy aggression
    trade_id: int = 0

    @property
    def is_buy(self) -> bool:
        """True if buyer was the aggressor (market buy)."""
        return not self.is_buyer_maker

    @property
    def is_sell(self) -> bool:
        """True if seller was the aggressor (market sell)."""
        return self.is_buyer_maker

    @property
    def notional(self) -> float:
        """Trade value in quote currency."""
        return self.price * self.quantity

    @property
    def signed_quantity(self) -> float:
        """Positive for buys, negative for sells."""
        return self.quantity if self.is_buy else -self.quantity


@dataclass(slots=True)
class OrderbookLevel:
    """Single price level in orderbook."""

    price: float
    quantity: float

    @property
    def notional(self) -> float:
        return self.price * self.quantity


@dataclass
class OrderbookSnapshot:
    """
    Orderbook snapshot with analysis helpers.

    Kept sparse (top 10-25 levels) for efficiency.
    """

    timestamp_ms: int
    bids: List[OrderbookLevel]  # Sorted by price descending (best bid first)
    asks: List[OrderbookLevel]  # Sorted by price ascending (best ask first)
    last_update_id: int = 0

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2 if self.bids and self.asks else 0.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid if self.bids and self.asks else 0.0

    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        mid = self.mid_price
        return (self.spread / mid) * 10000 if mid > 0 else 0.0

    def bid_depth(self, levels: int = 10) -> float:
        """Total bid quantity within top N levels."""
        return sum(b.quantity for b in self.bids[:levels])

    def ask_depth(self, levels: int = 10) -> float:
        """Total ask quantity within top N levels."""
        return sum(a.quantity for a in self.asks[:levels])

    def bid_value(self, levels: int = 10) -> float:
        """Total bid value (notional) within top N levels."""
        return sum(b.notional for b in self.bids[:levels])

    def ask_value(self, levels: int = 10) -> float:
        """Total ask value (notional) within top N levels."""
        return sum(a.notional for a in self.asks[:levels])

    def imbalance(self, levels: int = 10) -> float:
        """
        Orderbook imbalance ratio.

        Returns: (bid_depth - ask_depth) / (bid_depth + ask_depth)
        Range: [-1, +1], positive = bid heavy
        """
        bid_d = self.bid_depth(levels)
        ask_d = self.ask_depth(levels)
        total = bid_d + ask_d
        return (bid_d - ask_d) / total if total > 0 else 0.0

    def depth_at_bps(self, bps: float) -> tuple[float, float]:
        """
        Get bid/ask depth within N basis points of mid.

        Args:
            bps: Basis points from mid price

        Returns:
            (bid_depth_within_bps, ask_depth_within_bps)
        """
        mid = self.mid_price
        if mid <= 0:
            return 0.0, 0.0

        threshold = mid * bps / 10000

        bid_depth = sum(b.quantity for b in self.bids if mid - b.price <= threshold)
        ask_depth = sum(a.quantity for a in self.asks if a.price - mid <= threshold)

        return bid_depth, ask_depth


@dataclass(slots=True)
class OIUpdate:
    """Open Interest update."""

    timestamp_ms: int
    open_interest: float
    open_interest_value: float  # OI * mark price


@dataclass(slots=True)
class FundingUpdate:
    """Funding rate update."""

    timestamp_ms: int
    funding_rate: float
    mark_price: float
    next_funding_time: int

    @property
    def funding_rate_percent(self) -> float:
        return self.funding_rate * 100

    @property
    def annualized_rate(self) -> float:
        """Annualized funding rate (3 payments per day)."""
        return self.funding_rate * 1095 * 100


# =============================================================================
# COMPUTED SIGNALS (from engines)
# =============================================================================


class SignalDirection(Enum):
    """Directional bias of a signal."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class VolumeSignal:
    """Signal from Volume Engine."""

    timestamp_ms: int

    # Raw metrics
    total_volume: float
    buy_volume: float
    sell_volume: float

    # Computed
    delta: float  # buy - sell
    delta_ratio: float  # (buy - sell) / total
    relative_volume: float  # current vs average

    # Derived signals
    direction: SignalDirection
    strength: float  # 0-100
    is_climax: bool  # Potential reversal volume

    # Window info
    window_seconds: int


@dataclass
class DeltaSignal:
    """Signal from Delta Engine (cumulative volume delta)."""

    timestamp_ms: int

    # CVD metrics
    cvd: float  # Cumulative volume delta
    cvd_change: float  # Change over window
    cvd_slope: float  # Rate of change

    # Divergence detection
    price_direction: SignalDirection
    cvd_direction: SignalDirection
    is_divergent: bool  # Price and CVD disagree

    # Interpretation
    who_aggressing: str  # "buyers", "sellers", "balanced"
    direction: SignalDirection
    strength: float


@dataclass
class BookSignal:
    """Signal from Orderbook Engine."""

    timestamp_ms: int

    # Raw metrics
    bid_depth: float
    ask_depth: float
    imbalance: float  # [-1, +1]

    # Computed
    absorption_detected: bool
    absorption_side: Optional[str]  # "bid" or "ask"
    spoof_detected: bool
    spoof_side: Optional[str]

    # Liquidity analysis
    path_of_least_resistance: str  # "up", "down", "unclear"
    nearest_support: Optional[float]
    nearest_resistance: Optional[float]

    # Signal
    direction: SignalDirection
    strength: float


@dataclass
class OIFundingSignal:
    """Combined OI + Funding signal (context layer)."""

    timestamp_ms: int

    # OI metrics
    current_oi: float
    oi_change_percent: float
    oi_direction: str  # "rising", "falling", "flat"

    # Funding metrics
    funding_rate: float
    funding_percentile: float
    crowd_position: str  # "heavily_long", "moderately_long", "balanced", etc.

    # OI Regime
    regime: str  # "new_longs", "short_covering", "new_shorts", "long_liquidation"

    # Warnings
    squeeze_risk: Optional[str]  # "up", "down", None
    is_extreme: bool

    # Signal (this is a context layer - doesn't trigger, only allows/disallows)
    allows_long: bool
    allows_short: bool
    strength: float


# =============================================================================
# AGGREGATED MARKET STATE
# =============================================================================


@dataclass
class MarketState:
    """
    Complete market state at a point in time.

    This is what the state machine consumes to make decisions.
    """

    timestamp_ms: int
    symbol: str
    current_price: float

    # Volatility (for dynamic risk/stop sizing)
    atr: Optional[float] = None
    atr_percent: Optional[float] = None

    # Signals from engines
    volume_signal: Optional[VolumeSignal] = None
    delta_signal: Optional[DeltaSignal] = None
    book_signal: Optional[BookSignal] = None
    oi_funding_signal: Optional[OIFundingSignal] = None

    # Unified score (computed from all signals)
    unified_score: float = 0.0
    confidence: float = 0.0

    # Meta
    data_quality: float = 1.0  # 0-1, how complete the data is

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp_ms": self.timestamp_ms,
            "symbol": self.symbol,
            "price": self.current_price,
            "atr": self.atr,
            "atr_percent": self.atr_percent,
            "unified_score": self.unified_score,
            "confidence": self.confidence,
            "volume": (
                {
                    "delta_ratio": self.volume_signal.delta_ratio if self.volume_signal else None,
                    "relative_volume": (
                        self.volume_signal.relative_volume if self.volume_signal else None
                    ),
                }
                if self.volume_signal
                else None
            ),
            "book": (
                {
                    "imbalance": self.book_signal.imbalance if self.book_signal else None,
                    "path": self.book_signal.path_of_least_resistance if self.book_signal else None,
                }
                if self.book_signal
                else None
            ),
            "oi_funding": (
                {
                    "regime": self.oi_funding_signal.regime if self.oi_funding_signal else None,
                    "funding_percentile": (
                        self.oi_funding_signal.funding_percentile
                        if self.oi_funding_signal
                        else None
                    ),
                }
                if self.oi_funding_signal
                else None
            ),
        }


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class IngestionConfig:
    """Configuration for data ingestion rates."""

    # Orderbook snapshot interval
    orderbook_interval_ms: int = 250  # 4 snapshots per second
    orderbook_depth: int = 50  # Depth levels to fetch (default 50 for better liquidity analysis)

    # OI poll interval
    oi_interval_ms: int = 10_000  # Every 10 seconds

    # Funding update mode
    funding_on_update: bool = True  # Event-driven, not polled

    # Buffer sizes
    trades_buffer_size: int = 10_000  # ~1-5 minutes of trades
    orderbook_buffer_size: int = 200  # ~50 seconds at 250ms
    oi_buffer_size: int = 100  # ~16 minutes at 10s
    funding_buffer_size: int = 50  # ~1 week of 8h funding

    # Max age for data (beyond this, data is stale)
    trades_max_age_ms: int = 300_000  # 5 minutes
    orderbook_max_age_ms: int = 60_000  # 1 minute


@dataclass
class WindowConfig:
    """Configuration for rolling windows."""

    # Layer 1 - Micro (Execution)
    micro_windows_seconds: List[int] = field(default_factory=lambda: [5, 15, 30])

    # Layer 2 - Decision Frame
    decision_windows_seconds: List[int] = field(default_factory=lambda: [60, 120, 180])

    # Layer 3 - Context Frame
    context_windows_seconds: List[int] = field(default_factory=lambda: [900, 3600])  # 15min, 1h

    @property
    def all_windows(self) -> List[int]:
        """All window sizes in seconds."""
        return (
            self.micro_windows_seconds
            + self.decision_windows_seconds
            + self.context_windows_seconds
        )
