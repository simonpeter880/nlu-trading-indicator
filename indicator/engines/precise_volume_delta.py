"""
Precise Volume Delta Calculation from Aggregated Trades
Uses real order flow data (aggTrades) instead of candle approximations.

Core Metrics:
1. Vbuy  = Σ(q) for trades with m=false (buyer aggressed into ask)
2. Vsell = Σ(q) for trades with m=true (seller aggressed into bid)
3. ΔV    = Vbuy - Vsell (signed delta)
4. ΔVr   = (Vbuy - Vsell) / (Vbuy + Vsell) (normalized to [-1, +1])
5. CVD   = Cumulative Volume Delta (running total)

Noise Control:
- Min notional filter: Ignore trades with p*q < threshold
- Percentile filter: Ignore bottom X% trade sizes
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING
from enum import Enum
from .data_fetcher import AggTradeData

if TYPE_CHECKING:
    from .indicator_config import IndicatorConfig

from .indicator_config import DEFAULT_CONFIG

class PreciseAggressionBias(Enum):
    """Precise aggression bias from real trade data."""
    STRONG_BUY = "strong_buy"       # Strong positive delta ratio
    BUY = "buy"                      # Positive delta ratio
    NEUTRAL = "neutral"              # Balanced delta ratio
    SELL = "sell"                    # Negative delta ratio
    STRONG_SELL = "strong_sell"      # Strong negative delta ratio


class VolumeLevel(Enum):
    """Relative volume levels."""
    DEAD = "dead"           # Very low RV
    LOW = "low"             # Low RV
    NORMAL = "normal"       # Normal RV
    HIGH = "high"           # High RV
    EXTREME = "extreme"     # Extreme RV


class AccelerationState(Enum):
    """Volume acceleration state."""
    ACCELERATING = "accelerating"   # Acceleration above threshold
    STEADY = "steady"               # Within steady range
    DECELERATING = "decelerating"   # Acceleration below threshold


@dataclass
class BarVolumeDelta:
    """Volume delta for a single time bar."""
    timestamp: int
    v_buy: float            # Total buy volume (buyer was aggressor)
    v_sell: float           # Total sell volume (seller was aggressor)
    v_total: float          # Total volume
    delta: float            # Vbuy - Vsell
    delta_ratio: float      # Normalized delta [-1, +1]
    cvd: float              # Cumulative volume delta
    num_trades: int
    avg_trade_size: float

    # Price context
    open_price: float
    close_price: float
    high_price: float
    low_price: float


@dataclass
class PreciseVolumeDeltaResult:
    """Complete precise volume delta analysis."""
    # Core metrics
    delta: float                    # Net delta (Vbuy - Vsell)
    delta_ratio: float              # Normalized [-1, +1]
    aggression_bias: PreciseAggressionBias
    cvd: float                      # Cumulative volume delta

    # Volume analysis
    v_buy: float
    v_sell: float
    v_total: float
    relative_volume: float          # vs average
    volume_level: VolumeLevel

    # Acceleration
    acceleration: AccelerationState
    acceleration_ratio: float       # Current vs previous

    # Context
    bars_analyzed: int
    trades_analyzed: int
    trades_filtered: int            # Noise filtered out

    # Divergence
    delta_divergence: bool          # Delta disagrees with price

    # Interpretation
    description: str
    interpretation: str
    confidence: float


@dataclass
class AbsorptionDetectionResult:
    """Tightened absorption detection using exact formulas."""
    is_absorption: bool
    absorption_side: Optional[str]  # 'bid' or 'ask'

    # Metrics
    relative_volume: float          # RV
    price_move_pct: float           # |close - open| / open
    efficiency: float               # PriceMove / RV
    atr_pct: float                  # ATR as percentage
    move_vs_atr: float              # PriceMove / ATR%

    # Delta context
    delta_ratio: float

    # Thresholds used
    rv_threshold: float             # 1.5
    k1_threshold: float             # 0.25 (move vs ATR)
    k2_threshold: float             # 0.10 (efficiency)

    confidence: float
    description: str
    action: str


@dataclass
class SweepConfirmResult:
    """Sweep + confirm detection with tight rules."""
    is_sweep: bool
    sweep_direction: Optional[str]  # 'up' or 'down'
    is_confirmed: bool              # Post-sweep confirmation

    # Sweep metrics
    swept_level: Optional[float]
    closed_back_below: bool         # For upswing
    closed_back_above: bool         # For downswing
    bars_to_close_back: int

    # Confirmation metrics
    post_sweep_rv: float
    delta_flipped: bool             # Delta opposite of sweep direction

    # Interpretation
    trapped_side: Optional[str]     # 'longs' or 'shorts'
    confidence: float
    description: str
    action: str


class PreciseVolumeDeltaEngine:
    """
    Calculate precise volume delta from aggTrades data.

    This replaces candle-based approximations with real order flow.
    """

    def __init__(
        self,
        min_notional: Optional[float] = None,
        percentile_filter: Optional[float] = None,
        config: Optional['IndicatorConfig'] = None
    ):
        """
        Initialize engine with noise filters.

        Args:
            min_notional: Minimum trade value to include (USD equivalent)
            percentile_filter: Filter bottom X% of trades by size
        """
        self.config = config or DEFAULT_CONFIG
        cfg = self.config.precise_delta
        self.min_notional = cfg.min_notional if min_notional is None else min_notional
        self.percentile_filter = cfg.percentile_filter if percentile_filter is None else percentile_filter
        self.thresholds = cfg

    def filter_noise(
        self,
        trades: List[AggTradeData],
        min_notional: Optional[float] = None
    ) -> List[AggTradeData]:
        """
        Apply noise filters to trades.

        1. Min notional filter: Remove trades with p*q < threshold
        2. Percentile filter: Remove bottom X% by trade size
        """
        if not trades:
            return []

        threshold = min_notional if min_notional is not None else self.min_notional

        # Filter 1: Min notional
        filtered = [t for t in trades if t.notional >= threshold]

        if not filtered:
            return []

        # Filter 2: Percentile (by quantity)
        if self.percentile_filter > 0:
            quantities = sorted([t.quantity for t in filtered])
            cutoff_idx = int(len(quantities) * self.percentile_filter)
            if cutoff_idx < len(quantities):
                min_qty = quantities[cutoff_idx]
                filtered = [t for t in filtered if t.quantity >= min_qty]

        return filtered

    def calculate_bar_delta(
        self,
        trades: List[AggTradeData],
        bar_start: int,
        bar_end: int,
        apply_filters: bool = True
    ) -> BarVolumeDelta:
        """
        Calculate volume delta for a single time bar.

        Args:
            trades: List of trades
            bar_start: Bar start timestamp (ms)
            bar_end: Bar end timestamp (ms)
            apply_filters: Whether to apply noise filters
        """
        # Filter trades in this bar
        bar_trades = [t for t in trades if bar_start <= t.timestamp < bar_end]

        if apply_filters:
            bar_trades = self.filter_noise(bar_trades)

        if not bar_trades:
            return BarVolumeDelta(
                timestamp=bar_start,
                v_buy=0, v_sell=0, v_total=0,
                delta=0, delta_ratio=0, cvd=0,
                num_trades=0, avg_trade_size=0,
                open_price=0, close_price=0, high_price=0, low_price=0
            )

        # Ensure bar_trades are sorted by timestamp for correct OHLC
        # (Should already be sorted from bucket_trades_to_bars, but defensive)
        bar_trades_sorted = sorted(bar_trades, key=lambda t: t.timestamp)

        # Separate buy and sell aggression
        buy_trades = [t for t in bar_trades_sorted if t.is_buy_aggression]
        sell_trades = [t for t in bar_trades_sorted if t.is_sell_aggression]

        v_buy = sum(t.quantity for t in buy_trades)
        v_sell = sum(t.quantity for t in sell_trades)
        v_total = v_buy + v_sell

        # Calculate delta
        delta = v_buy - v_sell

        # Delta ratio (normalized)
        epsilon = 1e-9
        delta_ratio = delta / (v_total + epsilon) if v_total > 0 else 0

        # Price data (using sorted trades for correct OHLC)
        prices = [t.price for t in bar_trades_sorted]
        open_price = bar_trades_sorted[0].price
        close_price = bar_trades_sorted[-1].price
        high_price = max(prices)
        low_price = min(prices)

        avg_trade_size = sum(t.quantity for t in bar_trades_sorted) / len(bar_trades_sorted)

        return BarVolumeDelta(
            timestamp=bar_start,
            v_buy=v_buy,
            v_sell=v_sell,
            v_total=v_total,
            delta=delta,
            delta_ratio=delta_ratio,
            cvd=0,  # Will be calculated cumulatively
            num_trades=len(bar_trades),
            avg_trade_size=avg_trade_size,
            open_price=open_price,
            close_price=close_price,
            high_price=high_price,
            low_price=low_price
        )

    def bucket_trades_to_bars(
        self,
        trades: List[AggTradeData],
        bar_size_ms: int,
        apply_filters: bool = True,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None
    ) -> List[BarVolumeDelta]:
        """
        Bucket trades into time bars and calculate delta for each.

        Args:
            trades: List of all trades (will be sorted by timestamp)
            bar_size_ms: Bar size in milliseconds (e.g., 60000 = 1 minute)
            apply_filters: Whether to apply noise filters
            start_time_ms: Optional bar-aligned start time override (ms)
            end_time_ms: Optional bar-aligned end time override (ms)
        """
        if not trades:
            return []

        # CRITICAL: Sort trades by timestamp to ensure correct ordering
        # Binance API should return sorted data, but defensive sorting prevents subtle bugs
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)

        # Determine time range
        start_time = start_time_ms if start_time_ms is not None else sorted_trades[0].timestamp
        end_time = end_time_ms if end_time_ms is not None else sorted_trades[-1].timestamp

        if start_time > end_time:
            return []

        # Filter trades to window if explicit bounds provided
        if start_time_ms is not None or end_time_ms is not None:
            sorted_trades = [t for t in sorted_trades if start_time <= t.timestamp <= end_time]
            if not sorted_trades:
                return []

        # Create bars
        bars = []
        cvd = 0

        current_bar_start = (start_time // bar_size_ms) * bar_size_ms

        while current_bar_start <= end_time:
            bar_end = current_bar_start + bar_size_ms

            bar_delta = self.calculate_bar_delta(
                sorted_trades, current_bar_start, bar_end, apply_filters
            )

            # Update CVD
            cvd += bar_delta.delta
            bar_delta.cvd = cvd

            bars.append(bar_delta)
            current_bar_start = bar_end

        return bars

    def analyze_volume_delta(
        self,
        bars: List[BarVolumeDelta],
        lookback: int = 20
    ) -> PreciseVolumeDeltaResult:
        """
        Analyze volume delta across multiple bars.

        Args:
            bars: List of bar deltas
            lookback: Number of bars for average calculations
        """
        if not bars:
            return PreciseVolumeDeltaResult(
                delta=0, delta_ratio=0,
                aggression_bias=PreciseAggressionBias.NEUTRAL,
                cvd=0, v_buy=0, v_sell=0, v_total=0,
                relative_volume=0,
                volume_level=VolumeLevel.DEAD,
                acceleration=AccelerationState.STEADY,
                acceleration_ratio=1.0,
                bars_analyzed=0, trades_analyzed=0, trades_filtered=0,
                delta_divergence=False,
                description="No data",
                interpretation="Insufficient data",
                confidence=0
            )

        # Recent window (last N bars or less)
        recent_window = min(self.thresholds.recent_window_bars, len(bars))
        recent_bars = bars[-recent_window:]

        # Aggregate recent metrics
        total_v_buy = sum(b.v_buy for b in recent_bars)
        total_v_sell = sum(b.v_sell for b in recent_bars)
        total_v = total_v_buy + total_v_sell

        recent_delta = total_v_buy - total_v_sell
        epsilon = 1e-9
        recent_delta_ratio = recent_delta / (total_v + epsilon) if total_v > 0 else 0

        # CVD (from last bar)
        cvd = bars[-1].cvd

        # Determine aggression bias
        if recent_delta_ratio > self.thresholds.strong_buy_threshold:
            bias = PreciseAggressionBias.STRONG_BUY
        elif recent_delta_ratio > self.thresholds.buy_threshold:
            bias = PreciseAggressionBias.BUY
        elif recent_delta_ratio < self.thresholds.strong_sell_threshold:
            bias = PreciseAggressionBias.STRONG_SELL
        elif recent_delta_ratio < self.thresholds.sell_threshold:
            bias = PreciseAggressionBias.SELL
        else:
            bias = PreciseAggressionBias.NEUTRAL

        # Relative volume
        lookback_bars = bars[-lookback:] if len(bars) >= lookback else bars
        avg_volume = sum(b.v_total for b in lookback_bars) / len(lookback_bars)
        current_volume = bars[-1].v_total
        rv = current_volume / avg_volume if avg_volume > 0 else 1.0

        # Volume level
        if rv < self.thresholds.rv_dead:
            vol_level = VolumeLevel.DEAD
        elif rv < self.thresholds.rv_low:
            vol_level = VolumeLevel.LOW
        elif rv <= self.thresholds.rv_normal_max:
            vol_level = VolumeLevel.NORMAL
        elif rv <= self.thresholds.rv_extreme:
            vol_level = VolumeLevel.HIGH
        else:
            vol_level = VolumeLevel.EXTREME

        # Volume acceleration
        if len(bars) >= 2:
            prev_volume = bars[-2].v_total
            va = current_volume / prev_volume if prev_volume > 0 else 1.0
        else:
            va = 1.0

        if va > self.thresholds.va_accelerating:
            accel = AccelerationState.ACCELERATING
        elif va < self.thresholds.va_decelerating:
            accel = AccelerationState.DECELERATING
        else:
            accel = AccelerationState.STEADY

        # Delta divergence (price vs delta)
        price_change = bars[-1].close_price - bars[-recent_window].open_price if len(bars) >= recent_window else 0
        delta_divergence = (price_change > 0 and recent_delta < 0) or (price_change < 0 and recent_delta > 0)

        # Counts
        total_trades = sum(b.num_trades for b in bars)

        # Confidence
        confidence = min(95, 50 + abs(recent_delta_ratio) * 100 + (rv - 1) * 10)

        # Interpretation
        if delta_divergence:
            if price_change > 0:
                interp = "HIDDEN DISTRIBUTION - Price rising but sellers aggressive (delta negative)"
            else:
                interp = "HIDDEN ACCUMULATION - Price falling but buyers aggressive (delta positive)"
        else:
            if bias == PreciseAggressionBias.STRONG_BUY:
                interp = "STRONG BUYING PRESSURE - Buyers aggressively lifting asks"
            elif bias == PreciseAggressionBias.BUY:
                interp = "MODERATE BUYING - Buyers showing initiative"
            elif bias == PreciseAggressionBias.STRONG_SELL:
                interp = "STRONG SELLING PRESSURE - Sellers aggressively hitting bids"
            elif bias == PreciseAggressionBias.SELL:
                interp = "MODERATE SELLING - Sellers showing initiative"
            else:
                interp = "BALANCED FLOW - No clear aggressor"

        desc = f"ΔVr: {recent_delta_ratio:+.3f} | RV: {rv:.2f}x | VA: {va:.2f}x | CVD: {cvd:+,.0f}"

        return PreciseVolumeDeltaResult(
            delta=recent_delta,
            delta_ratio=recent_delta_ratio,
            aggression_bias=bias,
            cvd=cvd,
            v_buy=total_v_buy,
            v_sell=total_v_sell,
            v_total=total_v,
            relative_volume=rv,
            volume_level=vol_level,
            acceleration=accel,
            acceleration_ratio=va,
            bars_analyzed=len(bars),
            trades_analyzed=total_trades,
            trades_filtered=0,  # Would need to track separately
            delta_divergence=delta_divergence,
            description=desc,
            interpretation=interp,
            confidence=confidence
        )

    def detect_absorption(
        self,
        bar: BarVolumeDelta,
        avg_volume: float,
        atr_pct: float
    ) -> AbsorptionDetectionResult:
        """
        Detect absorption using exact formulas.

        Absorption = high volume + low movement

        Conditions:
        1. RV >= 1.5 (high volume)
        2. PriceMove <= k1 * ATR% (small move)
        3. Efficiency <= k2 (low efficiency)

        Direction:
        - ΔVr > +0.1 but price can't rise → ASK absorption (bearish)
        - ΔVr < -0.1 but price can't fall → BID absorption (bullish)
        """
        # Calculate metrics
        rv = bar.v_total / avg_volume if avg_volume > 0 else 1.0

        price_move_pct = abs(bar.close_price - bar.open_price) / bar.open_price if bar.open_price > 0 else 0

        epsilon = 1e-9
        efficiency = price_move_pct / (rv + epsilon) if rv > 0 else 0

        move_vs_atr = price_move_pct / atr_pct if atr_pct > 0 else 0

        # Check conditions
        high_volume = rv >= self.thresholds.absorption_rv
        small_move = move_vs_atr <= self.thresholds.absorption_k1
        low_efficiency = efficiency <= self.thresholds.absorption_k2

        is_absorption = high_volume and small_move and low_efficiency

        # Determine side
        absorption_side = None
        if is_absorption:
            if bar.delta_ratio > self.thresholds.absorption_delta_ratio and bar.close_price <= bar.open_price:
                # Buyers aggressive but price can't rise → asks absorbing
                absorption_side = "ask"
            elif bar.delta_ratio < -self.thresholds.absorption_delta_ratio and bar.close_price >= bar.open_price:
                # Sellers aggressive but price can't fall → bids absorbing
                absorption_side = "bid"

        confidence = 0
        if is_absorption:
            confidence = min(
                95,
                50 + (rv - self.thresholds.absorption_rv) * 20
                + (self.thresholds.absorption_k2 - efficiency) * 200
            )

        if is_absorption and absorption_side:
            if absorption_side == "ask":
                desc = f"ASK ABSORPTION - High volume ({rv:.1f}x) but price rejected (eff: {efficiency:.3f})"
                action = "BEARISH - Smart money defending resistance"
            else:
                desc = f"BID ABSORPTION - High volume ({rv:.1f}x) but price holding (eff: {efficiency:.3f})"
                action = "BULLISH - Smart money defending support"
        elif is_absorption:
            desc = f"ABSORPTION detected - RV {rv:.1f}x, low movement ({price_move_pct*100:.2f}%)"
            action = "CAUTION - Large volume with no follow-through"
        else:
            desc = "No absorption detected"
            action = "Normal price action"

        return AbsorptionDetectionResult(
            is_absorption=is_absorption,
            absorption_side=absorption_side,
            relative_volume=rv,
            price_move_pct=price_move_pct,
            efficiency=efficiency,
            atr_pct=atr_pct,
            move_vs_atr=move_vs_atr,
            delta_ratio=bar.delta_ratio,
            rv_threshold=self.thresholds.absorption_rv,
            k1_threshold=self.thresholds.absorption_k1,
            k2_threshold=self.thresholds.absorption_k2,
            confidence=confidence,
            description=desc,
            action=action
        )

    def detect_sweep_and_confirm(
        self,
        bars: List[BarVolumeDelta],
        swing_highs: List[float],
        swing_lows: List[float]
    ) -> SweepConfirmResult:
        """
        Detect liquidity sweep + confirmation.

        Sweep detection:
        1. High breaks previous swing high (or equal highs)
        2. Closes back below within N bars

        Confirmation:
        1. Post-sweep bar has RV > 1.2
        2. Delta flips opposite of sweep direction

        Trap signature = sweep + confirm
        """
        if len(bars) < 3:
            return SweepConfirmResult(
                is_sweep=False, sweep_direction=None, is_confirmed=False,
                swept_level=None, closed_back_below=False, closed_back_above=False,
                bars_to_close_back=0, post_sweep_rv=0, delta_flipped=False,
                trapped_side=None, confidence=0,
                description="Insufficient bars for sweep detection",
                action="Need more data"
            )

        # Check for upward sweep (high breaks swing high, then closes back below)
        recent_high = bars[-1].high_price
        swept_level_up = None

        for swing_high in swing_highs:
            if recent_high > swing_high:
                swept_level_up = swing_high
                break

        # Check for downward sweep
        recent_low = bars[-1].low_price
        swept_level_down = None

        for swing_low in swing_lows:
            if recent_low < swing_low:
                swept_level_down = swing_low
                break

        # Determine if swept
        is_sweep = False
        sweep_direction = None
        swept_level = None
        closed_back = False
        bars_to_close = 0

        # Check upward sweep
        if swept_level_up:
            # Check if closed back below within N bars
            for i in range(1, min(self.thresholds.sweep_max_bars + 1, len(bars))):
                if bars[-i].close_price < swept_level_up:
                    is_sweep = True
                    sweep_direction = "up"
                    swept_level = swept_level_up
                    closed_back = True
                    bars_to_close = i
                    break

        # Check downward sweep (if no upward sweep found)
        if not is_sweep and swept_level_down:
            for i in range(1, min(self.thresholds.sweep_max_bars + 1, len(bars))):
                if bars[-i].close_price > swept_level_down:
                    is_sweep = True
                    sweep_direction = "down"
                    swept_level = swept_level_down
                    closed_back = True
                    bars_to_close = i
                    break

        # Check confirmation
        is_confirmed = False
        post_sweep_rv = 0
        delta_flipped = False
        trapped_side = None

        if is_sweep and len(bars) > bars_to_close:
            # Get post-sweep bar
            post_sweep_idx = -bars_to_close if bars_to_close > 0 else -1
            post_sweep_bar = bars[post_sweep_idx]

            # Calculate RV for post-sweep bar
            lookback_bars = bars[:-bars_to_close-1] if bars_to_close > 0 else bars[:-1]
            if lookback_bars:
                lookback_window = min(self.thresholds.sweep_avg_volume_lookback, len(lookback_bars))
                avg_vol = sum(b.v_total for b in lookback_bars[-lookback_window:]) / lookback_window
                post_sweep_rv = post_sweep_bar.v_total / avg_vol if avg_vol > 0 else 0

            # Check delta flip
            # For upward sweep: expect delta to be negative (sellers now in control)
            # For downward sweep: expect delta to be positive (buyers now in control)
            if sweep_direction == "up" and post_sweep_bar.delta_ratio < -self.thresholds.sweep_delta_flip_ratio:
                delta_flipped = True
                trapped_side = "longs"
            elif sweep_direction == "down" and post_sweep_bar.delta_ratio > self.thresholds.sweep_delta_flip_ratio:
                delta_flipped = True
                trapped_side = "shorts"

            # Confirm if RV high enough and delta flipped
            is_confirmed = post_sweep_rv > self.thresholds.sweep_confirm_rv and delta_flipped

        confidence = 0
        if is_confirmed:
            confidence = min(95, 60 + post_sweep_rv * 10 + abs(bars[post_sweep_idx].delta_ratio) * 50)
        elif is_sweep:
            confidence = 40

        if is_confirmed:
            desc = f"{sweep_direction.upper()} SWEEP CONFIRMED - Swept ${swept_level:.2f}, delta flipped, RV {post_sweep_rv:.1f}x"
            action = f"TRAP DETECTED - {trapped_side.upper()} likely trapped, prepare for reversal"
        elif is_sweep:
            desc = f"{sweep_direction.upper()} SWEEP detected but not confirmed - Watch for delta flip"
            action = "CAUTION - Potential trap, wait for confirmation"
        else:
            desc = "No sweep detected"
            action = "Normal price action"

        return SweepConfirmResult(
            is_sweep=is_sweep,
            sweep_direction=sweep_direction,
            is_confirmed=is_confirmed,
            swept_level=swept_level,
            closed_back_below=(sweep_direction == "up" and closed_back),
            closed_back_above=(sweep_direction == "down" and closed_back),
            bars_to_close_back=bars_to_close,
            post_sweep_rv=post_sweep_rv,
            delta_flipped=delta_flipped,
            trapped_side=trapped_side,
            confidence=confidence,
            description=desc,
            action=action
        )


# Convenience functions
def calculate_precise_delta(
    trades: List[AggTradeData],
    bar_size_ms: int = 60000,
    min_notional: Optional[float] = None,
    config: Optional['IndicatorConfig'] = None
) -> PreciseVolumeDeltaResult:
    """
    Quick calculation of precise volume delta.

    Args:
        trades: List of aggTrades
        bar_size_ms: Bar size in milliseconds (default 1 minute)
        min_notional: Minimum trade value to include
        config: Optional IndicatorConfig for thresholds
    """
    engine = PreciseVolumeDeltaEngine(min_notional=min_notional, config=config)
    bars = engine.bucket_trades_to_bars(trades, bar_size_ms)
    return engine.analyze_volume_delta(bars)


def is_absorption_present(
    bar: BarVolumeDelta,
    avg_volume: float,
    atr_pct: float,
    config: Optional['IndicatorConfig'] = None
) -> Tuple[bool, Optional[str]]:
    """
    Quick check: Is absorption happening?

    Returns:
        (is_absorption, absorption_side)
    """
    engine = PreciseVolumeDeltaEngine(config=config)
    result = engine.detect_absorption(bar, avg_volume, atr_pct)
    return result.is_absorption, result.absorption_side
