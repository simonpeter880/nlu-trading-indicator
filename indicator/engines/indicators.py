"""
Technical Indicators Module
Implements all indicator calculations for trading analysis.
"""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

from .calculations import calculate_ema, calculate_rolling_mean_std, calculate_sma
from .signals import Signal

if TYPE_CHECKING:
    from .indicator_config import IndicatorConfig

from .indicator_config import (
    DEFAULT_CONFIG,
    FundingThresholds,
    OpenInterestThresholds,
    OrderbookThresholds,
    VolumeThresholds,
    safe_divide,
)


@dataclass
class IndicatorResult:
    """Base result for indicator calculations."""

    name: str
    value: float
    signal: Signal
    strength: float  # 0-100
    description: str


# =============================================================================
# VOLUME INDICATORS
# =============================================================================


class VolumeIndicators:
    """Volume-based indicators: Volume analysis, OI, Funding, Liquidity."""

    @staticmethod
    def analyze_volume(
        volumes: List[float],
        closes: List[float],
        period: int = 20,
        config: Optional["IndicatorConfig"] = None,
    ) -> IndicatorResult:
        """
        Analyze volume patterns.

        Returns:
            Volume analysis with trend and relative strength
        """
        cfg = (config or DEFAULT_CONFIG).volume

        if len(volumes) < period:
            return IndicatorResult(
                name="Volume",
                value=volumes[-1] if volumes else 0,
                signal=Signal.NEUTRAL,
                strength=50,
                description="Insufficient data",
            )

        current_vol = volumes[-1]
        avg_vol = sum(volumes[-period:]) / period
        vol_ratio = safe_divide(current_vol, avg_vol, default=1.0)

        # Check if volume confirms price direction
        price_change = closes[-1] - closes[-2] if len(closes) >= 2 else 0
        vol_change = volumes[-1] - volumes[-2] if len(volumes) >= 2 else 0

        # Volume expanding with price = confirmation
        vol_confirms = (price_change > 0 and vol_change > 0) or (
            price_change < 0 and vol_change > 0
        )

        if vol_ratio > cfg.high_ratio:
            signal = Signal.BULLISH if price_change > 0 else Signal.BEARISH
            strength = min(
                cfg.high_volume_max_strength,
                cfg.high_volume_base_strength
                + (vol_ratio - 1) * cfg.high_volume_strength_multiplier,
            )
            if vol_confirms:
                desc = f"Very high volume ({vol_ratio:.1f}x avg) - Strong {signal.value} conviction (CONFIRMED)"
            else:
                desc = f"Very high volume ({vol_ratio:.1f}x avg) - Strong {signal.value} conviction (divergence warning)"
                strength = max(
                    60, strength - cfg.divergence_strength_penalty
                )  # Reduce strength on divergence
        elif vol_ratio > cfg.moderate_ratio:
            signal = Signal.BULLISH if price_change > 0 else Signal.BEARISH
            strength = cfg.moderate_volume_strength
            if vol_confirms:
                desc = f"High volume ({vol_ratio:.1f}x avg) - {signal.value.capitalize()} pressure (confirmed)"
            else:
                desc = f"High volume ({vol_ratio:.1f}x avg) - {signal.value.capitalize()} pressure (weak confirmation)"
                strength = 55  # Reduce strength on divergence
        elif vol_ratio < cfg.low_ratio:
            signal = Signal.NEUTRAL
            strength = cfg.low_volume_strength
            desc = f"Low volume ({vol_ratio:.1f}x avg) - Lack of conviction"
        else:
            signal = Signal.NEUTRAL
            strength = 50
            desc = f"Normal volume ({vol_ratio:.1f}x avg)"

        return IndicatorResult(
            name="Volume", value=vol_ratio, signal=signal, strength=strength, description=desc
        )

    @staticmethod
    def analyze_open_interest(
        current_oi: float,
        oi_history: List[float],
        price_change_percent: float,
        config: Optional["IndicatorConfig"] = None,
    ) -> IndicatorResult:
        """
        Analyze open interest changes.

        OI Up + Price Up = New longs (bullish)
        OI Up + Price Down = New shorts (bearish)
        OI Down + Price Up = Short covering (weak bullish)
        OI Down + Price Down = Long liquidation (weak bearish)
        """
        cfg = (config or DEFAULT_CONFIG).open_interest

        if not oi_history:
            return IndicatorResult(
                name="Open Interest",
                value=current_oi,
                signal=Signal.NEUTRAL,
                strength=cfg.neutral_strength,
                description="No historical OI data",
            )

        oi_change = safe_divide(current_oi - oi_history[0], oi_history[0], default=0.0) * 100

        if oi_change > cfg.significant_change_pct and price_change_percent > 0:
            signal = Signal.BULLISH
            strength = min(85, cfg.strong_oi_base_strength + abs(oi_change))
            desc = f"OI +{oi_change:.1f}% with price up - New longs entering (strong bullish)"
        elif oi_change > cfg.significant_change_pct and price_change_percent < 0:
            signal = Signal.BEARISH
            strength = min(85, cfg.strong_oi_base_strength + abs(oi_change))
            desc = f"OI +{oi_change:.1f}% with price down - New shorts entering (strong bearish)"
        elif oi_change < -cfg.significant_change_pct and price_change_percent > 0:
            signal = Signal.BULLISH
            strength = cfg.weak_oi_strength
            desc = f"OI {oi_change:.1f}% with price up - Short covering (weak bullish)"
        elif oi_change < -cfg.significant_change_pct and price_change_percent < 0:
            signal = Signal.BEARISH
            strength = cfg.weak_oi_strength
            desc = f"OI {oi_change:.1f}% with price down - Long liquidation (weak bearish)"
        else:
            signal = Signal.NEUTRAL
            strength = cfg.neutral_strength
            desc = f"OI change {oi_change:.1f}% - No significant positioning change"

        return IndicatorResult(
            name="Open Interest",
            value=oi_change,
            signal=signal,
            strength=strength,
            description=desc,
        )

    @staticmethod
    def analyze_funding_rate(
        funding_rate: float,
        funding_history: Optional[List[float]] = None,
        config: Optional["IndicatorConfig"] = None,
    ) -> IndicatorResult:
        """
        Analyze perpetual funding rate.

        Positive funding = longs pay shorts (market is long-heavy)
        Negative funding = shorts pay longs (market is short-heavy)
        Extreme funding often precedes reversals.
        """
        cfg = (config or DEFAULT_CONFIG).funding

        rate_percent = funding_rate * 100
        annualized = funding_rate * 1095 * 100  # 3 funding periods per day * 365

        # Check for extreme funding
        if rate_percent > cfg.extreme_positive_pct:  # Very positive
            signal = Signal.BEARISH  # Contrarian - too many longs
            strength = min(
                cfg.extreme_max_strength,
                cfg.extreme_base_strength + abs(rate_percent) * cfg.extreme_strength_multiplier,
            )
            desc = f"Very positive funding ({rate_percent:.4f}%) - Crowded long, reversal risk"
        elif rate_percent > cfg.positive_pct:
            signal = Signal.NEUTRAL
            strength = cfg.moderate_strength
            desc = f"Positive funding ({rate_percent:.4f}%) - Longs paying shorts"
        elif rate_percent < cfg.extreme_negative_pct:  # Very negative
            signal = Signal.BULLISH  # Contrarian - too many shorts
            strength = min(
                cfg.extreme_max_strength,
                cfg.extreme_base_strength + abs(rate_percent) * cfg.extreme_strength_multiplier,
            )
            desc = f"Very negative funding ({rate_percent:.4f}%) - Crowded short, squeeze risk"
        elif rate_percent < cfg.negative_pct:
            signal = Signal.NEUTRAL
            strength = cfg.moderate_strength
            desc = f"Negative funding ({rate_percent:.4f}%) - Shorts paying longs"
        else:
            signal = Signal.NEUTRAL
            strength = cfg.neutral_strength
            desc = f"Neutral funding ({rate_percent:.4f}%) - Balanced positioning"

        return IndicatorResult(
            name="Funding Rate",
            value=rate_percent,
            signal=signal,
            strength=strength,
            description=desc + f" | Annualized: {annualized:.1f}%",
        )

    @staticmethod
    def analyze_orderbook(
        bid_depth: float,
        ask_depth: float,
        bid_depth_value: float,
        ask_depth_value: float,
        spread_percent: float,
        config: Optional["IndicatorConfig"] = None,
    ) -> IndicatorResult:
        """
        Analyze orderbook liquidity and imbalance.
        """
        cfg = (config or DEFAULT_CONFIG).orderbook

        total_depth = bid_depth + ask_depth
        imbalance = safe_divide(bid_depth - ask_depth, total_depth, default=0.0)
        imbalance_percent = imbalance * 100

        total_value = bid_depth_value + ask_depth_value
        value_imbalance = safe_divide(bid_depth_value - ask_depth_value, total_value, default=0.0)

        # Determine signal based on imbalance
        if imbalance > cfg.strong_imbalance:
            signal = Signal.BULLISH
            strength = min(
                cfg.strong_imbalance_max_strength,
                cfg.strong_imbalance_base_strength + imbalance * cfg.strong_imbalance_multiplier,
            )
            desc = f"Strong bid support ({imbalance_percent:.1f}% imbalance) - Buyers dominating"
        elif imbalance > cfg.moderate_imbalance:
            signal = Signal.BULLISH
            strength = cfg.moderate_imbalance_strength
            desc = f"Moderate bid support ({imbalance_percent:.1f}% imbalance)"
        elif imbalance < -cfg.strong_imbalance:
            signal = Signal.BEARISH
            strength = min(
                cfg.strong_imbalance_max_strength,
                cfg.strong_imbalance_base_strength
                + abs(imbalance) * cfg.strong_imbalance_multiplier,
            )
            desc = f"Strong ask pressure ({imbalance_percent:.1f}% imbalance) - Sellers dominating"
        elif imbalance < -cfg.moderate_imbalance:
            signal = Signal.BEARISH
            strength = cfg.moderate_imbalance_strength
            desc = f"Moderate ask pressure ({imbalance_percent:.1f}% imbalance)"
        else:
            signal = Signal.NEUTRAL
            strength = cfg.neutral_strength
            desc = f"Balanced orderbook ({imbalance_percent:.1f}% imbalance)"

        # Add spread context
        if spread_percent > cfg.wide_spread_pct:
            desc += f" | Wide spread ({spread_percent:.3f}%) - Low liquidity"
        elif spread_percent < cfg.tight_spread_pct:
            desc += f" | Tight spread ({spread_percent:.4f}%) - High liquidity"

        return IndicatorResult(
            name="Orderbook/Liquidity",
            value=imbalance_percent,
            signal=signal,
            strength=strength,
            description=desc,
        )


# =============================================================================
# TREND INDICATORS
# =============================================================================


class TrendIndicators:
    """Trend-following indicators: MA, VWAP, Supertrend."""

    @staticmethod
    def calculate_sma(prices: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average."""
        return calculate_sma(prices, period)

    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average."""
        return calculate_ema(prices, period)

    @staticmethod
    def analyze_moving_averages(
        closes: List[float], short_period: int = 20, long_period: int = 50
    ) -> IndicatorResult:
        """
        Analyze moving average crossovers and price position.
        """
        if len(closes) < long_period:
            return IndicatorResult(
                name="Moving Average",
                value=0,
                signal=Signal.NEUTRAL,
                strength=50,
                description="Insufficient data for MA analysis",
            )

        current_price = closes[-1]
        short_ma = TrendIndicators.calculate_ema(closes, short_period)
        long_ma = TrendIndicators.calculate_ema(closes, long_period)

        if not short_ma or not long_ma:
            return IndicatorResult(
                name="Moving Average",
                value=0,
                signal=Signal.NEUTRAL,
                strength=50,
                description="Insufficient data",
            )

        short_current = short_ma[-1]
        long_current = long_ma[-1]

        # Check previous values for crossover detection
        short_prev = short_ma[-2] if len(short_ma) > 1 else short_current
        long_prev = long_ma[-2] if len(long_ma) > 1 else long_current

        # Price position relative to MAs
        above_short = current_price > short_current
        above_long = current_price > long_current
        ma_bullish = short_current > long_current

        # Crossover detection
        golden_cross = short_prev <= long_prev and short_current > long_current
        death_cross = short_prev >= long_prev and short_current < long_current

        # Distance from short MA
        distance_percent = (
            safe_divide(current_price - short_current, short_current, default=0.0) * 100
        )

        if golden_cross:
            signal = Signal.BULLISH
            strength = 85
            desc = f"Golden Cross! EMA{short_period} crossed above EMA{long_period}"
        elif death_cross:
            signal = Signal.BEARISH
            strength = 85
            desc = f"Death Cross! EMA{short_period} crossed below EMA{long_period}"
        elif above_short and above_long and ma_bullish:
            signal = Signal.BULLISH
            strength = 75
            desc = f"Price above both MAs, uptrend confirmed | {distance_percent:+.1f}% from EMA{short_period}"
        elif not above_short and not above_long and not ma_bullish:
            signal = Signal.BEARISH
            strength = 75
            desc = f"Price below both MAs, downtrend confirmed | {distance_percent:+.1f}% from EMA{short_period}"
        elif above_long and not above_short:
            signal = Signal.NEUTRAL
            strength = 55
            desc = f"Price between MAs - consolidation | {distance_percent:+.1f}% from EMA{short_period}"
        else:
            signal = Signal.NEUTRAL
            strength = 50
            desc = f"Mixed MA signals | {distance_percent:+.1f}% from EMA{short_period}"

        return IndicatorResult(
            name="Moving Average",
            value=distance_percent,
            signal=signal,
            strength=strength,
            description=desc,
        )

    @staticmethod
    def calculate_vwap(
        highs: List[float], lows: List[float], closes: List[float], volumes: List[float]
    ) -> Tuple[float, List[float]]:
        """
        Calculate Volume Weighted Average Price.
        Returns current VWAP and cumulative VWAP values.
        """
        if not all([highs, lows, closes, volumes]):
            return 0, []

        typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
        cumulative_tpv = []
        cumulative_vol = []
        vwap_values = []

        running_tpv = 0
        running_vol = 0

        for tp, vol in zip(typical_prices, volumes):
            running_tpv += tp * vol
            running_vol += vol
            vwap_values.append(safe_divide(running_tpv, running_vol, default=tp))

        return vwap_values[-1] if vwap_values else 0, vwap_values

    @staticmethod
    def analyze_vwap(
        highs: List[float], lows: List[float], closes: List[float], volumes: List[float]
    ) -> IndicatorResult:
        """
        Analyze price relative to VWAP.
        """
        vwap, vwap_values = TrendIndicators.calculate_vwap(highs, lows, closes, volumes)

        if not vwap or not closes:
            return IndicatorResult(
                name="VWAP",
                value=0,
                signal=Signal.NEUTRAL,
                strength=50,
                description="Insufficient data for VWAP",
            )

        current_price = closes[-1]
        distance_percent = safe_divide(current_price - vwap, vwap, default=0.0) * 100

        # Check how long price has been on one side
        crosses = 0
        for i in range(1, min(20, len(closes))):
            if len(vwap_values) > i:
                if (closes[-i] > vwap_values[-i]) != (closes[-i - 1] > vwap_values[-i - 1]):
                    crosses += 1

        if distance_percent > 2:
            signal = Signal.BULLISH
            strength = min(80, 60 + abs(distance_percent) * 5)
            desc = f"Price {distance_percent:.2f}% above VWAP - Institutional buyers active"
        elif distance_percent > 0.5:
            signal = Signal.BULLISH
            strength = 60
            desc = f"Price {distance_percent:.2f}% above VWAP - Mild bullish bias"
        elif distance_percent < -2:
            signal = Signal.BEARISH
            strength = min(80, 60 + abs(distance_percent) * 5)
            desc = f"Price {distance_percent:.2f}% below VWAP - Institutional sellers active"
        elif distance_percent < -0.5:
            signal = Signal.BEARISH
            strength = 60
            desc = f"Price {distance_percent:.2f}% below VWAP - Mild bearish bias"
        else:
            signal = Signal.NEUTRAL
            strength = 50
            desc = f"Price at VWAP ({distance_percent:.2f}%) - Fair value zone"

        return IndicatorResult(
            name="VWAP", value=distance_percent, signal=signal, strength=strength, description=desc
        )

    @staticmethod
    def calculate_supertrend(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 10,
        multiplier: float = 3.0,
    ) -> Tuple[List[float], List[bool]]:
        """
        Calculate Supertrend indicator.
        Returns (supertrend_values, is_uptrend_list)
        """
        if len(closes) < period + 1:
            return [], []

        # Calculate ATR
        tr_values = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])
            )
            tr_values.append(tr)

        atr_values = TrendIndicators.calculate_sma(tr_values, period)
        if not atr_values:
            return [], []

        # Align arrays - ATR starts at index period
        start_idx = period

        supertrend = []
        is_uptrend = []

        for i in range(len(atr_values)):
            idx = start_idx + i
            atr = atr_values[i]

            # Calculate basic bands
            hl2 = (highs[idx] + lows[idx]) / 2
            upper_band = hl2 + multiplier * atr
            lower_band = hl2 - multiplier * atr

            if i == 0:
                supertrend.append(lower_band)
                is_uptrend.append(True)
            else:
                prev_supertrend = supertrend[-1]
                prev_uptrend = is_uptrend[-1]
                prev_close = closes[idx - 1]

                # Determine trend
                if prev_uptrend:
                    if closes[idx] < prev_supertrend:
                        supertrend.append(upper_band)
                        is_uptrend.append(False)
                    else:
                        supertrend.append(max(lower_band, prev_supertrend))
                        is_uptrend.append(True)
                else:
                    if closes[idx] > prev_supertrend:
                        supertrend.append(lower_band)
                        is_uptrend.append(True)
                    else:
                        supertrend.append(min(upper_band, prev_supertrend))
                        is_uptrend.append(False)

        return supertrend, is_uptrend

    @staticmethod
    def analyze_supertrend(
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 10,
        multiplier: float = 3.0,
    ) -> IndicatorResult:
        """
        Analyze Supertrend for trend direction and signals.
        """
        supertrend, is_uptrend = TrendIndicators.calculate_supertrend(
            highs, lows, closes, period, multiplier
        )

        if not supertrend or not is_uptrend:
            return IndicatorResult(
                name="Supertrend",
                value=0,
                signal=Signal.NEUTRAL,
                strength=50,
                description="Insufficient data for Supertrend",
            )

        current_price = closes[-1]
        current_st = supertrend[-1]
        current_trend = is_uptrend[-1]
        distance_percent = safe_divide(current_price - current_st, current_st, default=0.0) * 100

        # Check for trend flip
        trend_flipped = len(is_uptrend) > 1 and is_uptrend[-1] != is_uptrend[-2]

        # Count consecutive bars in trend
        consecutive = 1
        for i in range(2, min(20, len(is_uptrend) + 1)):
            if is_uptrend[-i] == current_trend:
                consecutive += 1
            else:
                break

        if trend_flipped and current_trend:
            signal = Signal.BULLISH
            strength = 85
            desc = f"Supertrend flipped BULLISH! Price {distance_percent:+.1f}% from ST line"
        elif trend_flipped and not current_trend:
            signal = Signal.BEARISH
            strength = 85
            desc = f"Supertrend flipped BEARISH! Price {distance_percent:+.1f}% from ST line"
        elif current_trend:
            signal = Signal.BULLISH
            strength = min(75, 55 + consecutive * 2)
            desc = (
                f"Supertrend bullish ({consecutive} bars) | Price {distance_percent:+.1f}% above ST"
            )
        else:
            signal = Signal.BEARISH
            strength = min(75, 55 + consecutive * 2)
            desc = (
                f"Supertrend bearish ({consecutive} bars) | Price {distance_percent:+.1f}% below ST"
            )

        return IndicatorResult(
            name="Supertrend",
            value=1 if current_trend else -1,
            signal=signal,
            strength=strength,
            description=desc,
        )


# =============================================================================
# MOMENTUM INDICATORS
# =============================================================================


class MomentumIndicators:
    """Momentum indicators: RSI, MACD, Stochastic RSI."""

    @staticmethod
    def calculate_rsi(closes: List[float], period: int = 14) -> List[float]:
        """Calculate Relative Strength Index."""
        if len(closes) < period + 1:
            return []

        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        # Initial averages
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        rsi_values = []

        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            rs = safe_divide(avg_gain, avg_loss, default=float("inf"))
            if rs == float("inf"):
                rsi_values.append(100)
            else:
                rsi_values.append(100 - (100 / (1 + rs)))

        return rsi_values

    @staticmethod
    def analyze_rsi(closes: List[float], period: int = 14) -> IndicatorResult:
        """
        Analyze RSI for overbought/oversold conditions and divergences.
        """
        rsi_values = MomentumIndicators.calculate_rsi(closes, period)

        if not rsi_values:
            return IndicatorResult(
                name="RSI",
                value=50,
                signal=Signal.NEUTRAL,
                strength=50,
                description="Insufficient data for RSI",
            )

        current_rsi = rsi_values[-1]
        prev_rsi = rsi_values[-2] if len(rsi_values) > 1 else current_rsi

        # Determine zone and signal
        if current_rsi > 80:
            signal = Signal.BEARISH
            strength = min(85, 60 + (current_rsi - 80))
            desc = f"RSI {current_rsi:.1f} - Extremely overbought, reversal likely"
        elif current_rsi > 70:
            signal = Signal.BEARISH
            strength = 65
            desc = f"RSI {current_rsi:.1f} - Overbought territory"
        elif current_rsi < 20:
            signal = Signal.BULLISH
            strength = min(85, 60 + (20 - current_rsi))
            desc = f"RSI {current_rsi:.1f} - Extremely oversold, bounce likely"
        elif current_rsi < 30:
            signal = Signal.BULLISH
            strength = 65
            desc = f"RSI {current_rsi:.1f} - Oversold territory"
        elif 45 <= current_rsi <= 55:
            signal = Signal.NEUTRAL
            strength = 50
            desc = f"RSI {current_rsi:.1f} - Neutral zone"
        else:
            # Check momentum direction
            if current_rsi > prev_rsi and current_rsi > 50:
                signal = Signal.BULLISH
                strength = 55
                desc = f"RSI {current_rsi:.1f} - Bullish momentum building"
            elif current_rsi < prev_rsi and current_rsi < 50:
                signal = Signal.BEARISH
                strength = 55
                desc = f"RSI {current_rsi:.1f} - Bearish momentum building"
            else:
                signal = Signal.NEUTRAL
                strength = 50
                desc = f"RSI {current_rsi:.1f} - Neutral"

        return IndicatorResult(
            name="RSI", value=current_rsi, signal=signal, strength=strength, description=desc
        )

    @staticmethod
    def calculate_macd(
        closes: List[float], fast: int = 12, slow: int = 26, signal_period: int = 9
    ) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate MACD, Signal line, and Histogram.
        Returns (macd_line, signal_line, histogram)
        """
        if len(closes) < slow:
            return [], [], []

        fast_ema = TrendIndicators.calculate_ema(closes, fast)
        slow_ema = TrendIndicators.calculate_ema(closes, slow)

        # Align EMAs
        offset = slow - fast
        macd_line = [fast_ema[i + offset] - slow_ema[i] for i in range(len(slow_ema))]

        if len(macd_line) < signal_period:
            return macd_line, [], []

        signal_line = TrendIndicators.calculate_ema(macd_line, signal_period)

        # Align histogram
        hist_offset = signal_period - 1
        histogram = [macd_line[i + hist_offset] - signal_line[i] for i in range(len(signal_line))]

        return macd_line, signal_line, histogram

    @staticmethod
    def analyze_macd_histogram(
        closes: List[float], fast: int = 12, slow: int = 26, signal_period: int = 9
    ) -> IndicatorResult:
        """
        Analyze MACD histogram for momentum and crossovers.
        """
        macd_line, signal_line, histogram = MomentumIndicators.calculate_macd(
            closes, fast, slow, signal_period
        )

        if not histogram:
            return IndicatorResult(
                name="MACD Histogram",
                value=0,
                signal=Signal.NEUTRAL,
                strength=50,
                description="Insufficient data for MACD",
            )

        current_hist = histogram[-1]
        prev_hist = histogram[-2] if len(histogram) > 1 else current_hist
        hist_change = current_hist - prev_hist

        current_macd = macd_line[-1] if macd_line else 0
        current_signal = signal_line[-1] if signal_line else 0

        # Check for crossover
        if len(macd_line) > 1 and len(signal_line) > 1:
            prev_macd = macd_line[-2]
            prev_signal = signal_line[-2]

            bullish_cross = prev_macd <= prev_signal and current_macd > current_signal
            bearish_cross = prev_macd >= prev_signal and current_macd < current_signal
        else:
            bullish_cross = bearish_cross = False

        if bullish_cross:
            signal = Signal.BULLISH
            strength = 85
            desc = f"MACD bullish crossover! Histogram: {current_hist:.6f}"
        elif bearish_cross:
            signal = Signal.BEARISH
            strength = 85
            desc = f"MACD bearish crossover! Histogram: {current_hist:.6f}"
        elif current_hist > 0 and hist_change > 0:
            signal = Signal.BULLISH
            strength = 70
            desc = f"MACD histogram positive and rising ({current_hist:.6f}) - Bullish momentum"
        elif current_hist > 0 and hist_change < 0:
            signal = Signal.BULLISH
            strength = 55
            desc = f"MACD histogram positive but falling ({current_hist:.6f}) - Momentum weakening"
        elif current_hist < 0 and hist_change < 0:
            signal = Signal.BEARISH
            strength = 70
            desc = f"MACD histogram negative and falling ({current_hist:.6f}) - Bearish momentum"
        elif current_hist < 0 and hist_change > 0:
            signal = Signal.BEARISH
            strength = 55
            desc = f"MACD histogram negative but rising ({current_hist:.6f}) - Momentum weakening"
        else:
            signal = Signal.NEUTRAL
            strength = 50
            desc = f"MACD histogram flat ({current_hist:.6f})"

        return IndicatorResult(
            name="MACD Histogram",
            value=current_hist,
            signal=signal,
            strength=strength,
            description=desc,
        )

    @staticmethod
    def calculate_stochastic_rsi(
        closes: List[float],
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_period: int = 3,
        d_period: int = 3,
    ) -> Tuple[List[float], List[float]]:
        """
        Calculate Stochastic RSI (%K and %D).
        Returns (stoch_k, stoch_d)
        """
        rsi_values = MomentumIndicators.calculate_rsi(closes, rsi_period)

        if len(rsi_values) < stoch_period:
            return [], []

        from collections import deque

        stoch_k = []
        min_deque: deque[int] = deque()
        max_deque: deque[int] = deque()

        for i, value in enumerate(rsi_values):
            window_start = i - stoch_period + 1

            while min_deque and min_deque[0] < window_start:
                min_deque.popleft()
            while max_deque and max_deque[0] < window_start:
                max_deque.popleft()

            while min_deque and rsi_values[min_deque[-1]] >= value:
                min_deque.pop()
            min_deque.append(i)

            while max_deque and rsi_values[max_deque[-1]] <= value:
                max_deque.pop()
            max_deque.append(i)

            if i >= stoch_period - 1:
                min_rsi = rsi_values[min_deque[0]]
                max_rsi = rsi_values[max_deque[0]]
                if max_rsi == min_rsi:
                    stoch_k.append(50)
                else:
                    stoch_k.append((value - min_rsi) / (max_rsi - min_rsi) * 100)

        # Smooth %K to get final %K
        if len(stoch_k) >= k_period:
            smoothed_k = TrendIndicators.calculate_sma(stoch_k, k_period)
        else:
            smoothed_k = stoch_k

        # Calculate %D (SMA of smoothed %K)
        if len(smoothed_k) >= d_period:
            stoch_d = TrendIndicators.calculate_sma(smoothed_k, d_period)
        else:
            stoch_d = smoothed_k

        return smoothed_k, stoch_d

    @staticmethod
    def analyze_stochastic_rsi(
        closes: List[float],
        rsi_period: int = 14,
        stoch_period: int = 14,
        k_period: int = 3,
        d_period: int = 3,
    ) -> IndicatorResult:
        """
        Analyze Stochastic RSI for momentum extremes and crossovers.
        """
        stoch_k, stoch_d = MomentumIndicators.calculate_stochastic_rsi(
            closes, rsi_period, stoch_period, k_period, d_period
        )

        if not stoch_k or not stoch_d:
            return IndicatorResult(
                name="Stochastic RSI",
                value=50,
                signal=Signal.NEUTRAL,
                strength=50,
                description="Insufficient data for Stochastic RSI",
            )

        current_k = stoch_k[-1]
        current_d = stoch_d[-1] if stoch_d else current_k
        prev_k = stoch_k[-2] if len(stoch_k) > 1 else current_k
        prev_d = stoch_d[-2] if len(stoch_d) > 1 else current_d

        # Check for crossover
        bullish_cross = prev_k <= prev_d and current_k > current_d
        bearish_cross = prev_k >= prev_d and current_k < current_d

        if bullish_cross and current_k < 30:
            signal = Signal.BULLISH
            strength = 85
            desc = (
                f"StochRSI bullish crossover in oversold zone! K:{current_k:.1f} D:{current_d:.1f}"
            )
        elif bearish_cross and current_k > 70:
            signal = Signal.BEARISH
            strength = 85
            desc = f"StochRSI bearish crossover in overbought zone! K:{current_k:.1f} D:{current_d:.1f}"
        elif current_k > 90:
            signal = Signal.BEARISH
            strength = 75
            desc = f"StochRSI extremely overbought K:{current_k:.1f} D:{current_d:.1f}"
        elif current_k > 80:
            signal = Signal.BEARISH
            strength = 65
            desc = f"StochRSI overbought K:{current_k:.1f} D:{current_d:.1f}"
        elif current_k < 10:
            signal = Signal.BULLISH
            strength = 75
            desc = f"StochRSI extremely oversold K:{current_k:.1f} D:{current_d:.1f}"
        elif current_k < 20:
            signal = Signal.BULLISH
            strength = 65
            desc = f"StochRSI oversold K:{current_k:.1f} D:{current_d:.1f}"
        elif bullish_cross:
            signal = Signal.BULLISH
            strength = 60
            desc = f"StochRSI bullish crossover K:{current_k:.1f} D:{current_d:.1f}"
        elif bearish_cross:
            signal = Signal.BEARISH
            strength = 60
            desc = f"StochRSI bearish crossover K:{current_k:.1f} D:{current_d:.1f}"
        else:
            signal = Signal.NEUTRAL
            strength = 50
            desc = f"StochRSI neutral K:{current_k:.1f} D:{current_d:.1f}"

        return IndicatorResult(
            name="Stochastic RSI",
            value=current_k,
            signal=signal,
            strength=strength,
            description=desc,
        )


# =============================================================================
# VOLATILITY INDICATORS
# =============================================================================


class VolatilityIndicators:
    """Volatility indicators: ATR, Bollinger Bands."""

    @staticmethod
    def calculate_atr(
        highs: List[float], lows: List[float], closes: List[float], period: int = 14
    ) -> List[float]:
        """Calculate Average True Range."""
        if len(closes) < period + 1:
            return []

        tr_values = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])
            )
            tr_values.append(tr)

        # Use Wilder's smoothing (similar to EMA but different multiplier)
        atr = [sum(tr_values[:period]) / period]

        for i in range(period, len(tr_values)):
            atr.append((atr[-1] * (period - 1) + tr_values[i]) / period)

        return atr

    @staticmethod
    def analyze_atr(
        highs: List[float], lows: List[float], closes: List[float], period: int = 14
    ) -> IndicatorResult:
        """
        Analyze ATR for volatility assessment.
        """
        atr_values = VolatilityIndicators.calculate_atr(highs, lows, closes, period)

        if not atr_values:
            return IndicatorResult(
                name="ATR",
                value=0,
                signal=Signal.NEUTRAL,
                strength=50,
                description="Insufficient data for ATR",
            )

        current_atr = atr_values[-1]
        current_price = closes[-1]
        atr_percent = (current_atr / current_price) * 100

        # Compare to historical ATR
        avg_atr = sum(atr_values[-20:]) / min(20, len(atr_values))
        atr_ratio = safe_divide(current_atr, avg_atr, default=1.0)

        if atr_ratio > 1.5:
            signal = Signal.NEUTRAL  # High volatility is neither bullish nor bearish
            strength = 75
            desc = f"ATR {current_atr:.2f} ({atr_percent:.2f}%) - HIGH volatility ({atr_ratio:.1f}x avg)"
        elif atr_ratio > 1.2:
            signal = Signal.NEUTRAL
            strength = 60
            desc = f"ATR {current_atr:.2f} ({atr_percent:.2f}%) - Elevated volatility"
        elif atr_ratio < 0.6:
            signal = Signal.NEUTRAL
            strength = 60
            desc = f"ATR {current_atr:.2f} ({atr_percent:.2f}%) - LOW volatility, breakout brewing"
        elif atr_ratio < 0.8:
            signal = Signal.NEUTRAL
            strength = 55
            desc = f"ATR {current_atr:.2f} ({atr_percent:.2f}%) - Decreased volatility"
        else:
            signal = Signal.NEUTRAL
            strength = 50
            desc = f"ATR {current_atr:.2f} ({atr_percent:.2f}%) - Normal volatility"

        return IndicatorResult(
            name="ATR", value=atr_percent, signal=signal, strength=strength, description=desc
        )

    @staticmethod
    def calculate_bollinger_bands(
        closes: List[float], period: int = 20, std_dev: float = 2.0
    ) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Calculate Bollinger Bands.
        Returns (middle_band, upper_band, lower_band, bandwidth)
        """
        if len(closes) < period:
            return [], [], [], []

        middle_band, stds = calculate_rolling_mean_std(closes, period)

        upper_band = []
        lower_band = []
        bandwidth = []

        for mean, std in zip(middle_band, stds):
            upper = mean + std_dev * std
            lower = mean - std_dev * std

            upper_band.append(upper)
            lower_band.append(lower)
            bandwidth.append((upper - lower) / mean * 100)

        return middle_band, upper_band, lower_band, bandwidth

    @staticmethod
    def analyze_bollinger_bands(
        closes: List[float], period: int = 20, std_dev: float = 2.0
    ) -> IndicatorResult:
        """
        Analyze Bollinger Bands for volatility and price position.
        """
        middle, upper, lower, bandwidth = VolatilityIndicators.calculate_bollinger_bands(
            closes, period, std_dev
        )

        if not middle or not upper or not lower:
            return IndicatorResult(
                name="Bollinger Bands",
                value=0,
                signal=Signal.NEUTRAL,
                strength=50,
                description="Insufficient data for Bollinger Bands",
            )

        current_price = closes[-1]
        current_middle = middle[-1]
        current_upper = upper[-1]
        current_lower = lower[-1]
        current_bandwidth = bandwidth[-1]

        # Calculate %B (position within bands)
        band_range = current_upper - current_lower
        percent_b = safe_divide(current_price - current_lower, band_range, default=0.5) * 100

        # Check bandwidth for squeeze
        avg_bandwidth = sum(bandwidth[-20:]) / min(20, len(bandwidth))
        bandwidth_ratio = safe_divide(current_bandwidth, avg_bandwidth, default=1.0)

        squeeze = bandwidth_ratio < 0.7

        if current_price > current_upper:
            signal = Signal.BULLISH
            strength = 70
            desc = f"Price ABOVE upper band - Strong momentum or reversal setup"
        elif current_price < current_lower:
            signal = Signal.BEARISH
            strength = 70
            desc = f"Price BELOW lower band - Strong momentum or reversal setup"
        elif percent_b > 80:
            signal = Signal.BEARISH
            strength = 60
            desc = f"Price near upper band (%B: {percent_b:.1f}%) - Potential resistance"
        elif percent_b < 20:
            signal = Signal.BULLISH
            strength = 60
            desc = f"Price near lower band (%B: {percent_b:.1f}%) - Potential support"
        else:
            signal = Signal.NEUTRAL
            strength = 50
            desc = f"Price mid-band (%B: {percent_b:.1f}%)"

        if squeeze:
            desc += " | SQUEEZE detected - Breakout imminent"
            strength = min(strength + 10, 90)

        desc += f" | Bandwidth: {current_bandwidth:.2f}%"

        return IndicatorResult(
            name="Bollinger Bands",
            value=percent_b,
            signal=signal,
            strength=strength,
            description=desc,
        )
