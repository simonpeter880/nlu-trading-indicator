"""
Analysis orchestration for trading pair indicators.
Contains the main analyze_pair function that coordinates all indicator analyses.
"""

import asyncio
import logging
from typing import List, Optional

from indicator.engines.data_fetcher import (
    BinanceIndicatorFetcher,
    OHLCVData,
    BinanceAPIError,
    BinanceTimeoutError,
    BinanceConnectionError,
)

logger = logging.getLogger(__name__)
from indicator.engines.indicators import (
    VolumeIndicators,
    TrendIndicators,
    MomentumIndicators,
    VolatilityIndicators,
    IndicatorResult
)
from indicator.engines.volume_analysis import AdvancedVolumeAnalyzer
from indicator.engines.volume_engine import InstitutionalVolumeEngine
from indicator.engines.unified_score import calculate_unified_score
from indicator.engines.breakout_validation import (
    BreakoutValidator,
    BreakoutFeatures,
)
from indicator.engines.oi_analysis import AdvancedOIAnalyzer
from indicator.engines.funding_analysis import AdvancedFundingAnalyzer
from indicator.engines.orderbook_analysis import AdvancedOrderbookAnalyzer, AbsorptionSide
from indicator.engines.atr_expansion import (
    ATRExpansionEngine,
    ATRExpansionConfig,
    Candle,
    print_atr_expansion,
)

from indicator.display import (
    Colors,
    print_volume_deep_dive,
    print_oi_deep_dive,
    print_funding_deep_dive,
    print_volume_engine_deep_dive,
    print_orderbook_deep_dive,
    print_unified_score,
    print_breakout_validation,
    print_header,
    print_section,
    print_indicator,
    print_summary,
)
from indicator.engines.signals import Signal, coerce_signal


def _period_for_timeframe(timeframe: str) -> str:
    """Map analysis timeframe to the closest supported history period."""
    supported = {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"}
    if timeframe in supported:
        return timeframe
    if timeframe == "1m":
        return "5m"
    return "5m"


def _htf_for_timeframe(timeframe: str) -> Optional[str]:
    """Map analysis timeframe to a higher timeframe for MTF volume confirmation."""
    mapping = {
        "1m": "5m",
        "3m": "15m",
        "5m": "15m",
        "15m": "1h",
        "30m": "2h",
        "1h": "4h",
        "2h": "4h",
        "4h": "1d",
        "6h": "1d",
        "8h": "1d",
        "12h": "1d",
        "1d": "1w",
        "3d": "1w",
        "1w": "1M",
    }
    return mapping.get(timeframe)


def _normalize_history_limit(requested: Optional[int], max_limit: int, default: int = 30) -> int:
    """Normalize history limit against available data."""
    if requested is None or requested <= 0:
        requested = default
    if max_limit < 2:
        return max_limit
    return max(2, min(requested, max_limit))


def _price_change_percent(closes: List[float], window: int) -> float:
    """Compute price change percent over the given window size."""
    if len(closes) < 2:
        return 0.0
    window = min(window, len(closes))
    if window < 2:
        return 0.0
    start = closes[-window]
    end = closes[-1]
    return (end - start) / start * 100 if start > 0 else 0.0


def _oi_change_percent(oi_history: List[float]) -> Optional[float]:
    """Compute OI change percent over the available history."""
    if len(oi_history) < 2:
        return None
    try:
        oi_start = oi_history[0]
        oi_end = oi_history[-1]
        return (oi_end - oi_start) / oi_start * 100 if oi_start > 0 else 0.0
    except (TypeError, ZeroDivisionError) as e:
        logger.debug(f"OI history data parsing failed: {e}")
        return None


def _timeframe_to_ms(timeframe: str) -> int:
    """Convert timeframe string to milliseconds."""
    units = {
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
        "M": 2_592_000_000,  # 30d approximation for monthly
    }
    try:
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        if unit not in units or value <= 0:
            raise ValueError("Invalid timeframe")
        return value * units[unit]
    except (ValueError, IndexError, TypeError):
        return 60_000


async def _fetch_agg_trades_window(
    fetcher: BinanceIndicatorFetcher,
    symbol: str,
    start_time: int,
    end_time: int,
    futures: bool = True,
    limit: int = 1000
) -> List:
    """Fetch aggTrades covering the full time window, paginating as needed."""
    agg_trades = []
    next_start = start_time
    next_from_id = None

    while next_start <= end_time:
        if next_from_id is None:
            batch = await fetcher.get_agg_trades(
                symbol,
                limit=limit,
                start_time=next_start,
                end_time=end_time,
                futures=futures
            )
        else:
            batch = await fetcher.get_agg_trades(
                symbol,
                limit=limit,
                from_id=next_from_id,
                end_time=end_time,
                futures=futures
            )
        if not batch:
            break

        # Filter trades beyond the requested window (for fromId pagination)
        if batch[-1].timestamp > end_time:
            batch = [t for t in batch if t.timestamp <= end_time]
            if not batch:
                break

        agg_trades.extend(batch)
        next_from_id = batch[-1].agg_trade_id + 1
        next_start = batch[-1].timestamp

        if batch[-1].timestamp >= end_time:
            break
        if len(batch) < limit:
            break

    return agg_trades


async def analyze_pair(
    symbol: str,
    timeframe: str = "1h",
    kline_limit: Optional[int] = 100,
    oi_history_limit: Optional[int] = 30
):
    """
    Perform full indicator analysis on a trading pair.

    Args:
        symbol: Trading pair symbol (e.g., BTCUSDT)
        timeframe: Timeframe for analysis (e.g., 1h, 4h, 1d)
        kline_limit: Number of klines to fetch (max 1500). Use None for max.
        oi_history_limit: Number of OI history points to fetch
    """
    async with BinanceIndicatorFetcher() as fetcher:
        # Validate symbol
        print(f"\n{Colors.DIM}Validating symbol...{Colors.RESET}")
        if not await fetcher.validate_symbol(symbol, futures=True):
            print(f"{Colors.RED}Error: Symbol '{symbol}' not found on Binance Futures.{Colors.RESET}")
            print(f"{Colors.DIM}Try formats like: BTCUSDT, ETHUSDT, SOLUSDT{Colors.RESET}")
            return

        # Fetch all data
        print(f"{Colors.DIM}Fetching market data for {symbol}...{Colors.RESET}")
        history_period = _period_for_timeframe(timeframe)
        data = await fetcher.get_all_data(
            symbol,
            timeframe,
            kline_limit,
            ls_ratio_period=history_period,
            taker_volume_period=history_period
        )

        if data['errors']['klines']:
            print(f"{Colors.RED}Error fetching klines: {data['errors']['klines']}{Colors.RESET}")
            return

        klines: List[OHLCVData] = data['klines']

        # Extract price arrays
        opens = [k.open for k in klines]
        highs = [k.high for k in klines]
        lows = [k.low for k in klines]
        closes = [k.close for k in klines]
        volumes = [k.volume for k in klines]
        timestamps = [k.timestamp for k in klines]

        # Higher timeframe volumes for MTF confirmation
        htf_volumes = None
        htf_timeframe = _htf_for_timeframe(timeframe)
        htf_task = None
        if htf_timeframe:
            htf_limit = 200 if kline_limit is None else max(5, min(kline_limit, 200))
            htf_task = asyncio.create_task(
                fetcher.get_klines(
                    symbol,
                    interval=htf_timeframe,
                    limit=htf_limit,
                    futures=True
                )
            )

        # Get current price info
        ticker = data['ticker']
        current_price = ticker['last_price'] if ticker else closes[-1]

        # OI history (use selected timeframe period)
        oi_hist_period = history_period
        available_klines = len(closes)
        kline_cap = available_klines if kline_limit is None else min(kline_limit, available_klines)
        oi_hist_limit = _normalize_history_limit(oi_history_limit, max_limit=kline_cap)
        oi_history = []
        oi_hist_data = []
        oi_hist_task = None
        if data['open_interest']:
            oi_hist_task = asyncio.create_task(
                fetcher.get_open_interest_history(
                    symbol,
                    period=oi_hist_period,
                    limit=oi_hist_limit
                )
            )

        # Funding history (for percentile calculation)
        funding_history = []
        funding_hist_task = None
        if data['funding_rate']:
            funding_hist_task = asyncio.create_task(
                fetcher.get_funding_rate_history(symbol, limit=100)
            )

        # aggTrades for precise delta calculation
        agg_trades = None
        agg_trade_coverage = 0.0
        bar_size_ms = _timeframe_to_ms(timeframe)
        start_time = klines[0].timestamp
        end_time = klines[-1].timestamp + bar_size_ms - 1
        agg_task = asyncio.create_task(
            _fetch_agg_trades_window(
                fetcher,
                symbol,
                start_time=start_time,
                end_time=end_time,
                futures=True
            )
        )

        tasks = {}
        if htf_task:
            tasks["htf"] = htf_task
        if oi_hist_task:
            tasks["oi_hist"] = oi_hist_task
        if funding_hist_task:
            tasks["fund_hist"] = funding_hist_task
        if agg_task:
            tasks["agg_trades"] = agg_task

        if tasks:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for key, result in zip(tasks.keys(), results):
                if key == "htf":
                    if isinstance(result, Exception):
                        logger.debug(f"HTF klines fetch failed: {result}")
                    else:
                        htf_volumes = [k.volume for k in result]
                elif key == "oi_hist":
                    if isinstance(result, Exception):
                        logger.debug(f"OI history fetch failed: {result}")
                    else:
                        oi_hist_data = result
                        try:
                            oi_history = [d['sum_open_interest'] for d in oi_hist_data]
                        except (KeyError, TypeError) as e:
                            logger.debug(f"OI history data parsing failed: {e}")
                            oi_history = []
                elif key == "fund_hist":
                    if isinstance(result, Exception):
                        logger.debug(f"Funding rate history fetch failed: {result}")
                    else:
                        try:
                            funding_history = [d['funding_rate'] for d in result]
                        except (KeyError, TypeError) as e:
                            logger.debug(f"Funding rate history data parsing failed: {e}")
                            funding_history = []
                elif key == "agg_trades":
                    if isinstance(result, Exception):
                        logger.debug(f"aggTrades fetch failed: {result}")
                        print(f"  {Colors.DIM}[aggTrades unavailable, using candle approximation]{Colors.RESET}")
                    else:
                        agg_trades = result
                        if agg_trades:
                            trade_start = min(t.timestamp for t in agg_trades)
                            trade_end = max(t.timestamp for t in agg_trades)
                            window_span = max(1, end_time - start_time)
                            trade_span = max(0, trade_end - trade_start)
                            agg_trade_coverage = trade_span / window_span
                        print(f"  {Colors.CYAN}[Using PRECISE delta from {len(agg_trades)} aggTrades]{Colors.RESET}")

        # Timeframe-aligned price change (match OI history window when available)
        if len(oi_history) >= 2:
            price_change_pct = _price_change_percent(closes, window=len(oi_history))
        else:
            price_change_pct = _price_change_percent(closes, window=2)

        # Print header
        print_header(symbol, current_price, price_change_pct, timeframe)

        all_results: List[IndicatorResult] = []

        # =================================================================
        # DEEP VOLUME ANALYSIS - "Was the move REAL?"
        # =================================================================
        volume_analyzer = AdvancedVolumeAnalyzer()
        volume_summary = volume_analyzer.full_analysis(opens, highs, lows, closes, volumes)
        print_volume_deep_dive(volume_summary)

        # Convert volume analysis to IndicatorResult for summary
        vol_signal = coerce_signal(volume_summary.signal, default=Signal.NEUTRAL)
        if vol_signal == Signal.SUSPICIOUS:
            vol_signal = Signal.BEARISH  # Treat suspicious as bearish for summary

        vol_deep_result = IndicatorResult(
            name="Volume Analysis",
            value=volume_summary.relative_volume.relative_ratio,
            signal=vol_signal,
            strength=volume_summary.confidence,
            description=volume_summary.summary
        )
        all_results.append(vol_deep_result)

        # =================================================================
        # INSTITUTIONAL VOLUME ENGINE - "Who initiated, who absorbed, who trapped?"
        # =================================================================
        # Get OI change for exhaustion detection (will try to fetch later if not available)
        oi_change_for_engine = _oi_change_percent(oi_history) if data['open_interest'] else None

        # Run institutional volume engine
        volume_engine = InstitutionalVolumeEngine()

        min_coverage = 0.8
        has_precise_window = agg_trades and len(agg_trades) >= 10 and agg_trade_coverage >= min_coverage
        if agg_trades and len(agg_trades) >= 10 and agg_trade_coverage < min_coverage:
            print(f"  {Colors.DIM}[aggTrades window coverage {agg_trade_coverage:.0%} too low, using candle approximation]{Colors.RESET}")

        if has_precise_window:
            # Use precise delta calculation with aggTrades
            engine_result = volume_engine.full_analysis_with_precise_delta(
                agg_trades=agg_trades,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                volumes=volumes,
                bar_size_ms=bar_size_ms,
                htf_volumes=htf_volumes,
                oi_change_percent=oi_change_for_engine,
                window_start_ms=start_time,
                window_end_ms=end_time
            )
        else:
            # Fallback to candle-based approximation
            engine_result = volume_engine.full_analysis(
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                volumes=volumes,
                htf_volumes=htf_volumes,
                oi_change_percent=oi_change_for_engine
            )

        print_volume_engine_deep_dive(engine_result)

        # Convert engine result to IndicatorResult for summary
        engine_signal = engine_result.signal
        if engine_signal == Signal.CAUTION:
            engine_signal = Signal.NEUTRAL

        engine_indicator = IndicatorResult(
            name="Volume Engine",
            value=engine_result.delta.delta_percent,
            signal=engine_signal,
            strength=engine_result.confidence,
            description=f"{engine_result.who_initiated} initiated | {engine_result.volume_quality} quality"
        )
        all_results.append(engine_indicator)

        # =================================================================
        # DEEP OI ANALYSIS - "Is money entering or leaving?"
        # =================================================================
        if data['open_interest']:
            oi = data['open_interest']
            if oi_history:
                oi_analyzer = AdvancedOIAnalyzer()
                oi_summary = oi_analyzer.full_analysis(
                    prices=closes,
                    oi_history=oi_history,
                    highs=highs,
                    lows=lows,
                    price_change_percent=price_change_pct
                )
                print_oi_deep_dive(oi_summary)

                # Convert OI analysis to IndicatorResult for summary
                oi_signal = oi_summary.overall_signal
                if oi_signal == Signal.CAUTION:
                    oi_signal = Signal.NEUTRAL

                oi_deep_result = IndicatorResult(
                    name="Open Interest",
                    value=oi_summary.rate_of_change.oi_change_percent,
                    signal=oi_signal,
                    strength=oi_summary.confidence,
                    description=oi_summary.summary
                )
                all_results.append(oi_deep_result)
            else:
                # Fallback to basic OI analysis
                oi_result = VolumeIndicators.analyze_open_interest(
                    oi.open_interest,
                    [],
                    price_change_pct
                )
                print_section("OPEN INTEREST", "📊")
                print_indicator(oi_result)
                all_results.append(oi_result)
        else:
            print(f"  {Colors.DIM}Open Interest: Data unavailable{Colors.RESET}\n")

        # =================================================================
        # DEEP FUNDING ANALYSIS - "Where is the crowd leaning?"
        # =================================================================
        oi_change_for_funding = oi_change_for_engine  # Reuse OI change for Funding+OI combo

        # Funding Rate Deep Analysis
        if data['funding_rate']:
            fr = data['funding_rate']

            funding_analyzer = AdvancedFundingAnalyzer()
            funding_summary = funding_analyzer.full_analysis(
                current_rate=fr.funding_rate,
                historical_rates=funding_history if funding_history else None,
                oi_change_percent=oi_change_for_funding
            )
            print_funding_deep_dive(funding_summary, oi_change_for_funding)

            # Convert to IndicatorResult for summary
            fr_signal = funding_summary.overall_signal
            if fr_signal == Signal.WARNING:
                fr_signal = Signal.NEUTRAL  # Warnings are neutral for bias calc

            fr_deep_result = IndicatorResult(
                name="Funding Rate",
                value=funding_summary.percentile.current_rate_percent,
                signal=fr_signal,
                strength=funding_summary.confidence,
                description=funding_summary.summary
            )
            all_results.append(fr_deep_result)
        else:
            print(f"  {Colors.DIM}Funding Rate: Data unavailable{Colors.RESET}\n")

        # =================================================================
        # DEEP ORDERBOOK ANALYSIS - "Where is price FORCED to go?"
        # =================================================================
        ob_summary = None
        if data['orderbook']:
            ob = data['orderbook']

            # Get recent volume from ticker
            recent_volume = ticker['volume'] if ticker else 0

            orderbook_analyzer = AdvancedOrderbookAnalyzer()
            ob_summary = orderbook_analyzer.full_analysis(
                bids=ob.bids,
                asks=ob.asks,
                recent_volume=recent_volume,
                price_change_percent=price_change_pct,
                oi_change_percent=oi_change_for_funding,  # Reuse OI change from earlier
                previous_snapshots=None  # Would need history for spoof detection
            )
            print_orderbook_deep_dive(ob_summary)

            # Convert to IndicatorResult for summary
            ob_signal = ob_summary.overall_signal
            if ob_signal == Signal.TRAP:
                ob_signal = Signal.BEARISH  # Traps are bearish for trend following

            ob_deep_result = IndicatorResult(
                name="Orderbook",
                value=ob_summary.imbalance.ratio,
                signal=ob_signal,
                strength=ob_summary.confidence,
                description=f"Path: {ob_summary.where_price_forced} | {ob_summary.summary}"
            )
            all_results.append(ob_deep_result)
        else:
            print(f"  {Colors.DIM}Orderbook: Data unavailable{Colors.RESET}\n")

        # =================================================================
        # TREND INDICATORS
        # =================================================================
        print_section("TREND INDICATORS", "📈")

        # Moving Average
        ma_result = TrendIndicators.analyze_moving_averages(closes, short_period=20, long_period=50)
        print_indicator(ma_result)
        all_results.append(ma_result)

        # VWAP
        vwap_result = TrendIndicators.analyze_vwap(highs, lows, closes, volumes)
        print_indicator(vwap_result)
        all_results.append(vwap_result)

        # Supertrend
        st_result = TrendIndicators.analyze_supertrend(highs, lows, closes)
        print_indicator(st_result)
        all_results.append(st_result)

        # =================================================================
        # MOMENTUM INDICATORS
        # =================================================================
        print_section("MOMENTUM INDICATORS", "⚡")

        # RSI
        rsi_result = MomentumIndicators.analyze_rsi(closes)
        print_indicator(rsi_result)
        all_results.append(rsi_result)

        # MACD Histogram
        macd_result = MomentumIndicators.analyze_macd_histogram(closes)
        print_indicator(macd_result)
        all_results.append(macd_result)

        # Stochastic RSI
        stoch_result = MomentumIndicators.analyze_stochastic_rsi(closes)
        print_indicator(stoch_result)
        all_results.append(stoch_result)

        # =================================================================
        # VOLATILITY INDICATORS
        # =================================================================
        print_section("VOLATILITY INDICATORS", "📉")

        # ATR
        atr_result = VolatilityIndicators.analyze_atr(highs, lows, closes)
        print_indicator(atr_result)
        all_results.append(atr_result)

        # Bollinger Bands
        bb_result = VolatilityIndicators.analyze_bollinger_bands(closes)
        print_indicator(bb_result)
        all_results.append(bb_result)

        # =================================================================
        # ATR EXPANSION - Volatility Timing Gate
        # =================================================================
        print_section("ATR EXPANSION - Volatility Timing", "⚡")

        # Convert klines to Candle objects
        candles = [
            Candle(
                timestamp=k.timestamp,
                open=k.open,
                high=k.high,
                low=k.low,
                close=k.close,
                volume=k.volume
            )
            for k in klines
        ]

        # Initialize ATR engine with config for batch analysis
        atr_config = ATRExpansionConfig(
            timeframes=[timeframe],  # Use the analysis timeframe
            atr_period=14,
            sma_period=20,
        )
        atr_engine = ATRExpansionEngine(atr_config)

        # Warmup with historical candles
        atr_states = atr_engine.warmup({timeframe: candles})

        # Display ATR expansion states
        print_atr_expansion(atr_states)

        # Interpret timing
        if timeframe in atr_states:
            atr_state = atr_states[timeframe]
            if atr_state.vol_state == "EXPANSION":
                print(f"  {Colors.GREEN}✅ TIMING: Volatility expanding - good for breakout attempts{Colors.RESET}")
            elif atr_state.vol_state == "EXTREME":
                print(f"  {Colors.YELLOW}⚡ TIMING: Extreme volatility - move is ON, use tight stops{Colors.RESET}")
            elif atr_state.vol_state == "SQUEEZE":
                print(f"  {Colors.DIM}⏸️  TIMING: Low volatility squeeze - wait for expansion{Colors.RESET}")
            elif atr_state.vol_state == "FADE_RISK":
                print(f"  {Colors.RED}⚠️  TIMING: Expansion fading - consider taking profits{Colors.RESET}")
            else:  # NORMAL
                print(f"  {Colors.CYAN}➡️  TIMING: Normal volatility - standard risk management{Colors.RESET}")

            # Show TR shock if detected
            if atr_state.debug.get("shock_now"):
                print(f"  {Colors.BOLD}{Colors.YELLOW}🔥 TR SHOCK DETECTED - Immediate volatility spike!{Colors.RESET}")

        print()  # Add spacing

        # =================================================================
        # UNIFIED MARKET SCORE - THE DECISIVE SIGNAL
        # =================================================================
        # Collect data for unified score calculation
        delta_ratio = engine_result.delta.delta_percent / 100.0  # Convert back to ratio
        relative_volume = volume_summary.relative_volume.relative_ratio

        # OI change
        oi_change_pct = None
        if oi_change_for_engine is not None:
            oi_change_pct = oi_change_for_engine

        # Funding data (pass historical funding for advanced analysis)
        current_funding = None
        historical_funding = None
        if data['funding_rate']:
            current_funding = data['funding_rate'].funding_rate
            # Pass funding history if available (for percentile-based analysis)
            if funding_history and len(funding_history) >= 10:
                historical_funding = funding_history

        # Orderbook imbalance
        depth_imbalance = None
        absorption_bullish = False
        absorption_bearish = False
        if data['orderbook']:
            # Calculate near-price depth imbalance (top 10 levels)
            bid_depth = data['orderbook'].bid_depth(10)
            ask_depth = data['orderbook'].ask_depth(10)
            total_depth = bid_depth + ask_depth
            if total_depth > 0:
                depth_imbalance = (bid_depth - ask_depth) / total_depth

            # Check for absorption from orderbook analysis
            if ob_summary is not None:
                absorption_bullish = (
                    ob_summary.absorption.detected
                    and ob_summary.absorption.side == AbsorptionSide.BID_ABSORPTION
                )
                absorption_bearish = (
                    ob_summary.absorption.detected
                    and ob_summary.absorption.side == AbsorptionSide.ASK_ABSORPTION
                )

        # Calculate unified score
        unified = calculate_unified_score(
            delta_ratio=delta_ratio,
            relative_volume=relative_volume,
            price_change_pct=price_change_pct,
            oi_change_pct=oi_change_pct,
            current_funding=current_funding,
            historical_funding=historical_funding,
            depth_imbalance=depth_imbalance,
            absorption_bullish=absorption_bullish,
            absorption_bearish=absorption_bearish
        )

        # Display unified score
        print_unified_score(unified)

        # =================================================================
        # BREAKOUT VALIDATION - Real or Fake?
        # =================================================================
        # Detect if we're at a potential breakout
        # Find swing high/low from recent data
        lookback = 20
        swing_high = max(highs[-lookback:]) if len(highs) >= lookback else max(highs)
        swing_low = min(lows[-lookback:]) if len(lows) >= lookback else min(lows)

        # Calculate ATR percentage
        atr = atr_result.value if atr_result else (highs[-1] - lows[-1])
        atr_pct = (atr / closes[-1]) * 100 if closes[-1] > 0 else 1.0

        # Detect breakout
        validator = BreakoutValidator()
        breakout_event = validator.detect_breakout(
            prices=closes,
            highs=highs,
            lows=lows,
            timestamps=timestamps,
            swing_high=swing_high,
            swing_low=swing_low,
            atr_pct=atr_pct
        )

        if breakout_event:
            # Build features for validation
            breakout_features = BreakoutFeatures(
                relative_volume=relative_volume,
                volume_acceleration=engine_result.acceleration.rate if engine_result.acceleration else 1.0,
                delta_ratio=delta_ratio,
                cvd_slope=engine_result.delta.cumulative_delta / len(closes) if len(closes) > 0 else 0,
                oi_change_pct=oi_change_pct if oi_change_pct is not None else 0.0,
                oi_acceleration=1.0,  # Would need OI history for this
                funding_z_score=(
                    (current_funding - funding_mean) / funding_std
                    if (current_funding is not None and funding_mean is not None and funding_std is not None and funding_std > 0)
                    else 0.0
                ),
                depth_imbalance_25bps=depth_imbalance if depth_imbalance is not None else 0.0,
                depth_imbalance_50bps=depth_imbalance if depth_imbalance is not None else 0.0,
                absorption_present=absorption_bullish or absorption_bearish,
                absorption_side='bid' if absorption_bullish else 'ask' if absorption_bearish else None,
                exhaustion_risk=engine_result.exhaustion.risk.value if engine_result.exhaustion else 'low',
                volume_score=unified.volume_score,
                oi_score=unified.oi_score,
                orderbook_score=unified.orderbook_score
            )

            # Validate breakout
            breakout_validation = validator.validate_breakout(
                event=breakout_event,
                features=breakout_features,
                volume_score=unified.volume_score,
                oi_score=unified.oi_score,
                orderbook_score=unified.orderbook_score
            )

            # Display validation
            print_breakout_validation(breakout_validation)

        # =================================================================
        # SUMMARY
        # =================================================================
        print_summary(all_results)
