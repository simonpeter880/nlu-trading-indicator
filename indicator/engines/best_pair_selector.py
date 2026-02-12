#!/usr/bin/env python3
"""
Best Pair Selector for Binance USDT-Margined Perpetual Futures

Scans all Binance USDⓈ-M perpetual futures and selects the single most tradable pair
based on a weighted multi-factor scoring system with strict priority ordering.

Priority Factors (in order):
1. Participation/Activity (35%) - MOST IMPORTANT
2. Liquidity Quality (25%)
3. Tradable Volatility (20%)
4. Cleanliness/Noise (10%)
5. Crowd Risk (10%) - LOWEST PRIORITY

Usage:
    python best_pair_selector.py

    Or import:
    from best_pair_selector import select_best_pair
    result = select_best_pair()
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for pair selection"""

    # API settings
    BASE_URL: str = "https://fapi.binance.com"
    REQUEST_TIMEOUT: int = 10
    CACHE_TTL: int = 10  # seconds

    # Factor weights (must sum to 1.0)
    WEIGHT_PARTICIPATION: float = 0.35
    WEIGHT_LIQUIDITY: float = 0.25
    WEIGHT_VOLATILITY: float = 0.20
    WEIGHT_CLEANLINESS: float = 0.10
    WEIGHT_CROWD: float = 0.10

    # Hard filters
    VOLUME_PERCENTILE_THRESHOLD: float = 0.30  # Top 30% by volume
    MAX_SPREAD_PCT: float = 0.03  # 0.03% max spread

    # Trade gate thresholds
    MIN_PARTICIPATION_GATE: float = 0.60
    MIN_LIQUIDITY_GATE: float = 0.60

    # Volatility preference (target range for optimal score)
    TARGET_VOLATILITY_MIN: float = 0.02  # 2% daily range
    TARGET_VOLATILITY_MAX: float = 0.08  # 8% daily range
    TARGET_VOLATILITY_OPTIMAL: float = 0.04  # 4% optimal

    # Display settings
    TOP_N_DISPLAY: int = 10


class BinanceDataCache:
    """Simple time-based cache for API responses"""

    def __init__(self, ttl: int = 10):
        self.ttl = ttl
        self._cache = {}
        self._timestamps = {}

    def get(self, key: str) -> Optional[any]:
        """Get cached value if not expired"""
        if key not in self._cache:
            return None

        if time.time() - self._timestamps[key] > self.ttl:
            del self._cache[key]
            del self._timestamps[key]
            return None

        return self._cache[key]

    def set(self, key: str, value: any):
        """Set cached value with current timestamp"""
        self._cache[key] = value
        self._timestamps[key] = time.time()

    def clear(self):
        """Clear all cached data"""
        self._cache.clear()
        self._timestamps.clear()


class BinanceFuturesAPI:
    """Binance USDT-Margined Futures API client with caching"""

    def __init__(self, config: Config):
        self.config = config
        self.cache = BinanceDataCache(ttl=config.CACHE_TTL)
        self.session = requests.Session()

    def _request(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make GET request with error handling and timeout"""
        url = f"{self.config.BASE_URL}{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=self.config.REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.error(f"Timeout requesting {endpoint}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Error requesting {endpoint}: {e}")
            raise

    def get_exchange_info(self) -> dict:
        """Get exchange trading rules and symbol information"""
        cache_key = "exchange_info"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        data = self._request("/fapi/v1/exchangeInfo")
        self.cache.set(cache_key, data)
        return data

    def get_24h_tickers(self) -> List[dict]:
        """Get 24-hour ticker data for all symbols"""
        cache_key = "24h_tickers"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        data = self._request("/fapi/v1/ticker/24hr")
        self.cache.set(cache_key, data)
        return data

    def get_book_ticker(self, symbol: Optional[str] = None) -> dict | List[dict]:
        """Get best bid/ask for symbol(s)"""
        cache_key = f"book_ticker_{symbol}" if symbol else "book_ticker_all"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        params = {"symbol": symbol} if symbol else {}
        data = self._request("/fapi/v1/ticker/bookTicker", params)
        self.cache.set(cache_key, data)
        return data

    def get_funding_rate(self, symbol: str) -> Optional[dict]:
        """Get current funding rate for symbol"""
        cache_key = f"funding_{symbol}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            # Get most recent funding rate
            data = self._request("/fapi/v1/fundingRate", {"symbol": symbol, "limit": 1})
            if data:
                self.cache.set(cache_key, data[0])
                return data[0]
        except Exception as e:
            logger.warning(f"Could not fetch funding rate for {symbol}: {e}")

        return None


class PairScorer:
    """Calculates tradability scores for futures pairs"""

    def __init__(self, config: Config):
        self.config = config

    def normalize_percentile(self, value: float, values: List[float]) -> float:
        """Convert value to percentile rank [0,1]"""
        if not values or len(values) == 0:
            return 0.5

        percentile = np.sum(np.array(values) <= value) / len(values)
        return percentile

    def score_participation(self, quote_volume: float, all_volumes: List[float]) -> float:
        """
        Score participation/activity (weight: 0.35)
        Higher volume = better, normalized by percentile
        """
        if quote_volume <= 0:
            return 0.0

        percentile = self.normalize_percentile(quote_volume, all_volumes)

        # Apply power curve to emphasize high-volume pairs
        score = percentile**0.8

        return max(0.0, min(1.0, score))

    def score_liquidity(
        self,
        spread_pct: float,
        bid_qty: float,
        ask_qty: float,
        all_spreads: List[float],
        all_depths: List[float],
    ) -> float:
        """
        Score liquidity quality (weight: 0.25)
        Tight spread + good depth = better
        """
        if spread_pct < 0 or bid_qty <= 0 or ask_qty <= 0:
            return 0.0

        # Spread score: lower is better (inverse percentile)
        spread_percentile = self.normalize_percentile(spread_pct, all_spreads)
        spread_score = 1.0 - spread_percentile

        # Depth score: higher total depth is better
        total_depth = bid_qty + ask_qty
        depth_percentile = self.normalize_percentile(total_depth, all_depths)

        # Combine with 70% weight on spread, 30% on depth
        score = 0.7 * spread_score + 0.3 * depth_percentile

        return max(0.0, min(1.0, score))

    def score_volatility(self, range_pct: float) -> float:
        """
        Score tradable volatility (weight: 0.20)
        Bell-shaped preference: moderate movement is best
        Too low = boring, too high = risky
        """
        if range_pct < 0:
            return 0.0

        target = self.config.TARGET_VOLATILITY_OPTIMAL
        min_target = self.config.TARGET_VOLATILITY_MIN
        max_target = self.config.TARGET_VOLATILITY_MAX

        # Bell curve with peak at optimal target
        if min_target <= range_pct <= max_target:
            # Within ideal range: score based on distance from optimal
            distance = abs(range_pct - target)
            max_distance = max(target - min_target, max_target - target)
            score = 1.0 - (distance / max_distance) * 0.3  # Max penalty 30% within range
        elif range_pct < min_target:
            # Too low: linear penalty
            score = range_pct / min_target * 0.5  # Max 50% score
        else:  # range_pct > max_target
            # Too high: exponential decay penalty
            excess = range_pct - max_target
            score = 0.7 * np.exp(-excess * 20)  # Rapid decay for extreme volatility

        return max(0.0, min(1.0, score))

    def score_cleanliness(self, price_change_pct: float, volume_percentile: float) -> float:
        """
        Score cleanliness/noise (weight: 0.10)
        Penalize huge price swings on low volume (manipulation proxy)
        """
        abs_change = abs(price_change_pct)

        # If large price change (>10%) on low volume (<40th percentile), penalize
        if abs_change > 10.0 and volume_percentile < 0.4:
            score = 0.3  # Heavy penalty for suspicious movement
        elif abs_change > 15.0:
            # Extreme moves are risky regardless of volume
            score = 0.5
        else:
            # Normal case: slight preference for moderate changes
            if abs_change < 2.0:
                score = 0.85  # Very stable
            elif abs_change < 5.0:
                score = 1.0  # Ideal movement
            else:
                score = 0.9 - (abs_change - 5.0) / 100.0  # Slight decay

        return max(0.0, min(1.0, score))

    def score_crowd_risk(self, funding_rate: Optional[float]) -> float:
        """
        Score crowd risk (weight: 0.10)
        Neutral funding = good, extreme funding = crowded trade risk
        """
        if funding_rate is None:
            # If funding data unavailable, return neutral
            logger.debug("Funding rate unavailable, using neutral crowd score")
            return 0.5

        # Convert funding rate to absolute percentage
        abs_funding_pct = abs(funding_rate * 100)

        # Ideal: near zero funding (balanced market)
        # Penalize: high positive (longs crowded) or negative (shorts crowded)
        if abs_funding_pct < 0.01:  # < 0.01% funding
            score = 1.0
        elif abs_funding_pct < 0.05:  # < 0.05% funding
            score = 0.9
        elif abs_funding_pct < 0.1:  # < 0.1% funding
            score = 0.7
        else:
            # Exponential penalty for extreme funding
            score = 0.7 * np.exp(-(abs_funding_pct - 0.1) * 10)

        return max(0.0, min(1.0, score))

    def calculate_tradability_score(
        self,
        participation: float,
        liquidity: float,
        volatility: float,
        cleanliness: float,
        crowd: float,
    ) -> float:
        """Calculate final weighted tradability score"""
        score = (
            self.config.WEIGHT_PARTICIPATION * participation
            + self.config.WEIGHT_LIQUIDITY * liquidity
            + self.config.WEIGHT_VOLATILITY * volatility
            + self.config.WEIGHT_CLEANLINESS * cleanliness
            + self.config.WEIGHT_CROWD * crowd
        )
        return score

    def check_trade_gate(self, participation: float, liquidity: float, spread_pct: float) -> bool:
        """
        Hard gate: trade only allowed if:
        - participation_score >= 0.60
        - liquidity_score >= 0.60
        - spread_pct <= threshold
        """
        if participation < self.config.MIN_PARTICIPATION_GATE:
            return False
        if liquidity < self.config.MIN_LIQUIDITY_GATE:
            return False
        if spread_pct > self.config.MAX_SPREAD_PCT:
            return False

        return True


class BestPairSelector:
    """Main class for selecting the best tradable pair"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.api = BinanceFuturesAPI(self.config)
        self.scorer = PairScorer(self.config)

    def get_eligible_symbols(self) -> List[dict]:
        """Get all eligible USDT perpetual futures symbols"""
        exchange_info = self.api.get_exchange_info()

        eligible = []
        for symbol_info in exchange_info["symbols"]:
            # Must be TRADING status
            if symbol_info["status"] != "TRADING":
                continue

            # Must be PERPETUAL contract
            if symbol_info.get("contractType") != "PERPETUAL":
                continue

            # Must be USDT quote asset
            if symbol_info.get("quoteAsset") != "USDT":
                continue

            # Exclude leveraged tokens and indices
            symbol = symbol_info["symbol"]
            if any(x in symbol for x in ["UP", "DOWN", "BEAR", "BULL", "DEFI", "MOVE"]):
                continue

            eligible.append(symbol_info)

        logger.info(f"Found {len(eligible)} eligible USDT perpetual symbols")
        return eligible

    def apply_liquidity_filters(self, tickers: List[dict]) -> Tuple[List[dict], dict]:
        """
        Apply hard liquidity filters:
        - Volume must be in top X percentile
        - Spread must be below threshold
        Returns: (filtered_tickers, filter_stats)
        """
        # Calculate volume threshold
        volumes = [float(t["quoteVolume"]) for t in tickers if float(t["quoteVolume"]) > 0]
        if not volumes:
            return [], {}

        volume_threshold = np.percentile(
            volumes, (1 - self.config.VOLUME_PERCENTILE_THRESHOLD) * 100
        )

        # Get book tickers for spread calculation
        book_tickers = self.api.get_book_ticker()
        book_map = (
            {bt["symbol"]: bt for bt in book_tickers}
            if isinstance(book_tickers, list)
            else {book_tickers["symbol"]: book_tickers}
        )

        filtered = []
        stats = {
            "total_input": len(tickers),
            "failed_volume": 0,
            "failed_spread": 0,
            "failed_book_data": 0,
            "passed": 0,
        }

        for ticker in tickers:
            symbol = ticker["symbol"]
            quote_volume = float(ticker["quoteVolume"])

            # Volume filter
            if quote_volume < volume_threshold:
                stats["failed_volume"] += 1
                continue

            # Get book data
            if symbol not in book_map:
                stats["failed_book_data"] += 1
                continue

            book = book_map[symbol]
            bid = float(book["bidPrice"])
            ask = float(book["askPrice"])

            if bid <= 0 or ask <= 0:
                stats["failed_book_data"] += 1
                continue

            # Calculate spread
            mid = (bid + ask) / 2
            spread_pct = ((ask - bid) / mid) * 100 if mid > 0 else float("inf")

            # Spread filter
            if spread_pct > self.config.MAX_SPREAD_PCT:
                stats["failed_spread"] += 1
                continue

            # Augment ticker with book data
            ticker["_bid"] = bid
            ticker["_ask"] = ask
            ticker["_bidQty"] = float(book["bidQty"])
            ticker["_askQty"] = float(book["askQty"])
            ticker["_spread_pct"] = spread_pct

            filtered.append(ticker)
            stats["passed"] += 1

        logger.info(f"Liquidity filter: {stats['passed']}/{stats['total_input']} symbols passed")
        return filtered, stats

    def score_symbols(self, tickers: List[dict]) -> pd.DataFrame:
        """Score all eligible symbols"""
        results = []

        # Collect all values for percentile calculations
        all_volumes = [float(t["quoteVolume"]) for t in tickers]
        all_spreads = [t["_spread_pct"] for t in tickers]
        all_depths = [t["_bidQty"] + t["_askQty"] for t in tickers]

        for ticker in tickers:
            symbol = ticker["symbol"]

            # Extract metrics
            quote_volume = float(ticker["quoteVolume"])
            price_change_pct = float(ticker["priceChangePercent"])
            high = float(ticker["highPrice"])
            low = float(ticker["lowPrice"])
            last_price = float(ticker["lastPrice"])
            spread_pct = ticker["_spread_pct"]
            bid_qty = ticker["_bidQty"]
            ask_qty = ticker["_askQty"]

            # Calculate range
            range_pct = ((high - low) / last_price) * 100 if last_price > 0 else 0

            # Calculate volume percentile for cleanliness score
            volume_percentile = self.scorer.normalize_percentile(quote_volume, all_volumes)

            # Get funding rate (optional, may be slow)
            funding_rate = None
            # Uncomment to enable funding rate (adds latency):
            # funding_data = self.api.get_funding_rate(symbol)
            # if funding_data:
            #     funding_rate = float(funding_data.get('fundingRate', 0))

            # Calculate individual scores
            participation_score = self.scorer.score_participation(quote_volume, all_volumes)
            liquidity_score = self.scorer.score_liquidity(
                spread_pct, bid_qty, ask_qty, all_spreads, all_depths
            )
            volatility_score = self.scorer.score_volatility(range_pct / 100)  # Convert to decimal
            cleanliness_score = self.scorer.score_cleanliness(price_change_pct, volume_percentile)
            crowd_score = self.scorer.score_crowd_risk(funding_rate)

            # Calculate final tradability score
            tradability_score = self.scorer.calculate_tradability_score(
                participation_score,
                liquidity_score,
                volatility_score,
                cleanliness_score,
                crowd_score,
            )

            # Check trade gate
            trade_allowed = self.scorer.check_trade_gate(
                participation_score, liquidity_score, spread_pct
            )

            results.append(
                {
                    "symbol": symbol,
                    "tradability_score": tradability_score,
                    "trade_allowed": trade_allowed,
                    "participation_score": participation_score,
                    "liquidity_score": liquidity_score,
                    "volatility_score": volatility_score,
                    "cleanliness_score": cleanliness_score,
                    "crowd_score": crowd_score,
                    "quote_volume": quote_volume,
                    "spread_pct": spread_pct,
                    "range_pct": range_pct,
                    "price_change_pct": price_change_pct,
                    "bid_qty": bid_qty,
                    "ask_qty": ask_qty,
                    "funding_rate": funding_rate if funding_rate is not None else 0.0,
                }
            )

        # Create DataFrame and sort by tradability score
        df = pd.DataFrame(results)
        df = df.sort_values("tradability_score", ascending=False).reset_index(drop=True)

        return df

    def select_best_pair(self) -> dict:
        """
        Main entry point: select the best tradable pair

        Returns:
            dict with:
                - best_symbol: str
                - best_score: float
                - trade_allowed: bool
                - top_n: pd.DataFrame
                - all_results: pd.DataFrame
                - stats: dict
        """
        start_time = time.time()

        logger.info("=" * 80)
        logger.info("Starting Best Pair Selection")
        logger.info("=" * 80)

        # Step 1: Get eligible symbols
        eligible_symbols = self.get_eligible_symbols()
        eligible_symbol_names = [s["symbol"] for s in eligible_symbols]

        # Step 2: Get 24h ticker data
        logger.info("Fetching 24h ticker data...")
        all_tickers = self.api.get_24h_tickers()

        # Filter to eligible symbols only
        tickers = [t for t in all_tickers if t["symbol"] in eligible_symbol_names]
        logger.info(f"Got ticker data for {len(tickers)} eligible symbols")

        # Step 3: Apply liquidity filters
        logger.info("Applying liquidity filters...")
        filtered_tickers, filter_stats = self.apply_liquidity_filters(tickers)

        if not filtered_tickers:
            logger.error("No symbols passed liquidity filters!")
            return {
                "best_symbol": None,
                "best_score": 0.0,
                "trade_allowed": False,
                "top_n": pd.DataFrame(),
                "all_results": pd.DataFrame(),
                "stats": filter_stats,
            }

        # Step 4: Score all filtered symbols
        logger.info(f"Scoring {len(filtered_tickers)} symbols...")
        results_df = self.score_symbols(filtered_tickers)

        # Step 5: Get best pair
        best_row = results_df.iloc[0]
        best_symbol = best_row["symbol"]
        best_score = best_row["tradability_score"]
        trade_allowed = best_row["trade_allowed"]

        # Step 6: Get top N
        top_n = results_df.head(self.config.TOP_N_DISPLAY).copy()

        elapsed = time.time() - start_time

        stats = {**filter_stats, "total_scored": len(results_df), "elapsed_seconds": elapsed}

        logger.info("=" * 80)
        logger.info(f"Selection complete in {elapsed:.2f}s")
        logger.info(f"Best pair: {best_symbol} (score: {best_score:.4f}, allowed: {trade_allowed})")
        logger.info("=" * 80)

        return {
            "best_symbol": best_symbol,
            "best_score": best_score,
            "trade_allowed": trade_allowed,
            "top_n": top_n,
            "all_results": results_df,
            "stats": stats,
        }


def print_results(result: dict):
    """Pretty print selection results"""
    print("\n" + "=" * 100)
    print("BINANCE FUTURES - BEST PAIR SELECTOR")
    print("=" * 100)

    # Best pair
    print(f"\n🏆 BEST PAIR: {result['best_symbol']}")
    print(f"   Tradability Score: {result['best_score']:.4f}")
    print(f"   Trade Allowed: {'✅ YES' if result['trade_allowed'] else '❌ NO'}")

    # Stats
    stats = result["stats"]
    print(f"\n📊 STATISTICS:")
    print(f"   Total symbols scanned: {stats['total_input']}")
    print(f"   Passed filters: {stats['passed']}")
    print(f"   Failed - volume: {stats['failed_volume']}")
    print(f"   Failed - spread: {stats['failed_spread']}")
    print(f"   Failed - book data: {stats['failed_book_data']}")
    print(f"   Elapsed time: {stats['elapsed_seconds']:.2f}s")

    # Top N table
    print(f"\n📈 TOP {len(result['top_n'])} RANKED PAIRS:")
    print("=" * 100)

    df = result["top_n"]

    # Format for display
    display_df = pd.DataFrame(
        {
            "Rank": range(1, len(df) + 1),
            "Symbol": df["symbol"],
            "Score": df["tradability_score"].apply(lambda x: f"{x:.4f}"),
            "Trade?": df["trade_allowed"].apply(lambda x: "✅" if x else "❌"),
            "Part.": df["participation_score"].apply(lambda x: f"{x:.3f}"),
            "Liq.": df["liquidity_score"].apply(lambda x: f"{x:.3f}"),
            "Vol.": df["volatility_score"].apply(lambda x: f"{x:.3f}"),
            "Clean": df["cleanliness_score"].apply(lambda x: f"{x:.3f}"),
            "Crowd": df["crowd_score"].apply(lambda x: f"{x:.3f}"),
            "Volume": df["quote_volume"].apply(lambda x: f"${x/1e6:.1f}M"),
            "Spread%": df["spread_pct"].apply(lambda x: f"{x:.4f}"),
            "Range%": df["range_pct"].apply(lambda x: f"{x:.2f}"),
        }
    )

    print(display_df.to_string(index=False))

    # Detailed breakdown for best pair
    print(f"\n🔍 DETAILED BREAKDOWN - {result['best_symbol']}:")
    print("=" * 100)
    best = df.iloc[0]

    print(f"\nFinal Score: {best['tradability_score']:.4f}")
    print(
        f"  ├─ Participation (35%):  {best['participation_score']:.4f}  [Volume: ${best['quote_volume']/1e6:.1f}M]"
    )
    print(
        f"  ├─ Liquidity (25%):      {best['liquidity_score']:.4f}  [Spread: {best['spread_pct']:.4f}%, Depth: {best['bid_qty']:.0f}/{best['ask_qty']:.0f}]"
    )
    print(
        f"  ├─ Volatility (20%):     {best['volatility_score']:.4f}  [Range: {best['range_pct']:.2f}%]"
    )
    print(
        f"  ├─ Cleanliness (10%):    {best['cleanliness_score']:.4f}  [Change: {best['price_change_pct']:.2f}%]"
    )
    print(
        f"  └─ Crowd Risk (10%):     {best['crowd_score']:.4f}  [Funding: {best['funding_rate']:.6f}]"
    )

    print(f"\nTrade Gate: {'✅ PASSED' if best['trade_allowed'] else '❌ FAILED'}")
    if not best["trade_allowed"]:
        print("  Reason: ", end="")
        if best["participation_score"] < 0.60:
            print("Participation score too low")
        elif best["liquidity_score"] < 0.60:
            print("Liquidity score too low")
        elif best["spread_pct"] > 0.03:
            print("Spread too wide")

    print("\n" + "=" * 100)
    print("⚠️  This is SELECTION ONLY - no trades are executed")
    print("=" * 100 + "\n")


def main():
    """CLI entry point"""
    try:
        selector = BestPairSelector()
        result = selector.select_best_pair()
        print_results(result)

        # Return best symbol as exit code signal (0 = success)
        return 0

    except KeyboardInterrupt:
        print("\n\nAborted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
