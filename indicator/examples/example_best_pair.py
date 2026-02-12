#!/usr/bin/env python3
"""
Example usage of the Best Pair Selector

This demonstrates various ways to use the selector programmatically.
"""

import sys
from indicator.engines import best_pair_selector as bps


def example_basic():
    """Basic usage: get the best pair"""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 80)

    selector = bps.BestPairSelector()
    result = selector.select_best_pair()

    print(f"\nBest pair: {result['best_symbol']}")
    print(f"Tradability score: {result['best_score']:.4f}")
    print(f"Trade allowed: {'YES' if result['trade_allowed'] else 'NO'}")


def example_custom_config():
    """Custom configuration: stricter requirements"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Custom Configuration (Stricter)")
    print("=" * 80)

    config = bps.Config()
    config.WEIGHT_PARTICIPATION = 0.40  # Increase participation importance
    config.MAX_SPREAD_PCT = 0.02        # Tighter spread requirement
    config.MIN_PARTICIPATION_GATE = 0.70  # Higher gate threshold

    selector = bps.BestPairSelector(config)
    result = selector.select_best_pair()

    print(f"\nWith stricter requirements:")
    print(f"Best pair: {result['best_symbol']}")
    print(f"Score: {result['best_score']:.4f}")
    print(f"Trade allowed: {'YES' if result['trade_allowed'] else 'NO'}")


def example_top_n():
    """Get top N pairs for portfolio"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Top N Pairs for Portfolio")
    print("=" * 80)

    selector = bps.BestPairSelector()
    result = selector.select_best_pair()

    # Get top 5 tradable pairs
    top_n = result['top_n']
    tradable = top_n[top_n['trade_allowed'] == True].head(5)

    print("\nTop 5 tradable pairs:")
    for idx, row in tradable.iterrows():
        print(f"  {idx+1}. {row['symbol']:12s} - Score: {row['tradability_score']:.4f} "
              f"(Vol: ${row['quote_volume']/1e6:.1f}M, Spread: {row['spread_pct']:.4f}%)")


def example_detailed_analysis():
    """Detailed factor breakdown"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Detailed Factor Analysis")
    print("=" * 80)

    selector = bps.BestPairSelector()
    result = selector.select_best_pair()

    best = result['top_n'].iloc[0]

    print(f"\nDetailed breakdown for {best['symbol']}:")
    print(f"  Final Score: {best['tradability_score']:.4f}")
    print(f"\n  Factor Scores:")
    print(f"    Participation (35%):  {best['participation_score']:.4f}")
    print(f"    Liquidity (25%):      {best['liquidity_score']:.4f}")
    print(f"    Volatility (20%):     {best['volatility_score']:.4f}")
    print(f"    Cleanliness (10%):    {best['cleanliness_score']:.4f}")
    print(f"    Crowd Risk (10%):     {best['crowd_score']:.4f}")
    print(f"\n  Raw Metrics:")
    print(f"    24h Volume:      ${best['quote_volume']/1e6:.1f}M")
    print(f"    Spread:          {best['spread_pct']:.4f}%")
    print(f"    Range:           {best['range_pct']:.2f}%")
    print(f"    Price Change:    {best['price_change_pct']:.2f}%")
    print(f"    Bid/Ask Depth:   {best['bid_qty']:.0f} / {best['ask_qty']:.0f}")


def example_volatility_preference():
    """Custom volatility targeting"""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: High Volatility Preference (Scalping)")
    print("=" * 80)

    config = bps.Config()
    # Target higher volatility for scalping
    config.TARGET_VOLATILITY_OPTIMAL = 0.06  # 6% optimal
    config.TARGET_VOLATILITY_MIN = 0.03      # 3% minimum
    config.TARGET_VOLATILITY_MAX = 0.12      # 12% maximum

    selector = bps.BestPairSelector(config)
    result = selector.select_best_pair()

    best = result['top_n'].iloc[0]

    print(f"\nWith high-volatility preference:")
    print(f"Best pair: {best['symbol']}")
    print(f"Range: {best['range_pct']:.2f}%")
    print(f"Volatility score: {best['volatility_score']:.4f}")


def example_cache_demo():
    """Demonstrate caching"""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Cache Performance")
    print("=" * 80)

    import time

    selector = bps.BestPairSelector()

    # First call (cold cache)
    start = time.time()
    result1 = selector.select_best_pair()
    elapsed1 = time.time() - start

    # Second call (warm cache)
    start = time.time()
    result2 = selector.select_best_pair()
    elapsed2 = time.time() - start

    print(f"\nFirst call (cold cache):  {elapsed1:.2f}s")
    print(f"Second call (warm cache): {elapsed2:.2f}s")
    print(f"Speedup: {elapsed1/elapsed2:.1f}x faster")
    print(f"Cache TTL: {selector.config.CACHE_TTL}s")


def example_stats():
    """Show selection statistics"""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Selection Statistics")
    print("=" * 80)

    selector = bps.BestPairSelector()
    result = selector.select_best_pair()

    stats = result['stats']

    print(f"\nMarket Scan Statistics:")
    print(f"  Total eligible symbols:     {stats['total_input']}")
    print(f"  Passed all filters:         {stats['passed']}")
    print(f"  Failed volume filter:       {stats['failed_volume']}")
    print(f"  Failed spread filter:       {stats['failed_spread']}")
    print(f"  Failed book data:           {stats['failed_book_data']}")
    print(f"  Final scored symbols:       {stats['total_scored']}")
    print(f"\n  Filter pass rate:           {stats['passed']/stats['total_input']*100:.1f}%")
    print(f"  Execution time:             {stats['elapsed_seconds']:.2f}s")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BEST PAIR SELECTOR - USAGE EXAMPLES")
    print("=" * 80)

    try:
        example_basic()
        example_top_n()
        example_detailed_analysis()
        example_custom_config()
        example_volatility_preference()
        example_stats()
        example_cache_demo()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80 + "\n")

    except KeyboardInterrupt:
        print("\n\nAborted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
