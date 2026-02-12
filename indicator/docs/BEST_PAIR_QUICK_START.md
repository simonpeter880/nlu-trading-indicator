# Best Pair Selector - Quick Start

## 🚀 Run It Now

```bash
cd /home/cymo/nlu
python indicator/best_pair_selector.py
```

## 📊 What You Get

```
🏆 BEST PAIR: BTCUSDT
   Tradability Score: 0.8483
   Trade Allowed: ✅ YES

📈 TOP 10 RANKED PAIRS:
   Detailed table with scores, volume, spread, etc.

🔍 DETAILED BREAKDOWN:
   Factor-by-factor analysis of the best pair
```

## ⚡ Use in Python

```python
import importlib.util

# Load module
spec = importlib.util.spec_from_file_location(
    'best_pair_selector',
    'indicator/best_pair_selector.py'
)
bps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bps)

# Get best pair
selector = bps.BestPairSelector()
result = selector.select_best_pair()

# Use it
best_symbol = result['best_symbol']
best_score = result['best_score']
trade_ok = result['trade_allowed']

print(f"Trade {best_symbol} - Score: {best_score:.4f}")
```

## 🎛️ Quick Config

```python
config = bps.Config()

# Adjust weights (must sum to 1.0)
config.WEIGHT_PARTICIPATION = 0.40  # Increase from 0.35
config.WEIGHT_LIQUIDITY = 0.20      # Decrease from 0.25

# Stricter filters
config.MAX_SPREAD_PCT = 0.02        # Tighter spread requirement
config.MIN_PARTICIPATION_GATE = 0.70  # Higher threshold

# Use custom config
selector = bps.BestPairSelector(config)
result = selector.select_best_pair()
```

## 📈 Get Top N Pairs

```python
result = selector.select_best_pair()
top_5 = result['top_n'].head(5)

for idx, row in top_5.iterrows():
    if row['trade_allowed']:
        print(f"{row['symbol']}: {row['tradability_score']:.4f}")
```

## 🔄 Run Examples

```bash
python indicator/example_best_pair.py
```

Shows 7 different usage patterns:
1. Basic usage
2. Custom configuration
3. Top N for portfolio
4. Detailed factor analysis
5. Volatility preferences
6. Cache performance
7. Selection statistics

## 📖 Full Documentation

- **[README](BEST_PAIR_SELECTOR_README.md)** - Complete guide
- **[SUMMARY](BEST_PAIR_SELECTOR_SUMMARY.md)** - Implementation details
- **Code** - [best_pair_selector.py](best_pair_selector.py) (730 lines, fully documented)

## 🎯 Key Points

- **Fast:** 5-6 seconds to scan 500+ pairs
- **Smart:** Multi-factor weighted scoring
- **Safe:** Hard trade gate for quality assurance
- **Cached:** 10-second TTL for rapid iterations
- **Configurable:** All parameters adjustable

## 🔧 Configuration Highlights

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WEIGHT_PARTICIPATION` | 0.35 | Volume/activity importance |
| `WEIGHT_LIQUIDITY` | 0.25 | Spread/depth importance |
| `WEIGHT_VOLATILITY` | 0.20 | Price movement preference |
| `WEIGHT_CLEANLINESS` | 0.10 | Manipulation detection |
| `WEIGHT_CROWD` | 0.10 | Funding rate extremes |
| `MAX_SPREAD_PCT` | 0.03% | Maximum allowed spread |
| `MIN_PARTICIPATION_GATE` | 0.60 | Minimum activity threshold |
| `MIN_LIQUIDITY_GATE` | 0.60 | Minimum liquidity threshold |
| `TARGET_VOLATILITY_OPTIMAL` | 4% | Ideal daily price range |

## ⚠️ Important

- **Selection only** - Does NOT execute trades
- **Public API** - No authentication required
- **Real-time** - Current market snapshot
- **Funding disabled** - Can be enabled (slower)

## 🎓 How Scoring Works

Each pair gets 5 scores (0 to 1):

1. **Participation (35%)** - Higher volume = better
2. **Liquidity (25%)** - Tighter spread + more depth = better
3. **Volatility (20%)** - Moderate movement (2-8%) = better
4. **Cleanliness (10%)** - Stable price action = better
5. **Crowd Risk (10%)** - Neutral funding = better

**Final Score** = weighted sum of all factors

**Trade Gate** = hard veto if participation/liquidity too low OR spread too wide

---

**Need Help?** See [BEST_PAIR_SELECTOR_README.md](BEST_PAIR_SELECTOR_README.md)
