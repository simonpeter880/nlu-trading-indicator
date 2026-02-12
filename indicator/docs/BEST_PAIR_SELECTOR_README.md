# Best Pair Selector for Binance Futures

A sophisticated market scanner that identifies the single most tradable USDT-margined perpetual futures pair on Binance using a multi-factor weighted scoring system.

## 🎯 Purpose

Automatically select the optimal trading pair RIGHT NOW based on:
- **Participation/Activity** (35%) - Market volume and engagement
- **Liquidity Quality** (25%) - Tight spreads and depth
- **Tradable Volatility** (20%) - Moderate, tradable price movement
- **Cleanliness** (10%) - Absence of manipulation signals
- **Crowd Risk** (10%) - Funding rate extremes

## 🚀 Quick Start

```bash
# Run the selector
python indicator/best_pair_selector.py

# Or from Python
from indicator.best_pair_selector import select_best_pair
result = select_best_pair()
best = result['best_symbol']
```

## 📊 Output

The selector provides:

1. **Best Symbol** - The #1 ranked pair
2. **Top 10 Table** - Ranked pairs with all factor scores
3. **Detailed Breakdown** - Per-factor analysis for the best pair
4. **Trade Gate Status** - Whether the pair passes hard participation/liquidity thresholds

### Example Output

```
🏆 BEST PAIR: BTCUSDT
   Tradability Score: 0.8483
   Trade Allowed: ✅ YES

📈 TOP 10 RANKED PAIRS:
Rank       Symbol  Score Trade? Part.  Liq.  Vol. Clean Crowd    Volume Spread% Range%
   1      BTCUSDT 0.8483      ✅ 1.000 0.695 0.873 1.000 0.500 $13274.9M  0.0001   5.69
   2      XRPUSDT 0.8420      ✅ 0.967 0.785 0.786 1.000 0.500  $1305.1M  0.0071   6.86
   ...
```

## ⚙️ Configuration

Edit the `Config` class in [best_pair_selector.py](best_pair_selector.py) to customize:

### Factor Weights
```python
WEIGHT_PARTICIPATION: float = 0.35  # 35%
WEIGHT_LIQUIDITY: float = 0.25      # 25%
WEIGHT_VOLATILITY: float = 0.20     # 20%
WEIGHT_CLEANLINESS: float = 0.10    # 10%
WEIGHT_CROWD: float = 0.10          # 10%
```

### Hard Filters
```python
VOLUME_PERCENTILE_THRESHOLD: float = 0.30  # Top 30% by volume
MAX_SPREAD_PCT: float = 0.03               # Max 0.03% spread
```

### Trade Gate
```python
MIN_PARTICIPATION_GATE: float = 0.60  # Min participation score
MIN_LIQUIDITY_GATE: float = 0.60      # Min liquidity score
```

### Volatility Preference
```python
TARGET_VOLATILITY_MIN: float = 0.02      # 2% daily range minimum
TARGET_VOLATILITY_MAX: float = 0.08      # 8% daily range maximum
TARGET_VOLATILITY_OPTIMAL: float = 0.04  # 4% optimal (peak score)
```

### Display & Performance
```python
TOP_N_DISPLAY: int = 10          # Number of pairs to show
CACHE_TTL: int = 10              # Cache duration (seconds)
REQUEST_TIMEOUT: int = 10        # API timeout (seconds)
```

## 🔍 Scoring Details

### 1. Participation Score (35% weight) - HIGHEST PRIORITY

**What it measures:** Market activity and volume engagement

**Calculation:**
- Uses 24h quote volume (USDT)
- Normalized to percentile rank among all eligible symbols
- Power curve (^0.8) to emphasize high-volume pairs

**Why it matters:** Low participation = dead markets with poor execution

### 2. Liquidity Score (25% weight)

**What it measures:** Execution quality

**Components:**
- **Spread** (70% sub-weight): Bid-ask spread as % of mid price
  - Lower spread = better score (inverse percentile)
- **Depth** (30% sub-weight): Top-of-book bid + ask quantities
  - Higher depth = better score

**Why it matters:** Tight spreads and good depth mean minimal slippage

### 3. Volatility Score (20% weight)

**What it measures:** Tradable price movement

**Calculation:**
- 24h high-low range as % of last price
- Bell-curve preference:
  - **Optimal:** 4% daily range (score = 1.0)
  - **Acceptable:** 2-8% range (score ≥ 0.7)
  - **Too low:** < 2% (boring, score ≤ 0.5)
  - **Too high:** > 8% (risky, exponential decay)

**Why it matters:** Need movement for profit, but not chaos

### 4. Cleanliness Score (10% weight)

**What it measures:** Absence of manipulation signals

**Logic:**
- Large price change (>10%) + low volume (<40th percentile) = **penalty** (manipulation proxy)
- Extreme moves (>15%) = **penalty** regardless of volume
- Moderate moves (2-5%) = **ideal**

**Why it matters:** Avoid manipulated or unstable markets

### 5. Crowd Risk Score (10% weight) - LOWEST PRIORITY

**What it measures:** Funding rate extremes (optional)

**Calculation:**
- Neutral funding (±0.01%) = **best** (balanced market)
- Moderate funding (±0.05%) = **good**
- High funding (±0.1%+) = **penalty** (crowded trade risk)

**Note:** Currently disabled by default (returns 0.5 neutral). Uncomment funding API calls to enable.

**Why it matters:** Extreme funding indicates overcrowded positions

## 🚫 Trade Gate (Hard Veto)

A pair is marked `TRADE_ALLOWED = false` if it fails ANY of:

1. **Participation score < 0.60** - Insufficient market activity
2. **Liquidity score < 0.60** - Poor execution quality
3. **Spread > 0.03%** - Too wide for efficient trading

Even high-scoring pairs can be vetoed by these hard constraints.

## 🔧 API Endpoints Used

All public REST endpoints (no authentication required):

| Endpoint | Purpose | Frequency |
|----------|---------|-----------|
| `/fapi/v1/exchangeInfo` | Get trading symbols and status | Once per run |
| `/fapi/v1/ticker/24hr` | 24h volume, price change, high/low | Once per run (all symbols) |
| `/fapi/v1/ticker/bookTicker` | Best bid/ask prices and quantities | Once per run (all symbols) |
| `/fapi/v1/fundingRate` | Current funding rate | Per-symbol (optional, disabled by default) |

**Performance:** Typical execution time: 5-6 seconds for 500+ symbols

## 💾 Caching

Built-in TTL cache (default: 10 seconds):
- Repeated calls within TTL reuse cached data
- Useful for dashboards or rapid iterations
- Clear cache: `selector.api.cache.clear()`

## 📝 Usage Examples

### Basic Usage
```python
from indicator.best_pair_selector import BestPairSelector

selector = BestPairSelector()
result = selector.select_best_pair()

print(f"Best pair: {result['best_symbol']}")
print(f"Score: {result['best_score']:.4f}")
print(f"Trade allowed: {result['trade_allowed']}")
```

### Custom Configuration
```python
from indicator.best_pair_selector import BestPairSelector, Config

config = Config()
config.WEIGHT_PARTICIPATION = 0.40  # Increase participation weight
config.MAX_SPREAD_PCT = 0.02        # Stricter spread requirement

selector = BestPairSelector(config)
result = selector.select_best_pair()
```

### Get Top 10 DataFrame
```python
result = selector.select_best_pair()
top_10 = result['top_n']

for idx, row in top_10.iterrows():
    print(f"{row['symbol']}: {row['tradability_score']:.4f}")
```

### Access All Scored Symbols
```python
result = selector.select_best_pair()
all_pairs = result['all_results']

# Filter for trade-allowed only
tradable = all_pairs[all_pairs['trade_allowed'] == True]
```

### Enable Funding Rate (slower)
```python
# In best_pair_selector.py, uncomment lines 360-363:
# funding_data = self.api.get_funding_rate(symbol)
# if funding_data:
#     funding_rate = float(funding_data.get('fundingRate', 0))
```

## 🎛️ Advanced Customization

### Adjust Volume Filter
```python
config.VOLUME_PERCENTILE_THRESHOLD = 0.20  # Top 20% instead of 30%
```

### Change Volatility Preference
```python
# Prefer higher volatility for scalping
config.TARGET_VOLATILITY_OPTIMAL = 0.06  # 6% optimal
config.TARGET_VOLATILITY_MAX = 0.12      # 12% max
```

### Modify Factor Weights
```python
# Example: Prioritize liquidity over participation
config.WEIGHT_PARTICIPATION = 0.25
config.WEIGHT_LIQUIDITY = 0.35
# Weights must sum to 1.0
```

### Stricter Trade Gate
```python
config.MIN_PARTICIPATION_GATE = 0.70  # Require 70th percentile
config.MIN_LIQUIDITY_GATE = 0.70
config.MAX_SPREAD_PCT = 0.02          # Max 0.02% spread
```

## 📈 Integration Ideas

### Dashboard
```python
import time
from rich.live import Live
from rich.table import Table

while True:
    result = selector.select_best_pair()
    table = create_rich_table(result['top_n'])
    live.update(table)
    time.sleep(10)
```

### Trading Bot Pair Rotation
```python
def select_trading_pair():
    result = selector.select_best_pair()

    if not result['trade_allowed']:
        return None  # No good pairs right now

    return result['best_symbol']

# Switch pairs every hour based on market conditions
current_pair = select_trading_pair()
```

### Multi-Pair Portfolio
```python
result = selector.select_best_pair()
top_pairs = result['top_n']

# Trade top 3 tradable pairs
tradable = top_pairs[top_pairs['trade_allowed'] == True].head(3)
portfolio = tradable['symbol'].tolist()
```

## ⚠️ Important Notes

1. **Selection Only** - This tool does NOT execute trades, only identifies pairs
2. **No Authentication** - Uses public API endpoints only
3. **Rate Limits** - Respects Binance rate limits via batch endpoints
4. **Market Conditions** - Results change with market activity
5. **Funding Rate** - Disabled by default for performance (adds 1-2s per symbol if enabled)

## 🔄 Maintenance

### Update Excluded Patterns
```python
# In get_eligible_symbols(), line 286:
if any(x in symbol for x in ['UP', 'DOWN', 'BEAR', 'BULL', 'DEFI', 'MOVE']):
    continue
```

Add new patterns as Binance introduces leveraged tokens or indices.

### Monitor API Changes
Binance occasionally updates API endpoints. Check:
- https://binance-docs.github.io/apidocs/futures/en/

## 📊 Statistics Returned

```python
result['stats'] = {
    'total_input': 536,          # Total eligible symbols
    'failed_volume': 375,        # Failed volume filter
    'failed_spread': 87,         # Failed spread filter
    'failed_book_data': 0,       # Missing book ticker data
    'passed': 74,                # Passed all filters
    'total_scored': 74,          # Actually scored
    'elapsed_seconds': 5.36      # Total runtime
}
```

## 🐛 Troubleshooting

### No symbols pass filters
- Lower `VOLUME_PERCENTILE_THRESHOLD` (e.g., 0.50 for top 50%)
- Increase `MAX_SPREAD_PCT` (e.g., 0.05%)

### Timeout errors
- Increase `REQUEST_TIMEOUT` (default: 10s)
- Check internet connection
- Binance API may be under load

### Unexpected best pair
- Review factor weights - participation and liquidity dominate by design
- Check the detailed breakdown for score components
- Verify market conditions align with your volatility preferences

## 📄 License

Part of the NLU indicator suite. Use at your own risk. No trading decisions should be based solely on this tool.

---

**Last Updated:** 2026-02-09
**Version:** 1.0.0
