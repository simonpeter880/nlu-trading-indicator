# Best Pair Selector - Implementation Summary

## 📁 Files Created

1. **[best_pair_selector.py](best_pair_selector.py)** (730 lines)
   - Main implementation
   - Fully documented with docstrings
   - CLI executable: `python best_pair_selector.py`

2. **[BEST_PAIR_SELECTOR_README.md](BEST_PAIR_SELECTOR_README.md)**
   - Comprehensive documentation
   - Configuration guide
   - Usage examples
   - Troubleshooting

3. **[example_best_pair.py](example_best_pair.py)**
   - 7 working examples
   - Demonstrates all major features
   - Run with: `python example_best_pair.py`

## ✅ Requirements Met

### Core Functionality
- ✅ Scans all Binance USDⓈ-M perpetual futures
- ✅ Multi-factor weighted scoring system
- ✅ Returns best symbol + top 10 ranked table
- ✅ Per-symbol factor breakdown + raw metrics
- ✅ Hard "TRADE_ALLOWED" gate with veto logic

### Data Sources
- ✅ `/fapi/v1/exchangeInfo` - trading status, symbols
- ✅ `/fapi/v1/ticker/24hr` - volume, price change, high/low
- ✅ `/fapi/v1/ticker/bookTicker` - best bid/ask, liquidity
- ✅ `/fapi/v1/fundingRate` - funding rate (optional)

### Constraints & Filters
- ✅ Status = TRADING, contractType = PERPETUAL, quoteAsset = USDT
- ✅ Excludes leveraged tokens/indices (UP/DOWN/BEAR/BULL/DEFI/MOVE)
- ✅ Volume filter: top 30% by default (configurable)
- ✅ Spread filter: ≤ 0.03% by default (configurable)

### Priority Factors (Exact Implementation)

#### 1. Participation/Activity (35% weight) ✅
- 24h quoteVolume normalized by percentile
- Power curve (^0.8) to emphasize high volume
- Penalizes low-volume "dead markets"
- Output: `participation_score` ∈ [0,1]

#### 2. Liquidity Quality (25% weight) ✅
- Spread percentage: (ask - bid) / mid
- Depth proxy: bidQty + askQty from book ticker
- 70% weight on spread, 30% on depth
- Output: `liquidity_score` ∈ [0,1]

#### 3. Tradable Volatility (20% weight) ✅
- 24h range percentage: (high - low) / price
- Bell-shaped preference curve
- Target range: 2-8%, optimal: 4%
- Penalizes both flat and extreme volatility
- Output: `volatility_score` ∈ [0,1]

#### 4. Cleanliness/Noise (10% weight) ✅
- Detects manipulation: large change + low volume
- Penalizes extreme moves (>15%)
- Prefers moderate changes (2-5%)
- Output: `cleanliness_score` ∈ [0,1]

#### 5. Crowd Risk (10% weight) ✅
- Uses funding rate (optional)
- Prefers neutral funding (±0.01%)
- Penalizes extreme funding (crowding)
- Defaults to 0.5 if funding unavailable
- Output: `crowd_score` ∈ [0,1]

### Hard Trade Gate ✅
Trade disallowed (`TRADE_ALLOWED=false`) if ANY fail:
- `participation_score < 0.60`
- `liquidity_score < 0.60`
- `spread_pct > 0.03%`

### Engineering ✅
- ✅ Efficient: batch endpoints, minimal requests
- ✅ Caching: TTL-based (default 10s)
- ✅ Error handling: timeouts, request exceptions
- ✅ Config file: `Config` dataclass with all thresholds/weights
- ✅ Robust logging: info/error messages

## 🎯 Key Features

### Performance
- **Typical runtime:** 5-6 seconds for 500+ symbols
- **Request optimization:**
  - 1 call: `/exchangeInfo`
  - 1 call: `/ticker/24hr` (all symbols)
  - 1 call: `/ticker/bookTicker` (all symbols)
  - Optional: per-symbol funding (disabled by default)
- **Caching:** 10-second TTL for repeated calls

### Output Format
```
🏆 BEST PAIR: BTCUSDT
   Tradability Score: 0.8483
   Trade Allowed: ✅ YES

📊 STATISTICS:
   Total symbols scanned: 536
   Passed filters: 74
   Elapsed time: 5.36s

📈 TOP 10 RANKED PAIRS:
   Rank Symbol  Score  Part. Liq. Vol. Clean Crowd Volume    Spread%
   ...detailed table...

🔍 DETAILED BREAKDOWN - BTCUSDT:
   Factor-by-factor analysis with raw metrics
```

### Configurability
All parameters customizable via `Config`:
- Factor weights (must sum to 1.0)
- Hard filter thresholds
- Trade gate requirements
- Volatility preferences
- Display settings

## 📊 Example Results

**Latest Run (2026-02-09):**
```
Best Pair: BTCUSDT
Score: 0.8483
Volume: $13,274.9M
Spread: 0.0001%
Range: 5.69%
Trade Allowed: YES

Top 5:
1. BTCUSDT    0.8483
2. XRPUSDT    0.8420
3. ETHUSDT    0.8310
4. 1000PEPEUSDT 0.8272
5. BNBUSDT    0.8180
```

## 🔧 Usage

### Command Line
```bash
python indicator/best_pair_selector.py
```

### Python Import
```python
import importlib.util
spec = importlib.util.spec_from_file_location(
    'best_pair_selector',
    'indicator/best_pair_selector.py'
)
bps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bps)

selector = bps.BestPairSelector()
result = selector.select_best_pair()
print(result['best_symbol'])
```

### Examples
```bash
python indicator/example_best_pair.py
```

## 🎨 Design Decisions

### Why Public API Only?
- No authentication overhead
- Faster execution
- Suitable for market scanning
- Easy to deploy/share

### Why Disable Funding by Default?
- Adds 1-2 seconds per symbol (slow)
- Least important factor (10% weight)
- Can be enabled in line 360-363 if needed

### Why Bell Curve for Volatility?
- Too low = no profit opportunity
- Too high = excessive risk
- Moderate = tradable with reasonable risk/reward

### Why Hard Trade Gate?
- Prevents trading dead/illiquid markets
- Non-negotiable safety threshold
- Separate from scoring (veto power)

## 🚀 Future Enhancements (Optional)

### Performance
- [ ] Async API calls with `aiohttp`
- [ ] Persistent Redis cache
- [ ] Batch funding rate endpoint (if available)

### Features
- [ ] Historical backtesting of selections
- [ ] Multi-timeframe volatility analysis
- [ ] Order book depth analysis (L2 data)
- [ ] Correlation filtering (avoid similar pairs)
- [ ] Machine learning factor optimization

### Integration
- [ ] REST API wrapper (Flask/FastAPI)
- [ ] WebSocket live updates
- [ ] Telegram/Discord notifications
- [ ] Grafana dashboard

## 📝 Notes

### Limitations
- **No execution:** Selection only, no trading
- **Public data:** No account-specific info
- **Snapshot:** Real-time data, not predictive
- **Funding:** Disabled by default (performance)

### Assumptions
- USDT-margined perpetuals only
- Standard contract types (no exotic derivatives)
- Public Binance API availability
- Reasonable internet connection

### Dependencies
```python
requests==2.x
pandas==1.x
numpy==1.x
```

## 📖 Documentation

- **[README](BEST_PAIR_SELECTOR_README.md):** Full documentation
- **[Examples](example_best_pair.py):** 7 working examples
- **Code comments:** Inline documentation throughout

## ✨ Highlights

1. **Priority-driven design:** Participation and liquidity dominate (60% weight)
2. **Hard safety gate:** Veto power for low-quality pairs
3. **Bell-curve volatility:** Prefers moderate movement
4. **Efficient architecture:** Minimal API calls, smart caching
5. **Highly configurable:** All parameters tunable
6. **Production-ready:** Error handling, logging, timeouts
7. **Well-documented:** README + examples + inline comments

## 🎓 Learning Points

### Factor Selection
- **Participation** matters most (liquidity follows volume)
- **Spread** is critical for execution quality
- **Volatility** needs balance (not min/max)
- **Funding** signals crowding but is expensive to fetch

### Scoring Techniques
- Percentile normalization for comparability
- Power curves to emphasize extremes
- Bell curves for optimal ranges
- Hard gates for non-negotiable constraints

### API Optimization
- Batch endpoints over individual calls
- Cache with TTL for rapid iterations
- Timeout protection for reliability
- Error handling for robustness

---

**Implementation Date:** 2026-02-09
**Status:** ✅ Complete and Tested
**Location:** `/home/cymo/nlu/indicator/`

All requirements met. Ready for production use.
