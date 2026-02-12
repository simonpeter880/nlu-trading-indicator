## EMA Filter Integration Guide

### Quick Start

```python
from indicator import EMAFilterEngine, EMAConfig, Candle

# Initialize engine with config
config = EMAConfig(
    ema_periods=[9, 21, 50],
    slope_threshold_factor=0.15,
    width_threshold_factor=0.10,
)

engine = EMAFilterEngine(config)

# Convert your data to Candles
candles_1m = [
    Candle(timestamp, open, high, low, close, volume)
    for ... in your_1m_data
]

candles_1h = [
    Candle(timestamp, open, high, low, close, volume)
    for ... in your_1h_data
]

# Warmup (one-time initialization)
states = engine.warmup({
    "1m": candles_1m,
    "1h": candles_1h,
})

# Access state
ltf_state = states["1m"]
print(f"Bias: {ltf_state.ema_bias.value}")
print(f"Regime: {ltf_state.ema_regime.value}")
print(f"Strength: {ltf_state.trend_strength_0_100:.0f}%")

# Incremental updates (real-time)
new_candle = Candle(...)
state = engine.on_candle_close("1m", new_candle, atr_percent=0.003)

# Multi-timeframe alignment
mtf = engine.get_mtf_state("1h", "1m")
print(f"MTF Alignment: {mtf.alignment_summary.value}")
```

---

## Integration into Continuous Runner

### Option 1: Add to your main analyzer

```python
# In your analyzer __init__
from indicator import EMAFilterEngine, EMAConfig

self.ema_config = EMAConfig(
    ema_periods=[9, 21, 50],
    slope_lookback_by_tf={"1m": 15, "5m": 10, "1h": 4},
    slope_threshold_factor=0.15,
    width_threshold_factor=0.10,
)

self.ema_engine = EMAFilterEngine(self.ema_config)
self._ema_warmed_up = False

# In your analysis loop (after candle aggregation):
def analyze(self):
    """Main analysis with EMA filter."""

    # Build candles from rolling windows
    ltf_candles = self._build_candles_from_window(
        self._windows.get_window(60),    # 1m window
        candle_size_seconds=60
    )

    htf_candles = self._build_candles_from_window(
        self._windows.get_window(3600),  # 1h window
        candle_size_seconds=3600
    )

    # Warmup once
    if not self._ema_warmed_up and len(ltf_candles) >= 50:
        self.ema_engine.warmup({
            "1m": ltf_candles,
            "1h": htf_candles,
        })
        self._ema_warmed_up = True

    # Get current state
    ltf_ema = self.ema_engine.get_state("1m")
    htf_ema = self.ema_engine.get_state("1h")
    mtf_ema = self.ema_engine.get_mtf_state("1h", "1m")

    # Print EMA block
    if ltf_ema and htf_ema:
        self._print_ema_block(ltf_ema, htf_ema, mtf_ema)

    # Use in trading logic
    if ltf_ema and ltf_ema.ema_bias == EMABias.BULL:
        if ltf_ema.pullback_zone_hit and not ltf_ema.extended:
            # Good long entry zone
            pass
```

### Option 2: Incremental updates on candle close

```python
# When a new candle closes (event-driven):
def on_candle_close(self, tf: str, candle: Candle):
    """Called when candle closes."""

    # Compute ATR% (optional, for adaptive thresholds)
    atr_pct = self._compute_atr_percent(tf)

    # Update EMA state
    ema_state = self.ema_engine.on_candle_close(
        tf=tf,
        candle=candle,
        atr_percent=atr_pct
    )

    # Check for regime changes
    if ema_state.ema_regime == EMARegime.TREND:
        self._on_trend_regime(tf, ema_state)
    else:
        self._on_range_regime(tf, ema_state)
```

---

## Helper: Build Candles from Rolling Window

```python
from indicator import Candle

def _build_candles_from_window(
    self,
    window: TradeWindow,
    candle_size_seconds: int
) -> List[Candle]:
    """Build OHLCV candles from trade window."""

    if not window.prices:
        return []

    candles = []
    candle_ms = candle_size_seconds * 1000

    # Group trades into candles
    start_time = window.start_time
    current_bucket_start = start_time

    bucket_trades = []
    bucket_volumes = []

    for i, price in enumerate(window.prices):
        # Estimate timestamp
        trade_time = start_time + (i / len(window.prices)) * (window.end_time - start_time)

        # Check if next bucket
        while trade_time >= current_bucket_start + candle_ms:
            # Finalize current bucket
            if bucket_trades:
                candles.append(Candle(
                    timestamp=int(current_bucket_start),
                    open=bucket_trades[0],
                    high=max(bucket_trades),
                    low=min(bucket_trades),
                    close=bucket_trades[-1],
                    volume=sum(bucket_volumes),
                ))

            current_bucket_start += candle_ms
            bucket_trades = []
            bucket_volumes = []

        bucket_trades.append(price)
        bucket_volumes.append(window.volumes[i])

    # Finalize last bucket
    if bucket_trades:
        candles.append(Candle(
            timestamp=int(current_bucket_start),
            open=bucket_trades[0],
            high=max(bucket_trades),
            low=min(bucket_trades),
            close=bucket_trades[-1],
            volume=sum(bucket_volumes),
        ))

    return candles
```

---

## Compact Print Block

```python
from indicator.ema_filter import print_ema_block

# In your display loop:
print_ema_block(self.ema_engine, ltf="1m", htf="1h")
```

**Output**:
```
EMA FILTER
  LTF(1m) Regime TREND | Bias BULL   | Align STACKED_UP  | Strength  74
    slope50=+0.00035 (thr=0.00020) width=0.0011 (thr=0.0006) ext21=0.0009 extended=NO pullback=YES
  HTF(1h) Regime TREND | Bias BULL   | Align STACKED_UP  | Strength  61
    slope50=+0.00082 (thr=0.00100) width=0.0019 (thr=0.0020) ext21=0.0003 extended=NO pullback=NO
  Multi-TF: ALIGNED
```

---

## Using EMA Filter for Trade Decisions

### Example 1: Trend Continuation Entry

```python
def check_long_entry(self, ema_state: EMAState, structure, volume_signal):
    """Check for long entry using EMA filter."""

    # 1. EMA must be bullish
    if ema_state.ema_bias != EMABias.BULL:
        return False, "EMA not bullish"

    # 2. Must be in TREND regime
    if ema_state.ema_regime != EMARegime.TREND:
        return False, "Not in trend regime"

    # 3. Require STACKED_UP alignment
    if ema_state.ema_alignment != EMAAlignment.STACKED_UP:
        return False, "EMAs not stacked"

    # 4. Strength threshold
    if ema_state.trend_strength_0_100 < 60:
        return False, f"Trend strength too low ({ema_state.trend_strength_0_100:.0f}%)"

    # 5. Prefer pullback zone (not extended)
    if ema_state.extended:
        return False, "Price too extended from EMA21"

    if not ema_state.pullback_zone_hit:
        # Wait for pullback
        return False, "Not in pullback zone"

    # 6. Confirm with structure and volume
    if structure.structure != StructureState.UP:
        return False, "Structure not bullish"

    if volume_signal < 0.3:
        return False, "Volume not confirming"

    return True, "Long entry confirmed"
```

### Example 2: Multi-Timeframe Filter

```python
def check_mtf_alignment(self, mtf: EMAMultiTFState):
    """Check if multi-timeframe allows trade."""

    if mtf.alignment_summary == MTFAlignment.RANGE_DOMINANT:
        # HTF ranging - reduce size or avoid
        return "scalp_only", "HTF in range"

    if mtf.alignment_summary == MTFAlignment.ALIGNED:
        # Full alignment - full size
        if mtf.htf_bias == EMABias.BULL:
            return "full_long", "HTF+LTF bullish"
        elif mtf.htf_bias == EMABias.BEAR:
            return "full_short", "HTF+LTF bearish"

    if mtf.alignment_summary == MTFAlignment.MIXED:
        # Mixed - reduced size
        return "reduced", "HTF and LTF disagree"

    return "wait", "No clear alignment"
```

### Example 3: Regime-Based Strategy Selection

```python
def select_strategy(self, ema_state: EMAState):
    """Select strategy based on EMA regime."""

    if ema_state.ema_regime == EMARegime.TREND:
        # Trend-following mode
        if ema_state.ema_bias == EMABias.BULL:
            return "trend_long"
        elif ema_state.ema_bias == EMABias.BEAR:
            return "trend_short"
        else:
            return "wait"  # Trend but neutral bias

    elif ema_state.ema_regime == EMARegime.RANGE:
        # Mean reversion mode
        return "mean_reversion"

    return "wait"
```

---

## Advanced: Dynamic Stop Loss Using EMA

```python
def calculate_stop_loss(self, direction: str, ema_state: EMAState, current_price: float):
    """Calculate stop loss using EMA levels."""

    if direction == "long":
        # Use EMA21 as dynamic support
        stop = ema_state.ema21

        # Add buffer based on extension
        buffer = ema_state.debug.get("pullback_band", 0.0005)
        stop = stop * (1 - buffer)

        return stop

    else:  # short
        # Use EMA21 as dynamic resistance
        stop = ema_state.ema21

        buffer = ema_state.debug.get("pullback_band", 0.0005)
        stop = stop * (1 + buffer)

        return stop
```

---

## Testing Your Integration

```bash
# Run EMA filter tests
pytest indicator/tests/test_ema_filter.py -v

# Run specific test
pytest indicator/tests/test_ema_filter.py::TestEMAComputation::test_incremental_equals_batch -v
```

---

## Configuration Tips

### For Scalping (1m/5m)
```python
config = EMAConfig(
    ema_periods=[9, 21, 50],
    slope_lookback_by_tf={"1m": 20, "5m": 15},
    slope_threshold_factor=0.10,  # Lower threshold, more sensitive
    width_threshold_factor=0.08,
)
```

### For Swing Trading (1h/4h)
```python
config = EMAConfig(
    ema_periods=[9, 21, 50],
    slope_lookback_by_tf={"1h": 6, "4h": 4},
    slope_threshold_factor=0.20,  # Higher threshold, less noise
    width_threshold_factor=0.15,
)
```

### For Adaptive Thresholds (Recommended)
```python
# Always pass ATR% when available
atr_pct = compute_atr_percent(candles, period=14)

state = engine.on_candle_close(
    "1m",
    candle,
    atr_percent=atr_pct  # Enables adaptive thresholds
)
```

---

## Summary

The EMA Filter provides:
- ✅ **Real-time incremental updates** (O(1) per candle)
- ✅ **Batch warmup** for initialization
- ✅ **Adaptive thresholds** based on ATR%
- ✅ **Multi-timeframe alignment**
- ✅ **Trend strength scoring** (0-100)
- ✅ **Pullback zone detection**
- ✅ **Extension detection**
- ✅ **Regime classification** (TREND/RANGE)

Integrate it into your pipeline to add **dynamic trend filtering** alongside your existing volume, OI, orderbook, and market structure analysis.
