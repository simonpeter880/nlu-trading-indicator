# Supertrend Filter Module - Implementation Complete

## Overview

A real-time Supertrend Engine used **ONLY for regime labeling (TREND/CHOP) and directional bias filtering (UP/DOWN)**. NOT used as entry trigger.

Provides context for other trading systems by identifying market regime and allowed trade direction.

## Deliverables

### 1. Core Module: [supertrend_filter.py](supertrend_filter.py)
- `SupertrendEngine` with incremental O(1) updates
- `SupertrendConfig` with per-timeframe overrides
- Wilder ATR smoothing
- Band locking (non-repainting)
- Flip-rate based regime detection
- Hysteresis for stability

### 2. Tests: [tests/test_supertrend_filter.py](tests/test_supertrend_filter.py)
**All 23 tests passing ✓**
- ATR calculation (Wilder smoothing)
- Band locking logic
- Direction flip detection
- Regime classification (TREND vs CHOP)
- Strength calculation
- Incremental vs batch equivalence
- Multi-timeframe support

### 3. Integration: [supertrend_integration.py](supertrend_integration.py)
- Streaming loop example
- Production template
- Compact dashboard output
- Multi-timeframe alignment logic

## Key Features

### Core Purpose

**Supertrend is used as a FILTER, not an entry trigger:**
- **Regime Label:** TREND or CHOP
- **Directional Bias:** UP or DOWN
- **Strength Score:** 0-100 confidence

**DO NOT:**
- Place trades on flip_event
- Use as standalone entry signal

**DO:**
- Filter trades by regime (avoid trend strategies in CHOP)
- Filter direction (only longs in UP, shorts in DOWN)
- Combine with other entry systems

### Supertrend Formula

**True Range:**
```
TR = max(high - low, |high - prev_close|, |low - prev_close|)
```

**ATR (Wilder Smoothing):**
```
First: ATR = SMA(TR, period)
Then:  ATR = (ATR_prev × (n-1) + TR) / n
```

**Basic Bands:**
```
HL2 = (high + low) / 2
basic_upper = HL2 + multiplier × ATR
basic_lower = HL2 - multiplier × ATR
```

**Final Bands (with locking):**
```
if (basic_upper < final_upper_prev) OR (close_prev > final_upper_prev):
    final_upper = basic_upper
else:
    final_upper = final_upper_prev  # Lock

if (basic_lower > final_lower_prev) OR (close_prev < final_lower_prev):
    final_lower = basic_lower
else:
    final_lower = final_lower_prev  # Lock
```

**Direction:**
```
if close > final_upper: direction = UP
elif close < final_lower: direction = DOWN
else: direction = previous  # Persist
```

**ST Line:**
```
st_line = final_lower  (if UP)
st_line = final_upper  (if DOWN)
```

### Regime Detection Logic

**CHOP regime if:**
- `flip_rate > flip_rate_chop` (default 10%)
- OR `atr_percent < atrp_min_chop AND flip_rate > 0.05`
- OR `distance_avg < st_distance_factor × atr_percent` (hugging line)

**TREND regime if:**
- `flip_rate < flip_rate_trend` (default 5%)
- AND `direction_hold_count >= min_hold_bars` (default 3)
- AND `distance_avg >= st_distance_factor × atr_percent` (default 15%)

**Otherwise:** CHOP (conservative default)

### Regime Strength (0-100)

Components:
```
hold_comp = clip(hold_count / (min_hold_bars×2), 0..1)
flip_comp = 1 - clip(flip_rate / flip_rate_chop, 0..1)
dist_comp = clip(distance_avg / (2×st_distance_factor×atr_percent), 0..1)

strength = 100 × (0.35×hold_comp + 0.35×flip_comp + 0.30×dist_comp)
```

**Caps:**
- CHOP regime: max 40
- TREND regime: 0-100

## Configuration

```python
SupertrendConfig(
    # Core parameters
    atr_period=10,
    multiplier=3.0,
    timeframes=["1m", "5m", "1h"],

    # Regime detection
    flip_window=20,
    flip_rate_chop=0.10,    # >10% flips => CHOP
    flip_rate_trend=0.05,   # <5% flips => TREND
    min_hold_bars=3,
    st_distance_factor=0.15,  # 15% of ATR%
    atrp_min_chop=0.0010,   # <0.1% ATR => CHOP-prone

    # Per-timeframe overrides
    per_tf_overrides={
        "1m": {"atr_period": 7, "multiplier": 2.5},
        "1h": {"atr_period": 14, "multiplier": 3.5}
    }
)
```

## Usage

### Basic Integration

```python
from supertrend_filter import SupertrendEngine, SupertrendConfig, Candle

# 1. Initialize
config = SupertrendConfig(
    atr_period=10,
    multiplier=3.0,
    flip_window=20
)
engine = SupertrendEngine(config)

# 2. Warmup with historical data
historical_candles = {
    "1m": [...],  # List[Candle]
    "5m": [...],
    "1h": [...]
}
engine.warmup(historical_candles)

# 3. Process new candles incrementally
def on_candle_close(tf: str, candle: Candle):
    st_state = engine.on_candle_close(tf, candle)

    # Use as filter (NOT entry trigger)
    if st_state.regime == Regime.CHOP:
        disable_trend_strategies()
    elif st_state.regime == Regime.TREND:
        if st_state.st_direction == Direction.UP:
            allow_long_entries_only()
        else:
            allow_short_entries_only()
```

### Compact Output

```python
from supertrend_filter import format_supertrend_output

result = engine.update(candles_by_tf)
print(format_supertrend_output(result, compact=True))
```

**Sample Output:**
```
SUPERTREND
1m: dir=UP regime=TREND strength=72 line=66310 atr%=0.22% flips20=1 (0.05) hold=6 dist_avg=0.09%
5m: dir=UP regime=CHOP  strength=28 flips20=4 (0.20) hold=2 dist_avg=0.03% hugging_line=YES
1h: dir=UP regime=TREND strength=85 line=66450 atr%=0.35% flips20=0 (0.00) hold=12 dist_avg=0.15%
```

## Trading Logic Examples

### Regime Filtering

```python
st_state = engine.on_candle_close("1m", candle)

if st_state.regime == Regime.TREND:
    # Enable trend-following strategies
    if st_state.st_direction == Direction.UP:
        # Allow long setups from EMA/VWAP systems
        allow_longs()
    else:
        # Allow short setups
        allow_shorts()

elif st_state.regime == Regime.CHOP:
    # Disable trend-following
    disable_trend_strategies()

    # Optional: Enable range strategies
    enable_range_strategies()
```

### Direction Filter

```python
# Only take trades aligned with Supertrend direction
if st_state.regime == Regime.TREND:
    if st_state.st_direction == Direction.UP:
        # Block shorts, allow longs
        if entry_signal_long():
            enter_long()
    else:
        # Block longs, allow shorts
        if entry_signal_short():
            enter_short()
```

### Flip Event (Use Carefully)

```python
# DO NOT use flip_event as entry trigger directly
# Use it for confirmation or to pause entries briefly

if st_state.flip_event:
    # Direction just flipped - wait for confirmation
    if st_state.direction_hold_count < min_hold_bars:
        # Too early, skip this entry
        pass
    else:
        # Confirmed flip, can proceed with aligned entries
        if st_state.st_direction == Direction.UP:
            # Now allowing longs after confirmed flip
            pass
```

### Multi-Timeframe Alignment

```python
def check_mtf_alignment(results):
    """Require multi-timeframe alignment"""

    htf = results["1h"]
    mtf = results["5m"]
    ltf = results["1m"]

    # Require HTF in TREND
    if htf.regime != Regime.TREND:
        return "BLOCK_ALL"

    # Require direction alignment
    if htf.st_direction == mtf.st_direction == ltf.st_direction:
        return f"ALLOW_{htf.st_direction.value}"

    # Mixed signals
    return "CAUTION"
```

### Strength-Based Risk Management

```python
st_state = engine.on_candle_close("1m", candle)

if st_state.regime == Regime.TREND:
    if st_state.regime_strength_0_100 > 70:
        # Strong trend - full position size
        position_size = 1.0
    elif st_state.regime_strength_0_100 > 50:
        # Moderate - reduce size
        position_size = 0.7
    else:
        # Weak trend - minimal size or avoid
        position_size = 0.3
```

### Flip Rate Analysis

```python
# High flip rate => choppy market
if st_state.flip_rate > 0.15:  # 15% flips
    # Very choppy - avoid trend trades
    disable_trend_entries()

# Low flip rate => clean trend
elif st_state.flip_rate < 0.05:  # 5% flips
    # Clean trend - aggressive trend-following
    enable_aggressive_entries()
```

## Performance

- **Time Complexity:** O(1) per candle update
- **Space Complexity:** O(flip_window) per timeframe
- **ATR Calculation:** Wilder smoothing (no full recalculation)
- **Band Updates:** Incremental with locking
- **Memory:** Small deques for flip/distance tracking

## Technical Highlights

1. **Wilder ATR Smoothing** - Proper exponential smoothing, not SMA
2. **Band Locking** - Non-repainting, matches TradingView
3. **Flip-Rate Detection** - Identifies choppy regimes
4. **Hysteresis** - min_hold_bars prevents rapid flipping
5. **Distance Tracking** - Detects price hugging ST line (noisy)
6. **Per-TF Overrides** - Different parameters per timeframe

## Files

- `supertrend_filter.py` - Core implementation (469 lines)
- `tests/test_supertrend_filter.py` - Comprehensive tests (432 lines)
- `supertrend_integration.py` - Integration examples (344 lines)
- `SUPERTREND_README.md` - This document

## Testing

Run all tests:
```bash
pytest tests/test_supertrend_filter.py -v
```

Result: **23/23 tests passing ✓**

## Integration Points

This module provides **regime context and directional filter** for:

1. **EMA System** - Only take EMA entries when ST regime = TREND
2. **VWAP System** - Only VWAP reclaims when ST direction aligned
3. **EMA Ribbon** - Require ST TREND + Ribbon HEALTHY for best setups
4. **Any Entry System** - Use ST as universal regime filter

## Important Notes

### What Supertrend IS

✓ Regime classifier (TREND vs CHOP)
✓ Directional filter (UP vs DOWN)
✓ Strength indicator (0-100)
✓ Flip-rate tracker
✓ Multi-timeframe alignment tool

### What Supertrend IS NOT

✗ Entry trigger (DO NOT trade on flip_event)
✗ Stop-loss tool
✗ Profit target
✗ Position sizing (except via strength)

### Best Practices

1. **Use as filter, not trigger**
   - Block trades against ST direction
   - Avoid trends in CHOP regime

2. **Require confirmation**
   - Wait for min_hold_bars after flip
   - Check regime strength > threshold

3. **Multi-timeframe alignment**
   - HTF regime for overall bias
   - LTF direction for entry timing

4. **Combine with other systems**
   - ST filters which entries are allowed
   - Other systems provide actual entry signals

## Next Steps

1. **Integrate into main loop**
   - Hook into candle close events
   - Update ST state per timeframe

2. **Apply regime filter**
   - Block trend strategies in CHOP
   - Block counter-trend in TREND

3. **Apply direction filter**
   - Only longs when UP
   - Only shorts when DOWN

4. **Multi-timeframe logic**
   - Require HTF + MTF alignment
   - Use LTF for entry timing

---

**Status:** ✓ Complete and tested
**Performance:** All tests passing, O(1) incremental updates
**Ready for:** Production integration as regime filter
