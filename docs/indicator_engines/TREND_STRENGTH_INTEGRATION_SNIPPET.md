# Trend Strength Integration Snippet

## Minimal Integration Example

```python
from trend_strength import TrendStrengthEngine, TrendStrengthConfig

# 1. Initialize
config = TrendStrengthConfig()
ts_engine = TrendStrengthEngine(config)

# 2. Warmup with historical data
historical_candles = {
    "1m": [...],  # List[Candle]
    "5m": [...],
    "1h": [...]
}
ts_engine.warmup(historical_candles)

# 3. On each candle close
def on_candle_close(tf: str, candle: Candle):
    # Get inputs from other systems
    slope_50 = ema_system.get_slope(tf)                    # EMA slope magnitude
    ribbon_wr = ema_ribbon.get_width_rate(tf)              # Ribbon expansion/contraction
    rv = volume_analyzer.get_rv(tf)                        # Relative volume
    oi_now, oi_prev = oi_tracker.get_oi(tf)               # Open interest snapshots
    bias = ema_system.get_bias(tf)                         # "BULL", "BEAR", or "NEUTRAL"

    # Update Trend Strength with directional bias
    ts_state = ts_engine.on_candle_close(
        tf,
        candle,
        slope_50=slope_50,
        ribbon_width_rate=ribbon_wr,
        rv=rv,
        oi_now=oi_now,
        oi_prev=oi_prev,
        bias=bias  # NEW: Directional bias
    )

    # Use signed strength for trading decisions
    if ts_state.direction_bias > 0:
        # Bullish momentum
        if ts_state.bucket == "STRONG":
            allow_long_entries(position_size=1.0)
        elif ts_state.bucket == "EMERGING":
            allow_long_entries(position_size=0.7)
    elif ts_state.direction_bias < 0:
        # Bearish momentum
        if ts_state.bucket == "STRONG":
            allow_short_entries(position_size=1.0)
        elif ts_state.bucket == "EMERGING":
            allow_short_entries(position_size=0.7)
    else:
        # Neutral - no directional conviction
        block_trend_entries()
```

## Directional Signing Feature (NEW)

### Purpose
The directional bias feature allows Trend Strength to produce **signed strength** values (-100 to +100) that indicate both momentum magnitude AND direction.

### Input Options

**Option 1: bias string** (from EMA or structure analysis)
```python
ts_state = ts_engine.on_candle_close(
    tf, candle,
    slope_50=0.0025,
    ribbon_width_rate=0.15,
    rv=1.5,
    bias="BULL"  # "BULL", "BEAR", or "NEUTRAL"
)
```

**Option 2: direction_bias int** (takes precedence)
```python
ts_state = ts_engine.on_candle_close(
    tf, candle,
    slope_50=0.0025,
    ribbon_width_rate=0.15,
    rv=1.5,
    direction_bias=+1  # +1 (bull), -1 (bear), 0 (neutral)
)
```

### Output Fields

```python
ts_state.strength_smooth    # 0-100 (unsigned magnitude)
ts_state.direction_bias     # -1, 0, or +1
ts_state.strength_signed    # -100 to +100 (direction_bias * strength_smooth)
```

### Examples

```python
# BULL bias with 75 strength
# → strength_smooth = 75
# → direction_bias = +1
# → strength_signed = +75

# BEAR bias with 75 strength
# → strength_smooth = 75
# → direction_bias = -1
# → strength_signed = -75

# NEUTRAL bias (no directional conviction)
# → strength_smooth = 75
# → direction_bias = 0
# → strength_signed = 0
```

### Trading Logic with Signed Strength

```python
ts_state = ts_engine.on_candle_close(tf, candle, ..., bias=ema_bias)

# Unified long/short scoring
if ts_state.strength_signed > 60:
    # Strong bullish momentum (60-100)
    allow_long_entries(size=1.0)
    block_short_entries()

elif ts_state.strength_signed > 30:
    # Emerging bullish momentum (30-60)
    allow_long_entries(size=0.7)
    block_short_entries()

elif ts_state.strength_signed < -60:
    # Strong bearish momentum (-60 to -100)
    allow_short_entries(size=1.0)
    block_long_entries()

elif ts_state.strength_signed < -30:
    # Emerging bearish momentum (-30 to -60)
    allow_short_entries(size=0.7)
    block_long_entries()

else:
    # Weak or neutral (-30 to +30)
    block_trend_entries()
```

## Compact Print Block

```python
from trend_strength import format_trend_strength_output

# Get states for all timeframes
results = {
    "1m": ts_engine.get_state("1m"),
    "5m": ts_engine.get_state("5m"),
    "1h": ts_engine.get_state("1h")
}

# Print compact output
print(format_trend_strength_output(results, compact=True))
```

**Sample Output:**
```
TREND STRENGTH
1m: +67 (STRONG) raw=71 smooth=67 bias=+1
  comps_norm: slope=0.82 ribbon=0.60 rv=0.55 oi=0.70
  comps_raw : slope=0.00034 wr=+0.18 RV=1.62 dOI=0.0041
5m: -42 (EMERGING) raw=45 smooth=42 bias=-1
  comps_norm: slope=0.40 ribbon=0.35 rv=0.50 oi=0.25
  comps_raw : slope=0.00021 wr=+0.08 RV=1.25 dOI=0.0015
```

## Integration with EMA and Structure

### Get Bias from EMA System

```python
from ema_filter import EMAEngine

ema_engine = EMAEngine(...)
ema_state = ema_engine.on_candle_close(tf, candle)

# Use EMA bias for Trend Strength
ts_state = ts_engine.on_candle_close(
    tf, candle,
    slope_50=abs(ema_state.slope_50),  # Use EMA slope magnitude
    ribbon_width_rate=ribbon_wr,
    rv=rv,
    bias=ema_state.ema_bias  # "BULL" / "BEAR" / "NEUTRAL"
)
```

### Get Bias from Structure Analysis

```python
# Structure-based bias
if price_above_structure_high:
    structure_bias = "BULL"
elif price_below_structure_low:
    structure_bias = "BEAR"
else:
    structure_bias = "NEUTRAL"

ts_state = ts_engine.on_candle_close(
    tf, candle,
    slope_50=slope,
    ribbon_width_rate=ribbon_wr,
    rv=rv,
    bias=structure_bias
)
```

## Complete Context: Supertrend + EMA Ribbon + Trend Strength

```python
# 1. Supertrend: Regime and direction filter
st_state = supertrend_engine.on_candle_close(tf, candle)

# 2. EMA Ribbon: Stack health
ribbon_state = ema_ribbon_engine.on_candle_close(tf, candle)

# 3. Trend Strength: Momentum with direction
ts_state = ts_engine.on_candle_close(
    tf, candle,
    slope_50=abs(ribbon_state.center_ema_slope),
    ribbon_width_rate=ribbon_state.ribbon_width_rate,
    rv=volume_analyzer.get_rv(tf),
    bias=ema_system.get_bias(tf)
)

# Decision matrix
if st_state.regime == "TREND":
    if ribbon_state.state == "HEALTHY":
        if ts_state.strength_signed > 60:
            # BEST SETUP: Trend confirmed, stack healthy, strong bullish momentum
            position_size = 1.0
            strategy = "AGGRESSIVE_LONG"
        elif ts_state.strength_signed < -60:
            # BEST SETUP: Trend confirmed, stack healthy, strong bearish momentum
            position_size = 1.0
            strategy = "AGGRESSIVE_SHORT"
        elif abs(ts_state.strength_signed) > 30:
            # GOOD SETUP: Emerging momentum
            position_size = 0.7
            strategy = "MODERATE"
    elif ribbon_state.state == "WEAKENING":
        # Ribbon weakening - reduce size even if strength strong
        position_size = 0.5
else:
    # Not in trend - skip
    position_size = 0.0
```

## Key Points

1. **Directional Bias Required**: Always provide `bias` or `direction_bias` for meaningful signed strength
2. **Source of Bias**: Get from EMA system, structure analysis, or Supertrend direction
3. **Signed Strength**: Use `strength_signed` (-100..+100) for unified long/short scoring
4. **Bucketing**: Still use `bucket` (WEAK/EMERGING/STRONG) for risk management
5. **Precedence**: `direction_bias` int parameter overrides `bias` string
6. **Default**: If no bias provided, defaults to NEUTRAL (0) → `strength_signed = 0`

## Testing

Run all tests including new directional tests:
```bash
pytest tests/test_trend_strength.py -v
```

**Result: 25/25 tests passing ✓** (includes 7 new directional signing tests)
