# Trend Strength Composite Module - Implementation Complete

## Overview

A lag-free trend strength measurement system that combines multiple momentum components into a single composite score (0-100) **with directional bias** (-100 to +100). Used for position sizing, risk management, and confirming trend quality.

**Key Principle:** This module provides a STRENGTH SCORE with direction, not entry signals. Use it alongside Supertrend (regime), EMA Ribbon (health), and other entry systems for complete context.

**NEW: Directional Signing** - Outputs `strength_signed` (-100 to +100) based on bias from EMA or structure analysis, enabling unified long/short momentum scoring.

## Deliverables

### 1. Core Module: [trend_strength.py](trend_strength.py)
- `TrendStrengthEngine` with incremental O(1) updates
- `TrendStrengthConfig` with component weights
- Component normalization (EMA slope, ribbon expansion, RV, OI)
- Weight renormalization when components missing
- EMA smoothing for stability
- Bucketing: WEAK / EMERGING / STRONG
- Safety caps for external conditions

### 2. Tests: [tests/test_trend_strength.py](tests/test_trend_strength.py)
**All 25 tests passing ✓** (includes 7 new directional signing tests)
- Component normalization tests
- Weighting and composite calculation
- Smoothing verification
- Bucketing logic
- Safety cap enforcement
- End-to-end scenarios
- Edge case handling

### 3. Integration: [trend_strength_integration.py](trend_strength_integration.py)
- Directional signing example (NEW)
- Basic usage example
- Multi-timeframe update
- Integration with EMA Ribbon and Supertrend
- Trading logic examples

### 4. Integration Snippet: [TREND_STRENGTH_INTEGRATION_SNIPPET.md](TREND_STRENGTH_INTEGRATION_SNIPPET.md)
- Minimal integration code
- Directional signing usage (NEW)
- Compact print block
- Complete context examples

## Key Features

### Directional Signing (NEW)

**Purpose:** Provides signed strength (-100 to +100) indicating both momentum magnitude AND direction.

**Inputs:**
- `bias`: String ("BULL", "BEAR", "NEUTRAL") from EMA or structure analysis
- `direction_bias`: Int (-1, 0, +1) - takes precedence over bias string

**Outputs:**
- `strength_smooth`: 0-100 (unsigned magnitude)
- `direction_bias`: -1, 0, or +1
- `strength_signed`: -100 to +100 (direction_bias × strength_smooth)

**Usage:**
```python
ts_state = engine.on_candle_close(
    tf, candle,
    slope_50=0.0025,
    ribbon_width_rate=0.15,
    rv=1.5,
    bias="BULL"  # From EMA or structure
)

# Use signed strength for unified long/short scoring
if ts_state.strength_signed > 60:
    allow_long_entries(size=1.0)  # Strong bullish
elif ts_state.strength_signed < -60:
    allow_short_entries(size=1.0)  # Strong bearish
else:
    reduce_or_block_entries()  # Weak or neutral
```

### Components (4 Total)

1. **EMA Slope Magnitude**
   - Measures directional velocity
   - Normalized by ATR% for cross-instrument comparison
   - Weight: 35% (default)

2. **Ribbon Width Rate (Expansion/Contraction)**
   - From EMA Ribbon module
   - Positive = expanding (bullish), negative = contracting
   - Weight: 25% (default)

3. **Relative Volume (RV)**
   - Current volume vs rolling average
   - RV > 1.5 = high participation
   - Weight: 25% (default)

4. **Open Interest (OI) Expansion** (Optional)
   - Change rate in OI
   - Positive = new money entering
   - Weight: 15% (default)

### Composite Formula

```
# Step 1: Normalize each component to [0, 1]
ema_slope_norm = clip(slope / (slope_strong_factor × ATR%), 0, 1)
ribbon_norm = clip((WR - WR_low) / (WR_high - WR_low), 0, 1)
rv_norm = clip((RV - RV_low) / (RV_high - RV_low), 0, 1)
oi_norm = clip(dOI / OI_strong, 0, 1)

# Step 2: Weight components
strength_raw = 100 × Σ(weight_i × component_norm_i)

# Step 3: Renormalize if components missing
# (e.g., if OI not available, redistribute its weight)

# Step 4: Apply EMA smoothing
strength_smooth = alpha × strength_raw + (1 - alpha) × strength_smooth_prev
where alpha = 2 / (smooth_period + 1)

# Step 5: Apply safety caps if needed
strength_smooth = min(strength_smooth, cap_value)
```

### Bucketing

- **WEAK** (0-30): Minimal momentum, avoid trend entries
- **EMERGING** (30-60): Developing momentum, reduced size
- **STRONG** (60-100): High momentum, full conviction trades

### Safety Caps

External conditions that cap strength:

1. **Structure Range Cap** (default: 40)
   - Applied when `structure_in_range = True`
   - Market stuck in range despite local momentum

2. **Supertrend CHOP Cap** (default: 35)
   - Applied when `supertrend_regime = CHOP`
   - Prevents false strength in choppy conditions

3. **RV Dead Cap** (default: 25)
   - Applied when `RV < rv_dead_threshold`
   - Low volume = unreliable momentum

## Configuration

```python
TrendStrengthConfig(
    # Timeframes
    timeframes=["1m", "5m", "1h"],

    # Smoothing and periods
    smooth_period=5,
    atr_period=14,
    rv_period=20,

    # EMA slope normalization
    ema_slope_k_by_tf={"1m": 15, "5m": 10, "1h": 4},  # Lookback per TF
    ema_slope_k_default=10,
    ema_slope_strong_factor=0.20,  # Strong slope = 0.20 × ATR%

    # Ribbon normalization
    ribbon_wr_low=-0.10,   # Contraction boundary
    ribbon_wr_high=0.20,   # Expansion boundary

    # RV normalization
    rv_low=0.8,            # Below average
    rv_high=2.0,           # Strong participation

    # OI normalization
    oi_ref_by_tf={"1m": 0.003, "5m": 0.006, "1h": 0.020},
    oi_ref_default=0.006,

    # Component weights
    w_ema_slope=0.35,
    w_ribbon=0.25,
    w_rv=0.20,
    w_oi=0.20,

    # Safety caps
    cap_when_structure_range=50,
    cap_when_supertrend_chop=50,
    cap_when_rv_dead=25,

    # Bucketing
    bucket_weak_max=30.0,
    bucket_emerging_max=60.0
)
```

## Usage

### Basic Integration

```python
from trend_strength import TrendStrengthEngine, TrendStrengthConfig, Candle

# 1. Initialize
config = TrendStrengthConfig()
engine = TrendStrengthEngine(config)

# 2. Warmup with historical data
historical_candles = {
    "1m": [...],  # List[Candle]
    "5m": [...],
    "1h": [...]
}
engine.warmup(historical_candles)

# 3. Process new candles with external indicators
def on_candle_close(tf: str, candle: Candle):
    # Get indicators from other systems
    slope = ema_ribbon_engine.get_slope(tf)
    ribbon_wr = ema_ribbon_engine.get_width_rate(tf)
    rv = volume_analyzer.get_rv(tf)
    oi_now = oi_tracker.get_current(tf)  # Optional
    oi_prev = oi_tracker.get_previous(tf)  # Optional
    bias = ema_system.get_bias(tf)  # "BULL", "BEAR", or "NEUTRAL"

    # Update Trend Strength with directional bias
    ts_state = engine.on_candle_close(
        tf,
        candle,
        slope_50=abs(slope),
        ribbon_width_rate=ribbon_wr,
        rv=rv,
        oi_now=oi_now,
        oi_prev=oi_prev,
        bias=bias  # NEW: Directional bias
    )

    # Use strength for position sizing
    if ts_state.bucket == TrendBucket.STRONG:
        position_size = 1.0  # Full size
    elif ts_state.bucket == TrendBucket.EMERGING:
        position_size = 0.7  # Reduced
    else:
        position_size = 0.3  # Minimal
```

### Compact Output

```python
from trend_strength import format_trend_strength_output

results = {
    "1m": engine.get_state("1m"),
    "5m": engine.get_state("5m"),
    "1h": engine.get_state("1h")
}

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
1h: +0 (WEAK) raw=28 smooth=28 bias=+0
  comps_norm: slope=0.22 ribbon=0.15 rv=0.35 oi=0.20
```

## Trading Logic Examples

### Position Sizing

```python
ts_state = engine.on_candle_close("1m", candle, ...)

if ts_state.bucket == TrendBucket.STRONG:
    # High conviction
    position_size = 1.0
    stop_distance = 2.0 * atr
    print("STRONG trend - full size, wider stops")

elif ts_state.bucket == TrendBucket.EMERGING:
    # Developing
    position_size = 0.7
    stop_distance = 1.5 * atr
    print("EMERGING trend - reduced size, moderate stops")

else:  # WEAK
    # Low conviction
    position_size = 0.3
    stop_distance = 1.0 * atr
    print("WEAK trend - minimal size, tight stops")
```

### Component Analysis

```python
ts_state = engine.on_candle_close("1m", candle, ...)

# Check individual components
if ts_state.components_norm['ema_slope'] > 0.8:
    print("EMA slope very strong - directional velocity high")

if ts_state.components_norm['ribbon'] > 0.7:
    print("Ribbon expanding well - trend structure building")

if ts_state.components_norm['rv'] > 0.7:
    print("High relative volume - strong participation")

if ts_state.components_norm.get('oi') is not None:
    if ts_state.components_norm['oi'] > 0.7:
        print("OI expanding - new money entering")
```

### Safety Caps

```python
ts_state = engine.on_candle_close(
    "1m", candle,
    slope_50=0.003,
    ribbon_width_rate=0.15,
    rv=0.5,  # Low volume
    structure_in_range=False,
    supertrend_regime="CHOP"
)

if ts_state.safety_caps_active:
    print(f"Strength capped at {ts_state.strength_smooth} due to:")
    # Raw strength might be 65, but capped to 35 due to CHOP
    if "supertrend_chop" in str(ts_state.safety_caps_active):
        print("  - Supertrend in CHOP regime")
    if "rv_dead" in str(ts_state.safety_caps_active):
        print("  - Volume too low (RV < 0.6)")
```

### Integration with Other Modules

```python
# Complete trading context
def evaluate_setup(tf: str, candle: Candle):
    # 1. Regime filter (Supertrend)
    st_state = supertrend_engine.on_candle_close(tf, candle)

    # 2. Stack health (EMA Ribbon)
    ribbon_state = ema_ribbon_engine.on_candle_close(tf, candle)

    # 3. Trend strength (This module)
    ts_state = trend_strength_engine.on_candle_close(
        tf, candle,
        slope_50=abs(ribbon_state.center_ema_slope),
        ribbon_width_rate=ribbon_state.ribbon_width_rate,
        rv=volume_analyzer.get_rv(tf)
    )

    # Decision matrix
    if st_state.regime == Regime.TREND:
        if st_state.st_direction == Direction.UP:
            if ribbon_state.state == RibbonState.HEALTHY:
                if ts_state.bucket == TrendBucket.STRONG:
                    return "BEST SETUP - Full size, aggressive"
                elif ts_state.bucket == TrendBucket.EMERGING:
                    return "GOOD SETUP - Moderate size"
                else:
                    return "WEAK SETUP - Wait for strength"
            else:
                return "RIBBON NOT HEALTHY - Skip"
    else:
        return "NOT IN TREND - Skip"
```

## Performance

- **Time Complexity:** O(1) per candle update
- **Space Complexity:** O(ema_center_period) per timeframe
- **Component Calculation:** All normalizations in O(1)
- **Weight Handling:** Dynamic renormalization if components missing
- **Smoothing:** Single EMA state, no full recalculation

## Technical Highlights

1. **ATR% Normalization** - Cross-instrument comparability (BTC vs ES vs NQ)
2. **Component Clipping** - All components normalized to [0, 1]
3. **Weight Renormalization** - Graceful handling of missing components
4. **EMA Smoothing** - Reduces noise without heavy lag
5. **Safety Caps** - External conditions override raw strength
6. **Incremental Updates** - O(1) per candle, no batch recomputation

## Files

- `trend_strength.py` - Core implementation (640 lines)
- `tests/test_trend_strength.py` - Comprehensive tests (468 lines)
- `trend_strength_integration.py` - Integration examples (285 lines)
- `TREND_STRENGTH_README.md` - This document

## Testing

Run all tests:
```bash
pytest tests/test_trend_strength.py -v
```

Result: **18/18 tests passing ✓**

## Integration Points

This module provides **strength measurement** for:

1. **Position Sizing** - Scale size based on strength bucket
2. **Risk Management** - Tighten stops in WEAK, wider in STRONG
3. **Entry Filtering** - Require EMERGING or STRONG for trend entries
4. **Exit Signals** - If strength drops to WEAK, consider exiting
5. **Trade Selection** - Prioritize highest strength opportunities

## Important Notes

### What Trend Strength IS

✓ Composite strength score (0-100)
✓ Momentum measurement from multiple sources
✓ Position sizing tool
✓ Risk management input
✓ Trend quality confirmation

### What Trend Strength IS NOT

✗ Entry signal (use with other systems)
✗ Direction indicator (use Supertrend)
✗ Regime classifier (use Supertrend)
✗ Stack health (use EMA Ribbon)
✗ Standalone trading system

### Best Practices

1. **Use with other modules**
   - Supertrend: Regime and direction
   - EMA Ribbon: Stack health
   - Trend Strength: Momentum quality

2. **Position sizing by bucket**
   - STRONG: Full size (100%)
   - EMERGING: Reduced (50-70%)
   - WEAK: Minimal (20-30%)

3. **Check individual components**
   - Look for divergences (high slope but low RV)
   - Use component insights for entry timing

4. **Respect safety caps**
   - If capped, external conditions are unfavorable
   - Don't fight the cap with oversized positions

5. **Multi-timeframe alignment**
   - HTF strength for overall bias
   - MTF strength for swing timing
   - LTF strength for entry precision

## Component Formulas

### EMA Slope Normalization

```
slope = (ema_current - ema_window_ago) / window
slope_magnitude = abs(slope)
atr_percent = atr / close

# Normalize
ema_slope_norm = clip(slope_magnitude / (slope_strong_factor × atr_percent), 0, 1)
```

**Interpretation:**
- 0.0 = No slope (flat)
- 1.0 = Slope ≥ 3× ATR% (very strong directional move)

### Ribbon Width Rate Normalization

```
# From EMA Ribbon module
ribbon_width_rate = (width_now - width_prev) / width_prev

# Normalize
ribbon_norm = clip((WR - WR_low) / (WR_high - WR_low), 0, 1)
```

**Interpretation:**
- WR < 0: Contracting (weakening)
- WR > 0: Expanding (strengthening)
- WR > 0.20: Strong expansion

### Relative Volume Normalization

```
RV = current_volume / avg_volume(rv_period)

# Normalize
rv_norm = clip((RV - rv_low) / (rv_high - rv_low), 0, 1)
```

**Interpretation:**
- RV < 0.8: Below average (weak)
- RV = 1.0: Average
- RV > 2.0: Very high participation

### OI Expansion Normalization

```
dOI = (oi_current - oi_window_ago) / oi_window_ago

# Normalize
oi_norm = clip(dOI / oi_expand_strong, 0, 1)
```

**Interpretation:**
- dOI < 0: OI decreasing (money leaving)
- dOI > 0: OI increasing (new money)
- dOI > 0.05: Strong expansion (5%+)

## Next Steps

1. **Integrate into main loop**
   - Hook into candle close events
   - Collect indicators from other modules
   - Update Trend Strength state

2. **Use for position sizing**
   - Scale position size by bucket
   - Adjust stops based on strength

3. **Combine with filters**
   - Supertrend regime (TREND/CHOP)
   - EMA Ribbon health
   - Trend Strength score
   - All three → highest conviction

4. **Monitor component divergences**
   - High slope but low RV → wait for confirmation
   - Strong ribbon but weak OI → proceed with caution

---

**Status:** ✓ Complete and tested
**Performance:** All tests passing, O(1) incremental updates
**Ready for:** Production integration as trend strength measurement
