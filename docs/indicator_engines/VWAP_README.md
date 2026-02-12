# VWAP Engine Module - Implementation Complete

## Overview

A real-time VWAP Engine supporting **session, weekly, and anchored VWAP** with volume-weighted standard deviation bands and interaction state tracking. Designed for continuous trading systems with incremental O(1) updates.

## Deliverables

### 1. Core Module: [vwap_engine.py](vwap_engine.py)
- `VWAPEngine` - Main engine class
- `VWAPConfig` - Configuration dataclass
- `VWAPState` - State for individual VWAP lines
- `VWAPMultiTFState` - Multi-timeframe state container
- Incremental VWAP calculation with volume-weighted variance
- Session/weekly boundary detection with timezone support
- Interaction state machine (ACCEPT/REJECT/RECLAIM/LOSS)

### 2. Tests: [tests/test_vwap_engine.py](tests/test_vwap_engine.py)
**All 31 tests passing ✓**
- VWAP formula correctness
- Incremental vs batch equivalence
- Session/weekly reset boundaries
- Anchored VWAP calculation
- Standard deviation bands (volume-weighted)
- ATR fallback bands
- Price position tracking
- Interaction state machine
- Multi-timeframe support
- Edge cases

### 3. Integration: [vwap_integration.py](vwap_integration.py)
- Complete streaming example
- Production integration template
- Compact dashboard output format
- Anchored VWAP creation on events

## Key Features

### VWAP Types

1. **Session VWAP** - Resets daily
   - UTC or local timezone support
   - Configurable session boundaries

2. **Weekly VWAP** - Resets weekly
   - ISO week (Monday start) by default
   - Timezone-aware

3. **Anchored VWAP** - User-defined anchor points
   - Create on market structure events (BOS, CHoCH, sweep, etc.)
   - Up to N anchors per timeframe
   - Auto-pruning by age or relevance

### VWAP Formula

```
Typical Price (TP) = (High + Low + Close) / 3
or
Price = Close  (configurable)

VWAP = Σ(TP × Volume) / Σ(Volume)
```

### Standard Deviation Bands

**Volume-weighted variance** (preferred):
```
Mean = VWAP
Variance = (Σ(TP² × Volume) / Σ(Volume)) - Mean²
Std = √Variance

Bands:
  Lower_k = VWAP - k×Std
  Upper_k = VWAP + k×Std
```

**ATR Fallback** (if insufficient data):
```
Band_k = k × ATR% × Close
Lower = VWAP - Band
Upper = VWAP + Band
```

### Interaction State Machine

States with hold confirmation (default 3 bars):

1. **RECLAIM** - Price crosses from BELOW to ABOVE and holds
   - Bullish signal
   - Entry opportunity after pullback

2. **LOSS** - Price crosses from ABOVE to BELOW and holds
   - Bearish signal
   - Exit or reversal signal

3. **ACCEPT** - Price maintains same side with hold
   - Trend continuation
   - Confirms direction

4. **REJECT** - Price touches VWAP then moves away
   - Rejection from level
   - Potential reversal

5. **NEUTRAL** - No clear interaction
   - Initial state or choppy price action

### Output Metrics (per VWAP line)

```python
VWAPState:
    vwap: float
    price_position: ABOVE | BELOW | AT
    interaction_state: ACCEPT | REJECT | RECLAIM | LOSS | NEUTRAL
    distance: {
        'pct': float,      # Percentage distance
        'sigma': float     # Sigma distance (if std available)
    }
    bands: VWAPBands {
        vwap: float
        std: float
        bands: {k: (lower, upper)}
        method: STD | ATR_FALLBACK | NONE
    }
    bar_count: int
    v_sum: float          # Total volume
    pv_sum: float         # Total price×volume
```

## Configuration

```python
VWAPConfig(
    # Price source
    price_source=PriceSource.TYPICAL,  # or CLOSE

    # Timeframes
    timeframes=["1m", "5m", "1h"],

    # Session/weekly reset
    session_reset="UTC_DAY",  # or "LOCAL_DAY"
    timezone="UTC",
    weekly_reset_day="MON",

    # Standard deviation bands
    enable_std_bands=True,
    std_band_multipliers=[1.0, 2.0],
    fallback_atr_band_multipliers=[0.5, 1.0],
    min_bars_for_std=30,

    # State machine
    hold_bars=3,
    reclaim_tolerance=0.0002,  # 0.02%
    touch_tolerance=0.0001,    # 0.01%

    # Anchored VWAP
    max_anchors_per_tf=3,
    anchor_expire_mode="AGE",  # or "RELEVANCE"
    anchor_max_age_bars={
        "1m": 1440,
        "5m": 1000,
        "1h": 300
    }
)
```

## Usage

### Basic Integration

```python
from vwap_engine import VWAPEngine, VWAPConfig, Candle, PriceSource

# 1. Initialize
config = VWAPConfig(
    price_source=PriceSource.TYPICAL,
    timeframes=["1m", "5m", "1h"],
    enable_std_bands=True
)
engine = VWAPEngine(config)

# 2. Warmup with historical data
historical_candles = {
    "1m": [...],  # List[Candle]
    "5m": [...],
    "1h": [...]
}
atr_percent_by_tf = {
    "1m": 0.008,  # 0.8%
    "5m": 0.012,  # 1.2%
    "1h": 0.020   # 2.0%
}
engine.warmup(historical_candles, atr_percent_by_tf)

# 3. Process new candles incrementally
def on_candle_close(tf: str, candle: Candle):
    result = engine.on_candle_close(
        tf=tf,
        candle=candle,
        atr_percent=get_atr_percent(tf)
    )

    # Access VWAP state
    session = result.session_by_tf[tf]
    weekly = result.weekly_by_tf[tf]

    # Use for trading decisions
    if session.interaction_state.value == "RECLAIM":
        # Bullish reclaim - look for longs
        pass
    elif session.interaction_state.value == "LOSS":
        # Bearish loss - look for shorts
        pass
```

### Adding Anchored VWAP

```python
# On market structure event (BOS, CHoCH, sweep, etc.)
def on_bos_detected(tf: str, timestamp: float, price: float):
    engine.add_anchor(
        tf=tf,
        anchor_time=timestamp,
        anchor_id=f"BOS_{price:.2f}",
        note=f"BOS@{price:.2f}",
        kind="BOS"
    )

# Access anchored VWAPs
result = engine.on_candle_close(tf, candle)
for anchor in result.anchors_by_tf[tf]:
    print(f"{anchor.anchor_note}: {anchor.vwap:.2f} | {anchor.interaction_state.value}")
```

### Compact Output

```python
from vwap_engine import format_vwap_output

result = engine.update(candles_by_tf, atr_percent_by_tf)
print(format_vwap_output(result, compact=True))
```

**Sample Output:**
```
VWAP CONTEXT
1m Session: ABOVE | RECLAIM | dist=+0.52% (+0.6σ) | bands=STD
1m Weekly : ABOVE | ACCEPT  | dist=+1.20% (+1.1σ)
1m Anchors:
  - BOS@110.7: ACCEPT | dist=+0.12% | vwap=110.57
  - SWEEP@109.5: REJECT | dist=-0.03% | vwap=110.73
5m Session: BELOW | LOSS    | dist=-0.85% (-0.8σ) | bands=STD
1h Session: ABOVE | ACCEPT  | dist=+2.10% (+1.8σ) | bands=STD
```

## Performance

- **O(1) per candle** - Incremental accumulation
- **Minimal memory** - Small state per VWAP line
- **Multi-timeframe** - Independent state per TF
- **Volume-weighted variance** - Streaming calculation without full history

## Integration Points

This module provides **VWAP context and state** (not entry signals). Use it to:

1. **Filter trades**
   - Only long when above session VWAP (ACCEPT/RECLAIM)
   - Only short when below session VWAP (LOSS)

2. **Multi-timeframe confirmation**
   - Require alignment across session/weekly VWAP
   - Use HTF VWAP for trend, LTF for entries

3. **Anchored VWAP for key levels**
   - Track from BOS/CHoCH points
   - Monitor reclaim/loss of anchored levels

4. **Mean reversion signals**
   - Watch for extended moves (>2σ)
   - Look for reversion to VWAP

5. **Rejection/acceptance patterns**
   - REJECT state = potential reversal
   - ACCEPT state = trend continuation

## Trading Logic Examples

### Session VWAP Context

```python
session = result.session_by_tf["1m"]

if session.interaction_state.value == "RECLAIM":
    # Price reclaimed VWAP from below - bullish
    # Look for long entries

elif session.interaction_state.value == "LOSS":
    # Price lost VWAP from above - bearish
    # Look for short entries or exit longs

elif session.interaction_state.value == "ACCEPT":
    if session.price_position.value == "ABOVE":
        # Accepting above - bullish continuation
        # Stay long, add on pullbacks to VWAP
    else:
        # Accepting below - bearish continuation
        # Stay short

elif session.interaction_state.value == "REJECT":
    # VWAP rejected price - potential reversal
    # Wait for confirmation
```

### Multi-Timeframe Alignment

```python
session_1m = result.session_by_tf["1m"]
session_1h = result.session_by_tf["1h"]

if (session_1m.price_position.value == "ABOVE" and
    session_1h.price_position.value == "ABOVE"):
    # Both timeframes above VWAP - strong bullish bias
    # Aggressive longs, tight stops

elif (session_1m.price_position.value == "ABOVE" and
      session_1h.price_position.value == "BELOW"):
    # Divergence - 1m pullback in 1h downtrend
    # Scalp only, or wait for 1h alignment
```

### Distance-Based Signals

```python
if session.distance.get('sigma') is not None:
    sigma = session.distance['sigma']

    if sigma > 2.0:
        # Extended >2σ above VWAP - mean reversion risk
        # Reduce position, tighten stops, look for shorts

    elif sigma < -2.0:
        # Extended >2σ below VWAP - mean reversion opportunity
        # Look for longs on stabilization

    elif abs(sigma) < 0.5:
        # Near VWAP - decision zone
        # Watch for RECLAIM/LOSS signals
```

### Anchored VWAP Usage

```python
for anchor in result.anchors_by_tf["1m"]:
    if "BOS" in anchor.anchor_note:
        # Break of structure anchor
        if anchor.interaction_state.value == "RECLAIM":
            # Price reclaimed BOS VWAP - continuation
            # Add to position

        elif anchor.interaction_state.value == "LOSS":
            # Price lost BOS VWAP - potential failure
            # Exit or reduce
```

## Files

- `vwap_engine.py` - Core implementation (754 lines)
- `tests/test_vwap_engine.py` - Comprehensive tests (693 lines)
- `vwap_integration.py` - Integration examples (397 lines)
- `VWAP_README.md` - This document

## Testing

Run all tests:
```bash
pytest tests/test_vwap_engine.py -v
```

Result: **31/31 tests passing ✓**

## Technical Highlights

1. **Incremental variance calculation**
   - Maintains Σ(P²×V) for O(1) variance updates
   - No need to store full price history

2. **Timezone-aware boundaries**
   - Uses pytz for robust timezone handling
   - Supports any timezone for session/weekly resets

3. **State machine with hold confirmation**
   - Prevents false signals from single-bar spikes
   - Tracks position history for transition detection

4. **Anchored VWAP management**
   - Automatic pruning by age or count
   - Each anchor maintains independent state

5. **Graceful degradation**
   - Falls back to ATR bands if std not ready
   - Handles zero volume candles safely

## Next Steps

1. **Integrate into streaming loop**
   - Hook into candle close events
   - Pass ATR% if available
   - Display compact output

2. **Add anchor creation logic**
   - Hook into BOS/CHoCH detection
   - Create anchors on key structure breaks
   - Monitor anchor states

3. **Use for trade filtering**
   - Require VWAP alignment for entries
   - Use RECLAIM/LOSS as confirmation
   - Watch distance for extension

4. **Multi-timeframe confirmation**
   - Check HTF VWAP for bias
   - Use LTF VWAP for timing
   - Monitor weekly VWAP for major trend

---

**Status:** ✓ Complete and tested
**Performance:** All tests passing, O(1) incremental updates
**Ready for:** Production integration
