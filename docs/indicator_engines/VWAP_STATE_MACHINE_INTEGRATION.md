# VWAP State Machine Integration Guide

## Minimal Integration Snippet

```python
from vwap_state_machine import VWAPStateMachine, VWAPStateConfig
from vwap_engine import VWAPEngine  # Your existing VWAP module

# Initialize VWAP State Machine
vwap_sm_config = VWAPStateConfig(
    window_crossings=20,
    touch_pct_by_tf={"1m": 0.0001, "5m": 0.00015, "1h": 0.0003},
    N_reclaim_trend=2,
    N_loss_trend=2,
    N_accept_trend=2,
    N_reject_trend=3,
    N_reclaim_range=3,
    N_loss_range=3,
    N_accept_range=3,
    N_reject_range=2
)

vwap_sm = VWAPStateMachine(vwap_sm_config)

# In your candle close handler
def on_candle_close(tf: str, candle: Candle):
    # 1. Update VWAP (your existing code)
    vwap_state = vwap_engine.on_candle_close(tf, candle, atr_percent=atrp)

    # 2. Update VWAP State Machine
    interaction = vwap_sm.on_update(
        tf=tf,
        timestamp=candle.timestamp,
        close=candle.close,
        vwap=vwap_state.vwap,
        std=vwap_state.bands.std if vwap_state.bands else None,
        atr_percent=atrp,  # From ATR module
        rv=rv,  # From volume analyzer (optional)
        delta_ok=delta_ok,  # From order flow (optional)
        oi_ok=oi_ok,  # From OI tracker (optional)
        regime=regime  # "TREND" or "RANGE" from regime filter (optional)
    )

    # 3. Use interaction state for trading decisions
    vwap_state.position = interaction.position
    vwap_state.interaction_state = interaction.state
    vwap_state.hold_count = interaction.hold_count
    vwap_state.distance = {
        "pct": interaction.dist_pct,
        "sigma": interaction.dist_sigma
    }
    vwap_state.crossings_20 = interaction.crossings_20

    # 4. Trading logic based on state
    if interaction.state == "RECLAIM":
        # Bullish - price reclaimed VWAP from below
        allow_long_entries()
    elif interaction.state == "LOSS":
        # Bearish - price lost VWAP from above
        allow_short_entries()
    elif interaction.state == "REJECT":
        # Price rejected VWAP after touch
        if interaction.position == "BELOW":
            allow_short_entries()  # Rejection downward
        else:
            allow_long_entries()   # Rejection upward
    elif interaction.state == "ACCEPT":
        # Price持续 holding one side
        if interaction.side == 1:
            allow_long_entries()  # Accepting above
        else:
            allow_short_entries()  # Accepting below
    elif interaction.state == "NEUTRAL":
        # No clear interaction
        wait_for_setup()

    # 5. Display (optional)
    print(format_vwap_interaction_output(interaction, compact=True))
```

## Output Format

**Compact:**
```
VWAP: ABOVE RECLAIM dist=+0.235% σ=1.52 hold=3 xings=2 touch_age=1 regime=TREND
```

**Detailed:**
```
VWAP INTERACTION STATE
  Position: ABOVE (side=+1)
  State: RECLAIM
  Hold Count: 3
  Distance: +0.235%
  Distance (σ): 1.52
  Crossings (20): 2
  Last Touch: 1 bars ago
  Debug:
    regime_used: TREND
    touch_pct: 0.0001
    reclaim_buffer: 0.0002
    conf_ok: True
```

## State Meanings

### Position
- **ABOVE**: Price above VWAP
- **BELOW**: Price below VWAP
- **AT**: Price touching VWAP (within tolerance)

### State (Priority Order)
1. **RECLAIM** (highest priority): Price crossed from below to above VWAP with confirmation hold
2. **LOSS**: Price crossed from above to below VWAP with confirmation hold
3. **REJECT**: Price touched VWAP recently, then moved away and held
4. **ACCEPT**: Price持续 holding one side of VWAP
5. **NEUTRAL**: No clear interaction pattern

## Regime-Aware Behavior

### TREND Regime
- Lower hold requirements (faster confirmation)
- RECLAIM/LOSS don't require volume/delta/OI confirmation
- N_reclaim = 2, N_loss = 2, N_accept = 2, N_reject = 3

### RANGE Regime
- Higher hold requirements (more confirmation needed)
- RECLAIM/LOSS require confirmation (RV >= 1.2 or delta_ok or oi_ok)
- N_reclaim = 3, N_loss = 3, N_accept = 3, N_reject = 2

## Chop Detection

High crossings (>= 4 in window) automatically:
- Infers RANGE regime if not explicitly provided
- Suppresses ACCEPT state unless strong confirmation
- Indicates choppy/noisy VWAP interaction

## Trading Examples

### RECLAIM Strategy
```python
if interaction.state == "RECLAIM":
    if interaction.hold_count >= 3:
        # Strong RECLAIM with confirmation
        enter_long(size=1.0)
    elif interaction.hold_count >= 2:
        # Emerging RECLAIM
        enter_long(size=0.7)
```

### LOSS Strategy
```python
if interaction.state == "LOSS":
    if interaction.debug['conf_ok']:
        # LOSS with volume/delta confirmation
        enter_short(size=1.0)
    else:
        # LOSS without confirmation (TREND only)
        enter_short(size=0.7)
```

### REJECT Filter
```python
# Use REJECT as filter - avoid entries against rejection
if interaction.state == "REJECT":
    if interaction.position == "BELOW":
        block_long_entries()  # Rejected downward
    else:
        block_short_entries()  # Rejected upward
```

### ACCEPT Confirmation
```python
# Use ACCEPT as trend confirmation
if interaction.state == "ACCEPT":
    if interaction.side == 1 and interaction.hold_count >= 5:
        # Strong acceptance above - add to longs
        scale_into_long_position()
```

### Distance-Based Filtering
```python
# Combine state with distance for entries
if interaction.state == "RECLAIM":
    if interaction.dist_sigma is not None:
        if interaction.dist_sigma > 0.5:
            # RECLAIM with enough distance from VWAP
            enter_long()
        else:
            # Too close to VWAP still
            wait_for_more_separation()
```

## Multi-Timeframe Alignment

```python
# Get interaction for multiple timeframes
interaction_1m = vwap_sm.on_update("1m", ts, close_1m, vwap_1m, ...)
interaction_5m = vwap_sm.on_update("5m", ts, close_5m, vwap_5m, ...)
interaction_1h = vwap_sm.on_update("1h", ts, close_1h, vwap_1h, ...)

# Require alignment for highest conviction
if interaction_1h.state == "RECLAIM":
    if interaction_5m.state in ["RECLAIM", "ACCEPT"]:
        if interaction_1m.state == "RECLAIM":
            # All timeframes aligned - BEST SETUP
            enter_long(size=1.0)
        elif interaction_1m.position == "ABOVE":
            # HTF/MTF RECLAIM, LTF above
            enter_long(size=0.7)
```

## Confirmation Sources

### RV (Relative Volume)
```python
rv = current_volume / sma(volume, 20)
# RV >= 1.2 counts as confirmation
```

### Delta OK (Order Flow)
```python
delta_ok = (delta > 0 and delta_pct > 0.6)  # For bullish
# Strong buy pressure confirms RECLAIM
```

### OI OK (Open Interest)
```python
oi_ok = (oi_change_rate > 0.01)  # Growing OI
# New money entering confirms directional moves
```

## Reset and Management

```python
# Reset timeframe state if needed
vwap_sm.reset("1m")

# Reset all timeframes
for tf in ["1m", "5m", "1h"]:
    vwap_sm.reset(tf)
```

## Performance Notes

- **O(1)** per update
- Minimal memory: deque of size `window_crossings` per timeframe
- No batch recalculation needed
- All state transitions are deterministic and instant

## Integration with Other Modules

### With Supertrend
```python
st_state = supertrend_engine.on_candle_close(tf, candle)
regime = "TREND" if st_state.regime == Regime.TREND else "RANGE"

interaction = vwap_sm.on_update(
    tf, ts, close, vwap,
    regime=regime  # Use Supertrend regime
)
```

### With Trend Strength
```python
ts_state = trend_strength_engine.on_candle_close(tf, candle, ...)

# Use VWAP interaction as confirmation
if interaction.state == "RECLAIM" and ts_state.strength_signed > 60:
    # Strong bullish momentum + VWAP RECLAIM
    enter_long_aggressive()
```

### With EMA Ribbon
```python
ribbon_state = ema_ribbon_engine.on_candle_close(tf, candle)

# Require healthy ribbon + VWAP RECLAIM
if (ribbon_state.state == "HEALTHY" and
    interaction.state == "RECLAIM"):
    # Best setup
    enter_long(size=1.0)
```

## Debug Information

Access debug fields for analysis:
```python
debug = interaction.debug

print(f"Regime used: {debug['regime_used']}")
print(f"Touch threshold: {debug['touch_pct']}")
print(f"Reclaim buffer: {debug['reclaim_buffer']}")
print(f"Confirmation OK: {debug['conf_ok']}")
print(f"Hold counters: reclaim={debug['reclaim_hold']} loss={debug['loss_hold']}")
```

## Key Points

1. **State Priority**: RECLAIM > LOSS > REJECT > ACCEPT > NEUTRAL (deterministic)
2. **Hysteresis**: All states require hold confirmation (anti-flip)
3. **Regime-Aware**: Automatically adjusts hold requirements based on regime
4. **Touch Sensitivity**: Multiple tolerance checks (%, sigma, ATR-based)
5. **Chop Detection**: Automatically detects choppy conditions via crossings
6. **Confirmation**: Optional volume/delta/OI confirmation for RECLAIM/LOSS in RANGE

## Testing

All 28 tests passing:
```bash
pytest tests/test_vwap_state_machine.py -v
```

Tests cover:
- Position classification (ABOVE/BELOW/AT)
- Crossings detection
- RECLAIM/LOSS state transitions
- REJECT state logic
- ACCEPT state logic
- Chop suppression
- Regime-aware behavior
- Stability and anti-flip
- State priority
- Hold count tracking
