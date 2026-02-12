# Market Structure Detection System

## Overview

**The Foundation of Trend Analysis**

Market structure defines what trades are allowed. Indicators validate structure.

> **RULE**: If structure ≠ clear → NO indicator can save the trade.

## Architecture

```
PRICE DATA
    ↓
SWING DETECTION (HH/HL/LH/LL)
    ↓
STRUCTURE CLASSIFICATION
├─ Trend (BOS/CHoCH)
├─ Range (Compression/Distribution/Accumulation)
└─ Acceptance/Rejection
    ↓
FVG DETECTION
    ↓
TIME MOMENTUM
    ↓
MULTI-TF ALIGNMENT
    ↓
FINAL STRUCTURE STATE
```

## Components

### 1. Swing Highs & Lows (HH/HL/LH/LL)

Foundation of structure analysis.

**Swing High**: `high[i] > high[i-n:i]` and `high[i] > high[i+1:i+n+1]`
**Swing Low**: `low[i] < low[i-n:i]` and `low[i] < low[i+1:i+n+1]`

**Trend Patterns**:
- **Uptrend**: HH (Higher Highs) + HL (Higher Lows)
- **Downtrend**: LH (Lower Highs) + LL (Lower Lows)
- **Range**: No clear HH/HL or LH/LL pattern

```python
from indicator import MarketStructureDetector

detector = MarketStructureDetector(swing_lookback=5)
swing_highs, swing_lows = detector.detect_swings(highs, lows, timestamps)
trend = detector.classify_trend(swing_highs, swing_lows)
```

### 2. Break of Structure (BOS) - Continuation

**BOS** signals trend continuation.

- **Uptrend BOS**: Price breaks above last swing high
- **Downtrend BOS**: Price breaks below last swing low

**What it means**: Trend is continuing with strength.

### 3. Change of Character (CHoCH) - Reversal Warning

**CHoCH** signals trend weakening or reversal.

**CHoCH Definition**:
- **In Uptrend**: Break BELOW last swing low (HL)
- **In Downtrend**: Break ABOVE last swing high (LH)

**Why critical**:
- BOS = Continuation confirmed
- CHoCH = Trend ending, prepare for reversal
- Prevents late trend entries

```python
# Detect BOS and CHoCH
structure_break = detector.detect_bos_and_choch(
    current_idx=len(highs) - 1,
    current_high=highs[-1],
    current_low=lows[-1],
    swing_highs=swing_highs,
    swing_lows=swing_lows,
    trend=trend
)

if structure_break.event_type == StructureEvent.BOS:
    print("Trend continuation!")
elif structure_break.event_type == StructureEvent.CHOCH:
    print("Warning: Trend may be reversing!")
```

### 4. Structural Acceptance vs Rejection

**Critical**: Break ≠ Acceptance

**Acceptance requires**:
1. Close beyond level
2. Volume confirmation (RV > threshold)
3. Price holds for N bars

**Programmatic Rule**:
```
After BOS:
if close > level
and RV > 1.2
and price holds N bars
→ ACCEPTED
else → REJECTED
```

This filters fake BOS and prevents false entries.

```python
accepted = detector.check_acceptance(
    structure_break=bos_event,
    closes=closes,
    volumes=volumes,
    current_idx=len(closes) - 1
)

if accepted:
    print("Structure break ACCEPTED - real move!")
else:
    print("Structure break REJECTED - likely fake!")
```

### 5. Range Structure Classification

Markets range more than they trend.

**What to detect**:
- Multiple rejections at same levels
- Decreasing range expansion
- Failed BOS attempts

**Range Types**:

| Type | Meaning | Action |
|------|---------|--------|
| **Compression** | Narrowing range, breakout imminent | Prepare for breakout |
| **Distribution** | Top formation | Bearish bias |
| **Accumulation** | Bottom formation | Bullish bias |
| **Neutral** | No clear bias | Wait |

**Rule**: If structure = RANGE → Disable trend trades, enable mean reversion

```python
in_range, range_type, high, low, tightness = detector.detect_range_structure(
    highs, lows, closes, volumes
)

if in_range:
    if range_type == RangeType.COMPRESSION:
        print(f"Compression! Tightness: {tightness*100:.0f}%")
        print("Breakout imminent - prepare for explosive move")
```

### 6. Fair Value Gaps (FVG) / Imbalance Zones

**What FVG really is**:
- Rapid price displacement
- Poorly traded area
- Price often revisits

**Bullish FVG**: `Low[i] > High[i-2]` (gap up)
**Bearish FVG**: `High[i] < Low[i-2]` (gap down)

**Why it matters**:
- Trend pullback targets
- Re-entry zones
- Expected reaction areas

**Rule**: You don't trade FVG — you expect reactions there.

```python
fvg = detector.detect_fvg(idx, highs, lows, closes)

if fvg:
    print(f"{'Bullish' if fvg.is_bullish else 'Bearish'} FVG detected")
    print(f"Gap: ${fvg.price_bottom:.2f} - ${fvg.price_top:.2f}")
    print(f"Size: {fvg.gap_size_pct:.2f}%")

    # Use as pullback target
    if fvg.is_bullish and trend == TrendDirection.UPTREND:
        print(f"Pullback target for long entry: ${fvg.price_bottom:.2f}")
```

### 7. Time-to-Followthrough (Structural Momentum)

**Time as structure**: If price fails to continue within expected time → structure weak.

**Metrics**:
- **FAST**: Quick followthrough (<100s)
- **SLOW**: Delayed followthrough (100-300s)
- **STALLED**: No followthrough (>300s) → likely trap or range

```python
momentum = detector.calculate_structural_momentum(
    structure_break=bos_event,
    current_price=closes[-1],
    time_elapsed=time_since_break
)

if momentum == StructuralMomentum.FAST:
    print("Strong structural momentum - aggressive entry OK")
elif momentum == StructuralMomentum.STALLED:
    print("Stalled momentum - reduce size or exit")
```

### 8. Multi-Timeframe Structure Alignment

**Rule**: LTF structure must align with HTF structure OR be explicit counter-trend play.

| HTF Trend | LTF Trend | Action | Position Size |
|-----------|-----------|--------|---------------|
| Uptrend | Uptrend | ✓ Full long bias | 100% |
| Uptrend | Range | ✓ Reduced long | 50% |
| Uptrend | Downtrend | ⚠ Scalp only | 25% |
| Range | Any | Mean reversion | 50% |
| Downtrend | Downtrend | ✓ Full short bias | 100% |

```python
# Analyze HTF first (context)
htf_state = detector.analyze(
    highs=htf_highs,
    lows=htf_lows,
    closes=htf_closes,
    volumes=htf_volumes
)

# Analyze LTF with HTF context
ltf_state = detector.analyze(
    highs=ltf_highs,
    lows=ltf_lows,
    closes=ltf_closes,
    volumes=ltf_volumes,
    htf_trend=htf_state.trend_direction  # Pass HTF context
)

# Check alignment
if ltf_state.tf_alignment == TimeframeAlignment.FULL_ALIGN:
    print("Full MTF alignment - high confidence trades")
elif ltf_state.tf_alignment == TimeframeAlignment.COUNTER_TREND:
    print("Counter-trend - scalp only or wait")
```

## Complete Structure Analysis

```python
from indicator import MarketStructureDetector, TrendDirection
from indicator import get_allowed_trade_direction, structure_veto_signal

# Initialize detector
detector = MarketStructureDetector(
    swing_lookback=5,
    bos_acceptance_bars=3,
    bos_acceptance_volume=1.2,
    followthrough_time_max=300,
    range_threshold_pct=2.0,
)

# Analyze structure
state = detector.analyze(
    highs=price_highs,
    lows=price_lows,
    closes=price_closes,
    volumes=volumes,
    timestamps=timestamps,
    htf_trend=htf_trend  # Optional: HTF context
)

# Check what trades are allowed
allowed = get_allowed_trade_direction(state)

if allowed == "long":
    print("LONG trades allowed - uptrend confirmed")
elif allowed == "short":
    print("SHORT trades allowed - downtrend confirmed")
elif allowed == "both":
    print("RANGE - mean reversion both directions")
else:
    print("NO TRADE - structure unclear")

# Test if structure vetos a signal
signal_direction = "long"
signal_confidence = 75

vetoed, reason = structure_veto_signal(state, signal_direction, signal_confidence)

if vetoed:
    print(f"VETOED: {reason}")
    # DO NOT take trade regardless of indicators
else:
    print("Structure allows trade - check indicators")
```

## Hard Veto Rules

Structure can veto ANY trade signal. These are NON-NEGOTIABLE:

1. **No Trade if Structure Unclear**
   - Confidence < 40% → NO TRADE

2. **Direction Mismatch**
   - Uptrend structure + SHORT signal → VETO
   - Downtrend structure + LONG signal → VETO

3. **Counter-Trend Requires High Confidence**
   - MTF counter-trend + confidence < 75% → VETO

4. **Recent CHoCH Warning**
   - CHoCH more recent than BOS + confidence < 70% → VETO

5. **Rejected Structure Break**
   - Last BOS/CHoCH rejected → VETO until new structure

```python
# Example: Structure vetos a long signal
vetoed, reason = structure_veto_signal(
    structure=state,
    signal_direction="long",
    signal_confidence=65
)

if vetoed:
    print(f"TRADE BLOCKED: {reason}")
    # Indicators say buy, but structure says NO
    # Structure WINS - do not enter
```

## Integration with Continuous Analysis

```python
from indicator.continuous import MarketStructureAdapter

# Create adapter
structure_adapter = MarketStructureAdapter(
    ltf_window_seconds=180,   # 3 minutes
    ltf_candle_seconds=15,    # 15s candles
    htf_window_seconds=3600,  # 1 hour
    htf_candle_seconds=60,    # 1m candles
)

# Compute structure from rolling windows
signal = structure_adapter.compute(
    ltf_window=ltf_trade_window,
    htf_window=htf_trade_window
)

# Check if structure vetos a trade
vetoed, reason = structure_adapter.veto_trade_signal(
    signal_direction="long",
    signal_confidence=75
)

# Get structure score for unified scoring
structure_score = signal.score  # -1 to +1
```

## Display Functions

```python
from indicator.display import (
    print_structure_deep_dive,
    print_structure_summary,
    print_structure_allowed_trades,
    print_structure_signal,
    get_structure_status_line,
)

# Deep dive analysis
print_structure_deep_dive(state)

# Compact summary
print_structure_summary(state)

# Trade guidance
print_structure_allowed_trades(state)

# Live signal (continuous runner)
print_structure_signal(signal)

# Status line for continuous display
status = get_structure_status_line(signal)
print(status)  # "Struct: UP ↑ +0.7"
```

## Example Usage

```bash
# Run market structure analysis
python example_market_structure.py BTCUSDT

# Example output:
# ════════════════════════════════════════════════════════════════════════════════
# MARKET STRUCTURE ANALYSIS - BTCUSDT
# ════════════════════════════════════════════════════════════════════════════════
#
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  MARKET STRUCTURE - THE FOUNDATION                                           │
# └──────────────────────────────────────────────────────────────────────────────┘
#
# Structure Summary
#   ────────────────────────────────────────────────────────────────
#   Trend: LTF: UPTREND  │  HTF: UPTREND  │  Alignment: ✓✓
#   Last Event: BOS ↑ $95,000.00 → ACCEPTED
#   Momentum: ⚡ FAST
#   Confidence: 85% ████████████████████
#
#   Recent Swings:
#     Highs: HH ($94,500.00 → $95,200.00)
#     Lows:  HL ($93,800.00 → $94,100.00)
#
#   Structure Events:
#     BOS    ↑ $95,000.00 → $95,200.00 │ ✓ ACCEPTED (RV: 1.45x) │ Follow: 45s
#
#   Active Fair Value Gaps (FVG):
#     1. Bullish FVG: $94,200.00 - $94,350.00 (0.16%)
#
# ────────────────────────────────────────────────────────────────
# Structure Trade Guidance
#   ────────────────────────────────────────────────────────────────
#   ✓ LONG ALLOWED
#     Uptrend structure confirmed
#     Pullback target: $94,200.00 - $94,350.00
```

## Best Practices

### 1. Always Check Structure First
```python
# WRONG: Check indicators first
if volume_signal == "bullish":
    enter_long()

# RIGHT: Check structure, then indicators
allowed = get_allowed_trade_direction(structure)
if allowed == "long" and volume_signal == "bullish":
    enter_long()
```

### 2. Respect Structure Vetos
```python
# Structure can override ANY indicator
vetoed, reason = structure_veto_signal(structure, "long", 80)

if vetoed:
    # DO NOT TRADE - even if all indicators are perfect
    log(f"Trade blocked: {reason}")
    return
```

### 3. Use Multi-Timeframe Alignment
```python
# Analyze HTF first for context
htf_state = detector.analyze(htf_data, ...)

# Then LTF with HTF context
ltf_state = detector.analyze(ltf_data, ..., htf_trend=htf_state.trend_direction)

# Check alignment
if ltf_state.tf_alignment == TimeframeAlignment.FULL_ALIGN:
    position_size = 1.0  # Full size
elif ltf_state.tf_alignment == TimeframeAlignment.PARTIAL_ALIGN:
    position_size = 0.5  # Half size
else:
    position_size = 0.25  # Scalp only
```

### 4. Wait for Acceptance
```python
if bos_event and bos_event.accepted is None:
    # Too early - wait for acceptance confirmation
    return

if bos_event and bos_event.accepted is False:
    # Rejected - avoid trades until new structure
    return

if bos_event and bos_event.accepted is True:
    # Confirmed - OK to trade in BOS direction
    enter_trade()
```

### 5. Use FVGs for Entry Timing
```python
# Don't chase - wait for pullback to FVG
if trend == TrendDirection.UPTREND:
    bullish_fvgs = [f for f in state.active_fvgs if f.is_bullish]
    if bullish_fvgs and near_fvg(current_price, bullish_fvgs[-1]):
        # Price at pullback target - good entry
        enter_long()
```

## Summary

Market structure is the FOUNDATION. The complete system provides:

✅ **Directional structure** (HH/HL/LH/LL, BOS)
✅ **Reversal warnings** (CHoCH)
✅ **Range detection** (Compression/Distribution/Accumulation)
✅ **Acceptance filtering** (Real breaks vs fake breaks)
✅ **Pullback targets** (FVG zones)
✅ **Momentum tracking** (Time-to-followthrough)
✅ **Multi-TF alignment** (HTF + LTF context)
✅ **Hard veto system** (Structure overrides indicators)

**REMEMBER**:
- Structure defines what trades are allowed
- Indicators validate structure
- If structure ≠ clear → NO indicator can save the trade

This is institutional-grade structure analysis.
