# Choppiness Index - Quick Reference

## What is CHOP?

**Regime filter** that detects whether the market is **trending** or **range-bound** (choppy).

Range: 0-100
- **High CHOP (>61.8)** = Range-bound, choppy
- **Low CHOP (<38.2)** = Trending
- **Mid CHOP (38.2-61.8)** = Transition

## Formula

```
CHOP = 100 * log10(SUM(TR, n) / (MaxHigh(n) - MinLow(n))) / log10(n)
```

## Quick Start

```python
from choppiness import ChoppinessEngine, Candle

engine = ChoppinessEngine()
state = engine.on_candle_close("1m", candle)

if state.chop_state == "CHOP":
    # Range-bound: use mean reversion
elif state.chop_state == "TREND":
    # Trending: use trend following
```

## States

| State | CHOP | Strategy |
|-------|------|----------|
| CHOP | ≥61.8 | Mean reversion, VWAP fades |
| TRANSITION | 38.2-61.8 | Reduce size, wait |
| TREND | ≤38.2 | Trend continuation, breakouts |

## Usage Patterns

### Pattern 1: Regime Filter
```python
if state.chop_state == "CHOP":
    allow_breakouts = False
elif state.chop_state == "TREND":
    allow_breakouts = True
```

### Pattern 2: With ATR
```python
if (chop_state.chop_state == "CHOP" and
    chop_state.chop_slope < 0 and
    atr_state.vol_state == "EXPANSION"):
    # Breakout setup
```

### Pattern 3: Position Sizing
```python
trendiness = 100 - state.chop
multiplier = 0.5 + (trendiness / 100.0)
```

## Output

```
CHOPPINESS
1m: chop=64.2 state=CHOP slope=+1.8 sumTR=123.4 range=56.7
5m: chop=41.7 state=TRANSITION slope=-0.9
1h: chop=33.5 state=TREND slope=-0.4
```

## Integration

**Continuous:**
```python
self.chop_engine = ChoppinessEngine()
chop_state = self.chop_engine.on_candle_close(tf, candle)
```

**Batch:**
```python
states = chop_engine.warmup({timeframe: candles})
print_choppiness(states)
```

## Key Points

✅ O(1) updates per candle
✅ Multi-timeframe support
✅ Regime filter, NOT entry signal
✅ Combine with directional indicators
✅ 26/26 tests passing

## Remember

**CHOP tells you WHAT TYPE of market, not WHERE to enter.**

Use it to choose the right strategy:
- High CHOP → Mean reversion
- Low CHOP → Trend following
