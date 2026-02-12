# ATR Expansion Module - Volatility Timing Gate

## Overview

The ATR Expansion module is a **timing gate** for detecting when volatility is "waking up" and moves are starting. It is **NOT an entry signal** - it's a **volatility regime classifier** that tells you WHEN to look for entries, not WHERE.

### Key Concept

> "The best trades happen when volatility expands. This module detects that expansion in real-time."

## What It Does

Computes per-timeframe (O(1) incremental):
- **TR** (True Range) - current volatility
- **ATR** (Wilder smoothed Average True Range)
- **ATR%** (ATR as percentage of price)
- **ATR Expansion Ratio** (ATR / SMA(ATR)) - is volatility expanding?
- **TR Spike Ratio** (TR / SMA(TR)) - sudden shock?
- **ATR Expansion Slope** - is expansion accelerating or fading?
- **Vol State** - SQUEEZE / NORMAL / EXPANSION / EXTREME / FADE_RISK
- **Vol Score** (0-100) - timing confidence

## Installation

```bash
# Files created:
indicator/atr_expansion.py              # Core module
indicator/tests/test_atr_expansion.py   # Comprehensive tests
indicator/atr_expansion_integration.py  # Integration examples
```

## Quick Start

```python
from atr_expansion import ATRExpansionEngine, ATRExpansionConfig, Candle

# 1. Initialize
config = ATRExpansionConfig(
    timeframes=["1m", "5m", "1h"],
    atr_period=14,
    sma_period=20,
)
engine = ATRExpansionEngine(config)

# 2. On each candle close
candle = Candle(timestamp=1234567890, open=100, high=105, low=95, close=102, volume=1000)
state = engine.on_candle_close("1m", candle)

# 3. Use state for timing decisions
if state.vol_state == "EXPANSION":
    print("✅ TIMING: Volatility expanding - good for breakouts")
elif state.vol_state == "SQUEEZE":
    print("⏸️ TIMING: Low volatility - wait")
```

## Volatility States

| State | Meaning | ATR Expansion | Action |
|-------|---------|---------------|--------|
| **SQUEEZE** | Low volatility compression | < 0.80 | Wait for expansion |
| **NORMAL** | Average volatility | 0.80 - 1.20 | Standard risk management |
| **EXPANSION** | Volatility waking up | 1.20 - 1.60 | **Good timing for entries** |
| **EXTREME** | High volatility | > 1.60 | Move is ON, tight stops |
| **FADE_RISK** | Expansion fading | High but falling | Consider taking profits |

## Vol Score Interpretation

| Score | Interpretation |
|-------|----------------|
| 0-30 | Squeeze - poor timing |
| 31-60 | Normal - standard timing |
| 61-85 | Expansion - **good timing** |
| 86-100 | Extreme - **excellent timing** but higher risk |

## TR Spike Detection

- **TR Spike > 1.50** → Immediate shock detected
- Use for ultra-fast entries when combined with other signals
- Indicates sudden increase in volatility RIGHT NOW

## Formulas

### True Range
```python
TR = max(
    high - low,
    abs(high - prev_close),
    abs(low - prev_close)
)
```

### Wilder ATR Smoothing
```python
# Seed with SMA of first N TRs
ATR_seed = mean(TRs[:atr_period])

# Then update with Wilder smoothing
ATR_t = (ATR_{t-1} * (atr_period - 1) + TR_t) / atr_period
```

### ATR Expansion
```python
ATR_exp = ATR / SMA(ATR, period=20)
```

### TR Spike
```python
TR_spike = TR / SMA(TR, period=20)
```

## Configuration

```python
@dataclass
class ATRExpansionConfig:
    timeframes: List[str] = ["1m", "5m", "1h"]
    atr_period: int = 14              # Wilder ATR period
    sma_period: int = 20              # SMA period for expansion ratio
    use_external_atr_percent: bool = False

    # State thresholds
    squeeze_thr: float = 0.80         # Below = SQUEEZE
    expansion_thr: float = 1.20       # Above = EXPANSION
    extreme_thr: float = 1.60         # Above = EXTREME
    tr_spike_thr: float = 1.50        # TR spike threshold
    fade_slope_thr: float = -0.05     # Negative slope = FADE_RISK
```

## Usage Patterns

### Pattern 1: Simple Timing Gate

```python
state = engine.on_candle_close("5m", candle)

if state.vol_state in ["EXPANSION", "EXTREME"]:
    # Enable breakout attempts
    allow_entries = True
else:
    # Wait for expansion
    allow_entries = False
```

### Pattern 2: Multi-Timeframe Confirmation

```python
state_1m = engine.get_state("1m")
state_5m = engine.get_state("5m")

# Require alignment
if (state_1m.vol_state == "EXPANSION" and
    state_5m.vol_state == "EXPANSION"):
    print("✅ STRONG TIMING: Multi-TF expansion confirmed")
```

### Pattern 3: Shock Detection

```python
state = engine.on_candle_close("1m", candle)

if state.debug.get("shock_now"):
    print("⚡ TR SHOCK: Move starting NOW!")
    # Fast entry with tight stops
```

### Pattern 4: Vol Score as Multiplier

```python
vol_confidence = state.vol_score_0_100 / 100.0

# Scale position size
position_size = base_size * vol_confidence

# Or scale stop distance
stop_distance = base_stop * (1 + vol_confidence)
```

## Integration Examples

### Continuous Runner

```python
# In continuous/orchestrator.py

from atr_expansion import ATRExpansionEngine, ATRExpansionConfig

class ContinuousAnalyzer:
    def __init__(self, symbol, config):
        # ... existing init ...

        # Add ATR expansion engine
        self.atr_engine = ATRExpansionEngine(ATRExpansionConfig(
            timeframes=["15s", "1m", "5m"]
        ))

    def _on_candle_close(self, tf, ohlcv):
        # ... existing logic ...

        # Update ATR state
        candle = Candle(ohlcv['timestamp'], ohlcv['open'], ohlcv['high'],
                       ohlcv['low'], ohlcv['close'], ohlcv['volume'])
        atr_state = self.atr_engine.on_candle_close(tf, candle)

        # Use in state machine
        if atr_state.vol_state == "EXPANSION":
            # Enable breakout validation
            pass
```

### Batch Analysis (analyze.py)

```python
# In runner.py analyze_pair() after fetching klines

from atr_expansion import ATRExpansionEngine, Candle, print_atr_expansion

# Convert klines to Candles
candles = [Candle(k.timestamp, k.open, k.high, k.low, k.close, k.volume)
           for k in klines]

# Analyze
atr_engine = ATRExpansionEngine()
states = atr_engine.warmup({"1h": candles})

# Display
print_atr_expansion(states)

# Use for decisions
if states["1h"].vol_state == "EXPANSION":
    print("\n✅ TIMING: Good time for breakout attempts")
```

## Output Format

### Compact Print

```
ATR EXPANSION
1m: state=EXPANSION score=72 atrp=0.22% atr_exp=1.31 slope=+0.07 TR_spike=1.62 shock=YES
5m: state=SQUEEZE  score=18 atrp=0.15% atr_exp=0.74 slope=-0.02 TR_spike=0.88 shock=NO
1h: state=FADE_RISK score=63 atr_exp=1.25 slope=-0.08 TR_spike=1.12 shock=NO
```

### Interpretation

- **1m: EXPANSION + shock** → Move starting NOW, fast entry
- **5m: SQUEEZE** → Wait for 5m confirmation before large position
- **1h: FADE_RISK** → Higher TF losing momentum, consider scaling out

## Testing

All tests pass (27/27):

```bash
pytest tests/test_atr_expansion.py -v
```

Tests cover:
- ✅ True Range calculation correctness
- ✅ Wilder ATR smoothing formula
- ✅ Rolling SMA correctness
- ✅ Ratio computations
- ✅ State classification logic
- ✅ Vol score calculation
- ✅ Incremental vs batch consistency
- ✅ Multi-timeframe support
- ✅ Edge cases (empty, zero vol, etc.)

## Performance

- **Complexity**: O(1) per candle (constant time updates)
- **Memory**: O(sma_period) per timeframe (~20 values typically)
- **Latency**: < 1ms per update

## Common Mistakes to Avoid

### ❌ DON'T: Use as entry signal

```python
# WRONG - ATR expansion alone is NOT an entry
if state.vol_state == "EXPANSION":
    enter_trade()  # ❌ Missing price action, structure, etc.
```

### ✅ DO: Use as timing gate

```python
# CORRECT - Combine with other signals
if (state.vol_state == "EXPANSION" and
    breakout_detected and
    volume_confirms):
    enter_trade()  # ✅ Complete setup
```

### ❌ DON'T: Ignore FADE_RISK

```python
# WRONG - Taking new positions during fade
if state.vol_state in ["EXPANSION", "EXTREME", "FADE_RISK"]:
    enter_trade()  # ❌ FADE_RISK is a warning, not confirmation
```

### ✅ DO: Respect FADE_RISK

```python
# CORRECT - Different actions for different states
if state.vol_state == "EXPANSION":
    # Take new positions
elif state.vol_state == "FADE_RISK":
    # Tighten stops, consider exits
```

## Advanced Usage

### Progressive Expansion Detection

```python
# Track expansion across timeframes
states = {
    "15m": engine.get_state("15m"),
    "5m": engine.get_state("5m"),
    "1m": engine.get_state("1m"),
}

# Ideal: Cascade from higher to lower TF
if (states["15m"].vol_state == "EXPANSION" and
    states["5m"].vol_state == "EXPANSION" and
    states["1m"].vol_state == "EXTREME"):
    print("🚀 PROGRESSIVE EXPANSION: 15m→5m→1m cascade")
    # Highest conviction timing
```

### Volatility Regime Tracking

```python
# Track state changes over time
prev_state = None

for candle in candles:
    state = engine.on_candle_close("5m", candle)

    if prev_state and prev_state.vol_state != state.vol_state:
        print(f"STATE CHANGE: {prev_state.vol_state} → {state.vol_state}")

        if state.vol_state == "EXPANSION":
            print("⚡ VOLATILITY WAKING UP")

    prev_state = state
```

## References

- **True Range**: Wilder, J. Welles. "New Concepts in Technical Trading Systems" (1978)
- **ATR Smoothing**: Wilder's exponential smoothing formula
- **Expansion Ratio**: Custom metric for volatility regime detection

## Support

For issues or questions:
1. Check test cases for usage examples
2. See `atr_expansion_integration.py` for integration patterns
3. Review formulas in module docstrings

## License

Same as parent project.

---

**Remember**: ATR Expansion is a **TIMING** tool, not an **ENTRY** signal. Use it to know WHEN to look for trades, not WHERE to enter.
