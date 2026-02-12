## Market Structure Implementations - Comparison

You now have **TWO** market structure implementations, each serving different purposes:

---

## 1. `market_structure.py` - Foundation/Simple Version

**Purpose**: Educational foundation, easier to understand and integrate

**Best for**:
- Learning market structure concepts
- Quick integration into existing systems
- When you want maximum flexibility
- Continuous streaming analysis (has adapter)

**Features**:
- ✅ Swing detection (HH/HL/LH/LL)
- ✅ BOS and CHoCH detection
- ✅ Range classification (Compression/Distribution/Accumulation)
- ✅ Acceptance vs Rejection
- ✅ Fair Value Gaps (FVG)
- ✅ Time-to-followthrough momentum
- ✅ Multi-timeframe alignment
- ✅ **Continuous adapter** for streaming
- ✅ Hard veto system
- ✅ Beautiful display functions

**Usage**:
```python
from indicator import MarketStructureDetector

detector = MarketStructureDetector()
state = detector.analyze(highs, lows, closes, volumes, timestamps)

allowed = get_allowed_trade_direction(state)
vetoed, reason = structure_veto_signal(state, "long", 75)
```

**Integration**:
```python
from indicator.continuous import MarketStructureAdapter

adapter = MarketStructureAdapter()
signal = adapter.compute(ltf_window, htf_window)
```

---

## 2. `institutional_structure.py` - Professional/Complete Engine

**Purpose**: Production-grade institutional implementation

**Best for**:
- Professional trading systems
- Backtesting and research
- Maximum precision and control
- When you need deterministic, unit-tested behavior

**Features**:
- ✅ **Pivot algorithm** with configurable L/R windows (no repainting)
- ✅ Swing strength calculation (distance/ATR based)
- ✅ **Liquidity sweep detection** with confirmation
- ✅ **Equal highs/lows** liquidity level detection
- ✅ Acceptance/Rejection with pending break tracking
- ✅ Range classification with ATR contraction
- ✅ FVG detection (optional, configurable)
- ✅ Time-to-followthrough with FAST/SLOW/STALLED
- ✅ **Multi-timeframe engine** with recommended trading mode
- ✅ **Complete dataclasses** for all outputs
- ✅ Helper functions (compute_atr, compute_rv, sma)
- ✅ **100% deterministic** - unit testable
- ✅ Stateful engine - handles multiple updates

**Usage**:
```python
from indicator import MarketStructureEngine, StructureConfig, Candle

# Configure
config = StructureConfig(
    pivot_left=3,
    pivot_right=3,
    bos_buffer_pct=0.05,
    enable_fvg=True,
)

# Create engine
engine = MarketStructureEngine(config)

# Convert your data to Candles
candles = [
    Candle(
        timestamp=int(timestamp),
        open=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=volume,
    )
    for ... in your_data
]

# Update engine (supports multiple timeframes)
states = engine.update({
    "LTF": ltf_candles,
    "HTF": htf_candles,
})

# Get results
ltf_state = states["LTF"]
print(f"Structure: {ltf_state.structure.value}")  # UP/DOWN/RANGE
print(f"Strength: {ltf_state.strength_0_100}%")
print(f"Momentum: {ltf_state.momentum.value}")

# Multi-TF alignment
alignment = engine.get_mtf_alignment("HTF", "LTF")
print(f"Mode: {alignment.recommended_mode.value}")  # TREND_MODE/RANGE_MODE/SCALP_ONLY
```

**Output Structure**:
```python
MarketStructureState(
    structure=StructureState.UP,           # UP/DOWN/RANGE/UNKNOWN
    strength_0_100=75.0,                   # Confidence
    regime="TREND",                        # TREND/RANGE/MIXED
    last_swing_high=SwingPoint(...),
    last_swing_low=SwingPoint(...),
    recent_swings=[...],
    last_bos=StructureEvent(...),         # Break of Structure
    last_choch=StructureEvent(...),       # Change of Character
    recent_events=[...],                   # All events (BOS/CHOCH/SWEEP/ACCEPT/REJECT)
    active_range=Zone(...),                # If in range
    active_fvgs=[Zone(...)],               # Fair Value Gaps
    momentum=StructuralMomentum.FAST,      # FAST/SLOW/STALLED
    structure_label="HH+HL",               # Pattern label
)
```

---

## Comparison Table

| Feature | market_structure.py | institutional_structure.py |
|---------|---------------------|---------------------------|
| **Swing Detection** | Simple lookback | Pivot algorithm (L/R windows) |
| **No Repainting** | ⚠️ Partial | ✅ Yes (configurable R lookahead) |
| **Swing Strength** | Basic | ✅ Distance/ATR based (0-1) |
| **BOS/CHoCH** | ✅ Yes | ✅ Yes (adaptive buffers) |
| **Liquidity Sweeps** | ❌ No | ✅ Yes (with confirmation) |
| **Equal Highs/Lows** | ❌ No | ✅ Yes |
| **Acceptance/Rejection** | ✅ Yes | ✅ Yes (pending break tracking) |
| **Range Classification** | ✅ Yes | ✅ Yes (with ATR contraction) |
| **FVG Detection** | ✅ Yes | ✅ Yes (optional) |
| **Multi-TF Alignment** | ✅ Yes | ✅ Yes (with trading mode) |
| **Streaming Adapter** | ✅ Yes | ❌ No (batch only) |
| **Stateful Engine** | ❌ No | ✅ Yes |
| **Unit Tests** | ❌ No | ✅ Yes (comprehensive) |
| **Helper Functions** | ❌ No | ✅ Yes (ATR, RV, SMA) |
| **Display Functions** | ✅ Yes | ❌ No (can add) |
| **Veto System** | ✅ Yes | ⚠️ Can implement |
| **Configurability** | Medium | ✅ High |

---

## When to Use Which?

### Use `market_structure.py` when:
- ✅ Integrating with **continuous runner** (streaming)
- ✅ Want **veto system** built-in
- ✅ Need **display functions** for terminal output
- ✅ Prefer simpler, easier-to-understand code
- ✅ Want rapid prototyping

### Use `institutional_structure.py` when:
- ✅ Building **production trading system**
- ✅ Need **deterministic, testable** behavior
- ✅ Want **no repainting** guarantees
- ✅ Need **liquidity sweep detection**
- ✅ Doing **backtesting** or research
- ✅ Want **maximum configurability**
- ✅ Need **stateful engine** for continuous updates
- ✅ Want **recommended trading modes** (TREND/RANGE/SCALP)

---

## Recommended Approach

**For your continuous runner**:
Use `market_structure.py` with `MarketStructureAdapter` - it's already integrated:

```python
from indicator.continuous import MarketStructureAdapter

structure_adapter = MarketStructureAdapter(
    ltf_window_seconds=180,
    htf_window_seconds=3600,
)

signal = structure_adapter.compute(ltf_window, htf_window)

# Check veto
vetoed, reason = structure_adapter.veto_trade_signal("long", 75)
```

**For backtesting and research**:
Use `institutional_structure.py` - it's deterministic and fully tested:

```python
from indicator import MarketStructureEngine, StructureConfig

engine = MarketStructureEngine(StructureConfig(...))

# Process historical data
for chunk in historical_candles:
    states = engine.update({"LTF": chunk})
    # Analyze states["LTF"]
```

**Best of both worlds**:
Use `institutional_structure.py` as the **engine** and create a simple adapter for streaming if needed later.

---

## Example Outputs

### institutional_structure.py Output:
```
═══════════════════════════════════════════════════════════════════
INSTITUTIONAL MARKET STRUCTURE - BTCUSDT
═══════════════════════════════════════════════════════════════════

LTF (5m) STRUCTURE
═══════════════════════════════════════════════════════════════════
  Structure: UP (HH+HL)
  Regime: TREND
  Strength: 78%
  Momentum: FAST

  Last BOS: BULL @ $95,123.50

  Active FVGs:
    1. BULL FVG: $94,850.00 - $95,012.00

  Recent Swings:
    High: $95,234.00 (strength: 0.78)
    Low:  $94,123.00 (strength: 0.65)

  Recent Events:
    Sweeps: 1
      BULL sweep @ $94,100.00 ✓

HTF (1h) STRUCTURE
═══════════════════════════════════════════════════════════════════
  Structure: UP (HH+HL)
  Regime: TREND
  Strength: 82%
  Momentum: SLOW

  Last BOS: BULL @ $94,500.00

MULTI-TIMEFRAME ALIGNMENT
═══════════════════════════════════════════════════════════════════
  Alignment: ALIGNED
  Recommended Mode: TREND_MODE

  HTF: UP
  LTF: UP

TRADING GUIDANCE
═══════════════════════════════════════════════════════════════════
  ✓ LONG BIAS
    Both timeframes in uptrend
    Pullback target: $94,850.00 - $95,012.00
```

---

## Migration Path

If you want to migrate from `market_structure.py` to `institutional_structure.py`:

1. **Replace imports**:
```python
# Old
from indicator import MarketStructureDetector

# New
from indicator import MarketStructureEngine, StructureConfig, Candle
```

2. **Convert data to Candles**:
```python
candles = [
    Candle(timestamp, open, high, low, close, volume)
    for ... in your_data
]
```

3. **Update engine instead of calling analyze**:
```python
# Old
state = detector.analyze(highs, lows, closes, ...)

# New
states = engine.update({"LTF": candles})
state = states["LTF"]
```

4. **Access results from dataclass**:
```python
# Old
state.trend_direction

# New
state.structure  # UP/DOWN/RANGE instead of uptrend/downtrend/range
```

---

## Summary

You have **two excellent implementations**:

1. **`market_structure.py`** - Simple, integrated, ready for continuous streaming
2. **`institutional_structure.py`** - Professional, deterministic, battle-tested

Choose based on your use case. Both implement the same core concepts - they just differ in complexity and integration approach.

**Most users should start with `market_structure.py`** for its simplicity and built-in continuous adapter. **Advanced users** doing backtesting or requiring maximum precision should use **`institutional_structure.py`**.
