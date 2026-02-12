# ATR Expansion - Quick Reference Card

## 🎯 Purpose
**Volatility Timing Gate** - Detects when volatility is "waking up" and moves are starting.
**NOT an entry signal** - tells you WHEN to look for trades, not WHERE.

---

## 📊 States

| State | ATR Exp | Action | Score Range |
|-------|---------|--------|-------------|
| **SQUEEZE** | < 0.80 | ⏸️ Wait | 0-30 |
| **NORMAL** | 0.80-1.20 | 👀 Monitor | 31-60 |
| **EXPANSION** | 1.20-1.60 | ✅ **Trade** | 61-85 |
| **EXTREME** | > 1.60 | 🔥 **Active** | 86-100 |
| **FADE_RISK** | High but ↓ | ⚠️ Exit | Varies |

---

## ⚡ Quick Start

```python
from atr_expansion import ATRExpansionEngine, Candle

# Initialize
engine = ATRExpansionEngine()

# On candle close
candle = Candle(ts, open, high, low, close, vol)
state = engine.on_candle_close("1m", candle)

# Check state
if state.vol_state == "EXPANSION":
    # Good timing for entries
    pass
```

---

## 🔑 Key Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **TR** | max(H-L, \|H-PC\|, \|L-PC\|) | Current volatility |
| **ATR** | Wilder smooth(TR) | Average volatility |
| **ATR_exp** | ATR / SMA(ATR) | Expansion ratio |
| **TR_spike** | TR / SMA(TR) | Shock detection |
| **Slope** | Δ(ATR_exp) | Acceleration |

---

## 💡 Usage Patterns

### 1️⃣ Simple Gate
```python
if state.vol_state in ["EXPANSION", "EXTREME"]:
    enable_entries = True
```

### 2️⃣ Shock Entry
```python
if state.debug.get("shock_now"):
    fast_entry_with_tight_stops()
```

### 3️⃣ Multi-TF
```python
if (state_1m.vol_state == "EXPANSION" and
    state_5m.vol_state == "EXPANSION"):
    high_conviction_entry()
```

### 4️⃣ Vol Multiplier
```python
confidence = state.vol_score_0_100 / 100
position_size = base * confidence
```

---

## ⚙️ Config

```python
ATRExpansionConfig(
    timeframes=["1m", "5m"],
    atr_period=14,        # ATR period
    sma_period=20,        # Expansion SMA
    squeeze_thr=0.80,     # < SQUEEZE
    expansion_thr=1.20,   # > EXPANSION
    extreme_thr=1.60,     # > EXTREME
    tr_spike_thr=1.50,    # Shock
    fade_slope_thr=-0.05, # FADE_RISK
)
```

---

## 📈 Interpretation

### Score Zones
- **0-30**: ⏸️ Poor timing (squeeze)
- **31-60**: 👀 Standard timing
- **61-85**: ✅ **Good timing** (expansion)
- **86-100**: 🔥 **Best timing** (extreme)

### State Flow
```
SQUEEZE → EXPANSION → EXTREME → FADE_RISK → NORMAL → SQUEEZE
   ⏸️        ✅           🔥          ⚠️        👀        ⏸️
```

---

## 🚫 Common Mistakes

### ❌ DON'T
```python
# Use alone as entry
if state.vol_state == "EXPANSION":
    enter()  # Missing price action!
```

### ✅ DO
```python
# Combine with signals
if (state.vol_state == "EXPANSION" and
    breakout and volume_confirms):
    enter()  # Complete setup
```

---

## 📊 Output Format

```
ATR EXPANSION
1m: state=EXPANSION score=72 atrp=0.22% atr_exp=1.31 slope=+0.07 TR_spike=1.62 shock=YES
5m: state=SQUEEZE  score=18 atr_exp=0.74 slope=-0.02 TR_spike=0.88 shock=NO
```

**Read as**: 1m volatility expanding with shock → fast entry opportunity

---

## 🎓 Decision Matrix

| 1m State | 5m State | Action |
|----------|----------|--------|
| EXPANSION | EXPANSION | ✅✅ **Best timing** |
| EXPANSION | SQUEEZE | ⚠️ Wait for 5m confirm |
| SQUEEZE | EXPANSION | 🔄 Watch for 1m trigger |
| SQUEEZE | SQUEEZE | ⏸️ No trades |
| EXTREME | EXTREME | 🔥 Active but tight stops |
| FADE_RISK | * | ⚠️ Tighten/exit |

---

## 🧪 Testing

```bash
# Run tests
pytest tests/test_atr_expansion.py -v

# Run demo
python3 demo_atr_expansion.py
```

---

## 📁 Files

| File | Purpose |
|------|---------|
| `atr_expansion.py` | Core module |
| `tests/test_atr_expansion.py` | Tests (27/27 ✅) |
| `atr_expansion_integration.py` | Integration examples |
| `ATR_EXPANSION_README.md` | Full documentation |
| `demo_atr_expansion.py` | Interactive demo |

---

## 🔧 Performance

- **Time**: O(1) per candle
- **Space**: O(sma_period) per TF
- **Latency**: < 1ms

---

## 💬 Remember

> "ATR Expansion tells you WHEN, not WHERE.
>  It's a timing gate, not an entry signal."

---

**Version**: 1.0
**Status**: ✅ Production Ready
**Tests**: 27/27 Passing
