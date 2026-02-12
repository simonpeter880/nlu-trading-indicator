"""
ROC Momentum - Minimal Compact Print Block

Copy-paste ready display functions for integrating ROC momentum into your output.
"""


def print_roc_status_line(roc_state, tf: str = "1m") -> str:
    """
    Ultra-compact single-line ROC status for continuous runner.

    Returns string like: "ROC:IMPULSE↑76 fast=+0.8% div=NONE"

    Usage:
        roc_line = print_roc_status_line(roc_state, "1m")
        print(f"... | {roc_line} | ...")
    """
    if roc_state is None:
        return "ROC:warming"

    # State with direction
    state = roc_state.momentum_state
    score = roc_state.momentum_score_0_100

    if state == "IMPULSE":
        direction = roc_state.debug.get("direction", "")
        arrow = "↑" if direction == "BULL" else "↓" if direction == "BEAR" else "·"
        state_str = f"{state}{arrow}{score:.0f}"
    else:
        state_str = f"{state}{score:.0f}"

    # Fast ROC
    if roc_state.roc:
        fast_lb = min(roc_state.roc.keys())
        roc_pct = roc_state.roc[fast_lb] * 100
        roc_str = f"{roc_pct:+.1f}%"
    else:
        roc_str = "N/A"

    # Divergence flag
    div = roc_state.debug.get("divergence", "NONE")
    div_str = div[0] if div != "NONE" else "-"  # B/L/- for bearish/bullish/none

    return f"ROC:{state_str} fast={roc_str} div={div_str}"


def print_roc_compact_block(roc_state, tf: str = "1m", width: int = 50) -> None:
    """
    Compact multi-line ROC block for analyze.py style output.

    Example output:
        ┌──────────────────────────────────────────────┐
        │  ROC MOMENTUM (1m)                           │
        └──────────────────────────────────────────────┘
        State: IMPULSE ↑  Score: 76/100
        ROC:  5=+0.08%  20=+0.22%  60=+0.60%
        Norm: 5=+1.10   ACC: 5=+0.0003
        Flags: blowoff=NO  divergence=NONE
    """
    if roc_state is None:
        print(f"┌{'─' * width}┐")
        print(f"│  ROC MOMENTUM ({tf}) - WARMING UP{' ' * (width - 33)}│")
        print(f"└{'─' * width}┘")
        return

    # Header
    print(f"┌{'─' * width}┐")
    title = f"  ROC MOMENTUM ({tf})"
    padding = width - len(title) - 2
    print(f"│{title}{' ' * padding}│")
    print(f"└{'─' * width}┘")

    # State and score
    state = roc_state.momentum_state
    score = roc_state.momentum_score_0_100

    if state == "IMPULSE":
        direction = roc_state.debug.get("direction", "")
        arrow = "↑" if direction == "BULL" else "↓" if direction == "BEAR" else "·"
        state_display = f"{state} {arrow}"
    else:
        state_display = state

    print(f"State: {state_display:12}  Score: {score:.0f}/100")

    # ROC values
    if roc_state.roc:
        roc_parts = []
        for lb in sorted(roc_state.roc.keys())[:3]:  # Show up to 3 lookbacks
            roc_pct = roc_state.roc[lb] * 100
            roc_parts.append(f"{lb}={roc_pct:+.2f}%")
        print(f"ROC:  {' '.join(roc_parts)}")

    # Normalized and acceleration
    if roc_state.roc_norm and roc_state.acc:
        fast_lb = min(roc_state.roc_norm.keys())
        norm = roc_state.roc_norm[fast_lb]
        acc = roc_state.acc[fast_lb]
        print(f"Norm: {fast_lb}={norm:+.2f}   ACC: {fast_lb}={acc:+.4f}")

    # Flags
    blowoff = roc_state.debug.get("blowoff", "NO")
    div = roc_state.debug.get("divergence", "NONE")
    print(f"Flags: blowoff={blowoff}  divergence={div}")

    # Optional context
    if "context" in roc_state.debug:
        print(f"  → {roc_state.debug['context']}")
    if "fade_type" in roc_state.debug:
        print(f"  → {roc_state.debug['fade_type']}")

    print()


def print_roc_mini(roc_state) -> str:
    """
    Minimal ROC info for cramped displays.

    Returns: "ROC:76↑ div:N"
    """
    if roc_state is None:
        return "ROC:--"

    score = roc_state.momentum_score_0_100
    state = roc_state.momentum_state

    arrow = ""
    if state == "IMPULSE":
        direction = roc_state.debug.get("direction", "")
        arrow = "↑" if direction == "BULL" else "↓" if direction == "BEAR" else ""

    div = roc_state.debug.get("divergence", "NONE")
    div_flag = div[0] if div != "NONE" else "N"

    return f"ROC:{score:.0f}{arrow} div:{div_flag}"


# =============================================================================
# INTEGRATION EXAMPLE
# =============================================================================

if __name__ == "__main__":
    from nlu_analyzer.indicators.roc_momentum import Candle, ROCConfig, ROCMomentumEngine

    # Create engine
    config = ROCConfig(timeframes=["1m"])
    engine = ROCMomentumEngine(config)

    # Generate sample data
    candles = []
    for i in range(70):
        close = 100.0 + i * 0.5
        candles.append(
            Candle(
                timestamp=float(i),
                open=close - 0.1,
                high=close + 0.2,
                low=close - 0.2,
                close=close,
                volume=1000.0,
            )
        )

    # Warmup
    engine.warmup({"1m": candles[:60]})

    # Process new candles
    for candle in candles[60:65]:
        state = engine.on_candle_close("1m", candle, atr_percent=0.01)

    state = engine.get_state("1m")

    # Test all display functions
    print("\n" + "=" * 60)
    print("ROC MOMENTUM - COMPACT DISPLAY FUNCTIONS")
    print("=" * 60)

    print("\n1. Status Line (for continuous runner):")
    print("-" * 60)
    line = print_roc_status_line(state, "1m")
    print(f"[15:30:45] BTCUSDT | {line} | Volume:+0.3M")

    print("\n2. Compact Block (for analyze.py):")
    print("-" * 60)
    print_roc_compact_block(state, "1m", width=50)

    print("3. Mini Display (for cramped space):")
    print("-" * 60)
    mini = print_roc_mini(state)
    print(f"Status: Price:$100.50 {mini} Vol:+3%")

    print("\n" + "=" * 60)
    print("COPY-PASTE READY ✓")
    print("=" * 60)
