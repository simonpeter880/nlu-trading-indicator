"""
ATR Expansion Adapter for Continuous System

Bridges the ATR Expansion engine with the continuous analysis architecture,
providing volatility regime signals for the state machine.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from atr_expansion import ATRExpansionConfig, ATRExpansionEngine, ATRExpansionState, Candle

from .data_types import SignalDirection, TradeEvent
from .rolling_window import TradeWindow


@dataclass
class ATRSignal:
    """ATR expansion signal for state machine."""

    timestamp_ms: int
    timeframe: str
    vol_state: str  # SQUEEZE / NORMAL / EXPANSION / EXTREME / FADE_RISK
    vol_score: float  # 0-100
    atr_exp: Optional[float]  # Expansion ratio
    atr_exp_slope: Optional[float]  # Rate of change
    tr_spike: Optional[float]  # Spike ratio
    shock_now: bool  # Immediate shock detected
    direction: SignalDirection  # Based on vol state
    strength: float  # 0-100


class ATRExpansionAdapter:
    """
    Adapter for ATR Expansion Engine in continuous system.

    Converts trade windows to candles and provides ATR signals
    for timing gates in the state machine.
    """

    def __init__(self, config: Optional[ATRExpansionConfig] = None):
        """Initialize adapter with config."""
        # Default config for continuous system
        if config is None:
            config = ATRExpansionConfig(
                timeframes=["15s", "1m", "5m"],
                atr_period=14,
                sma_period=20,
            )

        self._engine = ATRExpansionEngine(config)
        self._config = config

        # Store latest signals per timeframe
        self._latest_signals: Dict[str, ATRSignal] = {}

    def compute_from_window(
        self,
        timeframe: str,
        trade_window: TradeWindow,
    ) -> Optional[ATRSignal]:
        """
        Compute ATR signal from trade window.

        Args:
            timeframe: Timeframe identifier (e.g., "1m", "5m")
            trade_window: Rolling window of trades

        Returns:
            ATRSignal or None if insufficient data
        """
        trades = trade_window.items()
        if len(trades) < 2:
            return None

        # Convert window to candle
        candle = self._window_to_candle(trade_window)
        if candle is None:
            return None

        # Update ATR engine
        atr_state = self._engine.on_candle_close(timeframe, candle)

        # Convert to signal
        signal = self._state_to_signal(timeframe, atr_state)

        # Cache signal
        self._latest_signals[timeframe] = signal

        return signal

    def warmup(self, candles_by_tf: Dict[str, List[Candle]]) -> Dict[str, ATRSignal]:
        """
        Warmup engine with historical candles.

        Args:
            candles_by_tf: Dict mapping timeframe -> list of candles

        Returns:
            Dict mapping timeframe -> ATRSignal
        """
        states = self._engine.warmup(candles_by_tf)

        signals = {}
        for tf, state in states.items():
            signal = self._state_to_signal(tf, state)
            signals[tf] = signal
            self._latest_signals[tf] = signal

        return signals

    def get_signal(self, timeframe: str) -> Optional[ATRSignal]:
        """Get latest signal for a timeframe."""
        return self._latest_signals.get(timeframe)

    def get_all_signals(self) -> Dict[str, ATRSignal]:
        """Get all latest signals."""
        return self._latest_signals.copy()

    def get_state(self, timeframe: str) -> Optional[ATRExpansionState]:
        """Get raw ATR expansion state for a timeframe."""
        return self._engine.get_state(timeframe)

    def _window_to_candle(self, trade_window: TradeWindow) -> Optional[Candle]:
        """
        Convert trade window to candle.

        Uses window aggregation to create OHLCV candle.
        """
        if len(trade_window) == 0:
            return None

        trades = trade_window.items()

        # Calculate OHLCV from trades
        prices = [t.price for t in trades]
        timestamp = trades[-1].timestamp_ms

        candle = Candle(
            timestamp=timestamp,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=sum(t.quantity for t in trades),
        )

        return candle

    def _state_to_signal(self, timeframe: str, state: ATRExpansionState) -> ATRSignal:
        """
        Convert ATR expansion state to signal.

        Maps volatility states to signal directions and strengths
        for state machine consumption.
        """
        # Determine direction based on vol state
        if state.vol_state in ["EXPANSION", "EXTREME"]:
            direction = SignalDirection.BULLISH  # Volatility expansion favors moves
        elif state.vol_state == "SQUEEZE":
            direction = SignalDirection.NEUTRAL  # Low vol = no direction
        elif state.vol_state == "FADE_RISK":
            direction = SignalDirection.BEARISH  # Fading = caution
        else:
            direction = SignalDirection.NEUTRAL

        # Strength is vol score
        strength = state.vol_score_0_100 if state.vol_score_0_100 is not None else 0.0

        return ATRSignal(
            timestamp_ms=int(state.debug.get("prev_close", 0) * 1000),  # Placeholder
            timeframe=timeframe,
            vol_state=state.vol_state,
            vol_score=strength,
            atr_exp=state.atr_exp,
            atr_exp_slope=state.atr_exp_slope,
            tr_spike=state.tr_spike,
            shock_now=state.debug.get("shock_now", False),
            direction=direction,
            strength=strength,
        )


def format_atr_signals(signals: Dict[str, ATRSignal]) -> str:
    """
    Format ATR signals for compact display.

    Returns multi-line string for terminal output.
    """
    lines = []
    for tf, sig in sorted(signals.items()):
        parts = [f"{tf}:"]
        parts.append(f"state={sig.vol_state}")
        parts.append(f"score={sig.vol_score:.0f}")

        if sig.atr_exp is not None:
            parts.append(f"atr_exp={sig.atr_exp:.2f}")

        if sig.atr_exp_slope is not None:
            sign = "+" if sig.atr_exp_slope >= 0 else ""
            parts.append(f"slope={sign}{sig.atr_exp_slope:.2f}")

        if sig.tr_spike is not None:
            parts.append(f"TR_spike={sig.tr_spike:.2f}")

        shock = "YES" if sig.shock_now else "NO"
        parts.append(f"shock={shock}")

        lines.append(" ".join(parts))

    return "\n".join(lines)
