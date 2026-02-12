"""
Trading State Machine - Discrete decisions from continuous signals.

"Ingest continuously, decide discretely."

States represent market regimes. Transitions are triggered by
signal combinations, not individual indicators.
"""

import time
from typing import Optional, List, Dict, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from .data_types import (
    MarketState,
    VolumeSignal,
    DeltaSignal,
    BookSignal,
    OIFundingSignal,
    SignalDirection,
)


class MarketRegime(Enum):
    """
    Market regime states.

    These are mutually exclusive states that define
    what kind of market we're in RIGHT NOW.
    """
    # No trade states
    NO_TRADE = "no_trade"              # Default - no clear setup
    LOW_VOLUME = "low_volume"          # Volume too low for reliable signals
    CHOPPY = "choppy"                  # Conflicting signals, stay out

    # Pre-setup states (building energy)
    COMPRESSION = "compression"         # OI rising, price stalling - breakout imminent
    ACCUMULATION = "accumulation"       # Hidden buying, preparing for up move
    DISTRIBUTION = "distribution"       # Hidden selling, preparing for down move

    # Trade setup states
    SQUEEZE_SETUP_LONG = "squeeze_long"   # Short squeeze building
    SQUEEZE_SETUP_SHORT = "squeeze_short" # Long squeeze building
    TREND_CONTINUATION_LONG = "trend_long"   # Healthy uptrend, buy dips
    TREND_CONTINUATION_SHORT = "trend_short" # Healthy downtrend, sell rallies
    EXHAUSTION_LONG = "exhaustion_long"   # Uptrend exhausting, prepare for reversal
    EXHAUSTION_SHORT = "exhaustion_short" # Downtrend exhausting, prepare for reversal

    # Active states (in a position or just triggered)
    BREAKOUT_LONG = "breakout_long"    # Active long breakout
    BREAKOUT_SHORT = "breakout_short"  # Active short breakout
    REVERSAL_LONG = "reversal_long"    # Reversing to long
    REVERSAL_SHORT = "reversal_short"  # Reversing to short


@dataclass
class StateTransition:
    """Record of a state transition."""
    timestamp_ms: int
    from_state: MarketRegime
    to_state: MarketRegime
    trigger: str  # What caused the transition
    confidence: float
    signals: Dict[str, Any]  # Snapshot of signals at transition


@dataclass
class TradeSignal:
    """
    Actionable trade signal from state machine.

    This is what gets output when a tradeable state is entered.
    """
    timestamp_ms: int
    direction: str  # "long" or "short"
    signal_type: str  # "breakout", "squeeze", "trend", "reversal"
    regime: MarketRegime
    confidence: float

    # Entry guidance
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Context
    volume_confirms: bool = False
    book_confirms: bool = False
    oi_confirms: bool = False

    # Risk
    risk_level: str = "medium"  # "low", "medium", "high"
    warnings: List[str] = field(default_factory=list)


class TradingStateMachine:
    """
    State machine for trading decisions.

    Consumes MarketState snapshots and outputs:
    1. Current regime (always)
    2. Trade signals (when entering tradeable states)

    Key principle: State transitions require MULTIPLE confirming signals.
    Single indicators don't trigger trades.
    """

    # States that are "risk-off" (protective / exit states).
    # Transitions INTO these bypass min-duration and confirmation requirements.
    RISK_OFF_STATES = frozenset({
        MarketRegime.NO_TRADE,
        MarketRegime.LOW_VOLUME,
        MarketRegime.CHOPPY,
    })

    def __init__(
        self,
        min_confidence: float = 50.0,
        min_volume_ratio: float = 0.7,
        transition_cooldown_ms: int = 5000,
        min_state_duration_ms: int = 15000,
        confirmation_count: int = 3,
    ):
        """
        Initialize state machine.

        Args:
            min_confidence: Minimum unified score confidence for trades
            min_volume_ratio: Minimum relative volume to consider signals
            transition_cooldown_ms: Minimum time between transitions
            min_state_duration_ms: Minimum time in a risk-on state before
                allowing transition to a different risk-on state
            confirmation_count: Number of consecutive evaluations a new
                risk-on state must appear before the transition fires
        """
        self._current_state = MarketRegime.NO_TRADE
        self._state_entered_at: int = int(time.time() * 1000)
        self._min_confidence = min_confidence
        self._min_volume_ratio = min_volume_ratio
        self._cooldown_ms = transition_cooldown_ms
        self._min_state_duration_ms = min_state_duration_ms
        self._confirmation_count = confirmation_count

        # Consecutive confirmation tracking
        self._pending_state: Optional[MarketRegime] = None
        self._pending_count: int = 0

        # History
        self._transitions: deque[StateTransition] = deque(maxlen=100)
        self._last_transition_time: int = 0

        # Callbacks
        self._on_transition: List[Callable[[StateTransition], None]] = []
        self._on_trade_signal: List[Callable[[TradeSignal], None]] = []

    @property
    def current_state(self) -> MarketRegime:
        return self._current_state

    @property
    def state_duration_ms(self) -> int:
        """How long we've been in current state."""
        return int(time.time() * 1000) - self._state_entered_at

    @property
    def transitions(self) -> List[StateTransition]:
        return list(self._transitions)

    def on_transition(self, callback: Callable[[StateTransition], None]) -> None:
        """Register callback for state transitions."""
        self._on_transition.append(callback)

    def on_trade_signal(self, callback: Callable[[TradeSignal], None]) -> None:
        """Register callback for trade signals."""
        self._on_trade_signal.append(callback)

    def _is_risk_off(self, state: MarketRegime) -> bool:
        """True if state is a protective/exit state."""
        return state in self.RISK_OFF_STATES

    def update(self, market_state: MarketState) -> Optional[TradeSignal]:
        """
        Process new market state and potentially transition.

        Hysteresis rules:
        - Risk-off transitions (NO_TRADE, LOW_VOLUME, CHOPPY) are immediate
          — only the base cooldown applies.
        - Risk-on transitions require:
          1. The current state has been held for min_state_duration_ms
          2. The new state has been evaluated N consecutive times (confirmation)

        Args:
            market_state: Current aggregated market state

        Returns:
            TradeSignal if entering a tradeable state, else None
        """
        # Check base cooldown
        now = int(time.time() * 1000)
        if now - self._last_transition_time < self._cooldown_ms:
            return None

        # Determine what state we should be in
        new_state, trigger, confidence = self._evaluate_state(market_state)

        # No change — reset pending confirmation
        if new_state == self._current_state:
            self._pending_state = None
            self._pending_count = 0
            return None

        # Risk-off transitions are immediate (protect capital first)
        if self._is_risk_off(new_state):
            self._pending_state = None
            self._pending_count = 0
            return self._transition_to(new_state, trigger, confidence, market_state)

        # Risk-on transition from another risk-on state:
        # enforce minimum state duration to prevent flip-flopping
        if not self._is_risk_off(self._current_state):
            time_in_state = now - self._state_entered_at
            if time_in_state < self._min_state_duration_ms:
                return None

        # Consecutive confirmation: require N evaluations agreeing
        # on the same new state before committing
        if new_state == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = new_state
            self._pending_count = 1

        if self._pending_count < self._confirmation_count:
            return None

        # Confirmed — execute transition
        self._pending_state = None
        self._pending_count = 0
        return self._transition_to(new_state, trigger, confidence, market_state)

    def _evaluate_state(
        self,
        state: MarketState,
    ) -> tuple[MarketRegime, str, float]:
        """
        Evaluate what state we should be in based on signals.

        Returns:
            (new_state, trigger_reason, confidence)
        """
        vol = state.volume_signal
        delta = state.delta_signal
        book = state.book_signal
        oi_fund = state.oi_funding_signal
        score = state.unified_score
        conf = state.confidence

        # === NO TRADE CONDITIONS ===

        # Low volume - don't trade
        if vol and vol.relative_volume < self._min_volume_ratio:
            return MarketRegime.LOW_VOLUME, "volume_too_low", 80

        # Low confidence - stay out
        if conf < self._min_confidence * 0.5:
            return MarketRegime.NO_TRADE, "low_confidence", conf

        # === COMPRESSION DETECTION ===
        # OI rising + price flat + volume neutral
        if oi_fund and oi_fund.oi_direction == "rising":
            if vol and abs(vol.delta_ratio) < 0.1:
                # Check for price stalling (low directional movement)
                return MarketRegime.COMPRESSION, "oi_rising_price_flat", conf

        # === SQUEEZE SETUPS ===
        if oi_fund:
            # Short squeeze setup: crowded shorts + OI rising + bullish flow
            if (oi_fund.crowd_position in ["heavily_short", "moderately_short"] and
                oi_fund.oi_direction == "rising"):
                if vol and vol.direction == SignalDirection.BULLISH:
                    return MarketRegime.SQUEEZE_SETUP_LONG, "crowded_shorts_bullish_flow", conf

            # Long squeeze setup: crowded longs + OI rising + bearish flow
            if (oi_fund.crowd_position in ["heavily_long", "moderately_long"] and
                oi_fund.oi_direction == "rising"):
                if vol and vol.direction == SignalDirection.BEARISH:
                    return MarketRegime.SQUEEZE_SETUP_SHORT, "crowded_longs_bearish_flow", conf

        # === EXHAUSTION DETECTION ===
        if vol and vol.is_climax:
            if vol.direction == SignalDirection.BULLISH:
                return MarketRegime.EXHAUSTION_LONG, "volume_climax_bullish", conf
            elif vol.direction == SignalDirection.BEARISH:
                return MarketRegime.EXHAUSTION_SHORT, "volume_climax_bearish", conf

        # === TREND CONTINUATION ===
        if score >= 0.55 and conf >= self._min_confidence:
            # Check for healthy trend (not exhaustion)
            if oi_fund and oi_fund.oi_direction == "rising":
                if oi_fund.allows_long:
                    return MarketRegime.TREND_CONTINUATION_LONG, "strong_bullish_trend", conf

        if score <= -0.55 and conf >= self._min_confidence:
            if oi_fund and oi_fund.oi_direction == "rising":
                if oi_fund.allows_short:
                    return MarketRegime.TREND_CONTINUATION_SHORT, "strong_bearish_trend", conf

        # === BREAKOUT STATES ===
        # Strong unified score + volume + book alignment
        if score >= 0.7 and conf >= 70:
            if vol and vol.relative_volume >= 1.5:
                if book and book.path_of_least_resistance == "up":
                    return MarketRegime.BREAKOUT_LONG, "strong_bullish_breakout", conf

        if score <= -0.7 and conf >= 70:
            if vol and vol.relative_volume >= 1.5:
                if book and book.path_of_least_resistance == "down":
                    return MarketRegime.BREAKOUT_SHORT, "strong_bearish_breakout", conf

        # === ACCUMULATION / DISTRIBUTION ===
        if delta and delta.is_divergent:
            if delta.cvd_direction == SignalDirection.BULLISH:
                return MarketRegime.ACCUMULATION, "hidden_accumulation", conf
            elif delta.cvd_direction == SignalDirection.BEARISH:
                return MarketRegime.DISTRIBUTION, "hidden_distribution", conf

        # === CHOPPY - CONFLICTING SIGNALS ===
        # Volume says one thing, book says another
        if vol and book:
            vol_bullish = vol.direction == SignalDirection.BULLISH
            book_bullish = book.direction == SignalDirection.BULLISH

            if vol_bullish != book_bullish and vol.strength > 50 and book.strength > 50:
                return MarketRegime.CHOPPY, "conflicting_volume_book", conf

        # === DEFAULT: NO TRADE ===
        return MarketRegime.NO_TRADE, "no_clear_setup", conf

    def _transition_to(
        self,
        new_state: MarketRegime,
        trigger: str,
        confidence: float,
        market_state: MarketState,
    ) -> Optional[TradeSignal]:
        """
        Execute state transition.

        Returns:
            TradeSignal if entering tradeable state
        """
        now = int(time.time() * 1000)

        # Record transition
        transition = StateTransition(
            timestamp_ms=now,
            from_state=self._current_state,
            to_state=new_state,
            trigger=trigger,
            confidence=confidence,
            signals={
                "unified_score": market_state.unified_score,
                "price": market_state.current_price,
            },
        )
        self._transitions.append(transition)

        # Update state
        old_state = self._current_state
        self._current_state = new_state
        self._state_entered_at = now
        self._last_transition_time = now

        # Notify transition callbacks
        for callback in self._on_transition:
            try:
                callback(transition)
            except Exception:
                pass

        # Generate trade signal if entering tradeable state
        trade_signal = self._generate_trade_signal(new_state, market_state, confidence)

        if trade_signal:
            for callback in self._on_trade_signal:
                try:
                    callback(trade_signal)
                except Exception:
                    pass

        return trade_signal

    def _generate_trade_signal(
        self,
        state: MarketRegime,
        market_state: MarketState,
        confidence: float,
    ) -> Optional[TradeSignal]:
        """Generate trade signal for tradeable states."""
        now = int(time.time() * 1000)
        price = market_state.current_price

        # Map state to trade parameters
        trade_params = self._get_trade_params(state)
        if trade_params is None:
            return None

        direction, signal_type, risk_level = trade_params

        # Calculate stops and targets
        atr_estimate = market_state.atr if market_state.atr else price * 0.01  # 1% fallback
        if direction == "long":
            stop_loss = price - (atr_estimate * 1.5)
            take_profit = price + (atr_estimate * 3)
        else:
            stop_loss = price + (atr_estimate * 1.5)
            take_profit = price - (atr_estimate * 3)

        # Check confirmations
        vol = market_state.volume_signal
        book = market_state.book_signal
        oi_fund = market_state.oi_funding_signal

        volume_confirms = vol is not None and vol.relative_volume >= 1.2
        book_confirms = book is not None and (
            (direction == "long" and book.path_of_least_resistance == "up") or
            (direction == "short" and book.path_of_least_resistance == "down")
        )
        oi_confirms = oi_fund is not None and (
            (direction == "long" and oi_fund.allows_long) or
            (direction == "short" and oi_fund.allows_short)
        )

        # Gather warnings
        warnings = []
        if vol and vol.is_climax:
            warnings.append("Volume climax - potential reversal")
        if oi_fund and oi_fund.is_extreme:
            warnings.append(f"Extreme funding - {oi_fund.crowd_position}")
        if book and book.spoof_detected:
            warnings.append(f"Spoof detected on {book.spoof_side} side")

        return TradeSignal(
            timestamp_ms=now,
            direction=direction,
            signal_type=signal_type,
            regime=state,
            confidence=confidence,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            volume_confirms=volume_confirms,
            book_confirms=book_confirms,
            oi_confirms=oi_confirms,
            risk_level=risk_level,
            warnings=warnings,
        )

    def _get_trade_params(
        self,
        state: MarketRegime,
    ) -> Optional[tuple[str, str, str]]:
        """
        Get trade parameters for a state.

        Returns:
            (direction, signal_type, risk_level) or None if not tradeable
        """
        params = {
            MarketRegime.SQUEEZE_SETUP_LONG: ("long", "squeeze", "high"),
            MarketRegime.SQUEEZE_SETUP_SHORT: ("short", "squeeze", "high"),
            MarketRegime.TREND_CONTINUATION_LONG: ("long", "trend", "medium"),
            MarketRegime.TREND_CONTINUATION_SHORT: ("short", "trend", "medium"),
            MarketRegime.BREAKOUT_LONG: ("long", "breakout", "medium"),
            MarketRegime.BREAKOUT_SHORT: ("short", "breakout", "medium"),
            MarketRegime.REVERSAL_LONG: ("long", "reversal", "high"),
            MarketRegime.REVERSAL_SHORT: ("short", "reversal", "high"),
            # Exhaustion = prepare to fade
            MarketRegime.EXHAUSTION_LONG: ("short", "reversal", "high"),
            MarketRegime.EXHAUSTION_SHORT: ("long", "reversal", "high"),
        }
        return params.get(state)

    def force_state(self, state: MarketRegime, reason: str = "manual") -> None:
        """Force transition to a specific state (for testing/override)."""
        if state != self._current_state:
            now = int(time.time() * 1000)
            transition = StateTransition(
                timestamp_ms=now,
                from_state=self._current_state,
                to_state=state,
                trigger=f"forced: {reason}",
                confidence=100,
                signals={},
            )
            self._transitions.append(transition)
            self._current_state = state
            self._state_entered_at = now

    def reset(self) -> None:
        """Reset to initial state."""
        self._current_state = MarketRegime.NO_TRADE
        self._state_entered_at = int(time.time() * 1000)

    def snapshot(self) -> Dict[str, Any]:
        """Serialize state machine for persistence."""
        return {
            "current_state": self._current_state.value,
            "state_entered_at": self._state_entered_at,
            "last_transition_time": self._last_transition_time,
            "transitions": [
                {
                    "timestamp_ms": t.timestamp_ms,
                    "from_state": t.from_state.value,
                    "to_state": t.to_state.value,
                    "trigger": t.trigger,
                    "confidence": t.confidence,
                    "signals": t.signals,
                }
                for t in self._transitions
            ],
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore state machine from persisted snapshot."""
        try:
            state_value = snapshot.get("current_state", MarketRegime.NO_TRADE.value)
            self._current_state = MarketRegime(state_value)
        except Exception:
            self._current_state = MarketRegime.NO_TRADE

        self._state_entered_at = int(snapshot.get("state_entered_at", int(time.time() * 1000)))
        self._last_transition_time = int(snapshot.get("last_transition_time", 0))

        transitions = snapshot.get("transitions", [])
        self._transitions.clear()
        for item in transitions:
            try:
                transition = StateTransition(
                    timestamp_ms=int(item.get("timestamp_ms", 0)),
                    from_state=MarketRegime(item.get("from_state", MarketRegime.NO_TRADE.value)),
                    to_state=MarketRegime(item.get("to_state", MarketRegime.NO_TRADE.value)),
                    trigger=str(item.get("trigger", "")),
                    confidence=float(item.get("confidence", 0)),
                    signals=item.get("signals", {}),
                )
                self._transitions.append(transition)
            except Exception:
                continue
        self._transitions.clear()
