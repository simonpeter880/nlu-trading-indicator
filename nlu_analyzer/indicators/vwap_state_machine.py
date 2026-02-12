"""
VWAP Interaction State Machine

Robust state machine for classifying VWAP interaction patterns:
- Position: ABOVE / BELOW / AT
- State: NEUTRAL / ACCEPT / REJECT / RECLAIM / LOSS
- Anti-flip hysteresis with regime-aware tuning
- O(1) per update with deterministic priority rules.
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Optional


class Position(Enum):
    """Price position relative to VWAP."""

    ABOVE = "ABOVE"
    BELOW = "BELOW"
    AT = "AT"


class InteractionState(Enum):
    """VWAP interaction state."""

    NEUTRAL = "NEUTRAL"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    RECLAIM = "RECLAIM"
    LOSS = "LOSS"


class Regime(Enum):
    """Market regime."""

    TREND = "TREND"
    RANGE = "RANGE"


@dataclass
class VWAPStateConfig:
    """Configuration for VWAP State Machine."""

    # Crossings detection window
    window_crossings: int = 20

    # Touch tolerances (base) - percentage of price
    touch_pct_by_tf: Dict[str, float] = field(
        default_factory=lambda: {"1m": 0.0001, "5m": 0.00015, "1h": 0.0003}
    )
    touch_sigma: float = 0.15  # Sigma units
    touch_atr_factor: float = 0.05  # 0.05 × ATR%

    # Reclaim/Loss buffers (hysteresis)
    reclaim_buffer_pct_by_tf: Dict[str, float] = field(
        default_factory=lambda: {"1m": 0.0002, "5m": 0.00025, "1h": 0.0006}
    )
    reclaim_buffer_atr_factor: float = 0.10  # 0.10 × ATR%

    # Confirmation thresholds
    rv_confirm: float = 1.2  # RV >= 1.2 for confirmation

    # Hold bars - TREND regime
    N_reclaim_trend: int = 2
    N_loss_trend: int = 2
    N_accept_trend: int = 2
    N_reject_trend: int = 3

    # Hold bars - RANGE regime
    N_reclaim_range: int = 3
    N_loss_range: int = 3
    N_accept_range: int = 3
    N_reject_range: int = 2

    # Chop detection
    crossings_chop_threshold: int = 4


@dataclass
class VWAPInteractionState:
    """Output state from VWAP interaction analysis."""

    # Core classification
    position: str  # "ABOVE", "BELOW", "AT"
    state: str  # "NEUTRAL", "ACCEPT", "REJECT", "RECLAIM", "LOSS"

    # Metrics
    hold_count: int  # Consecutive bars on current side
    side: int  # +1 (above), -1 (below), 0 (at)
    dist_pct: float  # Distance as percentage
    dist_sigma: Optional[float]  # Distance in sigma units
    crossings_20: int  # Sign crossings in window

    # Context
    last_touch_age: Optional[int]  # Bars since last touch
    debug: Dict = field(default_factory=dict)


class _TimeframeState:
    """Internal state for a single timeframe."""

    def __init__(self, config: VWAPStateConfig, tf: str):
        self.config = config
        self.tf = tf

        # Position tracking
        self.prev_side: int = 0
        self.prev_nonzero_side: int = 0  # Track last non-AT side for hold_count
        self.prev_position: str = "NEUTRAL"
        self.prev_state: str = "NEUTRAL"
        self.hold_count: int = 0

        # Crossings detection
        self.sign_deque: Deque[int] = deque(maxlen=config.window_crossings)
        self.bar_index: int = 0

        # Touch tracking
        self.last_touch_index: Optional[int] = None

        # State machine hold counters
        self.reclaim_hold: int = 0
        self.loss_hold: int = 0
        self.reject_hold: int = 0
        self.accept_hold_above: int = 0
        self.accept_hold_below: int = 0

        # For RECLAIM/LOSS hysteresis
        self.in_reclaim_process: bool = False
        self.in_loss_process: bool = False


class VWAPStateMachine:
    """VWAP Interaction State Machine."""

    def __init__(self, config: VWAPStateConfig):
        """
        Initialize VWAP State Machine.

        Args:
            config: Configuration parameters.
        """
        self.config = config
        self._states: Dict[str, _TimeframeState] = {}
        self._eps = 1e-12

    def reset(self, tf: str) -> None:
        """
        Reset state for a timeframe.

        Args:
            tf: Timeframe string.
        """
        if tf in self._states:
            del self._states[tf]

    def _get_or_create_state(self, tf: str) -> _TimeframeState:
        """Get or create timeframe state."""
        if tf not in self._states:
            self._states[tf] = _TimeframeState(self.config, tf)
        return self._states[tf]

    def on_update(
        self,
        tf: str,
        timestamp: int,
        close: float,
        vwap: float,
        std: Optional[float] = None,
        atr_percent: Optional[float] = None,
        rv: Optional[float] = None,
        delta_ok: Optional[bool] = None,
        oi_ok: Optional[bool] = None,
        regime: Optional[str] = None,
    ) -> VWAPInteractionState:
        """
        Process VWAP update and return interaction state.

        Args:
            tf: Timeframe string
            timestamp: Bar timestamp
            close: Close price
            vwap: VWAP value
            std: Volume-weighted standard deviation (optional)
            atr_percent: ATR as percentage (ATR/close) (optional)
            rv: Relative volume (optional)
            delta_ok: Order flow confirmation (optional)
            oi_ok: Open interest confirmation (optional)
            regime: Market regime "TREND" or "RANGE" (optional, inferred if None)

        Returns:
            VWAPInteractionState with classification and metrics.
        """
        state = self._get_or_create_state(tf)

        # A) Compute distances
        dist_pct = (close - vwap) / (vwap + self._eps)
        dist_sigma = None
        if std is not None and std > 0:
            dist_sigma = (close - vwap) / (std + self._eps)

        # B) Determine touch tolerances
        touch_pct = self.config.touch_pct_by_tf.get(tf, 0.00015)
        touch_atr = None
        if atr_percent is not None:
            touch_atr = self.config.touch_atr_factor * atr_percent

        # Check if touching VWAP
        is_touch = False
        if abs(dist_pct) <= touch_pct:
            is_touch = True
        elif dist_sigma is not None and abs(dist_sigma) <= self.config.touch_sigma:
            is_touch = True
        elif touch_atr is not None and abs(dist_pct) <= touch_atr:
            is_touch = True

        # C) Determine position and side
        if is_touch:
            position = Position.AT.value
            side = 0
        elif dist_pct > 0:
            position = Position.ABOVE.value
            side = 1
        else:
            position = Position.BELOW.value
            side = -1

        # D) Update crossings detector
        # Use sign_side for crossings tracking
        if dist_pct > 0:
            sign_side = 1
        elif dist_pct < 0:
            sign_side = -1
        else:
            # Keep previous nonzero sign if at zero
            sign_side = state.sign_deque[-1] if state.sign_deque else 0

        state.sign_deque.append(sign_side)

        # Count crossings
        crossings = 0
        for i in range(1, len(state.sign_deque)):
            if state.sign_deque[i] != state.sign_deque[i - 1]:
                crossings += 1
        crossings_20 = crossings

        # E) Regime selection
        if regime is not None:
            regime_used = regime
        else:
            # Infer from crossings
            if crossings_20 >= self.config.crossings_chop_threshold:
                regime_used = Regime.RANGE.value
            else:
                regime_used = Regime.TREND.value

        # F) Choose hold params based on regime
        if regime_used == Regime.TREND.value:
            N_reclaim = self.config.N_reclaim_trend
            N_loss = self.config.N_loss_trend
            N_accept = self.config.N_accept_trend
            N_reject = self.config.N_reject_trend
        else:  # RANGE
            N_reclaim = self.config.N_reclaim_range
            N_loss = self.config.N_loss_range
            N_accept = self.config.N_accept_range
            N_reject = self.config.N_reject_range

        # G) Compute reclaim buffer
        buffer_pct = self.config.reclaim_buffer_pct_by_tf.get(tf, 0.00025)
        buffer_atr = None
        if atr_percent is not None:
            buffer_atr = self.config.reclaim_buffer_atr_factor * atr_percent
        reclaim_buffer = max(buffer_pct, buffer_atr if buffer_atr else 0.0)

        # H) Confirmations
        rv_ok = rv is not None and rv >= self.config.rv_confirm
        conf_ok = rv_ok or (delta_ok is True) or (oi_ok is True)

        # Track touch
        if is_touch:
            state.last_touch_index = state.bar_index

        # Recent touch check (allow up to 3 bars for REJECT to build)
        recent_touch = (
            state.last_touch_index is not None and (state.bar_index - state.last_touch_index) <= 3
        )

        # I) State machine logic (deterministic priority)
        final_state = InteractionState.NEUTRAL.value

        # 1) RECLAIM (priority 1)
        # Start RECLAIM process if coming from below (checking last nonzero side)
        if state.prev_nonzero_side == -1 and side == 1:
            state.in_reclaim_process = True
            state.reclaim_hold = 0

        # Continue RECLAIM if in process and above buffer
        if state.in_reclaim_process and dist_pct >= reclaim_buffer:
            state.reclaim_hold += 1
        else:
            # Reset if close back below -buffer during reclaim process
            if state.in_reclaim_process and dist_pct <= -reclaim_buffer:
                state.reclaim_hold = 0
                state.in_reclaim_process = False

        # Check RECLAIM trigger
        if state.reclaim_hold >= N_reclaim and (regime_used == Regime.TREND.value or conf_ok):
            final_state = InteractionState.RECLAIM.value
            # Reset other holds and exit their processes
            state.loss_hold = 0
            state.reject_hold = 0
            state.accept_hold_above = 0
            state.accept_hold_below = 0
            state.in_loss_process = False
            # Don't reset reclaim process until we leave reclaim state
        elif not state.in_reclaim_process or dist_pct < reclaim_buffer:
            # If not in reclaim or dropped below buffer, allow REJECT to reset reclaim_hold
            if state.reclaim_hold > 0 and not state.in_reclaim_process:
                state.reclaim_hold = 0

        # 2) LOSS (priority 2)
        if final_state == InteractionState.NEUTRAL.value:
            # Start LOSS process if coming from above (checking last nonzero side)
            if state.prev_nonzero_side == 1 and side == -1:
                state.in_loss_process = True
                state.loss_hold = 0

            # Continue LOSS if in process and below -buffer
            if state.in_loss_process and dist_pct <= -reclaim_buffer:
                state.loss_hold += 1
            else:
                # Reset if close back above +buffer during loss process
                if state.in_loss_process and dist_pct >= reclaim_buffer:
                    state.loss_hold = 0
                    state.in_loss_process = False

            # Check LOSS trigger
            if state.loss_hold >= N_loss and (regime_used == Regime.TREND.value or conf_ok):
                final_state = InteractionState.LOSS.value
                # Reset other holds and exit their processes
                state.reclaim_hold = 0
                state.reject_hold = 0
                state.accept_hold_above = 0
                state.accept_hold_below = 0
                state.in_reclaim_process = False
            elif not state.in_loss_process or dist_pct > -reclaim_buffer:
                # If not in loss or rose above -buffer, allow REJECT to reset loss_hold
                if state.loss_hold > 0 and not state.in_loss_process:
                    state.loss_hold = 0

        # 3) REJECT (priority 3)
        if final_state == InteractionState.NEUTRAL.value:
            # Don't process REJECT if RECLAIM or LOSS are in progress
            if not state.in_reclaim_process and not state.in_loss_process:
                # REJECT requires recent touch throughout the building process
                # If touch becomes non-recent, abandon REJECT (even if partially built)
                reject_in_zone = (position == Position.BELOW.value and dist_pct <= -touch_pct) or (
                    position == Position.ABOVE.value and dist_pct >= touch_pct
                )

                # Only build REJECT while touch remains recent
                if recent_touch and reject_in_zone:
                    state.reject_hold += 1
                else:
                    # Reset if: touch again, moved out of zone, or touch no longer recent
                    if is_touch or not reject_in_zone or not recent_touch:
                        state.reject_hold = 0

                # Check REJECT trigger
                if state.reject_hold >= N_reject:
                    final_state = InteractionState.REJECT.value
                    # Reset other holds when REJECT triggers
                    state.accept_hold_above = 0
                    state.accept_hold_below = 0
            else:
                # Reset REJECT hold if RECLAIM/LOSS in progress
                state.reject_hold = 0

        # 4) ACCEPT (priority 4)
        if final_state == InteractionState.NEUTRAL.value:
            # Don't process ACCEPT if higher priority states are building up
            # (RECLAIM/LOSS in process, or REJECT accumulating after recent touch)
            reject_building = (
                recent_touch and state.reject_hold > 0 and state.reject_hold < N_reject
            )

            if not (state.in_reclaim_process or state.in_loss_process or reject_building):
                # Update accept holds
                if position == Position.ABOVE.value:
                    state.accept_hold_above += 1
                    state.accept_hold_below = 0
                elif position == Position.BELOW.value:
                    state.accept_hold_below += 1
                    state.accept_hold_above = 0
                else:  # AT
                    # Don't reset immediately, just pause increment
                    pass

                # Check ACCEPT trigger
                # Suppress in chop unless strong confirmation
                chop_suppress = crossings_20 >= self.config.crossings_chop_threshold and not (
                    conf_ok
                    and state.accept_hold_above >= N_accept + 1
                    and state.accept_hold_below >= N_accept + 1
                )

                if not chop_suppress:
                    if state.accept_hold_above >= N_accept:
                        final_state = InteractionState.ACCEPT.value
                    elif state.accept_hold_below >= N_accept:
                        final_state = InteractionState.ACCEPT.value
            else:
                # Reset ACCEPT holds if higher priority processes active
                if state.in_reclaim_process or state.in_loss_process:
                    state.accept_hold_above = 0
                    state.accept_hold_below = 0

        # Update hold_count (consecutive bars on one side, excluding AT)
        if side != 0:
            # Check if same side as before (ignoring AT transitions)
            if side == state.prev_nonzero_side:
                state.hold_count += 1
            else:
                state.hold_count = 1
            # Update last nonzero side
            state.prev_nonzero_side = side
        else:
            # Don't reset on AT, keep previous count
            pass

        # Calculate last_touch_age
        last_touch_age = None
        if state.last_touch_index is not None:
            last_touch_age = state.bar_index - state.last_touch_index

        # Build debug info
        debug_info = {
            "regime_used": regime_used,
            "touch_pct": touch_pct,
            "touch_atr": touch_atr,
            "reclaim_buffer": reclaim_buffer,
            "is_touch": is_touch,
            "recent_touch": recent_touch,
            "rv_ok": rv_ok,
            "conf_ok": conf_ok,
            "N_reclaim": N_reclaim,
            "N_loss": N_loss,
            "N_accept": N_accept,
            "N_reject": N_reject,
            "reclaim_hold": state.reclaim_hold,
            "loss_hold": state.loss_hold,
            "reject_hold": state.reject_hold,
            "accept_hold_above": state.accept_hold_above,
            "accept_hold_below": state.accept_hold_below,
        }

        # Create result
        result = VWAPInteractionState(
            position=position,
            state=final_state,
            hold_count=state.hold_count,
            side=side,
            dist_pct=dist_pct,
            dist_sigma=dist_sigma,
            crossings_20=crossings_20,
            last_touch_age=last_touch_age,
            debug=debug_info,
        )

        # Update state for next iteration
        state.prev_side = side
        state.prev_position = position
        state.prev_state = final_state
        state.bar_index += 1

        return result


def format_vwap_interaction_output(state: VWAPInteractionState, compact: bool = True) -> str:
    """
    Format VWAP interaction state for display.

    Args:
        state: VWAPInteractionState to format
        compact: If True, use compact format

    Returns:
        Formatted string.
    """
    if compact:
        sigma_str = f"σ={state.dist_sigma:.2f}" if state.dist_sigma is not None else "σ=N/A"
        touch_age_str = (
            f"touch_age={state.last_touch_age}"
            if state.last_touch_age is not None
            else "never_touched"
        )

        line = (
            f"VWAP: {state.position} {state.state} "
            f"dist={state.dist_pct*100:.3f}% {sigma_str} "
            f"hold={state.hold_count} xings={state.crossings_20} {touch_age_str}"
        )

        if state.debug.get("regime_used"):
            line += f" regime={state.debug['regime_used']}"

        return line
    else:
        lines = ["VWAP INTERACTION STATE"]
        lines.append(f"  Position: {state.position} (side={state.side:+d})")
        lines.append(f"  State: {state.state}")
        lines.append(f"  Hold Count: {state.hold_count}")
        lines.append(f"  Distance: {state.dist_pct*100:.3f}%")
        if state.dist_sigma is not None:
            lines.append(f"  Distance (σ): {state.dist_sigma:.2f}")
        lines.append(f"  Crossings (20): {state.crossings_20}")
        if state.last_touch_age is not None:
            lines.append(f"  Last Touch: {state.last_touch_age} bars ago")
        else:
            lines.append("  Last Touch: Never")

        if state.debug:
            lines.append("  Debug:")
            for key, val in state.debug.items():
                if key not in [
                    "reclaim_hold",
                    "loss_hold",
                    "reject_hold",
                    "accept_hold_above",
                    "accept_hold_below",
                ]:
                    lines.append(f"    {key}: {val}")

        return "\n".join(lines)
