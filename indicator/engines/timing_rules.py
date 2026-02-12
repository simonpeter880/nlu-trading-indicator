"""
Timing Rules Engine - Consolidates Indicator Outputs into Timing Decisions

Consumes outputs from trend, VWAP, momentum, ROC, ATR expansion, CHOP, RSI, and MACD
modules to produce ONE deterministic timing decision per candle close.

CRITICAL:
- O(1) per candle close (no window scans)
- Deterministic with 2-of-3 logic and confirmation counters
- Does NOT place trades; only outputs labels + confidence + reasons
- Priority: EXHAUSTION > BREAKOUT > CONTINUATION > TRANSITION/NO_TRADE

Timing States:
- NO_TRADE: No setup present
- CONTINUATION_READY: Impulse restarting in established trend
- BREAKOUT_WINDOW: Squeeze-break expansion (compression → ignition)
- EXHAUSTION_WARNING: Don't chase (reversal risk)
- TRANSITION: Trend present but no clear timing signal

Usage:
    config = TimingRulesConfig(timeframes=["1m", "5m"])
    engine = TimingRulesEngine(config)

    inputs = {
        "trend_ok": True,
        "trend_bias": 1,
        "vwap_state": "RECLAIM",
        "vwap_position": "ABOVE",
        "roc_fast_norm": 0.45,
        "roc_fast_norm_prev": 0.30,
        "atr_exp": 1.15,
        # ... other fields
    }

    decision = engine.on_candle_close("1m", inputs)
    print(f"State: {decision.timing_state}, Dir: {decision.direction}, Conf: {decision.confidence_0_100:.0f}%")
    print(f"Reasons: {decision.reasons}")
"""

from dataclasses import dataclass, field
from typing import Optional

# ============================================================================
# CONFIG
# ============================================================================


@dataclass
class TimingRulesConfig:
    """Configuration for timing rules engine."""

    # Timeframes
    timeframes: list[str] = field(default_factory=lambda: ["1m", "5m", "1h"])

    # Continuation parameters
    roc_restart_thr: float = 0.30  # abs threshold for meaningful normalized ROC
    roc_impulse_thr: float = 0.80  # normalized impulse threshold
    atr_exp_continue_thr: float = 1.00
    use_macd_optional: bool = True
    macd_slope_thr: float = 0.0  # >0 for bull, <0 for bear
    require_vwap_alignment: bool = True  # vwap_position must match bias

    # Exhaustion parameters
    atr_exp_blowoff_thr: float = 1.60
    atr_exp_hot_thr: float = 1.20
    tr_spike_hot_thr: float = 1.50
    ribbon_compress_thr: float = 0.0  # <0 = compressing
    exhaustion_requires_2of3: bool = True

    # Squeeze-break parameters
    atr_exp_squeeze_thr: float = 0.80
    chop_high_thr: float = 61.8
    atr_exp_break_thr: float = 1.20
    tr_spike_break_thr: float = 1.50
    breakout_requires_roc_surge: bool = True
    breakout_confirm_bars: int = 2

    # Anti-flip / confirmation
    continuation_confirm_bars: int = 2
    exhaustion_confirm_bars: int = 2
    cooldown_bars_after_exhaustion: int = 2

    # Scoring
    enable_timing_score: bool = True
    score_act_thr: float = 60.0
    score_watch_thr: float = 30.0


# ============================================================================
# OUTPUT
# ============================================================================


@dataclass
class TimingDecision:
    """Timing decision output per timeframe."""

    timing_state: str  # NO_TRADE|CONTINUATION_READY|BREAKOUT_WINDOW|EXHAUSTION_WARNING|TRANSITION
    direction: int  # +1 bull, -1 bear, 0 neutral
    confidence_0_100: float
    timing_score_0_100: Optional[float]
    checklist: dict[str, bool]  # detailed booleans
    reasons: list[str]  # short strings
    debug: dict


# ============================================================================
# INTERNAL STATE
# ============================================================================


@dataclass
class _TimeframeState:
    """Internal state per timeframe (O(1))."""

    # Counters
    cont_count: int = 0
    ex_count: int = 0
    brk_count: int = 0
    cooldown_remaining: int = 0
    bar_index: int = 0

    # Last decision
    last_timing_state: str = "NO_TRADE"
    last_direction: int = 0

    # Breakout setup latch
    setup_armed: bool = False


# ============================================================================
# ENGINE
# ============================================================================


class TimingRulesEngine:
    """
    Timing Rules Engine - Consolidates indicator outputs into timing decisions.

    O(1) per candle close with deterministic 2-of-3 logic and confirmation counters.
    """

    def __init__(self, config: TimingRulesConfig):
        self.config = config
        self.states: dict[str, _TimeframeState] = {}

        for tf in config.timeframes:
            self.states[tf] = _TimeframeState()

    def reset(self, timeframe: str) -> None:
        """Reset state for a timeframe."""
        if timeframe in self.states:
            self.states[timeframe] = _TimeframeState()

    def on_candle_close(self, timeframe: str, inputs: dict) -> TimingDecision:
        """
        Process candle close and produce timing decision.

        Args:
            timeframe: Timeframe identifier
            inputs: Dict with indicator outputs (see module docstring)

        Returns:
            TimingDecision with state, direction, confidence, reasons
        """
        if timeframe not in self.states:
            self.states[timeframe] = _TimeframeState()

        state = self.states[timeframe]
        state.bar_index += 1

        # Extract inputs (treat missing as None)
        trend_ok = inputs.get("trend_ok")
        trend_bias = inputs.get("trend_bias", 0)
        ribbon_width_rate = inputs.get("ribbon_width_rate")

        vwap_state = inputs.get("vwap_state")
        vwap_position = inputs.get("vwap_position")

        roc_fast_norm = inputs.get("roc_fast_norm")
        roc_fast_norm_prev = inputs.get("roc_fast_norm_prev")
        acc_fast = inputs.get("acc_fast")

        atr_exp = inputs.get("atr_exp")
        atr_exp_slope = inputs.get("atr_exp_slope")
        tr_spike = inputs.get("tr_spike")

        chop_value = inputs.get("chop_value")
        chop_state = inputs.get("chop_state")

        rsi_divergence = inputs.get("rsi_divergence")
        structure_pivot_type = inputs.get("structure_pivot_type")

        macd_hist_slope = inputs.get("macd_hist_slope")
        macd_event = inputs.get("macd_event")

        # Initialize checklist and reasons
        checklist: dict[str, bool] = {}
        reasons: list[str] = []
        debug: dict = {"timeframe": timeframe, "bar_index": state.bar_index}

        # Update cooldown
        if state.cooldown_remaining > 0:
            state.cooldown_remaining -= 1
            debug["cooldown_remaining"] = state.cooldown_remaining

        # ====================================================================
        # PRIORITY 1: EXHAUSTION_WARNING
        # ====================================================================

        ex_decision = self._evaluate_exhaustion(
            state,
            trend_bias,
            roc_fast_norm,
            roc_fast_norm_prev,
            acc_fast,
            atr_exp,
            tr_spike,
            ribbon_width_rate,
            rsi_divergence,
            structure_pivot_type,
            checklist,
            reasons,
            debug,
        )

        if ex_decision is not None:
            state.last_timing_state = ex_decision.timing_state
            state.last_direction = ex_decision.direction
            return ex_decision

        # ====================================================================
        # PRIORITY 2: BREAKOUT_WINDOW
        # ====================================================================

        brk_decision = self._evaluate_breakout(
            state,
            atr_exp,
            tr_spike,
            chop_value,
            chop_state,
            roc_fast_norm,
            checklist,
            reasons,
            debug,
        )

        if brk_decision is not None:
            state.last_timing_state = brk_decision.timing_state
            state.last_direction = brk_decision.direction
            return brk_decision

        # ====================================================================
        # PRIORITY 3: CONTINUATION_READY
        # ====================================================================

        cont_decision = self._evaluate_continuation(
            state,
            trend_ok,
            trend_bias,
            vwap_state,
            vwap_position,
            roc_fast_norm,
            roc_fast_norm_prev,
            atr_exp,
            atr_exp_slope,
            macd_hist_slope,
            checklist,
            reasons,
            debug,
        )

        if cont_decision is not None:
            state.last_timing_state = cont_decision.timing_state
            state.last_direction = cont_decision.direction
            return cont_decision

        # ====================================================================
        # DEFAULT: TRANSITION / NO_TRADE
        # ====================================================================

        return self._default_state(trend_ok, checklist, reasons, debug)

    def get_state(self, timeframe: str) -> Optional[TimingDecision]:
        """Get last decision for timeframe (or None if never updated)."""
        if timeframe not in self.states:
            return None

        state = self.states[timeframe]
        if state.bar_index == 0:
            return None

        # Return minimal decision with last known state
        return TimingDecision(
            timing_state=state.last_timing_state,
            direction=state.last_direction,
            confidence_0_100=0.0,
            timing_score_0_100=None,
            checklist={},
            reasons=[],
            debug={"timeframe": timeframe, "cached": True},
        )

    # ========================================================================
    # EXHAUSTION LOGIC
    # ========================================================================

    def _evaluate_exhaustion(
        self,
        state: _TimeframeState,
        trend_bias: int,
        roc_fast_norm: Optional[float],
        roc_fast_norm_prev: Optional[float],
        acc_fast: Optional[float],
        atr_exp: Optional[float],
        tr_spike: Optional[float],
        ribbon_width_rate: Optional[float],
        rsi_divergence: Optional[str],
        structure_pivot_type: Optional[str],
        checklist: dict[str, bool],
        reasons: list[str],
        debug: dict,
    ) -> Optional[TimingDecision]:
        """Evaluate EXHAUSTION_WARNING (2-of-3 logic)."""

        if trend_bias == 0:
            checklist["ex_trend_context"] = False
            return None

        checklist["ex_trend_context"] = True

        # Condition 1: ROC decelerating
        roc_decelerating = False
        if roc_fast_norm is not None and acc_fast is not None:
            if trend_bias == 1:
                roc_decelerating = roc_fast_norm > self.config.roc_restart_thr and acc_fast < 0
            else:  # bear
                roc_decelerating = roc_fast_norm < -self.config.roc_restart_thr and acc_fast > 0
        checklist["ex_roc_decel"] = roc_decelerating

        # Condition 2: Blowoff
        hot = False
        if atr_exp is not None and atr_exp >= self.config.atr_exp_hot_thr:
            hot = True
        if tr_spike is not None and tr_spike >= self.config.tr_spike_hot_thr:
            hot = True

        extreme = atr_exp is not None and atr_exp >= self.config.atr_exp_blowoff_thr
        ribbon_compress = (
            ribbon_width_rate is not None and ribbon_width_rate < self.config.ribbon_compress_thr
        )

        blowoff = (extreme or hot) and ribbon_compress
        checklist["ex_blowoff"] = blowoff
        checklist["ex_hot"] = hot
        checklist["ex_extreme"] = extreme
        checklist["ex_ribbon_compress"] = ribbon_compress

        # Condition 3: RSI divergence at structure
        rsi_div_at_structure = False
        if rsi_divergence in {"REG_BEAR", "REG_BULL"} and structure_pivot_type in {
            "SWING_HIGH",
            "SWING_LOW",
        }:
            if trend_bias == 1:
                rsi_div_at_structure = (
                    rsi_divergence == "REG_BEAR" and structure_pivot_type == "SWING_HIGH"
                )
            else:  # bear
                rsi_div_at_structure = (
                    rsi_divergence == "REG_BULL" and structure_pivot_type == "SWING_LOW"
                )
        checklist["ex_rsi_div"] = rsi_div_at_structure

        # Count conditions
        conditions_met = sum([roc_decelerating, blowoff, rsi_div_at_structure])
        debug["ex_conditions_met"] = conditions_met

        if conditions_met >= 2:
            state.ex_count += 1
        else:
            state.ex_count = 0

        debug["ex_count"] = state.ex_count

        if state.ex_count >= self.config.exhaustion_confirm_bars:
            if roc_decelerating:
                reasons.append("roc_decelerating")
            if blowoff:
                reasons.append("blowoff")
            if rsi_div_at_structure:
                reasons.append("rsi_reg_div")

            confidence = 60.0 + 20.0 * (conditions_met - 2)
            if atr_exp is not None and atr_exp >= self.config.atr_exp_blowoff_thr:
                confidence += 10.0
            confidence = min(confidence, 100.0)

            # Set cooldown and reset continuation
            state.cooldown_remaining = self.config.cooldown_bars_after_exhaustion
            state.cont_count = 0

            timing_score = (
                self._compute_timing_score(
                    True, False, False, False, False, False, True, checklist, debug
                )
                if self.config.enable_timing_score
                else None
            )

            return TimingDecision(
                timing_state="EXHAUSTION_WARNING",
                direction=trend_bias,
                confidence_0_100=confidence,
                timing_score_0_100=timing_score,
                checklist=checklist,
                reasons=reasons,
                debug=debug,
            )

        return None

    # ========================================================================
    # BREAKOUT LOGIC
    # ========================================================================

    def _evaluate_breakout(
        self,
        state: _TimeframeState,
        atr_exp: Optional[float],
        tr_spike: Optional[float],
        chop_value: Optional[float],
        chop_state: Optional[str],
        roc_fast_norm: Optional[float],
        checklist: dict[str, bool],
        reasons: list[str],
        debug: dict,
    ) -> Optional[TimingDecision]:
        """Evaluate BREAKOUT_WINDOW (two-step: setup → ignition)."""

        # Step 1: Setup (compression)
        squeeze_setup = False
        if atr_exp is not None and atr_exp < self.config.atr_exp_squeeze_thr:
            if chop_state == "CHOP":
                squeeze_setup = True
            elif chop_value is not None and chop_value >= self.config.chop_high_thr:
                squeeze_setup = True

        checklist["brk_squeeze_setup"] = squeeze_setup

        if squeeze_setup:
            state.setup_armed = True
            reasons.append("squeeze_armed")

        debug["brk_setup_armed"] = state.setup_armed

        # Step 2: Ignition (only if armed)
        if not state.setup_armed:
            state.brk_count = 0
            return None

        # Volume ignition
        vol_ignite = False
        if atr_exp is not None and atr_exp >= self.config.atr_exp_break_thr:
            vol_ignite = True
        if tr_spike is not None and tr_spike >= self.config.tr_spike_break_thr:
            vol_ignite = True

        checklist["brk_vol_ignite"] = vol_ignite

        # ROC surge
        roc_surge = False
        if roc_fast_norm is not None and abs(roc_fast_norm) >= self.config.roc_impulse_thr:
            roc_surge = True

        checklist["brk_roc_surge"] = roc_surge

        # Ignition check
        ignition_ok = vol_ignite
        if self.config.breakout_requires_roc_surge:
            ignition_ok = ignition_ok and roc_surge

        checklist["brk_ignition_ok"] = ignition_ok

        # Direction
        direction = 0
        if roc_fast_norm is not None:
            direction = 1 if roc_fast_norm > 0 else -1 if roc_fast_norm < 0 else 0

        debug["brk_direction"] = direction

        # Confirmation
        if ignition_ok:
            state.brk_count += 1
        else:
            state.brk_count = 0

        debug["brk_count"] = state.brk_count

        if state.brk_count >= self.config.breakout_confirm_bars:
            if vol_ignite:
                reasons.append(
                    "atr_exp>=1.2"
                    if atr_exp and atr_exp >= self.config.atr_exp_break_thr
                    else "tr_spike>=1.5"
                )
            if roc_surge:
                reasons.append("roc_surge")

            confidence = 55.0
            if vol_ignite:
                confidence += 20.0
            if roc_surge:
                confidence += 20.0
            confidence = min(confidence, 100.0)

            # Disarm after firing
            state.setup_armed = False

            timing_score = (
                self._compute_timing_score(
                    False, False, False, False, False, not squeeze_setup, False, checklist, debug
                )
                if self.config.enable_timing_score
                else None
            )

            return TimingDecision(
                timing_state="BREAKOUT_WINDOW",
                direction=direction,
                confidence_0_100=confidence,
                timing_score_0_100=timing_score,
                checklist=checklist,
                reasons=reasons,
                debug=debug,
            )

        return None

    # ========================================================================
    # CONTINUATION LOGIC
    # ========================================================================

    def _evaluate_continuation(
        self,
        state: _TimeframeState,
        trend_ok: Optional[bool],
        trend_bias: int,
        vwap_state: Optional[str],
        vwap_position: Optional[str],
        roc_fast_norm: Optional[float],
        roc_fast_norm_prev: Optional[float],
        atr_exp: Optional[float],
        atr_exp_slope: Optional[float],
        macd_hist_slope: Optional[float],
        checklist: dict[str, bool],
        reasons: list[str],
        debug: dict,
    ) -> Optional[TimingDecision]:
        """Evaluate CONTINUATION_READY (2-of-3 momentum triggers)."""

        # Hard gates
        trend_gate = trend_ok is True and trend_bias in {1, -1}
        checklist["cont_trend_gate"] = trend_gate

        vwap_gate = vwap_state in {"ACCEPT", "RECLAIM"}
        checklist["cont_vwap_gate"] = vwap_gate

        alignment_gate = True
        if self.config.require_vwap_alignment and vwap_position is not None:
            if trend_bias == 1:
                alignment_gate = vwap_position == "ABOVE"
            elif trend_bias == -1:
                alignment_gate = vwap_position == "BELOW"
        checklist["cont_alignment_gate"] = alignment_gate

        cooldown_gate = state.cooldown_remaining == 0
        checklist["cont_cooldown_gate"] = cooldown_gate

        all_gates = trend_gate and vwap_gate and alignment_gate and cooldown_gate
        debug["cont_all_gates"] = all_gates

        if not all_gates:
            state.cont_count = 0
            return None

        # Momentum triggers (2 of 3)

        # Trigger 1: ROC restart
        roc_restart = False
        if roc_fast_norm is not None and roc_fast_norm_prev is not None:
            if trend_bias == 1:
                roc_restart = (
                    roc_fast_norm > roc_fast_norm_prev
                    and roc_fast_norm >= self.config.roc_restart_thr
                )
            elif trend_bias == -1:
                roc_restart = (
                    roc_fast_norm < roc_fast_norm_prev
                    and roc_fast_norm <= -self.config.roc_restart_thr
                )
        checklist["cont_roc_restart"] = roc_restart

        # Trigger 2: ATR support
        atr_support = False
        if atr_exp is not None and atr_exp >= self.config.atr_exp_continue_thr:
            if atr_exp_slope is None or atr_exp_slope >= 0:
                atr_support = True
        checklist["cont_atr_support"] = atr_support

        # Trigger 3: MACD support (optional)
        macd_support = False
        macd_available = False
        if self.config.use_macd_optional and macd_hist_slope is not None:
            macd_available = True
            if trend_bias == 1:
                macd_support = macd_hist_slope > self.config.macd_slope_thr
            elif trend_bias == -1:
                macd_support = macd_hist_slope < -self.config.macd_slope_thr
        checklist["cont_macd_support"] = macd_support
        checklist["cont_macd_available"] = macd_available

        # Count triggers (only count available ones)
        triggers = [roc_restart, atr_support]
        if macd_available:
            triggers.append(macd_support)

        triggers_met = sum(triggers)
        debug["cont_triggers_met"] = triggers_met
        debug["cont_triggers_available"] = len(triggers)

        if triggers_met >= 2:
            state.cont_count += 1
        else:
            state.cont_count = 0

        debug["cont_count"] = state.cont_count

        if state.cont_count >= self.config.continuation_confirm_bars:
            reasons.append("trend_ok")
            if vwap_state == "RECLAIM":
                reasons.append("vwap_reclaim")
            elif vwap_state == "ACCEPT":
                reasons.append("vwap_accept")
            if roc_restart:
                reasons.append("roc_turn_up" if trend_bias == 1 else "roc_turn_down")
            if atr_support:
                reasons.append("atr_exp>=1.0")
            if macd_support:
                reasons.append("macd_aligned")

            confidence = 50.0 + 15.0 * triggers_met
            if vwap_state == "RECLAIM":
                confidence += 10.0
            confidence = min(confidence, 100.0)

            timing_score = (
                self._compute_timing_score(
                    trend_gate,
                    vwap_gate,
                    roc_restart,
                    atr_support,
                    macd_support,
                    False,
                    False,
                    checklist,
                    debug,
                )
                if self.config.enable_timing_score
                else None
            )

            return TimingDecision(
                timing_state="CONTINUATION_READY",
                direction=trend_bias,
                confidence_0_100=confidence,
                timing_score_0_100=timing_score,
                checklist=checklist,
                reasons=reasons,
                debug=debug,
            )

        return None

    # ========================================================================
    # DEFAULT STATE
    # ========================================================================

    def _default_state(
        self, trend_ok: Optional[bool], checklist: dict[str, bool], reasons: list[str], debug: dict
    ) -> TimingDecision:
        """Return default state (TRANSITION or NO_TRADE)."""

        if trend_ok:
            timing_state = "TRANSITION"
            confidence = 30.0
            reasons.append("trend_present")
        else:
            timing_state = "NO_TRADE"
            confidence = 10.0
            reasons.append("no_setup")

        timing_score = (
            self._compute_timing_score(
                trend_ok or False, False, False, False, False, False, False, checklist, debug
            )
            if self.config.enable_timing_score
            else None
        )

        return TimingDecision(
            timing_state=timing_state,
            direction=0,
            confidence_0_100=confidence,
            timing_score_0_100=timing_score,
            checklist=checklist,
            reasons=reasons,
            debug=debug,
        )

    # ========================================================================
    # TIMING SCORE
    # ========================================================================

    def _compute_timing_score(
        self,
        trend_gate: bool,
        vwap_gate: bool,
        roc_restart: bool,
        atr_support: bool,
        macd_support: bool,
        chop_penalty: bool,
        exhaustion_active: bool,
        checklist: dict[str, bool],
        debug: dict,
    ) -> float:
        """Compute timing score 0-100 (for dashboards)."""

        score = 0.0

        if trend_gate:
            score += 25.0
        if vwap_gate:
            score += 25.0
        if roc_restart:
            score += 15.0
        if atr_support:
            score += 15.0
        if macd_support:
            score += 10.0

        if chop_penalty:
            score -= 30.0
        if exhaustion_active:
            score -= 40.0

        score = max(0.0, min(100.0, score))

        # Map to action levels (keep in debug only)
        if score < self.config.score_watch_thr:
            debug["score_level"] = "NO_TRADE"
        elif score < self.config.score_act_thr:
            debug["score_level"] = "WATCH"
        else:
            debug["score_level"] = "ACT"

        return score


# ============================================================================
# PRINT HELPERS
# ============================================================================


def print_timing_decisions(decisions: dict[str, TimingDecision]) -> None:
    """
    Print timing decisions in compact format.

    Example output:
    TIMING
    1m: state=CONTINUATION_READY dir=+1 conf=78 score=72 reasons=[trend_ok,vwap_reclaim,roc_turn_up,atr_exp>=1.0]
    5m: state=EXHAUSTION_WARNING dir=+1 conf=81 score=40 reasons=[roc_decelerating,blowoff,rsi_reg_div]
    1h: state=BREAKOUT_WINDOW dir=-1 conf=88 score=84 reasons=[squeeze_armed,atr_exp>=1.2,roc_surge]
    """
    print("TIMING")
    for tf, decision in decisions.items():
        print(format_timing_decision(tf, decision))


def format_timing_decision(timeframe: str, decision: TimingDecision) -> str:
    """Format single timing decision."""
    dir_str = f"{decision.direction:+d}" if decision.direction != 0 else " 0"
    score_str = (
        f"{decision.timing_score_0_100:.0f}" if decision.timing_score_0_100 is not None else "--"
    )
    reasons_str = ",".join(decision.reasons) if decision.reasons else "none"

    return (
        f"{timeframe}: state={decision.timing_state:<20s} dir={dir_str} "
        f"conf={decision.confidence_0_100:>2.0f} score={score_str:>2s} "
        f"reasons=[{reasons_str}]"
    )


def interpret_timing(decision: TimingDecision) -> str:
    """Interpret timing decision in human-readable format."""
    state = decision.timing_state
    direction = decision.direction
    conf = decision.confidence_0_100

    if state == "CONTINUATION_READY":
        bias = "bullish" if direction == 1 else "bearish" if direction == -1 else "neutral"
        return f"Continuation ready ({bias}, {conf:.0f}% confidence) - impulse restarting"
    elif state == "BREAKOUT_WINDOW":
        bias = "up" if direction == 1 else "down" if direction == -1 else "unknown"
        return f"Breakout window ({bias}, {conf:.0f}% confidence) - squeeze breaking"
    elif state == "EXHAUSTION_WARNING":
        bias = "bullish" if direction == 1 else "bearish" if direction == -1 else "neutral"
        return f"Exhaustion warning ({bias} trend, {conf:.0f}% confidence) - don't chase"
    elif state == "TRANSITION":
        return f"Transition ({conf:.0f}% confidence) - trend present but no timing signal"
    else:  # NO_TRADE
        return f"No trade ({conf:.0f}% confidence) - no setup"


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n╔" + "=" * 68 + "╗")
    print("║" + " " * 22 + "TIMING RULES ENGINE" + " " * 26 + "║")
    print("╚" + "=" * 68 + "╝\n")

    config = TimingRulesConfig(timeframes=["1m", "5m"])
    engine = TimingRulesEngine(config)

    print("Example 1: CONTINUATION_READY")
    print("-" * 70)
    inputs1 = {
        "trend_ok": True,
        "trend_bias": 1,
        "vwap_state": "RECLAIM",
        "vwap_position": "ABOVE",
        "roc_fast_norm": 0.45,
        "roc_fast_norm_prev": 0.30,
        "atr_exp": 1.15,
        "atr_exp_slope": 0.05,
    }

    # Feed 3 bars to confirm
    for i in range(3):
        decision1 = engine.on_candle_close("1m", inputs1)
    print(format_timing_decision("1m", decision1))
    print(f"→ {interpret_timing(decision1)}\n")

    print("Example 2: EXHAUSTION_WARNING")
    print("-" * 70)
    inputs2 = {
        "trend_ok": True,
        "trend_bias": 1,
        "roc_fast_norm": 0.50,
        "acc_fast": -0.20,
        "atr_exp": 1.70,
        "ribbon_width_rate": -0.05,
    }

    # Reset and feed
    engine.reset("5m")
    for i in range(3):
        decision2 = engine.on_candle_close("5m", inputs2)
    print(format_timing_decision("5m", decision2))
    print(f"→ {interpret_timing(decision2)}\n")

    print("Example 3: BREAKOUT_WINDOW")
    print("-" * 70)

    # Setup phase
    inputs3_setup = {
        "atr_exp": 0.75,
        "chop_state": "CHOP",
    }
    decision3a = engine.on_candle_close("1m", inputs3_setup)
    print(f"Setup: {format_timing_decision('1m', decision3a)}")

    # Ignition phase
    inputs3_ignite = {
        "atr_exp": 1.25,
        "roc_fast_norm": 0.85,
    }
    for i in range(3):
        decision3b = engine.on_candle_close("1m", inputs3_ignite)
    print(f"Ignition: {format_timing_decision('1m', decision3b)}")
    print(f"→ {interpret_timing(decision3b)}\n")

    print("=" * 70)
    print("✅ Timing Rules Engine Ready!")
    print("=" * 70)
