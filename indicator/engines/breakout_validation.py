"""
Breakout Validation System - Distinguish Real from Fake

Real breakouts have:
- Price breaks level + BUYER aggression (positive ΔVr)
- Price breaks level + OI EXPANDING (new positions)
- Volume acceleration + follow-through

Fake breakouts have:
- Price breaks up but ΔVr NEGATIVE (short covering only)
- Price breaks up but OI DROPS (exit liquidity)
- Heavy walls above + pull behavior
- Exhaustion signals at breakout

Key filter (HARD VETO):
- Breakout UP + (ΔVr < -0.05 OR S_oi < -0.3) → BLOCK LONG
- Breakout DOWN + (ΔVr > +0.05 OR S_oi > +0.3) → BLOCK SHORT
"""

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .indicator_config import IndicatorConfig

from .indicator_config import DEFAULT_CONFIG


class BreakoutType(Enum):
    """Type of breakout."""

    UPWARD = "upward"  # Breaking resistance
    DOWNWARD = "downward"  # Breaking support


class BreakoutOutcome(Enum):
    """Outcome classification."""

    TRUE_BREAKOUT = "true"  # Continued in breakout direction
    FAKE_BREAKOUT = "fake"  # Reversed and failed
    PENDING = "pending"  # Not enough bars to classify
    UNCLEAR = "unclear"  # Mixed signals


class BreakoutQuality(Enum):
    """Quality rating of breakout."""

    INSTITUTIONAL = "institutional"  # All signals aligned
    RETAIL = "retail"  # Weak signals, likely fake
    MIXED = "mixed"  # Some signals align


@dataclass
class BreakoutEvent:
    """A detected breakout event."""

    # Basic info
    timestamp: int
    breakout_type: BreakoutType
    breakout_price: float
    breakout_level: float  # The level that was broken
    breakout_margin_pct: float  # How far above/below level

    # Price context
    pre_breakout_range: float  # ATR or recent range
    close_above_level: bool  # For upward breakout
    close_below_level: bool  # For downward breakout

    # Outcome (if classified)
    outcome: BreakoutOutcome
    bars_to_classify: int
    max_favorable_excursion_pct: float  # MFE
    max_adverse_excursion_pct: float  # MAE
    final_return_pct: float  # Return after M bars


@dataclass
class BreakoutFeatures:
    """Features captured at breakout moment."""

    # Volume
    relative_volume: float  # RV
    volume_acceleration: float  # VA
    delta_ratio: float  # ΔVr
    cvd_slope: float  # CVD momentum (last 5 bars)

    # OI
    oi_change_pct: float  # ΔOI%
    oi_acceleration: float  # OI rate of change acceleration

    # Funding
    funding_z_score: float

    # Orderbook
    depth_imbalance_25bps: float  # Imbalance within 0.25%
    depth_imbalance_50bps: float  # Imbalance within 0.50%
    absorption_present: bool
    absorption_side: Optional[str]  # 'bid' or 'ask'

    # Exhaustion
    exhaustion_risk: str  # 'low', 'medium', 'high', 'extreme'

    # Derived scores
    volume_score: float
    oi_score: float
    orderbook_score: float


@dataclass
class BreakoutValidation:
    """Complete breakout validation result."""

    # Event
    event: BreakoutEvent
    features: BreakoutFeatures

    # Validation
    is_valid: bool  # Passed validation
    quality: BreakoutQuality
    confidence: float  # 0-100

    # Veto checks
    hard_veto: bool
    veto_reason: Optional[str]

    # Signals
    flow_aligned: bool  # Delta matches direction
    oi_aligned: bool  # OI expanding in direction
    book_aligned: bool  # Orderbook supports

    # Action
    action: str  # 'enter_long', 'enter_short', 'avoid', 'wait'
    entry_price: Optional[float]
    stop_loss: Optional[float]
    target: Optional[float]

    # Warnings
    warnings: List[str]
    description: str


class BreakoutValidator:
    """
    Validates breakouts using order flow, OI, and orderbook.

    Implements hard veto filter to block fake breakouts.
    """

    def __init__(self, config: Optional["IndicatorConfig"] = None):
        self.config = config or DEFAULT_CONFIG
        self.cfg = self.config.breakout

    def detect_breakout(
        self,
        prices: List[float],
        highs: List[float],
        lows: List[float],
        timestamps: Optional[List[int]],
        swing_high: float,
        swing_low: float,
        atr_pct: float,
    ) -> Optional[BreakoutEvent]:
        """
        Detect if a breakout occurred on the most recent bar.

        Args:
            prices: Close prices
            highs: High prices
            lows: Low prices
            timestamps: Candle timestamps (ms)
            swing_high: Recent swing high level
            swing_low: Recent swing low level
            atr_pct: ATR as percentage

        Returns:
            BreakoutEvent if breakout detected, None otherwise
        """
        if len(prices) < 2:
            return None

        current_price = prices[-1]
        current_high = highs[-1]
        current_low = lows[-1]

        # Check upward breakout
        if current_high > swing_high:
            margin_pct = (current_high - swing_high) / swing_high * 100
            atr_margin = atr_pct * self.cfg.atr_multiple

            if margin_pct >= self.cfg.min_margin_pct or margin_pct >= atr_margin:
                close_above = current_price > swing_high

                event_ts = timestamps[-1] if timestamps else int(time.time() * 1000)
                return BreakoutEvent(
                    timestamp=event_ts,
                    breakout_type=BreakoutType.UPWARD,
                    breakout_price=current_high,
                    breakout_level=swing_high,
                    breakout_margin_pct=margin_pct,
                    pre_breakout_range=atr_pct,
                    close_above_level=close_above,
                    close_below_level=False,
                    outcome=BreakoutOutcome.PENDING,
                    bars_to_classify=0,
                    max_favorable_excursion_pct=0,
                    max_adverse_excursion_pct=0,
                    final_return_pct=0,
                )

        # Check downward breakout
        if current_low < swing_low:
            margin_pct = (swing_low - current_low) / swing_low * 100
            atr_margin = atr_pct * self.cfg.atr_multiple

            if margin_pct >= self.cfg.min_margin_pct or margin_pct >= atr_margin:
                close_below = current_price < swing_low

                event_ts = timestamps[-1] if timestamps else int(time.time() * 1000)
                return BreakoutEvent(
                    timestamp=event_ts,
                    breakout_type=BreakoutType.DOWNWARD,
                    breakout_price=current_low,
                    breakout_level=swing_low,
                    breakout_margin_pct=margin_pct,
                    pre_breakout_range=atr_pct,
                    close_above_level=False,
                    close_below_level=close_below,
                    outcome=BreakoutOutcome.PENDING,
                    bars_to_classify=0,
                    max_favorable_excursion_pct=0,
                    max_adverse_excursion_pct=0,
                    final_return_pct=0,
                )

        return None

    def classify_breakout_outcome(
        self,
        event: BreakoutEvent,
        future_highs: List[float],
        future_lows: List[float],
        future_closes: List[float],
    ) -> BreakoutEvent:
        """
        Classify breakout outcome after M bars.

        True breakout: Continues +X% without dipping below level
        Fake breakout: Returns below level and travels -X%
        """
        if len(future_closes) < self.cfg.classification_bars:
            event.outcome = BreakoutOutcome.PENDING
            return event

        entry_price = event.breakout_price
        level = event.breakout_level

        # Track excursions
        mfe = 0.0  # Max favorable
        mae = 0.0  # Max adverse

        returned_below_level = False

        for i in range(min(self.cfg.classification_bars, len(future_closes))):
            high = future_highs[i]
            low = future_lows[i]

            if event.breakout_type == BreakoutType.UPWARD:
                # Favorable = upward
                favorable = (high - entry_price) / entry_price * 100
                adverse = (low - entry_price) / entry_price * 100

                mfe = max(mfe, favorable)
                mae = min(mae, adverse)

                # Check if returned below level
                if low < level:
                    returned_below_level = True

            else:  # DOWNWARD
                # Favorable = downward
                favorable = (entry_price - low) / entry_price * 100
                adverse = (entry_price - high) / entry_price * 100

                mfe = max(mfe, favorable)
                mae = min(mae, adverse)

                # Check if returned above level
                if high > level:
                    returned_below_level = True

        # Final return
        final_price = future_closes[min(self.cfg.classification_bars - 1, len(future_closes) - 1)]
        if event.breakout_type == BreakoutType.UPWARD:
            final_return = (final_price - entry_price) / entry_price * 100
        else:
            final_return = (entry_price - final_price) / entry_price * 100

        # Classify
        if mfe >= self.cfg.true_breakout_continuation_pct and not returned_below_level:
            outcome = BreakoutOutcome.TRUE_BREAKOUT
        elif returned_below_level and final_return < self.cfg.fake_breakout_reversal_pct:
            outcome = BreakoutOutcome.FAKE_BREAKOUT
        else:
            outcome = BreakoutOutcome.UNCLEAR

        event.outcome = outcome
        event.bars_to_classify = min(self.cfg.classification_bars, len(future_closes))
        event.max_favorable_excursion_pct = mfe
        event.max_adverse_excursion_pct = mae
        event.final_return_pct = final_return

        return event

    def validate_breakout(
        self,
        event: BreakoutEvent,
        features: BreakoutFeatures,
        volume_score: float,
        oi_score: float,
        orderbook_score: float,
    ) -> BreakoutValidation:
        """
        Validate breakout using flow, OI, and orderbook.

        Implements HARD VETO filter:
        - Breakout UP + (ΔVr < -0.05 OR S_oi < -0.3) → BLOCK
        - Breakout DOWN + (ΔVr > +0.05 OR S_oi > +0.3) → BLOCK
        """
        warnings = []
        hard_veto = False
        veto_reason = None

        # HARD VETO CHECK
        if event.breakout_type == BreakoutType.UPWARD:
            # Upward breakout: need positive delta and positive OI
            if features.delta_ratio < -self.cfg.veto_delta_ratio:
                hard_veto = True
                veto_reason = f"VETO: Upward breakout but ΔVr negative ({features.delta_ratio:+.3f}) - SHORT COVERING ONLY"
            elif oi_score < -self.cfg.veto_oi_score:
                hard_veto = True
                veto_reason = f"VETO: Upward breakout but OI score negative ({oi_score:+.2f}) - EXIT LIQUIDITY"

        else:  # DOWNWARD
            # Downward breakout: need negative delta and negative OI
            if features.delta_ratio > self.cfg.veto_delta_ratio:
                hard_veto = True
                veto_reason = f"VETO: Downward breakout but ΔVr positive ({features.delta_ratio:+.3f}) - LONG COVERING ONLY"
            elif oi_score > self.cfg.veto_oi_score:
                hard_veto = True
                veto_reason = f"VETO: Downward breakout but OI score positive ({oi_score:+.2f}) - EXIT LIQUIDITY"

        # If vetoed, return immediately
        if hard_veto:
            return BreakoutValidation(
                event=event,
                features=features,
                is_valid=False,
                quality=BreakoutQuality.RETAIL,
                confidence=0,
                hard_veto=True,
                veto_reason=veto_reason,
                flow_aligned=False,
                oi_aligned=False,
                book_aligned=False,
                action="avoid",
                entry_price=None,
                stop_loss=None,
                target=None,
                warnings=[veto_reason],
                description=f"FAKE BREAKOUT - {veto_reason}",
            )

        # Check alignment
        flow_aligned = False
        oi_aligned = False
        book_aligned = False

        if event.breakout_type == BreakoutType.UPWARD:
            # Need positive signals
            flow_aligned = (
                features.delta_ratio > self.cfg.flow_delta_ratio
                and volume_score > self.cfg.flow_volume_score
            )
            oi_aligned = (
                features.oi_change_pct > self.cfg.oi_change_pct and oi_score > self.cfg.oi_score
            )
            book_aligned = (
                features.depth_imbalance_25bps > self.cfg.book_depth_imbalance_25bps
                or orderbook_score > self.cfg.book_score
            )
        else:
            # Need negative signals
            flow_aligned = (
                features.delta_ratio < -self.cfg.flow_delta_ratio
                and volume_score < -self.cfg.flow_volume_score
            )
            oi_aligned = (
                features.oi_change_pct < -self.cfg.oi_change_pct and oi_score < -self.cfg.oi_score
            )
            book_aligned = (
                features.depth_imbalance_25bps < -self.cfg.book_depth_imbalance_25bps
                or orderbook_score < -self.cfg.book_score
            )

        # Count alignments
        num_aligned = sum([flow_aligned, oi_aligned, book_aligned])

        # Determine quality
        if num_aligned >= 3:
            quality = BreakoutQuality.INSTITUTIONAL
            confidence = self.cfg.institutional_confidence
        elif num_aligned >= 2:
            quality = BreakoutQuality.MIXED
            confidence = 60
        else:
            quality = BreakoutQuality.RETAIL
            confidence = self.cfg.retail_confidence

        # Adjust confidence based on features
        if features.relative_volume > self.cfg.relative_volume_boost_threshold:
            confidence += 10
        if abs(features.delta_ratio) > self.cfg.delta_ratio_boost_threshold:
            confidence += 10
        if features.exhaustion_risk in ["high", "extreme"]:
            confidence -= self.cfg.exhaustion_penalty
            warnings.append(f"Exhaustion risk: {features.exhaustion_risk}")

        confidence = max(0, min(100, confidence))

        # Determine action
        is_valid = confidence >= self.cfg.valid_confidence_threshold and not hard_veto

        if is_valid:
            if event.breakout_type == BreakoutType.UPWARD:
                action = "enter_long"
                entry_price = event.breakout_price
                stop_loss = event.breakout_level * (1 - self.cfg.stop_loss_pct / 100)
                target = entry_price * (1 + self.cfg.target_pct / 100)
            else:
                action = "enter_short"
                entry_price = event.breakout_price
                stop_loss = event.breakout_level * (1 + self.cfg.stop_loss_pct / 100)
                target = entry_price * (1 - self.cfg.target_pct / 100)
        else:
            action = "avoid" if confidence < self.cfg.avoid_confidence_threshold else "wait"
            entry_price = None
            stop_loss = None
            target = None

        # Warnings
        if not flow_aligned:
            warnings.append("Flow NOT aligned - delta doesn't match direction")
        if not oi_aligned:
            warnings.append("OI NOT aligned - positions not expanding in direction")
        if features.absorption_present:
            if event.breakout_type == BreakoutType.UPWARD and features.absorption_side == "ask":
                warnings.append("ASK ABSORPTION detected - resistance above")
            elif event.breakout_type == BreakoutType.DOWNWARD and features.absorption_side == "bid":
                warnings.append("BID ABSORPTION detected - support below")

        # Description
        aligned_str = f"{num_aligned}/3 signals aligned"
        desc = f"{quality.value.upper()} breakout ({aligned_str}) - {action.upper()}"

        return BreakoutValidation(
            event=event,
            features=features,
            is_valid=is_valid,
            quality=quality,
            confidence=confidence,
            hard_veto=hard_veto,
            veto_reason=veto_reason,
            flow_aligned=flow_aligned,
            oi_aligned=oi_aligned,
            book_aligned=book_aligned,
            action=action,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target=target,
            warnings=warnings,
            description=desc,
        )


@dataclass
class BreakoutBacktestResult:
    """Backtest results for breakout strategy."""

    total_breakouts: int
    true_breakouts: int
    fake_breakouts: int
    unclear: int

    # With veto filter
    total_entered: int
    correct_entries: int  # Entered true breakouts
    false_positives: int  # Entered fake breakouts
    missed_true: int  # Didn't enter true breakouts
    avoided_fake: int  # Correctly avoided fake breakouts

    # Metrics
    precision: float  # correct / total_entered
    recall: float  # correct / true_breakouts
    false_positive_rate: float

    # Excursions
    avg_mfe_true: float
    avg_mae_true: float
    avg_mfe_fake: float
    avg_mae_fake: float

    # P&L (if entered)
    avg_return: float
    win_rate: float
    profit_factor: float


class BreakoutBacktester:
    """
    Backtest breakout validation system.

    Tests precision, false positive rate, and excursions.
    """

    def __init__(self, validator: BreakoutValidator):
        self.validator = validator

    def backtest(
        self, events: List[BreakoutEvent], validations: List[BreakoutValidation]
    ) -> BreakoutBacktestResult:
        """
        Backtest breakout strategy.

        Assumes events have been classified with outcomes.
        """
        if len(events) != len(validations):
            raise ValueError("Events and validations must have same length")

        total_breakouts = len(events)
        true_breakouts = sum(1 for e in events if e.outcome == BreakoutOutcome.TRUE_BREAKOUT)
        fake_breakouts = sum(1 for e in events if e.outcome == BreakoutOutcome.FAKE_BREAKOUT)
        unclear = sum(1 for e in events if e.outcome == BreakoutOutcome.UNCLEAR)

        # Analyze entries (where validation said to enter)
        total_entered = sum(
            1 for v in validations if v.is_valid and v.action in ["enter_long", "enter_short"]
        )
        correct_entries = 0
        false_positives = 0
        missed_true = 0
        avoided_fake = 0

        mfe_true = []
        mae_true = []
        mfe_fake = []
        mae_fake = []

        returns = []
        wins = 0

        for i, (event, validation) in enumerate(zip(events, validations)):
            entered = validation.is_valid and validation.action in ["enter_long", "enter_short"]

            if event.outcome == BreakoutOutcome.TRUE_BREAKOUT:
                mfe_true.append(event.max_favorable_excursion_pct)
                mae_true.append(event.max_adverse_excursion_pct)

                if entered:
                    correct_entries += 1
                    returns.append(event.final_return_pct)
                    if event.final_return_pct > 0:
                        wins += 1
                else:
                    missed_true += 1

            elif event.outcome == BreakoutOutcome.FAKE_BREAKOUT:
                mfe_fake.append(event.max_favorable_excursion_pct)
                mae_fake.append(event.max_adverse_excursion_pct)

                if entered:
                    false_positives += 1
                    returns.append(event.final_return_pct)
                    if event.final_return_pct > 0:
                        wins += 1
                else:
                    avoided_fake += 1

        # Metrics
        precision = correct_entries / total_entered if total_entered > 0 else 0
        recall = correct_entries / true_breakouts if true_breakouts > 0 else 0
        fpr = false_positives / fake_breakouts if fake_breakouts > 0 else 0

        avg_mfe_true = sum(mfe_true) / len(mfe_true) if mfe_true else 0
        avg_mae_true = sum(mae_true) / len(mae_true) if mae_true else 0
        avg_mfe_fake = sum(mfe_fake) / len(mfe_fake) if mfe_fake else 0
        avg_mae_fake = sum(mae_fake) / len(mae_fake) if mae_fake else 0

        avg_return = sum(returns) / len(returns) if returns else 0
        win_rate = wins / len(returns) if returns else 0

        # Profit factor
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return BreakoutBacktestResult(
            total_breakouts=total_breakouts,
            true_breakouts=true_breakouts,
            fake_breakouts=fake_breakouts,
            unclear=unclear,
            total_entered=total_entered,
            correct_entries=correct_entries,
            false_positives=false_positives,
            missed_true=missed_true,
            avoided_fake=avoided_fake,
            precision=precision,
            recall=recall,
            false_positive_rate=fpr,
            avg_mfe_true=avg_mfe_true,
            avg_mae_true=avg_mae_true,
            avg_mfe_fake=avg_mfe_fake,
            avg_mae_fake=avg_mae_fake,
            avg_return=avg_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
        )


# Convenience functions
def is_breakout_valid(
    breakout_price: float,
    breakout_level: float,
    breakout_type: str,
    delta_ratio: float,
    oi_score: float,
    config: Optional["IndicatorConfig"] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Quick check: Is this breakout valid or fake?

    Returns:
        (is_valid, veto_reason)
    """
    cfg = (config or DEFAULT_CONFIG).breakout

    if breakout_type == "upward":
        if delta_ratio < -cfg.veto_delta_ratio:
            return False, "ΔVr negative - short covering only"
        if oi_score < -cfg.veto_oi_score:
            return False, "OI score negative - exit liquidity"
    else:  # downward
        if delta_ratio > cfg.veto_delta_ratio:
            return False, "ΔVr positive - long covering only"
        if oi_score > cfg.veto_oi_score:
            return False, "OI score positive - exit liquidity"

    return True, None
