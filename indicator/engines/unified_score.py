"""
Unified Market Score - ONE actionable number

Combines Volume, OI, Funding, and Orderbook into a single score in [-1, +1]
- Positive = Long bias
- Negative = Short bias
- Near zero = No trade

Weight distribution:
- Volume/Delta: 35% (most immediate/reliable)
- Orderbook:    30% (where price is forced)
- OI:           25% (money flow validation)
- Funding:      10% (contrarian crowding gauge)
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from .indicator_config import IndicatorConfig

from .funding_analysis import AdvancedFundingAnalyzer, FundingAnalysisSummary
from .indicator_config import DEFAULT_CONFIG, safe_divide
from .signals import Signal


@dataclass
class UnifiedScore:
    """Complete unified market score."""

    # Component scores [-1, +1]
    volume_score: float
    orderbook_score: float
    oi_score: float
    funding_score: float

    # Combined
    total_score: float  # Weighted combination [-1, +1]
    confidence: float  # 0-100

    # Action signals
    bias: str  # 'long', 'short', 'neutral'
    strength: str  # 'strong', 'moderate', 'weak'
    action: str  # Human-readable action

    # Breakdown
    volume_weight: float
    orderbook_weight: float
    oi_weight: float
    funding_weight: float

    # Interpretation
    description: str
    warning: Optional[str] = None


def clip(value: float, min_val: float, max_val: float) -> float:
    """Clip value to range."""
    return max(min_val, min(max_val, value))


def calculate_volume_score(
    delta_ratio: float, relative_volume: float, config: Optional["IndicatorConfig"] = None
) -> Tuple[float, str]:
    """
    Calculate volume score from delta and RV.

    Formula: S_vol = clip(ΔVr / 0.25, -1, +1) * clip((RV-1)/1.5, 0, 1)

    Interpretation: Delta matters more when RV is high.

    Args:
        delta_ratio: Volume delta ratio ΔVr (normalized to [-1, +1])
        relative_volume: RV (volume vs average)
        config: Optional IndicatorConfig for thresholds

    Returns:
        (score, description)
    """
    cfg = (config or DEFAULT_CONFIG).unified_score

    # Delta component (normalized by delta_normalization = 25% is "strong")
    delta_component = clip(
        safe_divide(delta_ratio, cfg.delta_normalization, default=0.0), -1.0, 1.0
    )

    # Volume amplification (RV > 1 amplifies, RV < 1 dampens)
    rv_multiplier = clip((relative_volume - 1.0) / cfg.rv_amplification, 0.0, 1.0)

    # Combined
    score = delta_component * rv_multiplier

    # Description
    if abs(score) < 0.1:
        if relative_volume < cfg.volume_low_rv:
            desc = "Low volume - delta unreliable"
        else:
            desc = "Balanced flow"
    elif score > cfg.volume_strong:
        desc = f"Strong buy pressure (ΔVr: {delta_ratio:+.2f}, RV: {relative_volume:.1f}x)"
    elif score > cfg.volume_moderate:
        desc = f"Moderate buy pressure (ΔVr: {delta_ratio:+.2f}, RV: {relative_volume:.1f}x)"
    elif score < -cfg.volume_strong:
        desc = f"Strong sell pressure (ΔVr: {delta_ratio:+.2f}, RV: {relative_volume:.1f}x)"
    elif score < -cfg.volume_moderate:
        desc = f"Moderate sell pressure (ΔVr: {delta_ratio:+.2f}, RV: {relative_volume:.1f}x)"
    else:
        desc = "Neutral volume"

    return score, desc


def calculate_oi_score(
    price_change_pct: float,
    oi_change_pct: float,
    oi_threshold: Optional[float] = None,
    config: Optional["IndicatorConfig"] = None,
) -> Tuple[float, str]:
    """
    Calculate OI score from price and OI change.

    Formula: S_oi = sign(ΔP) * clip(ΔOI / 0.02, -1, +1)

    Regime matrix:
    - ΔP > 0, ΔOI > 0 → Bullish (new longs entering)
    - ΔP > 0, ΔOI < 0 → Bearish trap (short covering, not real buying)
    - ΔP < 0, ΔOI > 0 → Bearish (new shorts entering)
    - ΔP < 0, ΔOI < 0 → Weak reversal potential (positions closing)

    Args:
        price_change_pct: Price change percentage
        oi_change_pct: OI change percentage
        oi_threshold: Threshold for "strong" OI change (default from config)
        config: Optional IndicatorConfig for thresholds

    Returns:
        (score, description)
    """
    cfg = (config or DEFAULT_CONFIG).unified_score
    threshold = oi_threshold if oi_threshold is not None else cfg.oi_threshold

    # Direction from price
    price_sign = 1.0 if price_change_pct > 0 else -1.0 if price_change_pct < 0 else 0.0

    # OI strength (threshold% change = 1.0)
    oi_component = clip(safe_divide(oi_change_pct, threshold, default=0.0), -1.0, 1.0)

    # Combined
    score = price_sign * oi_component

    # Description
    if abs(price_change_pct) < 0.1:
        desc = "Price stagnant - OI signal weak"
    elif price_change_pct > 0 and oi_change_pct > 1.0:
        desc = (
            f"BULLISH - Price up {price_change_pct:+.1f}%, OI up {oi_change_pct:+.1f}% (new longs)"
        )
    elif price_change_pct > 0 and oi_change_pct < -1.0:
        desc = f"TRAP - Price up {price_change_pct:+.1f}% but OI down {oi_change_pct:.1f}% (short covering only)"
    elif price_change_pct < 0 and oi_change_pct > 1.0:
        desc = f"BEARISH - Price down {price_change_pct:+.1f}%, OI up {oi_change_pct:+.1f}% (new shorts)"
    elif price_change_pct < 0 and oi_change_pct < -1.0:
        desc = f"EXHAUSTION - Price down {price_change_pct:+.1f}%, OI down {oi_change_pct:.1f}% (covering)"
    else:
        desc = f"Weak OI signal (ΔOI: {oi_change_pct:+.1f}%)"

    return score, desc


def calculate_funding_score(
    current_funding: float,
    historical_funding: Optional[List[float]] = None,
    oi_change_pct: Optional[float] = None,
    config: Optional["IndicatorConfig"] = None,
) -> Tuple[float, str]:
    """
    Calculate funding score using AdvancedFundingAnalyzer (CONTRARIAN).

    Delegates to funding_analysis.py for sophisticated percentile-based analysis,
    crowd positioning, and Funding+OI combos. Converts the result to [-1, +1] score.

    High funding → negative score (too bullish, fade it)
    Low/negative funding → positive score (too bearish, fade it)

    Args:
        current_funding: Current funding rate
        historical_funding: List of historical funding rates for percentile calculation
        oi_change_pct: OI change percentage (for Funding+OI combo analysis)
        config: Optional IndicatorConfig for thresholds

    Returns:
        (score, description)
    """
    # Use AdvancedFundingAnalyzer for full analysis
    analyzer = AdvancedFundingAnalyzer(config=config)
    funding_result = analyzer.full_analysis(
        current_rate=current_funding,
        historical_rates=historical_funding,
        oi_change_percent=oi_change_pct,
    )

    # Convert Signal to score [-1, +1]
    # funding_analysis returns contrarian signals:
    # - BEARISH = crowd is long, fade → negative score
    # - BULLISH = crowd is short, fade → positive score
    signal_to_score = {
        Signal.BULLISH: 0.7,  # Contrarian bullish (crowd short)
        Signal.BEARISH: -0.7,  # Contrarian bearish (crowd long)
        Signal.WARNING: 0.0,  # Extreme conditions, uncertain
        Signal.NEUTRAL: 0.0,
        Signal.CAUTION: 0.0,
        Signal.TRAP: 0.0,
    }

    base_score = signal_to_score.get(funding_result.overall_signal, 0.0)

    # Amplify if extreme and should NOT chase
    if funding_result.is_extreme and not funding_result.should_chase:
        base_score *= 1.3
    # Dampen if normal funding (can chase)
    elif funding_result.should_chase:
        base_score *= 0.5

    # Clip to [-1, +1]
    score = clip(base_score, -1.0, 1.0)

    # Use the summary from funding_analysis
    desc = funding_result.summary

    # Add Funding+OI combo context if available
    if funding_result.funding_oi_combo:
        combo = funding_result.funding_oi_combo
        if combo.probability > 65:
            desc += f" | {combo.expected_outcome}"

    return score, desc


def calculate_orderbook_score(
    depth_imbalance: float,
    absorption_bullish: bool = False,
    absorption_bearish: bool = False,
    imbalance_threshold: Optional[float] = None,
    is_bait: bool = False,
    spoof_detected: bool = False,
    config: Optional["IndicatorConfig"] = None,
) -> Tuple[float, str]:
    """
    Calculate orderbook score from depth imbalance and absorption.

    Formula:
    1. Base: imb = (bid_depth - ask_depth) / (bid_depth + ask_depth)
       S_book = clip(imb / 0.25, -1, +1)
    2. Absorption boost:
       - Bullish absorption: +0.2
       - Bearish absorption: -0.2
    3. If bait or spoof detected, neutralize the imbalance score
       (the imbalance is manipulation, not real conviction)
    4. Clip final score to [-1, +1]

    Args:
        depth_imbalance: (bids - asks) / (bids + asks) from near-price levels
        absorption_bullish: True if bids absorbing (bullish)
        absorption_bearish: True if asks absorbing (bearish)
        imbalance_threshold: Threshold for "strong" imbalance (default from config)
        is_bait: True if imbalance has no volume confirmation (likely manipulation)
        spoof_detected: True if spoof/wall-pull detected
        config: Optional IndicatorConfig for thresholds

    Returns:
        (score, description)
    """
    cfg = (config or DEFAULT_CONFIG).unified_score
    threshold = (
        imbalance_threshold
        if imbalance_threshold is not None
        else cfg.orderbook_imbalance_threshold
    )

    # Base imbalance score (threshold imbalance = 1.0)
    base_score = clip(safe_divide(depth_imbalance, threshold, default=0.0), -1.0, 1.0)

    # If bait or spoof detected, the imbalance is manipulation — don't trust it.
    # Spoof = contrarian signal (fake wall), bait = imbalance without volume.
    manipulation_desc = ""
    if spoof_detected:
        # Spoof is contrarian: fake wall on one side means pressure from the other.
        # Invert the imbalance direction and dampen.
        base_score = -base_score * 0.3
        manipulation_desc = " | SPOOF (contrarian)"
    elif is_bait:
        # Unconfirmed imbalance — dampen to near-zero
        base_score *= 0.2
        manipulation_desc = " | BAIT (unconfirmed)"

    # Absorption boost
    absorption_boost_val = 0.0
    absorption_desc = ""

    if absorption_bullish:
        absorption_boost_val = cfg.absorption_boost
        absorption_desc = " + BID ABSORPTION"
    elif absorption_bearish:
        absorption_boost_val = -cfg.absorption_boost
        absorption_desc = " + ASK ABSORPTION"

    # Final score (clipped)
    score = clip(base_score + absorption_boost_val, -1.0, 1.0)

    # Description
    if manipulation_desc:
        desc = f"Manipulation detected (imb: {depth_imbalance:+.2f}){manipulation_desc}{absorption_desc}"
    elif score > 0.5:
        desc = f"Strong bid support (imb: {depth_imbalance:+.2f}){absorption_desc}"
    elif score > 0.2:
        desc = f"Moderate bid support (imb: {depth_imbalance:+.2f}){absorption_desc}"
    elif score < -0.5:
        desc = f"Strong ask resistance (imb: {depth_imbalance:+.2f}){absorption_desc}"
    elif score < -0.2:
        desc = f"Moderate ask resistance (imb: {depth_imbalance:+.2f}){absorption_desc}"
    else:
        desc = f"Balanced orderbook (imb: {depth_imbalance:+.2f}){absorption_desc}"

    return score, desc


def calculate_unified_score(
    # Volume inputs
    delta_ratio: float,
    relative_volume: float,
    # OI inputs
    price_change_pct: float,
    oi_change_pct: Optional[float] = None,
    # Funding inputs
    current_funding: Optional[float] = None,
    historical_funding: Optional[List[float]] = None,
    # Orderbook inputs
    depth_imbalance: Optional[float] = None,
    absorption_bullish: bool = False,
    absorption_bearish: bool = False,
    is_bait: bool = False,
    spoof_detected: bool = False,
    # Weights (must sum to 1.0)
    volume_weight: float = 0.35,
    orderbook_weight: float = 0.30,
    oi_weight: float = 0.25,
    funding_weight: float = 0.10,
    # Config
    config: Optional["IndicatorConfig"] = None,
) -> UnifiedScore:
    """
    Calculate unified market score combining all indicators.

    Weights (default):
    - Volume:    35% (most immediate/reliable)
    - Orderbook: 30% (where price is forced)
    - OI:        25% (money flow validation)
    - Funding:   10% (contrarian crowding gauge)

    Action thresholds:
    - S_total >= +0.55 → Long bias
    - S_total <= -0.55 → Short bias
    - |S_total| < 0.25 → No trade

    Confidence:
    - Conf = 100 * min(1, |S_total| / 0.85)

    Returns:
        UnifiedScore with all components and final action
    """
    cfg = (config or DEFAULT_CONFIG).unified_score

    # Normalize weights
    total_weight = volume_weight + orderbook_weight + oi_weight + funding_weight
    if abs(total_weight - 1.0) > 0.01:
        # Renormalize
        volume_weight /= total_weight
        orderbook_weight /= total_weight
        oi_weight /= total_weight
        funding_weight /= total_weight

    # Calculate component scores
    vol_score, vol_desc = calculate_volume_score(delta_ratio, relative_volume, config)

    # OI score (if available)
    if oi_change_pct is not None:
        oi_score, oi_desc = calculate_oi_score(price_change_pct, oi_change_pct, config=config)
    else:
        oi_score = 0.0
        oi_desc = "OI data unavailable"

    # Funding score (if available)
    if current_funding is not None:
        fund_score, fund_desc = calculate_funding_score(
            current_funding=current_funding,
            historical_funding=historical_funding,
            oi_change_pct=oi_change_pct,
            config=config,
        )
    else:
        fund_score = 0.0
        fund_desc = "Funding data unavailable"

    # Orderbook score (if available)
    if depth_imbalance is not None:
        book_score, book_desc = calculate_orderbook_score(
            depth_imbalance,
            absorption_bullish,
            absorption_bearish,
            is_bait=is_bait,
            spoof_detected=spoof_detected,
            config=config,
        )
    else:
        book_score = 0.0
        book_desc = "Orderbook data unavailable"

    # Volume gating: orderbook/OI/funding can't be trusted without volume.
    # Two layers:
    #   1. Dynamic weights — scale non-volume weights by a volume factor when RV < 1.0
    #   2. Hard gate — if RV is truly dead (< 0.3), clamp to neutral regardless
    if relative_volume < 1.0:
        # Linear ramp: RV=0 → floor, RV=1.0 → 1.0 (no scaling)
        vol_factor = clip(
            cfg.volume_weight_floor + (1.0 - cfg.volume_weight_floor) * relative_volume,
            cfg.volume_weight_floor,
            1.0,
        )
        orderbook_weight *= vol_factor
        oi_weight *= vol_factor
        funding_weight *= vol_factor

        # Renormalize so weights still sum to original total
        adjusted_total = volume_weight + orderbook_weight + oi_weight + funding_weight
        if adjusted_total > 0:
            scale = 1.0 / adjusted_total
            volume_weight *= scale
            orderbook_weight *= scale
            oi_weight *= scale
            funding_weight *= scale

    # Weighted combination
    total_score = (
        volume_weight * vol_score
        + orderbook_weight * book_score
        + oi_weight * oi_score
        + funding_weight * fund_score
    )

    # Hard gate: truly dead volume → force neutral
    if relative_volume < cfg.volume_dead_rv:
        max_score = cfg.neutral_zone * 0.99  # Just below neutral zone edge
        total_score = clip(total_score, -max_score, max_score)

    # Confidence (0-100)
    confidence = min(100.0, 100.0 * abs(total_score) / cfg.confidence_divisor)

    # Determine bias
    if total_score >= cfg.long_threshold:
        bias = "long"
        strength = "strong" if total_score >= cfg.strong_threshold else "moderate"
        action = f"LONG (score: {total_score:+.2f})"
    elif total_score <= cfg.short_threshold:
        bias = "short"
        strength = "strong" if total_score <= -cfg.strong_threshold else "moderate"
        action = f"SHORT (score: {total_score:+.2f})"
    elif abs(total_score) < cfg.neutral_zone:
        bias = "neutral"
        strength = "weak"
        action = f"NO TRADE (score: {total_score:+.2f})"
    else:
        # neutral_zone <= |score| < long/short_threshold
        if total_score > 0:
            bias = "long"
            strength = "weak"
            action = f"WEAK LONG (score: {total_score:+.2f})"
        else:
            bias = "short"
            strength = "weak"
            action = f"WEAK SHORT (score: {total_score:+.2f})"

    # Build description
    desc_parts = []
    desc_parts.append(f"Volume: {vol_desc}")
    if oi_change_pct is not None:
        desc_parts.append(f"OI: {oi_desc}")
    if depth_imbalance is not None:
        desc_parts.append(f"Book: {book_desc}")
    if current_funding is not None:
        desc_parts.append(f"Funding: {fund_desc}")

    description = " | ".join(desc_parts)

    # Warnings
    warning = None

    # Warning: Volume disagrees with OI
    if (
        abs(vol_score) > cfg.divergence_warning_threshold
        and abs(oi_score) > cfg.divergence_warning_threshold
    ):
        if (vol_score > 0 and oi_score < 0) or (vol_score < 0 and oi_score > 0):
            warning = "Volume and OI diverging - potential trap"

    # Warning: Dead volume with directional bias from other components
    if relative_volume < cfg.volume_low_rv and abs(total_score) > cfg.neutral_zone:
        warning = f"Low volume (RV: {relative_volume:.2f}x) - signal lacks participation"

    # Warning: Extreme funding opposite to signal
    if abs(fund_score) > cfg.funding_extreme_threshold:
        if (fund_score > 0 and total_score < -cfg.divergence_warning_threshold) or (
            fund_score < 0 and total_score > cfg.divergence_warning_threshold
        ):
            warning = "Funding extreme opposite to signal - high squeeze risk"

    return UnifiedScore(
        volume_score=vol_score,
        orderbook_score=book_score,
        oi_score=oi_score,
        funding_score=fund_score,
        total_score=total_score,
        confidence=confidence,
        bias=bias,
        strength=strength,
        action=action,
        volume_weight=volume_weight,
        orderbook_weight=orderbook_weight,
        oi_weight=oi_weight,
        funding_weight=funding_weight,
        description=description,
        warning=warning,
    )


# Convenience function
def get_market_action(
    delta_ratio: float,
    relative_volume: float,
    price_change_pct: float,
    oi_change_pct: Optional[float] = None,
    depth_imbalance: Optional[float] = None,
    config: Optional["IndicatorConfig"] = None,
) -> Tuple[str, float, float]:
    """
    Quick check: What should I do?

    Returns:
        (action, score, confidence)
    """
    result = calculate_unified_score(
        delta_ratio=delta_ratio,
        relative_volume=relative_volume,
        price_change_pct=price_change_pct,
        oi_change_pct=oi_change_pct,
        depth_imbalance=depth_imbalance,
        config=config,
    )
    return result.action, result.total_score, result.confidence
