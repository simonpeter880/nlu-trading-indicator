"""
Indicator Configuration Module
Centralizes all magic numbers and thresholds for trading indicators.

This module provides a single source of truth for all configurable parameters,
making it easy to tune indicators without hunting through multiple files.
"""

from dataclasses import dataclass, field
from typing import Optional

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

EPSILON = 1e-9


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is near zero.

    Args:
        numerator: The numerator
        denominator: The denominator
        default: Value to return if division is unsafe (default: 0.0)

    Returns:
        numerator / denominator if safe, otherwise default
    """
    return numerator / denominator if abs(denominator) > EPSILON else default


@dataclass
class VolumeThresholds:
    """Volume-related thresholds."""

    # Volume ratio thresholds (current_vol / avg_vol)
    high_ratio: float = 2.0  # Very high volume
    moderate_ratio: float = 1.5  # Moderately high volume
    low_ratio: float = 0.5  # Low volume

    # Volume confirmation strength adjustments
    high_volume_base_strength: float = 50.0
    high_volume_strength_multiplier: float = 20.0
    high_volume_max_strength: float = 90.0
    moderate_volume_strength: float = 70.0
    low_volume_strength: float = 30.0
    divergence_strength_penalty: float = 15.0


@dataclass
class OpenInterestThresholds:
    """Open Interest thresholds."""

    # OI change percentage thresholds
    significant_change_pct: float = 5.0  # Significant OI change
    strong_oi_base_strength: float = 60.0
    weak_oi_strength: float = 55.0
    neutral_strength: float = 50.0

    # OI regime thresholds
    price_move_threshold_pct: float = 0.5  # % to consider price "moving"
    oi_change_threshold_pct: float = 1.0  # % to consider OI "changing"

    # Rate of change analysis thresholds
    rate_acceleration_periods: int = 5
    rate_vs_avg_significant: float = 2.0
    rate_vs_avg_elevated: float = 1.5
    rate_vs_avg_low: float = 0.5
    acceleration_positive_threshold: float = 0.3
    acceleration_negative_threshold: float = -0.3

    # High-edge signal thresholds
    compression_oi_change_pct: float = 3.0
    compression_price_volatility_pct: float = 2.0
    breakout_trap_breakout_pct: float = 0.2
    breakout_trap_recent_window_bars: int = 5
    breakout_trap_oi_drop_pct: float = -2.0
    expansion_oi_change_pct: float = 5.0
    expansion_price_move_pct: float = 1.0
    exhaustion_oi_change_pct: float = -5.0

    # Direction thresholds (rising/falling/flat)
    direction_change_pct: float = 0.3  # % OI change to count as rising/falling

    # Money flow epsilon - OI change below this is "FLAT" not "ENTERING/LEAVING"
    money_flow_epsilon_pct: float = 0.1


@dataclass
class FundingThresholds:
    """Funding rate thresholds."""

    # Funding rate percentage thresholds (rate * 100)
    extreme_positive_pct: float = 0.1  # Very positive funding (crowded long)
    high_positive_pct: float = 0.05  # High positive funding
    positive_pct: float = 0.03  # Moderately positive
    neutral_pct: float = 0.01  # Near-zero funding
    negative_pct: float = -0.03  # Moderately negative
    high_negative_pct: float = -0.05  # High negative funding
    extreme_negative_pct: float = -0.1  # Very negative funding (crowded short)

    # Percentile thresholds for zone classification
    extreme_percentile: float = 95.0
    high_percentile: float = 75.0

    # Strength values
    extreme_base_strength: float = 50.0
    extreme_strength_multiplier: float = 100.0
    extreme_max_strength: float = 80.0
    moderate_strength: float = 55.0
    neutral_strength: float = 50.0


@dataclass
class OrderbookThresholds:
    """Orderbook/liquidity thresholds."""

    # Depth imbalance thresholds (bid_depth - ask_depth) / total
    strong_imbalance: float = 0.3  # Strong support/resistance
    moderate_imbalance: float = 0.1  # Moderate support/resistance

    # Spread thresholds (percentage)
    wide_spread_pct: float = 0.1  # Wide spread (low liquidity)
    tight_spread_pct: float = 0.01  # Tight spread (high liquidity)

    # Strength values
    strong_imbalance_base_strength: float = 50.0
    strong_imbalance_multiplier: float = 100.0
    strong_imbalance_max_strength: float = 80.0
    moderate_imbalance_strength: float = 60.0
    neutral_strength: float = 50.0

    # Orderbook analysis thresholds
    imbalance_ratio_threshold: float = 2.0
    strong_imbalance_ratio: float = 3.0

    absorption_efficiency_max: float = 0.3
    absorption_volume_ratio: float = 1.5
    absorption_price_impact_pct: float = 0.3
    absorption_bid_imbalance_ratio: float = 1.5
    absorption_ask_imbalance_ratio: float = 0.67
    absorption_oi_confirm_max: float = 0.0

    spoof_min_confidence: float = 60.0
    spoof_wall_size_multiplier: float = 3.0
    spoof_price_tolerance_ratio: float = 0.001
    spoof_wall_persist_ratio: float = 0.5
    spoof_price_approach_pct: float = 1.0
    spoof_price_snap_min_bps: float = (
        2.0  # Minimum 2 bps move to confirm price snap (filters noise)
    )
    spoof_lookback_snapshots: int = 5
    spoof_oi_increase_threshold: float = 1.0

    thick_liquidity_multiplier: float = 2.0
    thin_liquidity_multiplier: float = 0.5

    # Path of least resistance calculation
    path_resistance_levels: int = 15  # Number of levels to sum for resistance (top 15 of 50)
    path_resistance_threshold: float = 0.7  # Ratio < 0.7 = clear path up, > 1.43 = clear path down
    path_hysteresis_factor: float = 0.85  # Harder to flip existing direction (15% hysteresis)

    volume_confirmation_price_change_pct: float = (
        0.1  # DEPRECATED: use volume_confirmation_ratio_threshold
    )
    volume_confirmation_ratio_threshold: float = (
        1.2  # Volume must be 1.2x average to confirm imbalance
    )
    directional_signal_ratio: float = 1.3


@dataclass
class DeltaThresholds:
    """Volume delta thresholds for InstitutionalVolumeEngine."""

    # Delta percentage thresholds
    strong_threshold_pct: float = 30.0  # Strong buy/sell bias
    weak_threshold_pct: float = 10.0  # Weak buy/sell bias

    # Strength calculation
    strong_base_strength: float = 70.0
    strong_strength_multiplier: float = 0.5
    strong_max_strength: float = 95.0
    weak_base_strength: float = 55.0
    weak_strength_multiplier: float = 0.75
    neutral_base_strength: float = 40.0
    neutral_strength_multiplier: float = 1.5


@dataclass
class AccelerationThresholds:
    """Volume acceleration thresholds."""

    accelerating: float = 1.5  # 1.5x for accelerating volume
    decelerating: float = 0.7  # 0.7x for decelerating volume
    climax: float = 3.0  # 3x for climax (potential reversal)

    # Bar counting thresholds
    accel_bar_threshold: float = 1.1  # Threshold for counting accelerating bars
    decel_bar_threshold: float = 0.9  # Threshold for counting decelerating bars


@dataclass
class MTFThresholds:
    """Multi-timeframe agreement thresholds."""

    high: float = 1.5  # 1.5x avg for "high" volume
    low: float = 0.7  # 0.7x avg for "low" volume


@dataclass
class ExhaustionThresholds:
    """Exhaustion detection thresholds."""

    body_shrink: float = 0.5  # 50% body reduction for exhaustion
    volume_spike: float = 1.5  # 1.5x avg volume for spike
    volume_decline: float = 0.7  # 70% of earlier volume = declining
    continuation_failure: float = 0.3  # Less than 30% continuation = failure
    oi_stagnant_pct: float = 1.0  # Less than 1% OI change = stagnant
    oi_dropping_pct: float = -2.0  # More than 2% OI drop


@dataclass
class UnifiedScoreThresholds:
    """Unified score action thresholds."""

    # Action thresholds
    long_threshold: float = 0.55  # Score >= this = long bias
    short_threshold: float = -0.55  # Score <= this = short bias
    neutral_zone: float = 0.25  # |score| < this = no trade
    strong_threshold: float = 0.75  # |score| >= this = strong signal

    # Confidence calculation
    confidence_divisor: float = 0.85  # Conf = 100 * |score| / this

    # Component score thresholds
    volume_strong: float = 0.5  # Strong volume pressure
    volume_moderate: float = 0.2  # Moderate volume pressure
    volume_low_rv: float = 0.7  # RV below this = low volume warning

    # Volume gating — prevents false bias when volume is dead
    volume_dead_rv: float = 0.3  # RV below this = hard gate (force neutral)
    volume_weight_floor: float = 0.3  # Minimum weight scaling at RV=0

    # Warning thresholds
    divergence_warning_threshold: float = 0.3  # Component score for divergence warning
    funding_extreme_threshold: float = 0.6  # Funding score for extreme warning

    # Calculation parameters
    delta_normalization: float = 0.25  # Delta ratio normalization factor
    rv_amplification: float = 1.5  # RV amplification factor
    oi_threshold: float = 2.0  # OI change percentage for "strong"
    funding_z_threshold: float = 2.5  # Z-score threshold for extreme funding
    orderbook_imbalance_threshold: float = 0.25  # Imbalance normalization factor
    absorption_boost: float = 0.2  # Absorption detection boost


@dataclass
class BreakoutThresholds:
    """Breakout validation thresholds."""

    # Detection thresholds
    min_margin_pct: float = 0.15
    atr_multiple: float = 0.25

    # Classification thresholds
    true_breakout_continuation_pct: float = 1.0
    fake_breakout_reversal_pct: float = -0.5
    classification_bars: int = 10

    # Hard veto thresholds
    veto_delta_ratio: float = 0.05
    veto_oi_score: float = 0.3

    # Alignment thresholds
    flow_delta_ratio: float = 0.05
    flow_volume_score: float = 0.2
    oi_change_pct: float = 0.5
    oi_score: float = 0.2
    book_depth_imbalance_25bps: float = 0.1
    book_score: float = 0.2

    # Confidence thresholds
    institutional_confidence: float = 85.0
    retail_confidence: float = 35.0
    valid_confidence_threshold: float = 50.0
    avoid_confidence_threshold: float = 40.0

    # Confidence adjustments
    relative_volume_boost_threshold: float = 2.0
    delta_ratio_boost_threshold: float = 0.25
    exhaustion_penalty: float = 20.0

    # Risk management defaults
    stop_loss_pct: float = 0.5
    target_pct: float = 1.5


@dataclass
class PreciseDeltaThresholds:
    """Precise volume delta thresholds."""

    # Noise filters
    min_notional: float = 50.0
    percentile_filter: float = 0.20

    # Delta ratio thresholds
    strong_buy_threshold: float = 0.30
    buy_threshold: float = 0.15
    neutral_threshold: float = 0.08
    sell_threshold: float = -0.15
    strong_sell_threshold: float = -0.30

    # Volume thresholds
    rv_dead: float = 0.3
    rv_low: float = 0.7
    rv_normal_max: float = 1.5
    rv_extreme: float = 3.0

    # Acceleration thresholds
    va_accelerating: float = 1.5
    va_decelerating: float = 0.7

    # Absorption thresholds
    absorption_rv: float = 1.5
    absorption_k1: float = 0.25
    absorption_k2: float = 0.10
    absorption_delta_ratio: float = 0.1

    # Sweep thresholds
    sweep_confirm_rv: float = 1.2
    sweep_max_bars: int = 3
    sweep_delta_flip_ratio: float = 0.1
    sweep_avg_volume_lookback: int = 20

    # Analysis windows
    recent_window_bars: int = 5


@dataclass
class IndicatorConfig:
    """
    Master configuration for all indicator thresholds.

    Usage:
        config = IndicatorConfig()
        # Use defaults

        # Or customize:
        config = IndicatorConfig(
            volume=VolumeThresholds(high_ratio=2.5),
            unified_score=UnifiedScoreThresholds(long_threshold=0.6)
        )
    """

    volume: VolumeThresholds = field(default_factory=VolumeThresholds)
    open_interest: OpenInterestThresholds = field(default_factory=OpenInterestThresholds)
    funding: FundingThresholds = field(default_factory=FundingThresholds)
    orderbook: OrderbookThresholds = field(default_factory=OrderbookThresholds)
    delta: DeltaThresholds = field(default_factory=DeltaThresholds)
    acceleration: AccelerationThresholds = field(default_factory=AccelerationThresholds)
    mtf: MTFThresholds = field(default_factory=MTFThresholds)
    exhaustion: ExhaustionThresholds = field(default_factory=ExhaustionThresholds)
    unified_score: UnifiedScoreThresholds = field(default_factory=UnifiedScoreThresholds)
    breakout: BreakoutThresholds = field(default_factory=BreakoutThresholds)
    precise_delta: PreciseDeltaThresholds = field(default_factory=PreciseDeltaThresholds)


# Global default config instance
DEFAULT_CONFIG = IndicatorConfig()


def get_config() -> IndicatorConfig:
    """Get the default configuration."""
    return DEFAULT_CONFIG


def create_aggressive_config() -> IndicatorConfig:
    """
    Create a more aggressive configuration with lower thresholds.
    Useful for scalping or high-frequency strategies.
    """
    return IndicatorConfig(
        volume=VolumeThresholds(high_ratio=1.5, moderate_ratio=1.2, low_ratio=0.6),
        unified_score=UnifiedScoreThresholds(
            long_threshold=0.45, short_threshold=-0.45, neutral_zone=0.15
        ),
        delta=DeltaThresholds(strong_threshold_pct=20.0, weak_threshold_pct=5.0),
    )


def create_conservative_config() -> IndicatorConfig:
    """
    Create a more conservative configuration with higher thresholds.
    Useful for swing trading or lower-frequency strategies.
    """
    return IndicatorConfig(
        volume=VolumeThresholds(high_ratio=2.5, moderate_ratio=2.0, low_ratio=0.4),
        unified_score=UnifiedScoreThresholds(
            long_threshold=0.65, short_threshold=-0.65, neutral_zone=0.35, strong_threshold=0.85
        ),
        delta=DeltaThresholds(strong_threshold_pct=40.0, weak_threshold_pct=15.0),
    )
