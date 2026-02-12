"""Trading Indicator Analysis Module.

Public symbols are exposed lazily so importing `indicator` does not eagerly
import optional network dependencies (for example `aiohttp` via data fetchers).
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple


__all__ = [
    # Data fetcher
    "BinanceIndicatorFetcher",
    "OHLCVData",
    "FundingRateData",
    "OpenInterestData",
    "OrderbookData",
    # Exceptions
    "BinanceAPIError",
    "BinanceRateLimitError",
    "BinanceTimeoutError",
    "BinanceConnectionError",
    # Request config
    "RequestConfig",
    "DEFAULT_REQUEST_CONFIG",
    # Basic indicators
    "IndicatorResult",
    "VolumeIndicators",
    "TrendIndicators",
    "MomentumIndicators",
    "VolatilityIndicators",
    # Advanced volume analysis
    "AdvancedVolumeAnalyzer",
    "VolumeContext",
    "VolumeLocation",
    "AbsorptionType",
    "RelativeVolumeResult",
    "VolumeLocationResult",
    "AbsorptionResult",
    "LiquiditySweepResult",
    "VolumeAnalysisSummary",
    "was_move_real",
    "get_volume_context",
    # Advanced OI analysis
    "AdvancedOIAnalyzer",
    "OIRegime",
    "OISignal",
    "OIRegimeResult",
    "OIRateOfChange",
    "OIHighEdgeSignal",
    "OIAnalysisSummary",
    "get_oi_regime",
    "is_money_entering",
    # Advanced Funding analysis
    "AdvancedFundingAnalyzer",
    "FundingZone",
    "CrowdPosition",
    "FundingWarning",
    "FundingOICombo",
    "FundingPercentileResult",
    "CrowdAnalysisResult",
    "FundingWarningResult",
    "FundingOIComboResult",
    "FundingAnalysisSummary",
    "should_i_chase",
    "get_crowd_lean",
    # Advanced Orderbook analysis
    "AdvancedOrderbookAnalyzer",
    "AbsorptionSide",
    "ImbalanceDirection",
    "SpoofType",
    "LiquidityZone",
    "OrderbookSnapshot",
    "OBAbsorptionResult",
    "ImbalanceResult",
    "SpoofResult",
    "LiquidityLadderResult",
    "OrderbookAnalysisSummary",
    "get_path_of_least_resistance",
    "is_imbalance_actionable",
    # Institutional Volume Engine
    "InstitutionalVolumeEngine",
    "AggressionBias",
    "VolumeAcceleration",
    "ExhaustionRisk",
    "MTFAgreement",
    "VolumeDeltaResult",
    "AccelerationResult",
    "MTFAgreementResult",
    "ExhaustionResult",
    "VolumeEngineResult",
    "get_aggression_bias",
    "is_volume_exhausted",
    # Precise Volume Delta
    "PreciseVolumeDeltaEngine",
    "PreciseAggressionBias",
    "VolumeLevel",
    "AccelerationState",
    "BarVolumeDelta",
    "PreciseVolumeDeltaResult",
    "AbsorptionDetectionResult",
    "SweepConfirmResult",
    "calculate_precise_delta",
    "is_absorption_present",
    "AggTradeData",
    # Unified Score
    "UnifiedScore",
    "calculate_unified_score",
    "get_market_action",
    # Breakout Validation
    "BreakoutValidator",
    "BreakoutType",
    "BreakoutOutcome",
    "BreakoutQuality",
    "BreakoutEvent",
    "BreakoutFeatures",
    "BreakoutValidation",
    "BreakoutBacktester",
    "BreakoutBacktestResult",
    "is_breakout_valid",
    # Signals
    "Signal",
    "coerce_signal",
    "signal_value",
    # Market Structure
    "MarketStructureDetector",
    "MarketStructureState",
    "SwingPoint",
    "SwingType",
    "TrendDirection",
    "StructureEvent",
    "RangeType",
    "StructuralMomentum",
    "TimeframeAlignment",
    "FairValueGap",
    "StructureBreak",
    "get_allowed_trade_direction",
    "structure_veto_signal",
    # Institutional Structure Engine
    "MarketStructureEngine",
    "StructureConfig",
    "Candle",
    "StructureState",
    "StructureSide",
    "EventType",
    "RangeClassification",
    "TradingMode",
    "ZoneStatus",
    "Zone",
    "MultiTFAlignment",
    "compute_atr",
    "compute_atr_pct",
    "compute_rv",
    "sma",
    # EMA Filter
    "EMAFilterEngine",
    "EMAConfig",
    "EMAState",
    "EMABias",
    "EMARegime",
    "EMAAlignment",
    "EMAMTFAlignment",
    "EMAMultiTFState",
    "create_candle",
    "print_ema_state",
    "print_ema_block",
    "compute_sma",
]


_EXPORT_TO_SOURCE: Dict[str, Tuple[str, str]] = {}


def _register(module: str, names: list[str], aliases: Dict[str, str] | None = None) -> None:
    for name in names:
        _EXPORT_TO_SOURCE[name] = (module, name)
    if aliases:
        for public_name, source_name in aliases.items():
            _EXPORT_TO_SOURCE[public_name] = (module, source_name)


_register(
    ".engines.data_fetcher",
    [
        "BinanceIndicatorFetcher",
        "OHLCVData",
        "FundingRateData",
        "OpenInterestData",
        "OrderbookData",
        "BinanceAPIError",
        "BinanceRateLimitError",
        "BinanceTimeoutError",
        "BinanceConnectionError",
        "RequestConfig",
        "DEFAULT_REQUEST_CONFIG",
        "AggTradeData",
    ],
)

_register(
    ".engines.indicators",
    [
        "IndicatorResult",
        "VolumeIndicators",
        "TrendIndicators",
        "MomentumIndicators",
        "VolatilityIndicators",
    ],
)

_register(
    ".engines.volume_analysis",
    [
        "AdvancedVolumeAnalyzer",
        "VolumeContext",
        "VolumeLocation",
        "AbsorptionType",
        "RelativeVolumeResult",
        "VolumeLocationResult",
        "AbsorptionResult",
        "LiquiditySweepResult",
        "VolumeAnalysisSummary",
        "was_move_real",
        "get_volume_context",
    ],
)

_register(
    ".engines.oi_analysis",
    [
        "AdvancedOIAnalyzer",
        "OIRegime",
        "OISignal",
        "OIRegimeResult",
        "OIRateOfChange",
        "OIHighEdgeSignal",
        "OIAnalysisSummary",
        "get_oi_regime",
        "is_money_entering",
    ],
)

_register(
    ".engines.funding_analysis",
    [
        "AdvancedFundingAnalyzer",
        "FundingZone",
        "CrowdPosition",
        "FundingWarning",
        "FundingOICombo",
        "FundingPercentileResult",
        "CrowdAnalysisResult",
        "FundingWarningResult",
        "FundingOIComboResult",
        "FundingAnalysisSummary",
        "should_i_chase",
        "get_crowd_lean",
    ],
)

_register(
    ".engines.orderbook_analysis",
    [
        "AdvancedOrderbookAnalyzer",
        "AbsorptionSide",
        "ImbalanceDirection",
        "SpoofType",
        "LiquidityZone",
        "OrderbookSnapshot",
        "ImbalanceResult",
        "SpoofResult",
        "LiquidityLadderResult",
        "OrderbookAnalysisSummary",
        "get_path_of_least_resistance",
        "is_imbalance_actionable",
    ],
    aliases={"OBAbsorptionResult": "AbsorptionResult"},
)

_register(
    ".engines.volume_engine",
    [
        "InstitutionalVolumeEngine",
        "AggressionBias",
        "VolumeAcceleration",
        "ExhaustionRisk",
        "MTFAgreement",
        "VolumeDeltaResult",
        "AccelerationResult",
        "MTFAgreementResult",
        "ExhaustionResult",
        "VolumeEngineResult",
        "get_aggression_bias",
        "is_volume_exhausted",
    ],
)

_register(
    ".engines.precise_volume_delta",
    [
        "PreciseVolumeDeltaEngine",
        "PreciseAggressionBias",
        "VolumeLevel",
        "AccelerationState",
        "BarVolumeDelta",
        "PreciseVolumeDeltaResult",
        "AbsorptionDetectionResult",
        "SweepConfirmResult",
        "calculate_precise_delta",
        "is_absorption_present",
    ],
)

_register(
    ".engines.unified_score",
    [
        "UnifiedScore",
        "calculate_unified_score",
        "get_market_action",
    ],
)

_register(
    ".engines.breakout_validation",
    [
        "BreakoutValidator",
        "BreakoutType",
        "BreakoutOutcome",
        "BreakoutQuality",
        "BreakoutEvent",
        "BreakoutFeatures",
        "BreakoutValidation",
        "BreakoutBacktester",
        "BreakoutBacktestResult",
        "is_breakout_valid",
    ],
)

_register(
    ".engines.signals",
    [
        "Signal",
        "coerce_signal",
        "signal_value",
    ],
)

_register(
    ".engines.market_structure",
    [
        "MarketStructureDetector",
        "MarketStructureState",
        "SwingPoint",
        "SwingType",
        "TrendDirection",
        "StructureEvent",
        "RangeType",
        "StructuralMomentum",
        "TimeframeAlignment",
        "FairValueGap",
        "StructureBreak",
        "get_allowed_trade_direction",
        "structure_veto_signal",
    ],
)

_register(
    ".engines.institutional_structure",
    [
        "MarketStructureEngine",
        "StructureConfig",
        "Candle",
        "StructureState",
        "StructureSide",
        "EventType",
        "RangeClassification",
        "TradingMode",
        "ZoneStatus",
        "Zone",
        "MultiTFAlignment",
        "compute_atr",
        "compute_atr_pct",
        "compute_rv",
        "sma",
    ],
)

_register(
    ".engines.ema_filter",
    [
        "EMAFilterEngine",
        "EMAConfig",
        "EMAState",
        "EMABias",
        "EMARegime",
        "EMAAlignment",
        "EMAMultiTFState",
        "create_candle",
        "print_ema_state",
        "print_ema_block",
        "compute_sma",
    ],
    aliases={"EMAMTFAlignment": "MTFAlignment"},
)


_missing_exports = [name for name in __all__ if name not in _EXPORT_TO_SOURCE]
if _missing_exports:
    raise RuntimeError(f"Lazy export map incomplete: {_missing_exports}")


def __getattr__(name: str):
    if name not in _EXPORT_TO_SOURCE:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, symbol_name = _EXPORT_TO_SOURCE[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, symbol_name)

    # Cache resolved symbol on module globals for subsequent fast access.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
