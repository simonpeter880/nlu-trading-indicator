# Refactoring TODO

## Large Files to Break Down (Future Work)

The following files exceed 1,000 lines and should be refactored into smaller, more maintainable modules:

### 1. `indicator/display/printers.py` (1,298 lines)
**Suggested split:**
- `indicator/display/volume_printers.py` - Volume-related printing functions
- `indicator/display/structure_printers.py` - Structure-related printing
- `indicator/display/score_printers.py` - Unified score printing
- `indicator/display/formatters.py` - Common formatting utilities

### 2. `indicator/continuous/engine_adapters.py` (1,215 lines)
**Suggested split by adapter classes:**
- `indicator/continuous/adapters/volume_adapter.py` - VolumeEngineAdapter (~287 lines)
- `indicator/continuous/adapters/delta_adapter.py` - DeltaEngineAdapter (~173 lines)
- `indicator/continuous/adapters/volume_analysis_adapter.py` - VolumeAnalysisAdapter (~91 lines)
- `indicator/continuous/adapters/book_adapter.py` - BookEngineAdapter (~135 lines)
- `indicator/continuous/adapters/oi_funding_adapter.py` - OIFundingEngineAdapter (~312 lines)
- `indicator/continuous/adapters/unified_score_adapter.py` - UnifiedScoreAdapter (~142 lines)
- `indicator/continuous/adapters/common.py` - Shared helper functions

### 3. `indicator/engines/institutional_structure.py` (1,199 lines)
**Suggested split:**
- `indicator/engines/structure/enums.py` - All Enum definitions (SwingType, StructureState, etc.)
- `indicator/engines/structure/models.py` - Data classes (SwingPoint, Zone, etc.)
- `indicator/engines/structure/helpers.py` - Helper functions (compute_atr, sma, etc.)
- `indicator/engines/structure/engine.py` - Main MarketStructureEngine class

### 4. `indicator/engines/indicators.py` (1,130 lines)
**Suggested split by indicator type:**
- `indicator/engines/volume_indicators.py` - Volume-based indicators
- `indicator/engines/trend_indicators.py` - Trend-following indicators
- `indicator/engines/momentum_indicators.py` - Momentum oscillators
- `indicator/engines/volatility_indicators.py` - Volatility measures

### 5. `indicator/engines/volume_engine.py` (1,020 lines)
**Suggested split:**
- `indicator/engines/volume/models.py` - Data classes and enums
- `indicator/engines/volume/analysis.py` - Analysis functions
- `indicator/engines/volume/engine.py` - Main InstitutionalVolumeEngine class

## Benefits of Refactoring

- **Improved maintainability**: Easier to locate and modify specific functionality
- **Better testing**: Smaller modules are easier to test in isolation
- **Reduced cognitive load**: Developers can focus on one aspect at a time
- **Faster IDE performance**: Smaller files load and parse faster
- **Easier code review**: Smaller diffs in pull requests

## Implementation Notes

- Keep all existing imports working (use `__init__.py` re-exports)
- Ensure backward compatibility for external users
- Add deprecation warnings if needed
- Update tests incrementally
- Run full test suite after each split

## Priority

- **Low priority** - Current files are functional and well-tested
- Consider refactoring when adding major new features to these modules
- Can be done incrementally over multiple releases
