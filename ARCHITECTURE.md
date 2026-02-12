# NLU Project Architecture

This repository now uses a package-first layout for the indicator engines that were previously scattered at the repository root.

## Top-level layout

- `nlu_analyzer/`
  - Main package for reusable analyzer components.
- `nlu_analyzer/indicators/`
  - Core engines: EMA ribbon, VWAP, Supertrend, Trend Strength, ROC momentum.
- `nlu_analyzer/integrations/`
  - Integration examples and adapters for streaming/continuous workflows.
- `nlu_analyzer/tools/`
  - Utility display helpers (for example compact ROC printers).
- `docs/indicator_engines/`
  - Documentation for the core engines and integration snippets.
- `indicator/`
  - Secondary indicator suite, now organized into subpackages:
    - `indicator/engines/`
    - `indicator/integrations/`
    - `indicator/examples/`
    - `indicator/apps/`
    - `indicator/docs/`
- `tests/`
  - Tests for the core engines.
- `indicator/tests/`
  - Tests for the `indicator/` suite.

## Backward compatibility

Root-level module names are preserved as shims:

- `ema_ribbon.py`
- `vwap_engine.py`
- `vwap_state_machine.py`
- `supertrend_filter.py`
- `trend_strength.py`
- `roc_momentum.py`
- `*_integration.py`
- `ROC_MOMENTUM_COMPACT_PRINT.py`

These files re-export from `nlu_analyzer` so old imports and scripts continue to work.

## Rules for new code

1. Put reusable engine logic in `nlu_analyzer/indicators/`.
2. Put adapters/examples in `nlu_analyzer/integrations/`.
3. Put display/utility helpers in `nlu_analyzer/tools/`.
4. Keep root-level files as compatibility shims only.
5. Add or update tests in `tests/` (or `indicator/tests/` for that suite).
