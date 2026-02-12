# Development Guide

This guide helps developers set up their environment and understand the project structure for effective development on NLU Trading Indicators.

## Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Architecture Overview](#architecture-overview)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Debugging](#debugging)
- [Common Tasks](#common-tasks)
- [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# Clone and setup
git clone https://github.com/simonpeter880/nlu-trading-indicator.git
cd nlu-trading-indicator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install everything
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# You're ready to develop! 🚀
```

## Project Structure

```
nlu-trading-indicator/
├── nlu_analyzer/              # High-level indicator engines
│   ├── indicators/            # Core indicator implementations
│   │   ├── ema_ribbon.py     # EMA-based trend analysis
│   │   ├── roc_momentum.py   # Rate of change momentum
│   │   ├── supertrend_filter.py  # Supertrend calculations
│   │   ├── trend_strength.py # Trend strength analysis
│   │   ├── vwap_engine.py    # VWAP calculations
│   │   └── vwap_state_machine.py # VWAP state logic
│   └── integrations/          # Integration examples
│       ├── ema_ribbon_integration.py
│       ├── roc_momentum_integration.py
│       └── supertrend_integration.py
│
├── indicator/                 # Core analysis engines
│   ├── engines/              # Technical analysis engines
│   │   ├── indicators.py     # ~1,130 lines - Basic indicators
│   │   ├── volume_engine.py  # ~1,020 lines - Volume analysis
│   │   ├── institutional_structure.py  # ~1,200 lines - Market structure
│   │   ├── ema_filter.py     # EMA filtering logic
│   │   ├── rsi_timing.py     # RSI-based timing
│   │   ├── data_fetcher.py   # Binance API integration
│   │   └── ... (20+ indicator engines)
│   ├── continuous/           # Real-time streaming
│   │   ├── engine_adapters.py  # ~1,215 lines - Adapters
│   │   ├── orchestrator.py   # Data orchestration
│   │   ├── state_machine.py  # State management
│   │   └── ...
│   ├── display/              # Output formatting
│   │   ├── printers.py       # ~1,298 lines - Display logic
│   │   ├── colors.py         # Terminal colors
│   │   └── formatters.py     # Formatting utilities
│   ├── apps/                 # Example applications
│   │   ├── analyze.py        # CLI analysis tool
│   │   └── continuous_runner.py  # Streaming example
│   └── integrations/         # Integration helpers
│
├── tests/                    # Test suite
│   ├── test_*.py            # Unit tests
│   └── integration/         # Integration tests
│
├── docs/                     # Documentation
│   └── indicator_engines/   # Engine-specific docs
│
├── .github/                  # GitHub configuration
│   └── workflows/           # CI/CD pipelines
│       └── ci.yml           # Main CI workflow
│
├── pyproject.toml           # Package configuration
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
├── .pre-commit-config.yaml  # Pre-commit hooks config
├── .gitignore              # Git ignore rules
├── README.md               # Project readme
├── CONTRIBUTING.md         # Contribution guidelines
├── DEVELOPMENT.md          # This file
├── CHANGELOG.md            # Version history
└── .env.example            # Environment template
```

## Architecture Overview

### Package Organization

The project is organized into two main packages:

1. **`nlu_analyzer`**: High-level, user-friendly indicator engines
   - Self-contained indicator implementations
   - Simple APIs for common use cases
   - Minimal dependencies between modules

2. **`indicator`**: Core analysis infrastructure
   - Lower-level building blocks
   - Shared utilities and calculations
   - Complex multi-indicator systems

### Key Design Patterns

#### 1. Engine Pattern

Most indicators follow an "engine" pattern:

```python
class IndicatorEngine:
    def __init__(self, config: Config):
        self.config = config
        self.state = State()

    def update(self, candle: Candle) -> IndicatorState:
        # Process new data
        # Update internal state
        # Return current state
        pass

    def reset(self) -> None:
        # Reset to initial state
        self.state = State()
```

#### 2. State Machine Pattern

Complex indicators use state machines:

```python
@dataclass
class State:
    trend: Trend
    strength: float
    last_signal: Signal

class StateMachine:
    def transition(self, event: Event) -> State:
        # Handle state transitions
        pass
```

#### 3. Configuration Pattern

All configurable parameters use dataclasses:

```python
@dataclass
class IndicatorConfig:
    period: int = 14
    threshold: float = 0.7

    def __post_init__(self):
        # Validate configuration
        if self.period < 1:
            raise ValueError("period must be positive")
```

### Data Flow

```
External Data (Binance API)
    ↓
DataFetcher
    ↓
Candle Objects (OHLCV)
    ↓
Individual Indicator Engines
    ↓
Indicator States/Results
    ↓
Unified Score Aggregator
    ↓
Trading Signals
    ↓
Display/Output
```

## Development Setup

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

```env
# Binance API (optional - for live data)
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here

# Data source
DATA_SOURCE=binance_live  # or local_csv, mock

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/nlu_trading.log

# Application
DEFAULT_TIMEFRAME=1h
MAX_CANDLES=500
```

### IDE Configuration

#### VS Code

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".mypy_cache": true,
        "*.egg-info": true
    }
}
```

Create `.vscode/launch.json` for debugging:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal"
        },
        {
            "name": "Python: Pytest Current File",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["${file}", "-v"],
            "console": "integratedTerminal"
        },
        {
            "name": "Python: Run Analyzer",
            "type": "python",
            "request": "launch",
            "module": "indicator.apps.analyze",
            "console": "integratedTerminal"
        }
    ]
}
```

#### PyCharm

1. Open project in PyCharm
2. Set interpreter: File → Settings → Project → Python Interpreter → Add → Virtualenv Environment → Existing
3. Enable pytest: Settings → Tools → Python Integrated Tools → Testing → pytest
4. Configure Black: Settings → Tools → Black → Enable
5. Enable type checking: Settings → Editor → Inspections → Python → Type Checker

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_indicators.py

# Run specific test
pytest tests/test_indicators.py::TestRSI::test_oversold

# Run tests matching pattern
pytest -k "rsi or macd"

# Run only failed tests from last run
pytest --lf

# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

### Coverage Analysis

```bash
# Run with coverage
pytest --cov=nlu_analyzer --cov=indicator

# Generate HTML coverage report
pytest --cov=nlu_analyzer --cov=indicator --cov-report=html
open htmlcov/index.html

# Coverage for specific module
pytest --cov=indicator.engines.indicators tests/test_indicators.py

# Show missing lines
pytest --cov --cov-report=term-missing

# Fail if coverage below threshold
pytest --cov --cov-fail-under=70
```

### Test Organization

```bash
tests/
├── test_indicators.py          # Core indicators
├── test_volume_engine.py       # Volume analysis
├── test_ema_ribbon.py          # EMA ribbon
├── test_roc_momentum.py        # ROC momentum
├── conftest.py                 # Shared fixtures
└── integration/                # Integration tests
    ├── test_full_pipeline.py
    └── test_streaming.py
```

## Debugging

### Using pdb

Insert breakpoint in code:

```python
import pdb; pdb.set_trace()
```

Or use Python 3.7+ built-in:

```python
breakpoint()
```

### Using pytest debugging

```bash
# Drop into pdb on failure
pytest --pdb

# Drop into pdb on first failure, then exit
pytest -x --pdb

# Show print statements
pytest -s
```

### Logging

Configure logging in your code:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

## Common Tasks

### Adding a New Indicator

1. **Create indicator file**

```python
# nlu_analyzer/indicators/my_indicator.py
from dataclasses import dataclass
from typing import List

@dataclass
class MyIndicatorState:
    value: float
    signal: str
    timestamp: int

class MyIndicatorEngine:
    def __init__(self, period: int = 14):
        self.period = period

    def update(self, candle: Candle) -> MyIndicatorState:
        # Implement your logic
        pass

    def reset(self) -> None:
        # Reset state
        pass
```

2. **Add tests**

```python
# tests/test_my_indicator.py
import pytest
from nlu_analyzer.indicators.my_indicator import MyIndicatorEngine

class TestMyIndicator:
    def test_basic_calculation(self):
        engine = MyIndicatorEngine(period=14)
        result = engine.update(sample_candle)
        assert result.value > 0
```

3. **Update package exports**

```python
# nlu_analyzer/indicators/__init__.py
from .my_indicator import MyIndicatorEngine, MyIndicatorState

__all__ = ["MyIndicatorEngine", "MyIndicatorState", ...]
```

4. **Add documentation**

Update README.md and add example usage.

### Running the Analyzer

```bash
# Interactive mode
python -m indicator.apps.analyze

# Programmatic usage
python -c "
from indicator.engines.indicators import MomentumIndicators
closes = [100, 101, 102, 103, 104]
result = MomentumIndicators.analyze_rsi(closes)
print(result)
"
```

### Updating Dependencies

```bash
# Check for outdated packages
pip list --outdated

# Update specific package
pip install --upgrade package-name

# Update requirements file
pip freeze > requirements.txt

# Or manually edit requirements.txt and reinstall
pip install -r requirements.txt
```

### Generating Documentation

```bash
# Install Sphinx (if not already)
pip install sphinx sphinx-rtd-theme

# Generate API documentation
cd docs
sphinx-apidoc -o source/ ../nlu_analyzer/ ../indicator/
make html

# View docs
open build/html/index.html
```

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'nlu_analyzer'`

**Solution**:
```bash
pip install -e .
```

### Test Failures

**Problem**: Tests fail with relative import errors

**Solution**: Make sure you're running tests from project root:
```bash
cd /path/to/nlu-trading-indicator
pytest
```

### Pre-commit Hooks Failing

**Problem**: `flake8` or `black` fails on commit

**Solution**:
```bash
# Auto-fix formatting
black .
isort .

# Check what's wrong
flake8 path/to/file.py

# Skip hooks (not recommended)
git commit --no-verify
```

### Type Checking Errors

**Problem**: `mypy` reports type errors

**Solution**:
```bash
# Run mypy to see errors
mypy nlu_analyzer/ indicator/

# Add type: ignore comment for unavoidable issues
result = legacy_function()  # type: ignore

# Or add to mypy.ini
[mypy-problematic_module.*]
ignore_errors = True
```

### Performance Issues

**Problem**: Tests or indicator calculations are slow

**Solution**:
```bash
# Profile code
python -m cProfile -s cumulative your_script.py

# Use line_profiler for detailed profiling
pip install line_profiler
kernprof -l -v your_script.py

# Time specific functions
import time
start = time.time()
result = slow_function()
print(f"Took {time.time() - start:.2f}s")
```

## Additional Resources

- [Python Best Practices](https://docs.python-guide.org/)
- [pytest Documentation](https://docs.pytest.org/)
- [Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
- [Black Code Style](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)

## Getting Help

- Check existing issues: https://github.com/simonpeter880/nlu-trading-indicator/issues
- Ask in discussions: https://github.com/simonpeter880/nlu-trading-indicator/discussions
- Read the code - it's well-documented!

Happy developing! 🚀
