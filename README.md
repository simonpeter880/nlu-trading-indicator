# NLU Trading Indicators

[![CI/CD Pipeline](https://github.com/simonpeter880/nlu-trading-indicator/actions/workflows/ci.yml/badge.svg)](https://github.com/simonpeter880/nlu-trading-indicator/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/simonpeter880/nlu-trading-indicator/branch/main/graph/badge.svg)](https://codecov.io/gh/simonpeter880/nlu-trading-indicator)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive Python-based cryptocurrency trading indicator analysis system designed for institutional-grade technical analysis of Binance trading pairs.

## Features

- **20+ Advanced Technical Indicators**: Market structure, volume analysis, momentum patterns, and institutional order flow detection
- **Multi-Timeframe Analysis**: Synchronize and analyze across multiple timeframes
- **Institutional Volume Analysis**: Detect smart money flow and institutional activity
- **Market Structure Detection**: Identify swing points, trends, and structure breaks
- **Unified Scoring System**: Combine multiple indicators into actionable signals
- **Real-time and Historical Analysis**: Support for both live streaming and backtesting

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Start

```bash
# Clone the repository
git clone https://github.com/simonpeter880/nlu-trading-indicator.git
cd nlu-trading-indicator

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run tests with coverage
pytest --cov
```

## Configuration

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` to add your Binance API credentials (optional, only needed for live data):

```bash
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
DATA_SOURCE=binance_live
```

## Usage

### Basic Analysis

```python
from nlu_analyzer.indicators import EMARibbonEngine, VWAPEngine
from indicator.engines.data_fetcher import DataFetcher

# Fetch OHLCV data
fetcher = DataFetcher()
ohlcv = await fetcher.fetch_ohlcv("BTCUSDT", "1h", limit=500)

# Run EMA Ribbon analysis
ema_engine = EMARibbonEngine()
ema_state = ema_engine.update(ohlcv[-1])
print(f"EMA Trend: {ema_state.trend}")

# Run VWAP analysis
vwap_engine = VWAPEngine()
vwap_state = vwap_engine.update(ohlcv[-1])
print(f"Price vs VWAP: {vwap_state.position}")
```

### Continuous Analysis

```python
from indicator.continuous.runner import ContinuousRunner

# Start continuous analysis for multiple pairs
runner = ContinuousRunner(
    pairs=["BTCUSDT", "ETHUSDT"],
    timeframes=["1m", "5m", "1h"],
    update_interval=60
)

await runner.start()
```

## Project Structure

```
nlu-trading-indicator/
├── nlu_analyzer/          # High-level indicator engines
│   ├── indicators/        # Individual indicator implementations
│   └── integrations/      # Integration helpers and examples
├── indicator/             # Core analysis engines
│   ├── engines/          # Technical analysis engines
│   ├── continuous/       # Real-time streaming adapters
│   └── display/          # Output formatting and printing
├── tests/                # Test suite
├── docs/                 # Documentation
└── .github/              # CI/CD workflows
```

## Key Indicators

### Trend Analysis
- **EMA Ribbon**: Multi-timeframe exponential moving average analysis
- **Supertrend Filter**: Trend direction and strength detection
- **Trend Strength**: Signed strength combining trend and momentum

### Volume Analysis
- **Institutional Volume Engine**: Detect large player activity
- **VWAP Engine**: Volume-weighted average price analysis
- **Delta Volume**: Buy/sell pressure analysis

### Momentum & Timing
- **ROC Momentum**: Rate of change with regime detection
- **RSI Timing**: Overbought/oversold with divergence detection
- **Stochastic Oscillator**: Momentum and potential reversals

### Market Structure
- **Institutional Structure**: Swing points, trends, and order blocks
- **Volume Analysis**: Volume profile and distribution

## Testing

The project includes comprehensive test coverage:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov --cov-report=html

# Run specific test file
pytest tests/test_ema_ribbon.py -v

# Run integration tests
pytest tests/integration/ -v
```

## Code Quality

This project follows strict code quality standards:

- **Formatting**: Black (PEP 8 compliant)
- **Import Sorting**: isort
- **Linting**: flake8 with PEP 257 docstrings (legacy issues being addressed)
- **Type Checking**: mypy for static type analysis
- **Security**: bandit for security vulnerability scanning
- **Pre-commit Hooks**: Automated checks before every commit

All checks run automatically in CI/CD on every push and pull request.

**Note**: This project is actively being modernized. Some flake8 warnings (E501 line length, D401 docstrings) exist in legacy code and will be addressed incrementally without affecting functionality.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and quality checks (`pytest && black . && flake8 .`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to your fork (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Documentation

- [Indicator Engines Documentation](docs/indicator_engines/)
- [Development Guide](DEVELOPMENT.md)
- [API Reference](https://nlu-trading-indicator.readthedocs.io/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Python 3.8+
- Uses aiohttp for asynchronous HTTP requests
- Powered by advanced technical analysis algorithms

## Support

For questions, issues, or feature requests, please [open an issue](https://github.com/simonpeter880/nlu-trading-indicator/issues) on GitHub.

---

**⚠️ Disclaimer**: This software is for educational and research purposes only. It is not financial advice. Trading cryptocurrencies carries risk. Always do your own research and never invest more than you can afford to lose.
