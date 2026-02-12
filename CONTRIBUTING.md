# Contributing to NLU Trading Indicators

Thank you for your interest in contributing to NLU Trading Indicators! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Requirements](#testing-requirements)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Welcome newcomers and help them learn
- Assume good intentions

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic understanding of technical analysis (for indicator contributions)

### Setting Up Your Development Environment

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/nlu-trading-indicator.git
cd nlu-trading-indicator
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

4. **Install pre-commit hooks**

```bash
pre-commit install
```

5. **Verify setup**

```bash
pytest  # Run tests
black --check .  # Check formatting
mypy nlu_analyzer/ indicator/  # Type checking
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Adding or updating tests

### 2. Make Your Changes

- Write clean, readable code
- Add docstrings to all public functions/classes
- Include type hints where applicable
- Update tests for your changes
- Update documentation if needed

### 3. Run Quality Checks

Before committing, ensure all checks pass:

```bash
# Format code
black .
isort .

# Lint
flake8 nlu_analyzer/ indicator/ tests/

# Type check
mypy nlu_analyzer/ indicator/ --ignore-missing-imports

# Security scan
bandit -r nlu_analyzer/ indicator/

# Run tests with coverage
pytest --cov --cov-report=term-missing
```

### 4. Commit Your Changes

```bash
git add .
git commit -m "Your descriptive commit message"
```

Pre-commit hooks will automatically run. Fix any issues they report.

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a pull request on GitHub.

## Code Style Guidelines

### Python Style

We follow PEP 8 with these specific guidelines:

- **Line length**: 88 characters (Black default)
- **Imports**: Sorted with isort
- **Quotes**: Use double quotes for strings
- **Docstrings**: PEP 257 compliant, with periods at the end

### Docstring Format

```python
def calculate_indicator(prices: List[float], period: int = 14) -> float:
    """
    Calculate a technical indicator value.

    Args:
        prices: List of historical prices
        period: Lookback period for calculation

    Returns:
        Calculated indicator value

    Raises:
        ValueError: If prices list is empty or period is invalid

    Example:
        >>> prices = [100, 101, 102, 103]
        >>> calculate_indicator(prices, period=2)
        101.5
    """
    pass
```

### Type Hints

Always include type hints for function parameters and return values:

```python
from typing import List, Optional, Tuple

def process_data(
    data: List[float],
    threshold: Optional[float] = None
) -> Tuple[float, str]:
    """Process data and return result."""
    pass
```

### Error Handling

- Use specific exception types
- Add context to error messages
- Don't catch exceptions you can't handle

```python
# Good
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Invalid value in calculation: {e}")
    raise

# Avoid
try:
    result = risky_operation()
except Exception:  # Too broad
    pass  # Silent failure
```

## Testing Requirements

### Test Coverage

- New features must include tests
- Bug fixes should include a regression test
- Aim for 70%+ coverage on new code
- Critical paths should have 90%+ coverage

### Test Structure

```python
class TestFeatureName:
    """Tests for feature X."""

    def test_normal_case(self):
        """Test typical usage scenario."""
        result = function_under_test(valid_input)
        assert result == expected_output

    def test_edge_case(self):
        """Test boundary conditions."""
        result = function_under_test(edge_case_input)
        assert result is not None

    def test_error_handling(self):
        """Test error conditions."""
        with pytest.raises(ValueError):
            function_under_test(invalid_input)
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_indicators.py

# Run with coverage
pytest --cov=nlu_analyzer --cov=indicator --cov-report=html

# Run only failed tests from last run
pytest --lf

# Run tests matching pattern
pytest -k "test_rsi"
```

## Commit Message Guidelines

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```
feat(indicators): add Ichimoku Cloud indicator

Implement full Ichimoku Cloud calculation including:
- Tenkan-sen (Conversion Line)
- Kijun-sen (Base Line)
- Senkou Span A and B (Leading Spans)
- Chikou Span (Lagging Span)

Closes #123
```

```
fix(volume_engine): handle zero volume gracefully

Previously crashed with division by zero when volume was 0.
Now returns neutral signal with appropriate warning message.

Fixes #456
```

## Pull Request Process

### Before Submitting

1. ✅ All tests pass locally
2. ✅ Code is formatted (black, isort)
3. ✅ No linting errors (flake8)
4. ✅ Type checking passes (mypy)
5. ✅ Documentation is updated
6. ✅ CHANGELOG.md is updated (if applicable)

### PR Title Format

Use the same format as commit messages:

```
feat(scope): brief description
```

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Added new tests
- [ ] All tests pass
- [ ] Coverage maintained/improved

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

### Review Process

1. Automated CI checks must pass
2. At least one maintainer review required
3. Address review comments
4. Squash commits if requested
5. Maintainer will merge when approved

## Reporting Issues

### Bug Reports

Include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces
- Code snippet (if applicable)

### Feature Requests

Include:
- Clear description of the feature
- Use case / motivation
- Proposed API / interface
- Alternatives considered

### Issue Template

```markdown
**Environment:**
- Python version: 3.11
- OS: Ubuntu 22.04
- Package version: 0.1.0

**Description:**
Clear description of the issue

**Steps to Reproduce:**
1. Step one
2. Step two
3. Step three

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Code Sample:**
```python
# Minimal code to reproduce
```

**Error Message:**
```
Full error traceback
```
```

## Development Tips

### Useful Commands

```bash
# Run tests in watch mode (requires pytest-watch)
ptw

# Generate coverage report
pytest --cov --cov-report=html
open htmlcov/index.html

# Check coverage for specific file
pytest --cov=indicator/engines/indicators.py tests/test_indicators.py

# Run linting on changed files only
git diff --name-only | grep '\.py$' | xargs flake8

# Update dependencies
pip list --outdated
pip install --upgrade package-name
```

### IDE Setup

#### VS Code

Recommended extensions:
- Python (Microsoft)
- Pylance
- Black Formatter
- isort
- GitLens

Settings (.vscode/settings.json):
```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.linting.mypyEnabled": true
}
```

#### PyCharm

1. Settings → Tools → Black → "Enable black formatter"
2. Settings → Editor → Code Style → Python → Set line length to 88
3. Settings → Tools → Python Integrated Tools → Default test runner: pytest

## Getting Help

- 📖 Read the [documentation](docs/)
- 💬 Ask questions in [GitHub Discussions](https://github.com/simonpeter880/nlu-trading-indicator/discussions)
- 🐛 Report bugs in [GitHub Issues](https://github.com/simonpeter880/nlu-trading-indicator/issues)
- 📧 Email: (add maintainer email if desired)

## Recognition

Contributors will be:
- Listed in CHANGELOG.md for their contributions
- Added to CONTRIBUTORS.md
- Recognized in release notes

Thank you for contributing to NLU Trading Indicators! 🎉
