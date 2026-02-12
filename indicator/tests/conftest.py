import os
import sys

import pytest


TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def rising_ohlcv():
    """Simple upward-trending OHLCV series."""
    opens = [100 + i for i in range(20)]
    highs = [o + 1 for o in opens]
    lows = [o - 1 for o in opens]
    closes = [h for h in highs]  # close at highs to bias buy aggression
    volumes = [1000 + i * 10 for i in range(20)]
    return opens, highs, lows, closes, volumes


@pytest.fixture
def rising_closes():
    """Monotonic increasing closes for trend tests."""
    return [float(i) for i in range(1, 61)]
