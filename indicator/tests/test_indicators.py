from indicators import TrendIndicators
from signals import Signal


def test_calculate_sma():
    prices = [1, 2, 3, 4, 5]
    assert TrendIndicators.calculate_sma(prices, 3) == [2.0, 3.0, 4.0]


def test_calculate_ema():
    prices = [1, 2, 3, 4, 5]
    assert TrendIndicators.calculate_ema(prices, 3) == [2.0, 3.0, 4.0]


def test_analyze_moving_averages_bullish(rising_closes):
    result = TrendIndicators.analyze_moving_averages(rising_closes, short_period=5, long_period=10)
    assert result.signal == Signal.BULLISH
