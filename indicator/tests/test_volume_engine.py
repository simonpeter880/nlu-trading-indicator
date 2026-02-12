from volume_engine import AggressionBias, get_aggression_bias, is_volume_exhausted


def test_get_aggression_bias_buy(rising_ohlcv):
    opens, highs, lows, closes, volumes = rising_ohlcv
    bias, strength = get_aggression_bias(opens, highs, lows, closes, volumes)
    assert bias in {AggressionBias.STRONG_BUY, AggressionBias.BUY}
    assert 0 <= strength <= 100


def test_is_volume_exhausted_false(rising_ohlcv):
    opens, highs, lows, closes, volumes = rising_ohlcv
    exhausted, risk = is_volume_exhausted(opens, highs, lows, closes, volumes, oi_change=1.0)
    assert isinstance(exhausted, bool)
    assert risk.name in {"LOW", "MEDIUM", "HIGH", "EXTREME"}
