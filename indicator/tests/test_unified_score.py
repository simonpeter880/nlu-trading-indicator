from unified_score import calculate_unified_score


def test_calculate_unified_score_long_bias():
    score = calculate_unified_score(
        delta_ratio=0.3,
        relative_volume=3.0,
        price_change_pct=1.5,
        oi_change_pct=2.0,
        depth_imbalance=0.5,
    )
    assert score.bias == "long"
    assert score.total_score > 0.55
