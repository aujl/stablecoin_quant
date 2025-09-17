from stable_yield_lab.core.models import Pool
from stable_yield_lab.risk_scoring import calculate_risk_score, score_pool


def test_calculate_risk_score_extremes() -> None:
    assert calculate_risk_score(1.0, 5, 0.0) == 1.0
    assert calculate_risk_score(0.0, 0, 1.0) == 3.0
    # Values outside expected ranges are clamped
    assert calculate_risk_score(2.0, -1, -5.0) == calculate_risk_score(1.0, 0, 0.0)


def test_score_pool_defaults() -> None:
    pool = Pool("Test", "Unknown", "USDC", 1.0, 0.1)
    scored = score_pool(pool)
    assert scored.risk_score == 2.0
