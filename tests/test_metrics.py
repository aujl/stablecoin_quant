import math

import pytest

from stable_yield_lab import Metrics, Pool, PoolRepository


def test_net_apy_flat_fee() -> None:
    val = Metrics.net_apy(0.10, 0.02, perf_fee_bps=200, mgmt_fee_bps=100)
    assert val == pytest.approx(0.12 * (1 - 0.01) * (1 - 0.02), rel=1e-12)


def test_net_apy_compounds_realised_returns() -> None:
    realised = [0.01, 0.02, -0.005]
    val = Metrics.net_apy(0.0, realized_returns=realised)
    expected = math.prod(1 + r for r in realised) - 1
    assert val == pytest.approx(expected, rel=1e-12)


def test_net_apy_tiered_fee_schedule() -> None:
    perf_schedule = [(0.05, 100), (0.10, 150), (float("inf"), 200)]
    mgmt_schedule = [(0.05, 25), (float("inf"), 100)]
    val = Metrics.net_apy(
        0.10,
        0.02,
        perf_fee_schedule=perf_schedule,
        mgmt_fee_schedule=mgmt_schedule,
    )
    expected = 0.12 * (1 - 0.01) * (1 - 0.02)
    assert val == pytest.approx(expected, rel=1e-12)


def test_net_apy_handles_negative_periods() -> None:
    realised = [-0.10, 0.05]
    val = Metrics.net_apy(0.0, realized_returns=realised)
    expected = math.prod(1 + r for r in realised) - 1
    assert val == pytest.approx(expected, rel=1e-12)


def test_hhi_basic() -> None:
    repo = PoolRepository(
        [
            Pool(name="A", chain="X", stablecoin="USDC", tvl_usd=50, base_apy=0.1),
            Pool(name="B", chain="X", stablecoin="USDT", tvl_usd=30, base_apy=0.1),
            Pool(name="C", chain="Y", stablecoin="DAI", tvl_usd=20, base_apy=0.1),
        ]
    )
    df = repo.to_dataframe()
    hhi_total = Metrics.hhi(df, value_col="tvl_usd")
    # Shares 0.5, 0.3, 0.2 -> HHI = 0.25 + 0.09 + 0.04 = 0.38
    assert abs(float(hhi_total["hhi"][0]) - 0.38) < 1e-9
