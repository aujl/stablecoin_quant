from stable_yield_lab import Metrics
from stable_yield_lab.core import Pool, PoolRepository


def test_net_apy() -> None:
    base = 0.10
    reward = 0.02
    val = Metrics.net_apy(base, reward, perf_fee_bps=200, mgmt_fee_bps=100)
    gross = base + reward
    expected = (1.0 + gross) * 0.97 - 1.0
    assert abs(val - expected) < 1e-12


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
