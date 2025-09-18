from __future__ import annotations

import math

import pandas as pd
import pytest

from stable_yield_lab.analytics.metrics import (
    Metrics,
    add_net_apy_column,
    hhi,
    net_apy,
    weighted_mean,
)
from stable_yield_lab.core import Pool, PoolRepository


def _sample_repository() -> PoolRepository:
    return PoolRepository(
        [
            Pool(name="A", chain="X", stablecoin="USDC", tvl_usd=50, base_apy=0.10),
            Pool(name="B", chain="X", stablecoin="USDT", tvl_usd=30, base_apy=0.08),
            Pool(name="C", chain="Y", stablecoin="DAI", tvl_usd=20, base_apy=0.07),
        ]
    )


def test_weighted_mean_matches_manual_result() -> None:
    values = [0.10, 0.08, 0.07]
    weights = [50, 30, 20]
    expected = sum(v * w for v, w in zip(values, weights)) / sum(weights)
    assert weighted_mean(values, weights) == pytest.approx(expected)


def test_weighted_mean_skips_nan_and_requires_positive_weight_sum() -> None:
    values = [0.1, math.nan, 0.05]
    weights = [0.0, 25.0, 0.0]
    assert math.isnan(weighted_mean(values, weights))

    values = [0.1, math.nan, 0.05]
    weights = [40.0, 60.0, 0.0]
    assert weighted_mean(values, weights) == pytest.approx(0.1)


def test_weighted_mean_mismatched_lengths_return_nan() -> None:
    assert math.isnan(weighted_mean([1.0, 2.0], [1.0]))
    assert math.isnan(Metrics.weighted_mean([], []))


def test_net_apy_applies_fees_and_clamps_losses() -> None:
    base = 0.10
    reward = 0.02
    val = net_apy(base, reward, perf_fee_bps=200, mgmt_fee_bps=100)
    gross = base + reward
    expected = (1.0 + gross) * 0.97 - 1.0
    assert val == pytest.approx(expected)

    stressed = net_apy(base, reward, perf_fee_bps=8_000, mgmt_fee_bps=4_000)
    assert stressed == -1.0


def test_net_apy_nan_inputs_propagate() -> None:
    assert math.isnan(net_apy(math.nan, 0.01))
    assert math.isnan(net_apy(0.05, 0.01, perf_fee_bps=math.nan))


def test_add_net_apy_column_handles_missing_columns() -> None:
    df = pd.DataFrame({"base_apy": [0.1, math.nan]})
    result = add_net_apy_column(df, perf_fee_bps=100)
    assert "net_apy" in result
    assert math.isnan(result.loc[1, "net_apy"])


def test_add_net_apy_column_extreme_fees() -> None:
    df = pd.DataFrame({"base_apy": [0.12], "reward_apy": [0.04]})
    result = add_net_apy_column(df, perf_fee_bps=8_000, mgmt_fee_bps=4_000)
    assert result.loc[0, "net_apy"] == -1.0


def test_hhi_total_and_grouped() -> None:
    repo = _sample_repository()
    df = repo.to_dataframe()

    hhi_total = hhi(df, value_col="tvl_usd")
    assert hhi_total.shape == (1, 1)
    assert hhi_total.iloc[0, 0] == pytest.approx(0.38)

    grouped = hhi(df, value_col="tvl_usd", group_col="chain")
    assert set(grouped["chain"]) == {"X", "Y"}
    assert grouped.loc[grouped["chain"] == "Y", "hhi"].iloc[0] == pytest.approx(1.0)


def test_hhi_zero_total_returns_nan() -> None:
    df = pd.DataFrame({"value": [0.0, 0.0]})
    result = hhi(df, value_col="value")
    assert math.isnan(result.loc[0, "hhi"])

    grouped = hhi(pd.DataFrame({"value": [0.0], "group": ["A"]}), value_col="value", group_col="group")
    assert math.isnan(grouped.loc[0, "hhi"])
