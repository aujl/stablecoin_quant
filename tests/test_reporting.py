from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stable_yield_lab import Pool, PoolRepository
from stable_yield_lab.reporting import cross_section_report


@pytest.mark.parametrize("with_returns", [False, True])
@pytest.mark.parametrize("with_horizon", [False, True])
def test_cross_section_report_history_outputs(
    tmp_path: Path, with_returns: bool, with_horizon: bool
) -> None:
    repo = PoolRepository(
        [
            Pool(
                name="PoolA",
                chain="Ethereum",
                stablecoin="USDC",
                tvl_usd=1_000_000,
                base_apy=0.03,
                timestamp=1_700_000_000,
            ),
            Pool(
                name="PoolB",
                chain="Ethereum",
                stablecoin="DAI",
                tvl_usd=500_000,
                base_apy=0.02,
                timestamp=1_700_000_000,
            ),
        ]
    )

    returns = pd.DataFrame(
        {
            "PoolA": [0.01, 0.02, 0.015],
            "PoolB": [0.005, -0.002, 0.01],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"], utc=True),
    )

    horizon = None
    if with_horizon:
        horizon = pd.DataFrame(
            {"Realised APY (custom)": [0.05, 0.04]}, index=["PoolA", "PoolB"]
        )

    paths = cross_section_report(
        repo,
        tmp_path,
        perf_fee_bps=0.0,
        mgmt_fee_bps=0.0,
        top_n=2,
        horizon_apys=horizon,
        returns=returns if with_returns else None,
        rolling_windows=(2,),
        periods_per_year=2,
    )

    base_keys = {"pools", "by_chain", "by_source", "by_stablecoin", "topN", "concentration"}
    assert base_keys.issubset(paths)

    pools_df = pd.read_csv(paths["pools"])
    if with_horizon:
        assert "Realised APY (custom)" in pools_df.columns
    else:
        assert "Realised APY (custom)" not in pools_df.columns

    if not with_returns:
        assert {"rolling_apy", "drawdowns", "drawdown_summary", "realised_vs_target"}.isdisjoint(paths)
        return

    # Rolling APY CSV
    rolling_df = pd.read_csv(paths["rolling_apy"], parse_dates=["timestamp"])
    assert sorted(rolling_df["window"].unique()) == [2]
    mask = (
        (rolling_df["name"] == "PoolA")
        & (rolling_df["window"] == 2)
        & (rolling_df["timestamp"] == pd.Timestamp("2024-01-08", tz="UTC"))
    )
    value = float(rolling_df.loc[mask, "rolling_apy"].iloc[0])
    expected_pool_a = ((1 + 0.01) * (1 + 0.02)) ** (2 / 2) - 1
    assert value == pytest.approx(expected_pool_a)

    # Drawdown time-series and summary
    drawdowns = pd.read_csv(paths["drawdowns"], parse_dates=["timestamp"])
    pool_b_dd = drawdowns[
        (drawdowns["name"] == "PoolB") & (drawdowns["timestamp"] == pd.Timestamp("2024-01-08", tz="UTC"))
    ]["drawdown"].iloc[0]
    expected_drawdown = (1 + 0.005) * (1 - 0.002) / (1 + 0.005) - 1
    assert pool_b_dd == pytest.approx(expected_drawdown)

    summary = pd.read_csv(paths["drawdown_summary"])
    pool_b_summary = summary[summary["name"] == "PoolB"].iloc[0]
    assert pool_b_summary["max_drawdown"] == pytest.approx(expected_drawdown)

    # Realised vs target APY comparison
    realised = pd.read_csv(paths["realised_vs_target"])
    pool_a_realised = realised[realised["name"] == "PoolA"].iloc[0]
    growth = (1 + returns["PoolA"]).prod()
    realised_expected = growth ** (2 / returns["PoolA"].count()) - 1
    assert pool_a_realised["realised_apy"] == pytest.approx(realised_expected)
    assert pool_a_realised["target_apy"] == pytest.approx(0.03)
    assert pool_a_realised["realised_minus_target"] == pytest.approx(
        realised_expected - 0.03
    )


def test_cross_section_report_requires_datetime_returns(tmp_path: Path) -> None:
    repo = PoolRepository(
        [
            Pool(
                name="PoolA",
                chain="Ethereum",
                stablecoin="USDC",
                tvl_usd=1_000_000,
                base_apy=0.03,
            )
        ]
    )

    returns = pd.DataFrame({"PoolA": [0.01, 0.02]}, index=[0, 1])

    with pytest.raises(ValueError, match="returns must be indexed by timestamps"):
        cross_section_report(repo, tmp_path, returns=returns)
