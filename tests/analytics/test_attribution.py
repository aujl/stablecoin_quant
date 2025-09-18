from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stable_yield_lab.analytics import attribution
from stable_yield_lab.core import Pool, PoolRepository
from stable_yield_lab.reporting import cross_section_report


def _synthetic_returns() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=4, freq="W")
    return pd.DataFrame(
        {
            "PoolA": [0.02, 0.01, -0.005, 0.015],
            "PoolB": [0.01, 0.012, 0.0, 0.008],
        },
        index=idx,
    )


def _synthetic_weights(returns: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PoolA": [0.6, 0.4],
            "PoolB": [0.4, 0.6],
        },
        index=[returns.index[0], returns.index[2]],
    )


def _synthetic_repo() -> PoolRepository:
    repo = PoolRepository(
        [
            Pool(
                name="PoolA",
                chain="TestChain",
                stablecoin="USDC",
                tvl_usd=1_000_000,
                base_apy=0.10,
                reward_apy=0.0,
                is_auto=True,
                source="synthetic",
            ),
            Pool(
                name="PoolB",
                chain="TestChain",
                stablecoin="USDT",
                tvl_usd=800_000,
                base_apy=0.08,
                reward_apy=0.0,
                is_auto=True,
                source="synthetic",
            ),
        ]
    )
    return repo


def test_compute_attribution_pool_and_window() -> None:
    returns = _synthetic_returns()
    weights = _synthetic_weights(returns)

    result = attribution.compute_attribution(returns, weights, periods_per_year=52)

    total_return = result.portfolio["total_return"]
    realized_apy = result.portfolio["realized_apy"]
    assert pytest.approx(total_return, rel=1e-6) == result.by_pool["return_contribution"].sum()
    assert pytest.approx(realized_apy, rel=1e-6) == result.by_pool["apy_contribution"].sum()
    assert pytest.approx(realized_apy, rel=1e-6) == result.by_window["apy_contribution"].sum()

    pool = result.by_pool.set_index("pool")
    assert pool.index.tolist() == ["PoolA", "PoolB"]
    assert pytest.approx(pool.loc["PoolA", "avg_weight"], rel=1e-6) == 0.5
    assert pytest.approx(pool.loc["PoolB", "avg_weight"], rel=1e-6) == 0.5
    assert pool.loc["PoolA", "return_contribution"] > pool.loc["PoolB", "return_contribution"]

    window = result.by_window
    assert len(window) == 2
    assert window.loc[0, "periods"] == 2
    assert window.loc[1, "periods"] == 2
    assert window.loc[0, "window_start"] < window.loc[1, "window_start"]
    assert all(window["window_end"] >= window["window_start"])
    assert all(window["window_return"].abs() < 0.1)


def test_cross_section_report_writes_attribution(tmp_path: Path) -> None:
    repo = _synthetic_repo()
    returns = _synthetic_returns()

    outdir = tmp_path / "report"
    outdir.mkdir()

    paths = cross_section_report(
        repo,
        outdir,
        perf_fee_bps=0.0,
        mgmt_fee_bps=0.0,
        top_n=5,
        returns=returns,
    )

    expected_outputs = {
        "pools",
        "warnings",
        "by_chain",
        "by_source",
        "by_stablecoin",
        "topN",
        "concentration",
    }
    assert expected_outputs.issubset(paths.keys())

    pools_df = pd.read_csv(paths["pools"])
    warnings_df = pd.read_csv(paths["warnings"])
    concentration_df = pd.read_csv(paths["concentration"])

    assert {"realised_apy", "realised_apy_observations", "realised_apy_warning"}.issubset(
        pools_df.columns
    )
    assert warnings_df.shape[0] == pools_df["name"].nunique()
    assert warnings_df["message"].str.contains("observations", case=False).all()

    required_metrics = {
        "scope",
        "hhi",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "negative_period_share",
    }
    assert required_metrics.issubset(concentration_df.columns)

    pool_scope = concentration_df.loc[concentration_df["scope"] == "pool:PoolA"]
    assert not pool_scope.empty
    assert pool_scope["sharpe_ratio"].notna().all()

    chain_scope = concentration_df.loc[concentration_df["scope"] == "chain:TestChain"]
    assert not chain_scope.empty
    assert chain_scope["negative_period_share"].between(0.0, 1.0).all()
