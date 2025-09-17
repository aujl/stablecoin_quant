from __future__ import annotations

import math
import statistics
from pathlib import Path

import pandas as pd
import pytest

from stable_yield_lab.core.models import Pool
from stable_yield_lab.core.repositories import PoolRepository
from stable_yield_lab.reporting import cross_section_report


def _expected_metrics(values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values)
    std = statistics.stdev(values)
    sharpe = mean / std if std > 0 else math.nan
    downside = [v for v in values if v < 0]
    if downside:
        downside_std = math.sqrt(sum(v * v for v in downside) / len(downside))
        sortino = mean / downside_std if downside_std > 0 else math.nan
    else:
        sortino = math.nan
    cumulative = 1.0
    max_nav = 1.0
    max_dd = 0.0
    for r in values:
        cumulative *= 1 + r
        if cumulative > max_nav:
            max_nav = cumulative
        drawdown = cumulative / max_nav - 1
        if drawdown < max_dd:
            max_dd = drawdown
    negative_share = sum(1 for v in values if v < 0) / len(values)
    return {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "negative_period_share": negative_share,
    }


def test_cross_section_report_adds_risk_metrics(tmp_path: Path) -> None:
    repo = PoolRepository(
        [
            Pool(
                name="PoolA",
                chain="Ethereum",
                stablecoin="USDC",
                tvl_usd=100.0,
                base_apy=0.1,
            ),
            Pool(
                name="PoolB",
                chain="Polygon",
                stablecoin="USDT",
                tvl_usd=300.0,
                base_apy=0.08,
            ),
        ]
    )

    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    returns = pd.DataFrame(
        {
            "PoolA": [0.1, -0.05, 0.0],
            "PoolB": [0.02, 0.03, -0.01],
        },
        index=index,
    )

    paths = cross_section_report(repo, tmp_path, returns=returns)
    conc = pd.read_csv(paths["concentration"])
    result = conc.set_index("scope")

    expected_total_returns = [
        0.25 * 0.1 + 0.75 * 0.02,
        0.25 * -0.05 + 0.75 * 0.03,
        0.25 * 0.0 + 0.75 * -0.01,
    ]

    expected_total = _expected_metrics(expected_total_returns)
    total_row = result.loc["total"]
    for key, value in expected_total.items():
        assert total_row[key] == pytest.approx(value, rel=1e-6, abs=1e-6)

    pool_a_row = result.loc["pool:PoolA"]
    expected_pool_a = _expected_metrics([0.1, -0.05, 0.0])
    for key, value in expected_pool_a.items():
        assert pool_a_row[key] == pytest.approx(value, rel=1e-6, abs=1e-6)
    assert math.isnan(pool_a_row["hhi"])

    chain_row = result.loc["chain:Ethereum"]
    for key, value in expected_pool_a.items():
        assert chain_row[key] == pytest.approx(value, rel=1e-6, abs=1e-6)

    assert result.loc["total", "hhi"] == pytest.approx(0.625)
    assert result.loc["chain:Ethereum", "hhi"] == pytest.approx(1.0)
    assert result.loc["stablecoin:USDC", "hhi"] == pytest.approx(1.0)


def test_cross_section_report_without_returns(tmp_path: Path) -> None:
    repo = PoolRepository(
        [
            Pool(
                name="Solo",
                chain="Ethereum",
                stablecoin="USDC",
                tvl_usd=1_000.0,
                base_apy=0.05,
            )
        ]
    )

    paths = cross_section_report(repo, tmp_path)
    conc = pd.read_csv(paths["concentration"])

    metrics_cols = {
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "negative_period_share",
    }
    assert metrics_cols.issubset(conc.columns)
    assert conc[list(metrics_cols)].isna().all().all()
