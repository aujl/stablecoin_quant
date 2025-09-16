from pathlib import Path

import pandas as pd
import pytest

from stable_yield_lab import (
    DataQualityError,
    DataQualityWarning,
    HistoricalCSVSource,
    Pipeline,
)


def test_historical_csv_parsing_and_alignment() -> None:
    csv_path = Path(__file__).resolve().parent.parent / "src" / "sample_yields.csv"
    src = HistoricalCSVSource(str(csv_path))
    df = Pipeline([src]).run_history()

    expected_columns = [
        "Aave USDT v3 (Polygon)",
        "Curve 3Pool Convex (ETH)",
        "Morpho USDC (ETH)",
    ]
    expected_index = pd.date_range("2023-01-02", "2024-07-08", freq="W-MON", tz="UTC", name="timestamp")

    assert list(df.columns) == expected_columns
    pd.testing.assert_index_equal(df.index, expected_index)
    assert df.shape == (len(expected_index), len(expected_columns))
    assert df.loc[pd.Timestamp("2024-01-08", tz="UTC"), "Aave USDT v3 (Polygon)"] == pytest.approx(0.001871)
    assert df.loc[pd.Timestamp("2024-01-15", tz="UTC"), "Morpho USDC (ETH)"] == pytest.approx(-0.01)

    for diag in src.last_diagnostics.values():
        assert diag.missing_periods == 0
        assert diag.filled_periods == 0
        assert diag.remaining_missing == 0


def test_historical_resample_and_fill_daily(tmp_path: Path) -> None:
    csv_path = tmp_path / "history.csv"
    csv_path.write_text(
        """timestamp,name,period_return\n"
        "2024-01-01,PoolA,0.010\n"
        "2024-01-03,PoolA,0.015\n"
        "2024-01-01,PoolB,0.020\n"
        "2024-01-04,PoolB,0.018\n"""
    )
    src = HistoricalCSVSource(str(csv_path), target_frequency="D", fill_strategy="ffill", min_observations=2)
    with pytest.warns(DataQualityWarning):
        df = Pipeline([src]).run_history()
    expected_index = pd.date_range("2024-01-01", "2024-01-04", freq="D", tz="UTC")
    assert list(df.index) == list(expected_index)
    assert df.loc[pd.Timestamp("2024-01-02", tz="UTC"), "PoolA"] == pytest.approx(0.010)
    assert df.loc[pd.Timestamp("2024-01-03", tz="UTC"), "PoolB"] == pytest.approx(0.020)
    diag_pool_a = src.last_diagnostics["PoolA"]
    assert diag_pool_a.missing_periods == 2
    assert diag_pool_a.filled_periods == 2
    assert diag_pool_a.remaining_missing == 0


def test_historical_min_observations_threshold(tmp_path: Path) -> None:
    csv_path = tmp_path / "history.csv"
    csv_path.write_text(
        """timestamp,name,period_return\n"
        "2024-01-01,PoolA,0.010\n"
        "2024-01-08,PoolA,0.012\n"""
    )
    src = HistoricalCSVSource(str(csv_path), target_frequency="W-MON", min_observations=3)
    with pytest.raises(DataQualityError, match="PoolA"):
        src.fetch()


def test_historical_zero_fill_strategy(tmp_path: Path) -> None:
    csv_path = tmp_path / "history.csv"
    csv_path.write_text(
        """timestamp,name,period_return\n"
        "2024-01-01,PoolA,0.010\n"
        "2024-01-04,PoolA,0.013\n"""
    )
    src = HistoricalCSVSource(str(csv_path), target_frequency="D", fill_strategy="zero", min_observations=1)
    with pytest.warns(DataQualityWarning):
        df = Pipeline([src]).run_history()
    assert df.loc[pd.Timestamp("2024-01-02", tz="UTC"), "PoolA"] == pytest.approx(0.0)
    assert df.loc[pd.Timestamp("2024-01-03", tz="UTC"), "PoolA"] == pytest.approx(0.0)
    diag_pool_a = src.last_diagnostics["PoolA"]
    assert diag_pool_a.filled_periods == 2
    assert diag_pool_a.remaining_missing == 0
