from pathlib import Path

import pandas as pd
import pytest

from stable_yield_lab.pipeline import Pipeline
from stable_yield_lab.sources import HistoricalCSVSource


def test_historical_csv_parsing_and_alignment() -> None:
    csv_path = Path(__file__).resolve().parent.parent / "src" / "sample_yields.csv"
    src = HistoricalCSVSource(str(csv_path))
    df = Pipeline([src]).run_history()

    expected_columns = {
        "Aave USDT v3 (Polygon)",
        "Curve 3Pool Convex (ETH)",
        "Morpho USDC (ETH)",
    }
    assert set(df.columns) == expected_columns
    assert df.index.tz is not None
    assert df.shape == (80, 3)

    first_ts = pd.Timestamp("2023-01-01", tz="UTC")
    last_ts = pd.Timestamp("2024-07-07", tz="UTC")
    assert df.index[0] == first_ts
    assert df.index[-1] == last_ts

    assert df.loc[first_ts, "Morpho USDC (ETH)"] == pytest.approx(0.0019, rel=1e-9)
    assert df.loc[first_ts, "Aave USDT v3 (Polygon)"] == pytest.approx(0.00133, rel=1e-9)
    assert df.loc[last_ts, "Morpho USDC (ETH)"] == pytest.approx(0.003164, rel=1e-6)
    assert df.loc[last_ts, "Curve 3Pool Convex (ETH)"] == pytest.approx(0.002438, rel=1e-6)
