from pathlib import Path

import pandas as pd

from stable_yield_lab import Pipeline
from stable_yield_lab.sources import HistoricalCSVSource


def test_historical_csv_parsing_and_alignment() -> None:
    csv_path = Path(__file__).resolve().parent.parent / "src" / "sample_yields.csv"
    src = HistoricalCSVSource(str(csv_path))
    df = Pipeline([src]).run_history()
    assert list(df.columns) == ["PoolA", "PoolB"]
    assert df.index.tz is not None
    assert df.shape == (3, 2)
    assert pd.isna(df.loc[pd.Timestamp("2024-01-08", tz="UTC"), "PoolB"])
    assert pd.isna(df.loc[pd.Timestamp("2024-01-15", tz="UTC"), "PoolA"])
    assert df.loc[pd.Timestamp("2024-01-08", tz="UTC"), "PoolA"] == 0.011
    assert df.loc[pd.Timestamp("2024-01-01", tz="UTC"), "PoolB"] == 0.008
