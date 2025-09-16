from pathlib import Path

import pandas as pd
import pytest

from stable_yield_lab import HistoricalCSVSource, Pipeline


def test_historical_csv_parsing_and_alignment() -> None:
    csv_path = Path(__file__).resolve().parent.parent / "src" / "sample_yields.csv"
    src = HistoricalCSVSource(str(csv_path))
    df = Pipeline([src]).run_history()

    expected = {"Morpho USDC (ETH)", "Aave USDT v3 (Polygon)", "Curve 3Pool Convex (ETH)"}
    assert expected.issubset(df.columns)
    assert df.index.tz is not None
    assert df.shape[0] >= 52

    morpho_window = df.loc[
        pd.Timestamp("2023-03-05", tz="UTC") : pd.Timestamp("2023-04-09", tz="UTC"),
        "Morpho USDC (ETH)",
    ]
    assert morpho_window.loc[pd.Timestamp("2023-03-12", tz="UTC")] == pytest.approx(-0.018, rel=1e-9)
    assert morpho_window.min() < 0

    curve_stress = df.loc[
        pd.Timestamp("2023-09-03", tz="UTC") : pd.Timestamp("2023-09-24", tz="UTC"),
        "Curve 3Pool Convex (ETH)",
    ]
    assert curve_stress.loc[pd.Timestamp("2023-09-10", tz="UTC")] == pytest.approx(-0.013, rel=1e-9)
    assert (df["Aave USDT v3 (Polygon)"] < 0).sum() >= 2
