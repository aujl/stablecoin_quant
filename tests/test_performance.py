from pathlib import Path
import json

import pandas as pd
import pytest

from stable_yield_lab import (
    HistoricalCSVSource,
    Pipeline,
    Visualizer,
    cumulative_return,
    nav_series,
    performance,
)

def test_nav_and_yield_trajectories(tmp_path: Path) -> None:
    csv_path = Path(__file__).resolve().parent.parent / "src" / "sample_yields.csv"
    returns = Pipeline([HistoricalCSVSource(str(csv_path))]).run_history()

    window = returns.loc[
        pd.Timestamp("2023-01-01", tz="UTC") : pd.Timestamp("2023-03-26", tz="UTC"),
        ["Morpho USDC (ETH)", "Aave USDT v3 (Polygon)"],
    ]

    nav = performance.nav_trajectories(window, initial_investment=100.0)
    yield_df = performance.yield_trajectories(window)

    assert nav.loc[pd.Timestamp("2023-02-12", tz="UTC"), "Morpho USDC (ETH)"] == pytest.approx(
        101.60988054, rel=1e-6
    )
    assert nav.loc[pd.Timestamp("2023-03-26", tz="UTC"), "Aave USDT v3 (Polygon)"] == pytest.approx(
        102.19848456, rel=1e-6
    )
    assert yield_df.loc[pd.Timestamp("2023-03-26", tz="UTC"), "Morpho USDC (ETH)"] == pytest.approx(
        -0.01167922286, rel=1e-6
    )
    assert yield_df.loc[pd.Timestamp("2023-03-26", tz="UTC"), "Aave USDT v3 (Polygon)"] == pytest.approx(
        0.02198484563, rel=1e-6
    )

    nav_path = tmp_path / "nav.png"
    yield_path = tmp_path / "yield.png"
    Visualizer.line_chart(
        nav.iloc[:12],
        title="NAV over time",
        ylabel="NAV (USD)",
        save_path=str(nav_path),
        show=False,
    )
    Visualizer.line_chart(
        (yield_df.iloc[:12]) * 100.0,
        title="Yield over time",
        ylabel="Yield (%)",
        save_path=str(yield_path),
        show=False,
    )

    assert nav_path.is_file()
    assert yield_path.is_file()


def test_cumulative_return_matches_manual_compounding() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    series = pd.Series([0.1, -0.05, 0.02], index=dates)
    expected = (1.0 + series).cumprod() - 1.0
    result = cumulative_return(series)
    pd.testing.assert_series_equal(result, expected)


def test_cumulative_return_empty_series() -> None:
    empty = pd.Series(dtype=float)
    result = cumulative_return(empty)
    assert result.empty


def test_nav_series_with_weights_and_initial_scaling() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    returns = pd.DataFrame(
        {
            "A": [0.1, 0.0, 0.02],
            "B": [0.0, 0.1, -0.01],
        },
        index=dates,
    )
    weights = pd.Series({"A": 0.6, "B": 0.4})
    nav_100 = nav_series(returns, weights, initial=100.0)
    nav_1 = nav_series(returns, weights, initial=1.0)

    clean_weights = weights / weights.sum()
    expected = 100.0 * (1.0 + returns.mul(clean_weights, axis=1).sum(axis=1)).cumprod()

    pd.testing.assert_series_equal(nav_100, expected)
    pd.testing.assert_series_equal(nav_100 / 100.0, nav_1)


def test_nav_series_defaults_and_empty() -> None:
    empty = pd.DataFrame()
    result = nav_series(empty, None, initial=1.0)
    assert result.empty

    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    returns = pd.DataFrame({"A": [0.05, 0.0], "B": [0.0, 0.05]}, index=dates)
    result = nav_series(returns, None, initial=1.0)

    expected_returns = returns.mul(0.5, axis=1).sum(axis=1)
    expected_nav = (1.0 + expected_returns).cumprod()
    pd.testing.assert_series_equal(result, expected_nav)


def test_sample_history_matches_expected_metrics() -> None:
    base_path = Path(__file__).resolve().parents[1]
    returns_csv = base_path / "src" / "sample_yields.csv"
    history_csv = base_path / "src" / "sample_pools_history.csv"
    expected_json = Path(__file__).resolve().parent / "fixtures" / "sample_history_expected.json"

    returns = Pipeline([HistoricalCSVSource(str(returns_csv))]).run_history()
    history = pd.read_csv(history_csv)
    expected = json.loads(expected_json.read_text())

    for row in history.to_dict(orient="records"):
        name = row["name"]
        start = pd.Timestamp(row["history_start"]).tz_localize("UTC")
        end = pd.Timestamp(row["history_end"]).tz_localize("UTC")
        series = returns.loc[start:end, name].dropna()

        assert len(series) == int(row["observations"])

        realized_return = float((1.0 + series).prod() - 1.0)
        realized_apy = float((1.0 + realized_return) ** (52 / len(series)) - 1.0)
        realized_vol = float(series.std(ddof=1) * (52**0.5))
        nav = (1.0 + series).cumprod()
        drawdown = (nav / nav.cummax()) - 1.0
        max_drawdown = float(drawdown.min())
        last_period_return = float(series.iloc[-1])

        assert realized_return == pytest.approx(row["realized_return_52w"], rel=1e-6, abs=1e-6)
        assert realized_apy == pytest.approx(row["realized_apy_52w"], rel=1e-6, abs=1e-6)
        assert realized_vol == pytest.approx(row["realized_volatility_52w"], rel=1e-6, abs=1e-6)
        assert max_drawdown == pytest.approx(row["max_drawdown_52w"], rel=1e-6, abs=1e-6)
        assert last_period_return == pytest.approx(row["last_period_return"], rel=1e-6, abs=1e-6)

        expected_metrics = expected[name]
        for key, value in expected_metrics.items():
            if isinstance(value, float):
                assert row[key] == pytest.approx(value, rel=1e-6, abs=1e-6)
            else:
                assert row[key] == value
