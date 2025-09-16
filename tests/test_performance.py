from collections.abc import Callable
from pathlib import Path

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


@pytest.fixture
def load_history() -> Callable[[str], pd.DataFrame]:
    """Load historical returns from the fixtures directory."""

    base_dir = Path(__file__).resolve().parent / "fixtures"

    def _loader(filename: str) -> pd.DataFrame:
        src = HistoricalCSVSource(str(base_dir / filename))
        return Pipeline([src]).run_history()

    return _loader

def test_nav_and_yield_trajectories(tmp_path: Path) -> None:
    csv_path = Path(__file__).resolve().parent.parent / "src" / "sample_yields.csv"
    returns = Pipeline([HistoricalCSVSource(str(csv_path))]).run_history()

    nav = performance.nav_trajectories(returns, initial_investment=100.0)
    yield_df = performance.yield_trajectories(returns)

    assert nav.loc[pd.Timestamp("2024-01-08", tz="UTC"), "PoolA"] == pytest.approx(
        102.111, rel=1e-6
    )
    assert yield_df.loc[pd.Timestamp("2024-01-15", tz="UTC"), "PoolB"] == pytest.approx(
        0.015056, rel=1e-6
    )

    nav_path = tmp_path / "nav.png"
    yield_path = tmp_path / "yield.png"
    Visualizer.line_chart(
        nav, title="NAV over time", ylabel="NAV (USD)", save_path=str(nav_path), show=False
    )
    Visualizer.line_chart(
        yield_df * 100.0,
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


@pytest.mark.parametrize(
    ("fixture_name", "return_sequences", "constant_checks", "metadata"),
    [
        pytest.param(
            "returns_irregular.csv",
            {
                "PoolIrregularA": [0.010, 0.015, -0.005],
                "PoolIrregularB": [0.020, 0.010],
            },
            [],
            {"check_irregular_spacing": True},
            id="irregular-sampling",
        ),
        pytest.param(
            "returns_gaps.csv",
            {
                "PoolGapA": [0.010, 0.020, 0.0, 0.0],
                "PoolGapB": [0.015, 0.0, -0.005, 0.0],
            },
            [
                (
                    pd.Timestamp("2024-02-07T00:00:00Z"),
                    "PoolGapA",
                    (1.0 + 0.010) * (1.0 + 0.020) - 1.0,
                ),
                (
                    pd.Timestamp("2024-02-04T00:00:00Z"),
                    "PoolGapB",
                    (1.0 + 0.015) - 1.0,
                ),
            ],
            {},
            id="gap-filling",
        ),
        pytest.param(
            "returns_negative.csv",
            {
                "PoolNeg": [0.020, -0.030, 0.010],
                "PoolFlat": [0.0, 0.0, -0.020],
            },
            [],
            {"expected_negative": {"PoolNeg", "PoolFlat"}},
            id="negative-returns",
        ),
    ],
)
def test_yield_trajectories_handle_messy_sampling(
    load_history: Callable[[str], pd.DataFrame],
    fixture_name: str,
    return_sequences: dict[str, list[float]],
    constant_checks: list[tuple[pd.Timestamp, str, float]],
    metadata: dict[str, object],
) -> None:
    returns = load_history(fixture_name)

    assert not returns.empty
    assert returns.index.tz is not None
    assert returns.index.is_monotonic_increasing

    if metadata.get("check_irregular_spacing"):
        diffs = returns.index.to_series().diff().dropna().unique()
        assert len(diffs) > 1

    yields = performance.yield_trajectories(returns)
    navs = performance.nav_trajectories(returns, initial_investment=100.0)

    assert yields.index.equals(returns.index)
    assert navs.index.equals(returns.index)

    for pool, sequence in return_sequences.items():
        growth = 1.0
        for r in sequence:
            growth *= 1.0 + float(r)
        expected_final = growth - 1.0

        final_yield = yields.iloc[-1][pool]
        final_nav = navs.iloc[-1][pool]

        assert final_yield == pytest.approx(expected_final, rel=1e-9)
        assert final_nav == pytest.approx(100.0 * (1.0 + expected_final), rel=1e-9)

    for timestamp, pool, expected_yield in constant_checks:
        assert yields.loc[timestamp, pool] == pytest.approx(expected_yield, rel=1e-9)
        assert navs.loc[timestamp, pool] == pytest.approx(
            100.0 * (1.0 + expected_yield), rel=1e-9
        )

    for pool in metadata.get("expected_negative", set()):
        assert yields.iloc[-1][pool] < 0.0
        assert navs.iloc[-1][pool] < 100.0
