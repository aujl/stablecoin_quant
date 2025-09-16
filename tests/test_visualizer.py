from pathlib import Path

import matplotlib
import pandas as pd

from stable_yield_lab import Visualizer

matplotlib.use("Agg")


def test_scatter_risk_return_creates_file(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "volatility": [0.1, 0.2, 0.15],
            "base_apy": [0.05, 0.1, 0.07],
            "tvl_usd": [1_000_000, 2_000_000, 1_500_000],
            "name": ["A", "B", "C"],
        }
    )
    out = tmp_path / "risk_return.png"
    Visualizer.scatter_risk_return(df, save_path=str(out), show=False)
    assert out.exists() and out.stat().st_size > 0


def test_line_yield_creates_file(tmp_path: Path) -> None:
    ts = pd.DataFrame(
        {
            "pool_a": [0.05, 0.06, 0.055],
            "pool_b": [0.04, 0.045, 0.05],
        },
        index=pd.date_range("2023-01-01", periods=3, freq="D"),
    )
    out = tmp_path / "yield.png"
    Visualizer.line_yield(ts, save_path=str(out), show=False)
    assert out.exists() and out.stat().st_size > 0


def test_line_nav_creates_file(tmp_path: Path) -> None:
    nav = pd.Series(
        [100.0, 101.5, 102.0],
        index=pd.date_range("2023-01-01", periods=3, freq="D"),
    )
    out = tmp_path / "nav.png"
    Visualizer.line_nav(nav, save_path=str(out), show=False)
    assert out.exists() and out.stat().st_size > 0


def test_hist_period_returns_creates_file(tmp_path: Path) -> None:
    returns = pd.DataFrame(
        {
            "PoolA": [0.01, 0.015, -0.005, 0.012],
            "PoolB": [0.008, 0.01, 0.011, 0.009],
        },
        index=pd.date_range("2023-01-01", periods=4, freq="W"),
    )
    out = tmp_path / "hist_returns.png"
    Visualizer.hist_period_returns(returns, bins=5, save_path=str(out), show=False)
    assert out.exists() and out.stat().st_size > 0


def test_boxplot_rolling_apy_creates_file(tmp_path: Path) -> None:
    returns = pd.DataFrame(
        {
            "PoolA": [0.01, 0.012, -0.004, 0.015, 0.009],
            "PoolB": [0.008, 0.007, 0.01, 0.011, 0.012],
        },
        index=pd.date_range("2023-01-01", periods=5, freq="W"),
    )
    out = tmp_path / "rolling_apy_box.png"
    Visualizer.boxplot_rolling_apy(
        returns,
        window=3,
        periods_per_year=52,
        save_path=str(out),
        show=False,
    )
    assert out.exists() and out.stat().st_size > 0
