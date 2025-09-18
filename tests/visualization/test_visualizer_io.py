from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

from stable_yield_lab.visualization import Visualizer

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


def test_nav_with_benchmarks_overlays_labels(tmp_path: Path) -> None:
    plt.close("all")
    returns = pd.DataFrame(
        {
            "PoolA": [0.01, 0.02, -0.005],
            "PoolB": [0.015, -0.01, 0.012],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    out = tmp_path / "nav_overlay.png"
    labels = {
        "rebalance": "Rebalanced",
        "buy_and_hold": "Buy & Hold",
        "cash": "Cash (0%)",
    }
    nav_df = Visualizer.nav_with_benchmarks(
        returns,
        initial_investment=1_000.0,
        cash_returns=0.0,
        labels=labels,
        save_path=str(out),
        show=False,
    )

    assert out.exists() and out.stat().st_size > 0
    assert list(nav_df.columns) == [labels["rebalance"], labels["buy_and_hold"], labels["cash"]]

    fig = plt.gcf()
    ax = fig.axes[-1]
    legend = ax.get_legend()
    assert legend is not None
    legend_labels = {text.get_text() for text in legend.get_texts()}
    assert legend_labels == set(nav_df.columns)
    plotted_labels = [line.get_label() for line in ax.get_lines()]
    assert plotted_labels == nav_df.columns.tolist()
