from pathlib import Path

import matplotlib
import pandas as pd

from stable_yield_lab import Visualizer, rebalance

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


def _synthetic_rebalance() -> rebalance.RebalanceResult:
    returns = pd.DataFrame(
        {
            "PoolA": [0.01, -0.004, 0.012, 0.003],
            "PoolB": [0.008, 0.006, -0.002, 0.005],
            "PoolC": [0.005, 0.007, 0.004, -0.001],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="W"),
    )
    return rebalance.run_rebalance(returns, trading_cost_bps=10.0)


def test_plot_weight_schedule_creates_file(tmp_path: Path) -> None:
    result = _synthetic_rebalance()
    out = tmp_path / "weights.png"
    Visualizer.plot_weight_schedule(result.target_weights, save_path=str(out), show=False)
    assert out.exists() and out.stat().st_size > 0


def test_plot_turnover_creates_file(tmp_path: Path) -> None:
    result = _synthetic_rebalance()
    turnover_df = pd.DataFrame({
        "turnover": result.turnover,
        "fees": result.fees,
    })
    out = tmp_path / "turnover.png"
    Visualizer.plot_turnover(turnover_df, save_path=str(out), show=False)
    assert out.exists() and out.stat().st_size > 0
