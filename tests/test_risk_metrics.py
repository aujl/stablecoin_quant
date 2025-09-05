import builtins
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import stable_yield_demo
from stable_yield_lab import risk_metrics


def test_requires_riskfolio(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import: Any = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "riskfolio":
            raise ImportError("riskfolio missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError):
        risk_metrics.summary_statistics(pd.DataFrame())


def test_summary_and_frontier_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePortfolio:
        def __init__(self, returns: pd.DataFrame) -> None:
            self.returns = returns

        def assets_stats(self, **kwargs: object) -> None:
            return None

        def efficient_frontier(
            self,
            model: str,
            rm: str,
            obj: str,
            rf: float,
            l: float,  # noqa: E741 - match riskfolio signature
            points: int,
        ) -> pd.DataFrame:
            return pd.DataFrame([[0.6, 0.4], [0.5, 0.5]], columns=["A", "B"], index=[0, 1])

    def fake_sharpe_risk(returns: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"A": [0.1], "B": [0.2]})

    fake_riskfolio = types.SimpleNamespace(Portfolio=FakePortfolio, Sharpe_Risk=fake_sharpe_risk)
    monkeypatch.setitem(sys.modules, "riskfolio", fake_riskfolio)

    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.03, 0.04]})
    stats = risk_metrics.summary_statistics(returns)
    frontier = risk_metrics.efficient_frontier(returns)

    assert list(stats.columns) == ["A", "B"]
    assert frontier.shape == (2, 2)
    assert all(abs(frontier.sum(axis=1) - 1.0) < 1e-9)


def test_demo_writes_risk_csvs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stats_df = pd.DataFrame({"A": [0.1]})
    frontier_df = pd.DataFrame({"A": [1.0]})
    monkeypatch.setattr(risk_metrics, "summary_statistics", lambda df: stats_df)
    monkeypatch.setattr(risk_metrics, "efficient_frontier", lambda df: frontier_df)

    csv_path = Path(__file__).resolve().parents[1] / "src/sample_pools.csv"
    outdir = tmp_path / "out"
    monkeypatch.setenv("STABLE_YIELD_CSV", str(csv_path))
    monkeypatch.setenv("STABLE_YIELD_OUTDIR", str(outdir))
    monkeypatch.setattr(sys, "argv", ["prog"])
    stable_yield_demo.main()

    assert (outdir / "risk_stats.csv").is_file()
    assert (outdir / "efficient_frontier.csv").is_file()


def test_demo_skips_risk_metrics_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - raises
        raise RuntimeError("riskfolio missing")

    monkeypatch.setattr(risk_metrics, "summary_statistics", fail)
    monkeypatch.setattr(risk_metrics, "efficient_frontier", fail)

    csv_path = Path(__file__).resolve().parents[1] / "src/sample_pools.csv"
    outdir = tmp_path / "out"
    monkeypatch.setenv("STABLE_YIELD_CSV", str(csv_path))
    monkeypatch.setenv("STABLE_YIELD_OUTDIR", str(outdir))
    monkeypatch.setattr(sys, "argv", ["prog"])
    stable_yield_demo.main()

    assert not (outdir / "risk_stats.csv").exists()
    assert not (outdir / "efficient_frontier.csv").exists()
