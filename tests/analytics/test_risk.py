from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import stable_yield_demo
from stable_yield_lab.analytics import risk as risk_metrics


def _write_pools_csv(path: Path, names: list[str]) -> None:
    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "name": name,
                "chain": "TestChain",
                "stablecoin": "USDC",
                "tvl_usd": 100_000 + i * 10_000,
                "base_apy": 0.07 + 0.01 * i,
                "reward_apy": 0.0,
                "is_auto": True,
                "source": "unit-test",
                "risk_score": 2.0,
                "timestamp": 1_700_000_000,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_history_csv(path: Path, rows: list[tuple[str, str, float]]) -> None:
    pd.DataFrame(rows, columns=["timestamp", "name", "period_return"]).to_csv(path, index=False)


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

    csv_path = tmp_path / "pools.csv"
    history_path = tmp_path / "history.csv"
    outdir = tmp_path / "out"
    _write_pools_csv(csv_path, ["PoolA"])
    _write_history_csv(
        history_path,
        [
            ("2024-01-01", "PoolA", 0.01),
            ("2024-01-08", "PoolA", 0.011),
        ],
    )
    monkeypatch.setenv("STABLE_YIELD_CSV", str(csv_path))
    monkeypatch.setenv("STABLE_YIELD_YIELDS_CSV", str(history_path))
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

    csv_path = tmp_path / "pools.csv"
    history_path = tmp_path / "history.csv"
    _write_pools_csv(csv_path, ["PoolA"])
    _write_history_csv(
        history_path,
        [
            ("2024-01-01", "PoolA", 0.01),
            ("2024-01-08", "PoolA", 0.011),
        ],
    )
    outdir = tmp_path / "out"
    monkeypatch.setenv("STABLE_YIELD_CSV", str(csv_path))
    monkeypatch.setenv("STABLE_YIELD_YIELDS_CSV", str(history_path))
    monkeypatch.setenv("STABLE_YIELD_OUTDIR", str(outdir))
    monkeypatch.setattr(sys, "argv", ["prog"])
    stable_yield_demo.main()

    assert not (outdir / "risk_stats.csv").exists()
    assert not (outdir / "efficient_frontier.csv").exists()


def test_demo_uses_realised_returns_for_risk_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "pools.csv"
    history_path = tmp_path / "history.csv"
    outdir = tmp_path / "out"
    _write_pools_csv(csv_path, ["PoolA", "PoolB"])
    _write_history_csv(
        history_path,
        [
            ("2024-01-01", "PoolA", 0.01),
            ("2024-01-08", "PoolA", 0.02),
            ("2024-01-01", "PoolB", 0.015),
            ("2024-01-08", "PoolB", 0.01),
        ],
    )

    captured: dict[str, pd.DataFrame] = {}

    def fake_summary(df: pd.DataFrame) -> pd.DataFrame:
        captured["summary"] = df.copy()
        return pd.DataFrame({"PoolA": [0.1], "PoolB": [0.2]})

    def fake_frontier(df: pd.DataFrame) -> pd.DataFrame:
        captured["frontier"] = df.copy()
        return pd.DataFrame({"PoolA": [0.6], "PoolB": [0.4]})

    monkeypatch.setattr(risk_metrics, "summary_statistics", fake_summary)
    monkeypatch.setattr(risk_metrics, "efficient_frontier", fake_frontier)
    monkeypatch.setenv("STABLE_YIELD_CSV", str(csv_path))
    monkeypatch.setenv("STABLE_YIELD_YIELDS_CSV", str(history_path))
    monkeypatch.setenv("STABLE_YIELD_OUTDIR", str(outdir))
    monkeypatch.setattr(sys, "argv", ["prog"])

    stable_yield_demo.main()

    assert "summary" in captured
    assert "frontier" in captured
    pd.testing.assert_frame_equal(captured["summary"], captured["frontier"])

    expected_index = pd.to_datetime(["2024-01-01", "2024-01-08"], utc=True)
    expected = pd.DataFrame(
        {
            "PoolA": [0.01, 0.02],
            "PoolB": [0.015, 0.01],
        },
        index=expected_index,
    )
    expected.index.name = "timestamp"
    expected.columns.name = "name"
    actual = captured["summary"].sort_index().sort_index(axis=1)
    expected = expected.sort_index().sort_index(axis=1)
    pd.testing.assert_frame_equal(actual, expected)


def test_demo_warns_when_history_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv_path = tmp_path / "pools.csv"
    history_path = tmp_path / "history.csv"
    _write_pools_csv(csv_path, ["PoolA"])
    _write_history_csv(history_path, [("2024-01-01", "PoolA", 0.01)])
    outdir = tmp_path / "out"

    monkeypatch.setenv("STABLE_YIELD_CSV", str(csv_path))
    monkeypatch.setenv("STABLE_YIELD_YIELDS_CSV", str(history_path))
    monkeypatch.setenv("STABLE_YIELD_OUTDIR", str(outdir))
    monkeypatch.setattr(sys, "argv", ["prog"])

    stable_yield_demo.main()

    warnings_path = outdir / "warnings.csv"
    assert warnings_path.is_file()
    warnings_df = pd.read_csv(warnings_path)
    assert "history" in warnings_df.loc[0, "message"].lower()
