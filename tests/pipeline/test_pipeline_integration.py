from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stable_yield_lab.pipeline import Pipeline
from stable_yield_lab.risk_scoring import score_pool
from stable_yield_lab.sources import CSVSource, HistoricalCSVSource


class FailingSource:
    def fetch(self) -> list[object]:
        raise RuntimeError("boom")


class FailingHistoricalSource:
    def fetch(self) -> list[object]:
        raise RuntimeError("history boom")


@pytest.fixture(scope="module")
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_pipeline_run_scores_snapshot_sources(project_root: Path) -> None:
    csv_path = project_root / "src" / "sample_pools.csv"
    pipeline = Pipeline([CSVSource(str(csv_path))])
    repo = pipeline.run()
    pools = list(repo)
    assert pools, "expected repository to contain pools"
    for pool in pools:
        expected = score_pool(pool).risk_score
        assert pool.risk_score == pytest.approx(expected)


def test_pipeline_run_history_returns_timeseries(project_root: Path) -> None:
    yields_path = project_root / "src" / "sample_yields.csv"
    pipeline = Pipeline([HistoricalCSVSource(str(yields_path))])
    returns = pipeline.run_history()
    assert not returns.empty
    csv_df = pd.read_csv(yields_path)
    expected_names = sorted(set(csv_df["name"]))
    assert sorted(returns.columns.tolist()) == expected_names
    assert returns.index.is_monotonic_increasing


def test_pipeline_logs_and_recovers_from_source_failure(
    caplog: pytest.LogCaptureFixture, project_root: Path
) -> None:
    csv_path = project_root / "src" / "sample_pools.csv"
    pipeline = Pipeline([FailingSource(), CSVSource(str(csv_path))])
    with caplog.at_level("WARNING", logger="stable_yield_lab.pipeline"):
        repo = pipeline.run()
    assert any("FailingSource" in rec.message for rec in caplog.records)
    assert len(repo) > 0


def test_pipeline_run_history_logs_and_recovers(
    caplog: pytest.LogCaptureFixture, project_root: Path
) -> None:
    yields_path = project_root / "src" / "sample_yields.csv"
    pipeline = Pipeline([HistoricalCSVSource(str(yields_path)), FailingHistoricalSource()])
    with caplog.at_level("WARNING", logger="stable_yield_lab.pipeline"):
        returns = pipeline.run_history()
    assert any("FailingHistoricalSource" in rec.message for rec in caplog.records)
    assert not returns.empty


def test_pipeline_shim_exposes_same_class() -> None:
    from stable_yield_lab import Pipeline as LegacyPipeline

    assert LegacyPipeline is Pipeline
