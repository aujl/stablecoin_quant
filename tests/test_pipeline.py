from pathlib import Path

from stable_yield_lab import Pipeline
from stable_yield_lab.sources import CSVSource
from stable_yield_lab.core import Pool
from stable_yield_lab.risk_scoring import calculate_risk_score


def test_pipeline_loads_sample_csv() -> None:
    csv_path = Path(__file__).resolve().parent.parent / "src" / "sample_pools.csv"
    src = CSVSource(str(csv_path))
    repo = Pipeline([src]).run()
    assert len(repo) > 0


def test_pipeline_applies_risk_scoring() -> None:
    class DummySource:
        def fetch(self) -> list[Pool]:
            return [Pool("P1", "Polygon", "USDC", 1.0, 0.1)]

    repo = Pipeline([DummySource()]).run()
    pools = list(repo._pools)
    assert len(pools) == 1
    expected = calculate_risk_score(0.7, 0, 0.0)
    assert pools[0].risk_score == expected
