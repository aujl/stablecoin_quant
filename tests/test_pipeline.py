from pathlib import Path

from stable_yield_lab import CSVSource, Pipeline


def test_pipeline_loads_sample_csv() -> None:
    csv_path = Path(__file__).resolve().parent.parent / "src" / "sample_pools.csv"
    src = CSVSource(str(csv_path))
    repo = Pipeline([src]).run()
    assert len(repo) > 0
