from __future__ import annotations

from pathlib import Path

from stable_yield_lab.core import PoolRepository
from stable_yield_lab.sources import CSVSource, HistoricalCSVSource

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_csv_source_preserves_values_and_flags() -> None:
    src = CSVSource(str(FIXTURES / "pools.csv"))
    pools = src.fetch()
    assert [p.name for p in pools] == ["Auto Pool", "Manual Pool"]
    assert pools[0].stablecoin == "USDC"
    assert pools[0].base_apy == 0.05
    assert pools[1].is_auto is False

    repo = PoolRepository(pools)
    auto_only = repo.filter(auto_only=True)
    assert [p.name for p in auto_only] == ["Auto Pool"]


def test_historical_csv_source_parses_returns_with_timezone() -> None:
    src = HistoricalCSVSource(str(FIXTURES / "returns_negative.csv"))
    rows = src.fetch()
    assert rows[0].timestamp.tz is not None
    assert rows[1].period_return == -0.03
    assert {row.name for row in rows} == {"PoolNeg", "PoolFlat"}
