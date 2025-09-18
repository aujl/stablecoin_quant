from __future__ import annotations

from pathlib import Path

import pytest

from stable_yield_lab.core import PoolRepository
from stable_yield_lab.sources import MorphoSource

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture()
def morpho_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MorphoSource:
    cache = tmp_path / "morpho.json"
    cache.write_bytes((FIXTURES / "morpho_markets.json").read_bytes())

    def _unexpected_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access should use cached response")

    monkeypatch.setattr(
        "stable_yield_lab.sources.morpho.urllib.request.urlopen",
        _unexpected_network,
    )
    return MorphoSource(cache_path=str(cache))


def test_morpho_converts_supply_apy_to_decimal(morpho_source: MorphoSource) -> None:
    pools = morpho_source.fetch()
    assert pools
    lookup = {pool.name: pool for pool in pools}
    key = "USDC-RST0001-UAE-DUBAI-BAYZ BY DANUBE-01"
    assert lookup[key].stablecoin == "USDC"
    assert lookup[key].base_apy == pytest.approx(0.0018021507649899118)


def test_morpho_auto_only_filter_preserves_all(morpho_source: MorphoSource) -> None:
    pools = morpho_source.fetch()
    repo = PoolRepository(pools)
    auto_only = repo.filter(auto_only=True)
    assert len(auto_only) == len(pools)
    assert all(pool.is_auto for pool in auto_only)
